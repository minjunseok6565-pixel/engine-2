from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import warnings

from .core import clamp


# ---------------------------------------------------------------------------
# Fatigue scaling (Issue #9)
#
# 목표
# - 9-A: 피로 곡선을 더 강하게(비선형) 만들어, 낮은 energy에서 체감이 확 커지게
# - 9-B: 스탯별 피로 민감도 차등(수비/핸들/패스/3점 더 민감, 포스트/피지컬 덜 민감)
#
# 적용 방식
# - scale = floor + (1 - floor) * (energy ** gamma)
#   (energy=1이면 1.0, energy=0이면 floor까지 하락)
# - 스탯 키(key)에 따라 (floor, gamma) 프로필을 다르게 선택
# ---------------------------------------------------------------------------

# Base profile (대부분의 스탯)
# - floor: 레드존(crit_e 이상)에서의 기본 하한
# - gamma: 곡선 형태(에너지 감소에 따른 하락 기울기)
# - floor_min: 레드존에서 energy=0에 가까울수록 추가로 내려가는 최저 하한
# - crit_e / crit_pow: 레드존 시작점 및 가속 곡선
FATIGUE_PROFILE_BASE = {"floor": 0.83, "gamma": 1.15, "floor_min": 0.58, "crit_e": 0.26, "crit_pow": 1.5}

# High sensitivity (피로 영향 큼): 수비/핸들/패스/3점
FATIGUE_PROFILE_HIGH = {"floor": 0.80, "gamma": 1.20, "floor_min": 0.52, "crit_e": 0.28, "crit_pow": 1.6}

# Low sensitivity (피로 영향 적음): 포스트/피지컬
FATIGUE_PROFILE_LOW = {"floor": 0.88, "gamma": 1.05, "floor_min": 0.70, "crit_e": 0.24, "crit_pow": 1.4}

# Exact key overrides (정확히 이 키면 우선 적용)
_FATIGUE_HIGH_EXACT = {
    "DEF_POA", "DEF_STEAL", "DEF_HELP", "DEF_RIM",
    "HANDLE_SAFE", "PASS_SAFE", "PASS_CREATE", "PNR_READ",
}
_FATIGUE_LOW_EXACT = {
    "PHYSICAL",
    "POST_CONTROL", "POST_SCORE",
    "DEF_POST",
}


def _fatigue_profile_for_key(key: str):
    """
    스탯 키에 따라 피로 프로필을 선택한다.
    - exact override 우선
    - 그 다음 prefix 기반 그룹핑
    - 마지막에 base
    """
    if not key:
        return FATIGUE_PROFILE_BASE

    # Exact overrides first
    if key in _FATIGUE_HIGH_EXACT:
        return FATIGUE_PROFILE_HIGH
    if key in _FATIGUE_LOW_EXACT:
        return FATIGUE_PROFILE_LOW

    # Prefix grouping (새 derived 키가 추가되어도 동작)
    if key.startswith("DEF_"):
        return FATIGUE_PROFILE_HIGH
    if key.startswith("HANDLE") or key.startswith("PASS_"):
        return FATIGUE_PROFILE_HIGH
    if key.startswith("SHOT_3"):
        return FATIGUE_PROFILE_HIGH
    if key.startswith("POST_") or key.startswith("PHYS"):
        return FATIGUE_PROFILE_LOW

    return FATIGUE_PROFILE_BASE


def _fatigue_scale(key: str, energy: float) -> float:
    """
    energy(0..1)에 따른 스탯 배율(0..1)을 계산한다.
    9-A(비선형) + 9-B(스탯별 차등)
    """
    e = clamp(float(energy), 0.0, 1.0)
    prof = _fatigue_profile_for_key(key)
    floor = float(prof.get("floor", 0.78))
    gamma = float(prof.get("gamma", 1.9))

    # Red-zone dynamic floor:
    # - energy < crit_e 구간에서만 floor가 floor_min 방향으로 추가 하락
    # - crit_pow로 레드존 가속 정도를 조절
    crit_e = float(prof.get("crit_e", 0.0))
    floor_min = float(prof.get("floor_min", floor))
    crit_pow = float(prof.get("crit_pow", 1.0))

    # safety: floor_min cannot exceed floor
    if floor_min > floor:
        floor_min = floor

    floor_eff = floor
    if crit_e > 1e-9 and e < crit_e:
        t = (crit_e - e) / crit_e  # 0 at crit_e, 1 at 0
        floor_eff = floor - (floor - floor_min) * (t ** crit_pow)

    # nonlinear curve: floor_eff + (1-floor_eff) * (energy^gamma)
    scale = floor_eff + (1.0 - floor_eff) * (e ** gamma)
    return clamp(scale, 0.0, 1.0)


def _default_possession_end_counts() -> Dict[str, int]:
    return {"FGA": 0, "TOV": 0, "FT_TRIP": 0, "OTHER": 0}


def _default_shot_zone_detail() -> Dict[str, Dict[str, int]]:
    zones = ["Restricted_Area", "Paint_Non_RA", "Mid_Range", "Corner_3", "ATB_3"]
    return {z: {"FGA": 0, "FGM": 0, "AST_FGM": 0} for z in zones}

# -------------------------
# Core Data Models
# -------------------------

DERIVED_DEFAULT = 50.0

@dataclass
class GameState:
    # Core clock context (current quarter remaining seconds).
    quarter: int
    clock_sec: float
    shot_clock_sec: float

    # Fixed team ids for this game (SSOT is external GameContext; GameState stores ids only).
    home_team_id: str
    away_team_id: str

    # Possession counter (monotonic, increments per possession)
    possession: int = 0

    # ---------------------------------------------------------------------
    # Replay / Play-by-play (single source of truth)
    # ---------------------------------------------------------------------
    # - replay_seq: event sequence counter (1..N) managed by emit_event()
    # - replay_events: append-only list of replay_event dicts
    replay_seq: int = 0
    replay_events: List[Dict[str, Any]] = field(default_factory=list)

    # ---------------------------------------------------------------------
    # Team/player state trackers (keyed by team_id)
    # ---------------------------------------------------------------------
    team_fouls: Dict[str, int] = field(default_factory=dict)                # {team_id: fouls}
    player_fouls: Dict[str, Dict[str, int]] = field(default_factory=dict)   # {team_id: {pid: fouls}}
    # Per-team, per-player minutes played tracked in **seconds**.
    # Use float to avoid systematic undercount from truncation when segment lengths are fractional.
    minutes_played_sec: Dict[str, Dict[str, float]] = field(default_factory=dict)  # {team_id: {pid: sec}}
    fatigue: Dict[str, Dict[str, float]] = field(default_factory=dict)             # {team_id: {pid: energy}}

    # Lineup versioning (for replay seeking / exact on-court reconstruction)
    # - lineup_version: global monotonic counter for any on-court change (both teams)
    # - lineup_version_by_team_id: per-team monotonic counter (useful when both teams sub in same stoppage)
    lineup_version: int = 0
    lineup_version_by_team_id: Dict[str, int] = field(default_factory=dict)

    # --- Timeouts (dead-ball only, v1) ---
    # State dictionaries (keyed by team_id).
    timeouts_remaining: Dict[str, int] = field(default_factory=dict)
    timeouts_used: Dict[str, int] = field(default_factory=dict)
    timeout_last_possession: Dict[str, int] = field(default_factory=dict)

    # --- Flow trackers for timeout AI ---
    # Run is tracked as "consecutive scoring points by the same team" (no opponent score in between).
    run_pts: Dict[str, int] = field(default_factory=dict)                   # {team_id: run_pts}
    # Consecutive team turnovers tracked per team possessions (only updates when that team is on offense).
    consecutive_team_tos: Dict[str, int] = field(default_factory=dict)       # {team_id: n}
    last_scoring_team_id: Optional[str] = None

    # --- Substitution system (rotation v1.0) ---
    # Smoothed indices (EMA) + dominant mode/levels (hysteresis)
    pressure_smoothed: float = 0.0
    garbage_smoothed: float = 0.0
    dominant_mode: str = "NEUTRAL"   # "NEUTRAL" | "CLUTCH" | "GARBAGE"
    clutch_level: str = "OFF"        # "OFF" | "MID" | "STRONG"
    garbage_level: str = "OFF"       # "OFF" | "MID" | "STRONG"

    # Rotation state trackers (per-team)
    # - rotation_last_sub_game_sec: last substitution game-time (elapsed seconds) per team_id
    # - rotation_last_in_game_sec: last time a player entered the court (elapsed seconds) per team_id
    # - rotation_checkpoint_mask: checkpoint processed flags (bitmask) per team_id for the current quarter
    # - rotation_checkpoint_quarter: last quarter number in which the team's checkpoint mask was initialized/reset
    rotation_last_sub_game_sec: Dict[str, int] = field(default_factory=dict)
    rotation_last_in_game_sec: Dict[str, Dict[str, int]] = field(default_factory=dict)
    rotation_checkpoint_mask: Dict[str, int] = field(default_factory=dict)
    rotation_checkpoint_quarter: Dict[str, int] = field(default_factory=dict)


@dataclass
class Player:
    pid: str
    name: str
    pos: str = "G"
    derived: Dict[str, float] = field(default_factory=dict)
    energy: float = 1.0  # 1.0 fresh -> 0.0 exhausted  (단일 스케일과 동일한 의미)

    def get(self, key: str, fatigue_sensitive: bool = True) -> float:
        v = float(self.derived.get(key, DERIVED_DEFAULT))
        if not fatigue_sensitive:
            return v

        # 더 강한 비선형 피로 + 스탯별 민감도 차등
        return v * _fatigue_scale(key, getattr(self, "energy", 1.0))

@dataclass
class TeamState:
    # Stable SSOT identifier (required). Never fall back to name.
    team_id: str
    name: str
    lineup: List[Player]
    roles: Dict[str, str]  # role -> pid (chosen via UI)
    tactics: "TacticsConfig"
    on_court_pids: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Strong contract: engine must be able to key all team-scoped dicts by team_id.
        if not str(self.team_id).strip():
            raise ValueError("TeamState.team_id is empty")


    # -------------------------
    # Rotation (user-configurable)
    # -------------------------
    # These are optional and can be supplied by UI/config.
    # - rotation_target_sec_by_pid: per-player target minutes in seconds.
    # - rotation_offense_role_by_pid: per-player offensive role name (one of the 12 roles).
    # - rotation_lock_pids: players that should never be auto-subbed out (except foul-out).
    rotation_target_sec_by_pid: Dict[str, int] = field(default_factory=dict)
    rotation_offense_role_by_pid: Dict[str, str] = field(default_factory=dict)
    rotation_lock_pids: List[str] = field(default_factory=list)


    # team totals
    pts: int = 0
    fgm: int = 0
    fga: int = 0
    tpm: int = 0
    tpa: int = 0
    ftm: int = 0
    fta: int = 0
    tov: int = 0
    orb: int = 0
    drb: int = 0
    possessions: int = 0
    ast: int = 0
    pitp: int = 0
    fastbreak_pts: int = 0
    second_chance_pts: int = 0
    points_off_tov: int = 0
    possession_end_counts: Dict[str, int] = field(default_factory=_default_possession_end_counts)
    shot_zone_detail: Dict[str, Dict[str, int]] = field(default_factory=_default_shot_zone_detail)

    # shot zones
    shot_zones: Dict[str, int] = field(default_factory=dict)  # rim/mid/3/corner3 attempts

    # breakdowns
    off_action_counts: Dict[str, int] = field(default_factory=dict)
    outcome_counts: Dict[str, int] = field(default_factory=dict)

    # player box
    player_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # internal debug (role fit)
    role_fit_pos_log: List[Dict[str, Any]] = field(default_factory=list)
    role_fit_role_counts: Dict[str, int] = field(default_factory=dict)
    role_fit_grade_counts: Dict[str, int] = field(default_factory=dict)
    role_fit_bad_totals: Dict[str, int] = field(default_factory=dict)  # {'TO': n, 'RESET': n}
    role_fit_bad_by_grade: Dict[str, Dict[str, int]] = field(default_factory=dict)  # grade -> {'TO': n, 'RESET': n}

    def find_player(self, pid: str) -> Optional[Player]:
        for p in self.lineup:
            if p.pid == pid:
                return p
        return None

    def get_player(self, pid: str) -> Optional[Player]:
        """Backward-compatible alias for find_player()."""
        return self.find_player(pid)


    def set_on_court(self, pids: List[str], strict: bool = False) -> None:
        roster_pids = [p.pid for p in self.lineup]
        roster_set = set(roster_pids)
        requested = [str(pid) for pid in (pids or []) if pid is not None]

        seen = set()
        normalized: List[str] = []
        dropped: List[str] = []
        for pid in requested:
            if pid in seen:
                dropped.append(pid)
                continue
            if pid not in roster_set:
                dropped.append(pid)
                continue
            seen.add(pid)
            normalized.append(pid)

        filled = []
        if len(normalized) < 5:
            for pid in roster_pids:
                if pid in seen:
                    continue
                normalized.append(pid)
                filled.append(pid)
                seen.add(pid)
                if len(normalized) >= 5:
                    break

        if len(normalized) > 5:
            normalized = normalized[:5]

        issues = []
        if dropped:
            issues.append(f"dropped={dropped}")
        if filled:
            issues.append(f"filled={filled}")
        if len(requested) != len(pids or []):
            issues.append("coerced_non_string")
        if len(normalized) != 5:
            issues.append(f"size={len(normalized)}")

        if issues:
            msg = f"{self.name}: on_court normalized ({'; '.join(issues)})"
            if strict:
                raise ValueError(msg)
            warnings.warn(msg)

        self.on_court_pids = normalized

    def on_court_players(self) -> List[Player]:
        if not self.on_court_pids or len(self.on_court_pids) != 5:
            self.set_on_court(self.on_court_pids, strict=False)
        return [p for pid in self.on_court_pids for p in [self.find_player(pid)] if p is not None]

    def is_on_court(self, pid: str) -> bool:
        return pid in self.on_court_pids

    def add_player_stat(self, pid: str, key: str, inc: int = 1) -> None:
        if pid not in self.player_stats:
            # Keep a stable set of tracked boxscore keys for every player.
            # (Other modules may read raw `player_stats` directly.)
            self.player_stats[pid] = {
                "PTS": 0,
                "FGM": 0, "FGA": 0,
                "3PM": 0, "3PA": 0,
                "FTM": 0, "FTA": 0,
                "ORB": 0, "DRB": 0,
                "AST": 0,
                "STL": 0,
                "BLK": 0,
                "TOV": 0,
            }
        self.player_stats[pid][key] = self.player_stats[pid].get(key, 0) + inc

    def get_role_player(self, role: str, fallback_rank_key: Optional[str] = None) -> Player:
        pid = self.roles.get(role)
        if pid:
            p = self.find_player(pid)
            if p:
                return p
        if fallback_rank_key:
            return max(self.lineup, key=lambda x: x.get(fallback_rank_key))
        return self.lineup[0]


# -------------------------
# Minimal role ranking keys (for fallbacks)
# -------------------------

ROLE_FALLBACK_RANK = {
    "ball_handler": "PNR_READ",
    "secondary_handler": "PASS_CREATE",
    "screener": "SHORTROLL_PLAY",
    "post": "POST_SCORE",
    "shooter": "SHOT_3_CS",
    "cutter": "FIRST_STEP",
    "rim_runner": "FIN_DUNK",
}
