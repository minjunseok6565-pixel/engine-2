from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import warnings

from .core import clamp


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
    quarter: int
    clock_sec: int
    shot_clock_sec: int
    score_home: int
    score_away: int
    possession: int = 0
    team_fouls: Dict[str, int] = field(default_factory=dict)
    player_fouls: Dict[str, Dict[str, int]] = field(default_factory=dict)
    minutes_played_sec: Dict[str, Dict[str, int]] = field(default_factory=dict)
    fatigue: Dict[str, Dict[str, float]] = field(default_factory=dict)
    on_court_home: List[str] = field(default_factory=list)
    on_court_away: List[str] = field(default_factory=list)
    targets_sec_home: Dict[str, int] = field(default_factory=dict)
    targets_sec_away: Dict[str, int] = field(default_factory=dict)

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

        # 단일 피로 스케일(0..1 에너지)을 능력치에 반영
        e = clamp(float(getattr(self, "energy", 1.0)), 0.0, 1.0)

        # energy=1.0 -> 1.00, energy=0.0 -> floor (기존 0.82 유지)
        floor = 0.82
        gamma = 1.35  # (선택) 피로가 후반에 더 급격히 체감되게 하는 커브. 원하면 1.0(선형)로.

        severity = (1.0 - e) ** gamma
        f = 1.0 - severity * (1.0 - floor)
        return v * f

@dataclass
class TeamState:
    name: str
    lineup: List[Player]
    roles: Dict[str, str]  # role -> pid (chosen via UI)
    tactics: "TacticsConfig"
    on_court_pids: List[str] = field(default_factory=list)


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
    def_action_counts: Dict[str, int] = field(default_factory=dict)
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
            self.player_stats[pid] = {"PTS":0,"FGM":0,"FGA":0,"3PM":0,"3PA":0,"FTM":0,"FTA":0,"TOV":0,"ORB":0,"DRB":0,"AST":0}
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
