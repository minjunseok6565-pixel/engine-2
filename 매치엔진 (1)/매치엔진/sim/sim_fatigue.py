from __future__ import annotations

"""Fatigue model utilities (drain, bench recovery, period/OT break recovery).

NOTE: Split from sim.py on 2025-12-27.
"""

from typing import Any, Dict, List, Mapping, Optional, Tuple

from .models import GameState, TeamState
from .team_keys import team_key


# Prefer to reuse the same role->group mapping as sim_rotation (keeps rotation + fatigue consistent).
try:
    # ROLE_TO_GROUPS maps offensive role names (12 roles) to rotation groups (Handler/Wing/Big).
    from .sim_rotation import ROLE_TO_GROUPS  # type: ignore
except Exception:  # pragma: no cover
    ROLE_TO_GROUPS = {}  # type: ignore


# Primary-group semantics:
# ROLE_TO_GROUPS values are tuples like ("Wing","Handler"). For fatigue we use ONLY the primary group (index 0).
_GROUP_PRIORITY: Dict[str, int] = {"Handler": 3, "Wing": 2, "Big": 1}


def _primary_group_for_role(role_name: str) -> str:
    """Return the primary rotation group for a 12-role name, or "" if unknown."""
    rn = str(role_name or "").strip()
    if not rn or not isinstance(ROLE_TO_GROUPS, dict):
        return ""
    groups = ROLE_TO_GROUPS.get(rn)
    if not groups:
        return ""
    # groups is expected to be Tuple[str, ...]
    try:
        return str(groups[0])
    except Exception:
        return ""


def _get_offense_role_by_pid(team: TeamState) -> Dict[str, str]:
    """Return pid -> offensive role name map if provided by UI/config.

    Priority:
    1) TeamState.rotation_offense_role_by_pid (explicit pid->role map)
    2) tactics.context (ROTATION_OFFENSE_ROLE_BY_PID / OFFENSE_ROLE_BY_PID)
    3) team.roles (role->pid), inverted (12-role only; primary-group semantics on conflicts)
    """
    m = getattr(team, "rotation_offense_role_by_pid", None)
    if isinstance(m, dict) and m:
        return {str(k): str(v) for k, v in m.items()}

    ctx = getattr(getattr(team, "tactics", None), "context", None)
    if isinstance(ctx, dict):
        rm = ctx.get("ROTATION_OFFENSE_ROLE_BY_PID") or ctx.get("OFFENSE_ROLE_BY_PID")
        if isinstance(rm, dict) and rm:
            return {str(k): str(v) for k, v in rm.items()}

    # Invert team.roles (12-role keys) -> pid -> role.
    roles = getattr(team, "roles", None)
    if isinstance(roles, dict) and roles and isinstance(ROLE_TO_GROUPS, dict) and ROLE_TO_GROUPS:
        pid_to_role_candidates: Dict[str, List[str]] = {}
        for role_name, pid in roles.items():
            if not pid:
                continue
            rn = str(role_name)
            if rn not in ROLE_TO_GROUPS:
                # Ignore legacy keys (ball_handler, screener, etc.) and unknown entries.
                continue
            pid_s = str(pid)
            pid_to_role_candidates.setdefault(pid_s, []).append(rn)

        if pid_to_role_candidates:
            out: Dict[str, str] = {}
            for pid_s, candidates in pid_to_role_candidates.items():
                # If multiple roles are assigned to the same pid, pick the "most fatiguing"
                # primary group to avoid under-estimating drain.
                best_role = candidates[0]
                best_score = _GROUP_PRIORITY.get(_primary_group_for_role(best_role), 0)
                for rn in candidates[1:]:
                    score = _GROUP_PRIORITY.get(_primary_group_for_role(rn), 0)
                    if score > best_score:
                        best_role, best_score = rn, score
                out[pid_s] = best_role
            if out:
                return out

    return {}



def _fatigue_archetype_for_pid(team: TeamState, pid: str, role_by_pid: Mapping[str, str]) -> str:
    """Classify a player as handler/wing/big for fatigue drain.

    If offensive roles are configured (12-role system), we derive archetype from ROLE_TO_GROUPS
    using ONLY the primary group (index 0):
      - primary == Handler => handler
      - primary == Big     => big
      - otherwise          => wing

    If no role is configured for this pid, fallback to legacy heuristics:
      - team.roles (ball_handler/secondary_handler/screener/post)
      - position override (C/F => big)
    """
    role_name = str(role_by_pid.get(pid, "") or "").strip()
    if role_name and isinstance(ROLE_TO_GROUPS, dict) and role_name in ROLE_TO_GROUPS:
        primary = _primary_group_for_role(role_name)
        if primary == "Handler":
            return "handler"
        if primary == "Big":
            return "big"
        return "wing"

    # --- legacy fallback ---
    role = "wing"
    if getattr(team, "roles", {}).get("ball_handler") == pid:
        role = "handler"
    elif getattr(team, "roles", {}).get("secondary_handler") == pid:
        role = "handler"
    elif getattr(team, "roles", {}).get("screener") == pid or getattr(team, "roles", {}).get("post") == pid:
        role = "big"

    p = team.find_player(pid)
    if p and p.pos in ("C", "F"):
        role = "big"
    return role



def _fatigue_loss_for_role(role: str, rules: Dict[str, Any]) -> float:
    fl = rules.get("fatigue_loss", {})
    if role == "handler":
        return float(fl.get("handler", 0.012))
    if role == "big":
        return float(fl.get("big", 0.009))
    return float(fl.get("wing", 0.010))

def _apply_fatigue_loss(
    team: TeamState,
    on_court: List[str],
    game_state: GameState,
    rules: Dict[str, Any],
    intensity: Dict[str, bool],
    elapsed_sec: float,  # ★ 추가: 실제 흘러간 시간(초)
    home: TeamState,
) -> None:
    if elapsed_sec <= 0:
        return

    def clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    def cap01(pid: str) -> float:
        p = team.find_player(pid)
        if not p:
            return 0.5
        cap = float(p.derived.get("FAT_CAPACITY", 50.0))
        return clamp01(cap / 100.0)

    def lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    # 포제션당 고정 소모량을 "시간"으로 스케일하기 위한 기준 시간(초)
    ref_sec = float(rules.get("fatigue_time_ref_sec", 10.0))

    # 벤치 회복(초당). 값은 튜닝 대상
    bench_rec_per_sec = float((rules.get("fatigue_recovery", {}) or {}).get("bench_per_sec", 0.0022))

    # FAT_CAPACITY가 낮을수록 더 빨리 닳고/덜 회복, 높을수록 덜 닳고/더 회복
    cap_cfg = rules.get("fatigue_capacity", {}) or {}
    drain_lo = float(cap_cfg.get("drain_mult_low_cap", 1.15))   # cap 낮으면 소모↑
    drain_hi = float(cap_cfg.get("drain_mult_high_cap", 0.85))  # cap 높으면 소모↓
    rec_lo = float(cap_cfg.get("rec_mult_low_cap", 0.90))       # cap 낮으면 회복↓
    rec_hi = float(cap_cfg.get("rec_mult_high_cap", 1.10))      # cap 높으면 회복↑

    # --- on-court: 소모 ---
    role_by_pid = _get_offense_role_by_pid(team)
    key = team_key(team, home)
    fat_map = game_state.fatigue.setdefault(key, {})

    for pid in on_court:
        # Use configured offensive roles if available; otherwise fallback to legacy role+position heuristics.
        role = _fatigue_archetype_for_pid(team, pid, role_by_pid)

        # 기존 룰(포제션당 소모)을 시간 비례로 변환
        loss = _fatigue_loss_for_role(role, rules) * (float(elapsed_sec) / ref_sec)

        if intensity.get("transition_emphasis"):
            loss += float(rules.get("fatigue_loss", {}).get("transition_emphasis", 0.001)) * (float(elapsed_sec) / ref_sec)
        if intensity.get("heavy_pnr") and role in ("handler", "big"):
            loss += float(rules.get("fatigue_loss", {}).get("heavy_pnr", 0.001)) * (float(elapsed_sec) / ref_sec)

        c01 = cap01(pid)
        loss *= lerp(drain_lo, drain_hi, c01)

        fat_map[pid] = clamp01(float(fat_map.get(pid, 1.0)) - float(loss))

    # --- bench: 회복 ---
    bench_pids = [p.pid for p in team.lineup if p.pid not in on_court]
    for pid in bench_pids:
        c01 = cap01(pid)
        rec = bench_rec_per_sec * float(elapsed_sec) * lerp(rec_lo, rec_hi, c01)
        fat_map[pid] = clamp01(float(fat_map.get(pid, 1.0)) + float(rec))

def _apply_break_recovery(
    team: TeamState,
    on_court: List[str],
    game_state: GameState,
    rules: Dict[str, Any],
    break_sec: float,
    home: TeamState,
) -> None:
    """Recover fatigue during period/OT breaks. No clock/minutes are consumed."""
    if break_sec <= 0:
        return

    def clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    def lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    def cap01(pid: str) -> float:
        p = team.find_player(pid)
        if not p:
            return 0.5
        cap = float(p.derived.get("FAT_CAPACITY", 50.0))
        return clamp01(cap / 100.0)

    # capacity-based recovery multipliers (reuse fatigue_capacity.rec_* if present)
    cap_cfg = rules.get("fatigue_capacity", {}) or {}
    rec_lo = float(cap_cfg.get("rec_mult_low_cap", 0.90))
    rec_hi = float(cap_cfg.get("rec_mult_high_cap", 1.10))

    br = rules.get("break_recovery", {}) or {}
    on_per_sec = float(br.get("on_court_per_sec", 0.0010))
    bench_per_sec = float(br.get("bench_per_sec", 0.0016))
    key = team_key(team, home)
    fat_map = game_state.fatigue.setdefault(key, {})

    # on-court players recover
    for pid in on_court:
        c01 = cap01(pid)
        rec = on_per_sec * float(break_sec) * lerp(rec_lo, rec_hi, c01)
        fat_map[pid] = clamp01(float(fat_map.get(pid, 1.0)) + float(rec))

    # bench players recover
    bench_pids = [p.pid for p in team.lineup if p.pid not in on_court]
    for pid in bench_pids:
        c01 = cap01(pid)
        rec = bench_per_sec * float(break_sec) * lerp(rec_lo, rec_hi, c01)
        fat_map[pid] = clamp01(float(fat_map.get(pid, 1.0)) + float(rec))
