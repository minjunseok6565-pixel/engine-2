from __future__ import annotations

"""Rotation utilities (on-court tracking, minutes accounting, auto-sub logic).

This module controls:
- per-player minutes tracking
- auto-substitution / rotation decisions
- on-court pid lists in GameState

NOTE: Split from sim.py on 2025-12-27.
"""

import random
from itertools import combinations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .core import clamp
from .models import GameState, TeamState
from .team_keys import team_key


# -------------------------
# Role -> Group mapping
# -------------------------
# Primary group is listed first; additional entries represent "hybrid" eligibility.
ROLE_TO_GROUPS: Dict[str, Tuple[str, ...]] = {
    # Handlers
    "Initiator_Primary": ("Handler",),
    "Transition_Handler": ("Handler",),

    # Handler/Wing hybrids
    "Initiator_Secondary": ("Handler", "Wing"),
    "Shot_Creator": ("Wing", "Handler"),
    "Connector_Playmaker": ("Wing", "Handler"),
    "Rim_Attacker": ("Wing", "Handler"),

    # Wings
    "Spacer_CatchShoot": ("Wing",),
    "Spacer_Movement": ("Wing",),

    # Bigs
    "Roller_Finisher": ("Big",),
    "ShortRoll_Playmaker": ("Big",),
    "Pop_Spacer_Big": ("Big",),
    "Post_Hub": ("Big",),
}


def _get_tactics_context(team: TeamState) -> Dict[str, Any]:
    """Safely access tactics.context if present."""
    tactics = getattr(team, "tactics", None)
    ctx = getattr(tactics, "context", None)
    return ctx if isinstance(ctx, dict) else {}


def _coerce_pid_to_int_map(value: Any) -> Dict[str, int]:
    """Best-effort conversion for {pid: number} inputs."""
    if not isinstance(value, dict):
        return {}
    out: Dict[str, int] = {}
    for k, v in value.items():
        if k is None:
            continue
        pid = str(k)
        try:
            out[pid] = int(float(v))
        except Exception:
            continue
    return out


def _coerce_pid_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value]
    return []


def _regulation_total_sec(rules: Mapping[str, Any]) -> int:
    quarters = int(rules.get("quarters", 4))
    quarter_len = int(float(rules.get("quarter_length", 720)))
    return max(1, quarters * quarter_len)


def _estimate_remaining_game_sec(game_state: GameState, rules: Mapping[str, Any]) -> int:
    """Estimate remaining seconds in the game, regulation-only.

    If already in OT, we use the remaining time in the current period only.
    """
    reg_quarters = int(rules.get("quarters", 4))
    quarter_len = int(float(rules.get("quarter_length", 720)))

    q = int(getattr(game_state, "quarter", 1))
    clock = int(float(getattr(game_state, "clock_sec", 0)))

    if q <= reg_quarters:
        # remaining in current quarter + remaining full quarters
        return max(0, clock + max(0, reg_quarters - q) * quarter_len)
    return max(0, clock)


def _fallback_groups_from_pos(pos: str) -> Tuple[str, ...]:
    p = (pos or "").upper()
    if p in {"C"}:
        return ("Big",)
    if p in {"PF"}:
        return ("Big", "Wing")
    if p in {"SF"}:
        return ("Wing",)
    if p in {"PG", "SG", "G"}:
        return ("Handler", "Wing")
    if p in {"F"}:
        return ("Wing", "Big")
    return ("Wing",)


def pick_desired_five_bruteforce(
    team: "TeamState",
    home: "TeamState",
    game_state: "GameState",
    rules: Mapping[str, Any],
    *,
    current_on_court: Sequence[str],
    eligible_pool: Sequence[str],
    must_keep: Set[str],
    mode: str,                  # "NEUTRAL" | "CLUTCH" | "GARBAGE"
    level: str,                 # "OFF" | "MID" | "STRONG"
    index: float,               # pressure_smoothed or garbage_smoothed (0~1)
    template_lineup: Sequence[str],  # clutch/garbage lineup pid list (can be empty)
    lock_mult: float = 1.0,     # coach preset multiplier for MID k
    # scoring knobs (start simple; tune later)
    w_talent: float = 1.00,
    w_urgency: float = 0.80,
    w_fatigue: float = 0.60,
    w_foul: float = 0.70,
    cont_bonus: float = 0.08,       # overlap bonus (stability)
    swap_penalty: float = 0.14,     # change penalty (stability)
    debug_topk: int = 0,            # 0이면 debug 저장 안 함
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Brute-force choose the best 'Desired 5' for the next stint.

    Returns:
        desired5: best lineup (5 pids)
        debug: diagnostics (scores, constraints, optional topK candidates)
    """
    debug: Dict[str, Any] = {}

    # -------------------------
    # 0) Prep maps / helpers
    # -------------------------
    foul_out = int(rules.get("foul_out", 6))
    q = int(getattr(game_state, "quarter", 1))

    key = team_key(team, home)

    pf_map = game_state.player_fouls.get(key, {})
    fat_map = game_state.fatigue.get(key, {})
    mins_map = game_state.minutes_played_sec.get(key, {})

    targets = game_state.targets_sec_home if team is home else game_state.targets_sec_away

    player_by_pid = {p.pid: p for p in team.lineup}

    # role assignment (pid -> role_name) for group fallback
    ctx = _get_tactics_context(team)
    role_by_pid: Dict[str, str] = {}
    team_roles = getattr(team, "rotation_offense_role_by_pid", None)
    if isinstance(team_roles, dict) and team_roles:
        role_by_pid = {str(pid): str(role) for pid, role in team_roles.items()}
    else:
        raw_roles = ctx.get("ROTATION_OFFENSE_ROLE_BY_PID") or ctx.get("OFFENSE_ROLE_BY_PID")
        if isinstance(raw_roles, dict):
            role_by_pid = {str(pid): str(role) for pid, role in raw_roles.items()}

    def groups_for(pid: str) -> Tuple[str, ...]:
        role = role_by_pid.get(pid)
        if role and role in ROLE_TO_GROUPS:
            return ROLE_TO_GROUPS[role]
        pos = getattr(player_by_pid.get(pid), "pos", "G")
        return _fallback_groups_from_pos(pos)

    def is_handler(pid: str) -> bool:
        return "Handler" in groups_for(pid)

    def is_big(pid: str) -> bool:
        return "Big" in groups_for(pid)

    def lineup_shape_ok(lineup: Sequence[str]) -> bool:
        # Hard constraint: at least 1 handler AND at least 1 big
        return any(is_handler(pid) for pid in lineup) and any(is_big(pid) for pid in lineup)

    def is_eligible(pid: str) -> bool:
        # Today: foul-out only. (Injury/unavailable hook can be added later.)
        return pf_map.get(pid, 0) < foul_out

    # -------------------------
    # 1) Normalize pools
    # -------------------------
    current_set = set(current_on_court)

    # Ensure must_keep are in current + eligible (if not, drop; forced subs should handle upstream)
    keep: Set[str] = set(
        pid for pid in must_keep
        if pid in current_set and pid in eligible_pool and is_eligible(pid)
    )

    eligible: List[str] = [pid for pid in eligible_pool if is_eligible(pid)]

    # Always include keep in eligible universe
    for pid in keep:
        if pid not in eligible:
            eligible.append(pid)

    # Guard: if keep already too many (should not happen). In worst case keep first 5 to stay safe.
    if len(keep) > 5:
        keep = set(list(keep)[:5])

    # Remaining candidates excluding keep
    cand = [pid for pid in eligible if pid not in keep]

    slots = 5 - len(keep)
    if slots < 0:
        slots = 0

    # -------------------------
    # 2) Template inclusion requirement (min_core_required)
    #    - shape constraint always wins over template inclusion
    # -------------------------
    core: Set[str] = set(pid for pid in template_lineup if pid in eligible)
    core_in_keep = len(core & keep)
    core_outside_keep = list(core - keep)

    max_core_total = core_in_keep + min(slots, len(core_outside_keep))

    if level == "STRONG" and mode in ("CLUTCH", "GARBAGE"):
        min_core_required = max_core_total  # "as many as possible"
    elif level == "MID" and mode in ("CLUTCH", "GARBAGE"):
        k = round(5.0 * float(index) * float(lock_mult))
        k = int(clamp(k, 2, 4))
        min_core_required = min(k, max_core_total)
    else:
        min_core_required = 0

    # Try required core count first; if infeasible, relax down to 0.
    req_core_try_list = list(range(min_core_required, -1, -1))

    # -------------------------
    # 3) Player score components
    # -------------------------
    remaining_est = _estimate_remaining_game_sec(game_state, rules)
    remaining_est = max(1, int(remaining_est))

    trouble_by_q = {1: 2, 2: 3, 3: 4, 4: 5}
    t = trouble_by_q.get(min(q, 4), 5)

    clutch_strong = (mode == "CLUTCH" and level == "STRONG")  # foul trouble penalty halved

    def talent_proxy(pid: str) -> float:
        # talent_proxy = 평균 derived (0~100 가정)
        p = player_by_pid.get(pid)
        d = getattr(p, "derived", None) if p else None
        if isinstance(d, dict) and d:
            vals = [float(v) for v in d.values()]
            return sum(vals) / max(1, len(vals))
        return 50.0

    def talent_norm(pid: str) -> float:
        return clamp(talent_proxy(pid) / 100.0, 0.0, 1.0)

    def fatigue(pid: str) -> float:
        return clamp(float(fat_map.get(pid, 1.0)), 0.0, 1.0)

    def played(pid: str) -> int:
        return int(mins_map.get(pid, 0))

    def target(pid: str) -> int:
        return int(targets.get(pid, 0))

    def urgency_norm(pid: str) -> float:
        # urgency = need/remaining, cap to 1.5 then normalize to 0..1
        need = max(0, target(pid) - played(pid))
        urg = need / float(remaining_est)
        urg = clamp(urg, 0.0, 1.5)
        return clamp(urg / 1.5, 0.0, 1.0)

    def foul_risk(pid: str) -> float:
        fouls = int(pf_map.get(pid, 0))
        if fouls < t:
            return 0.0
        denom = max(1, foul_out - (t - 1))
        r = (fouls - (t - 1)) / float(denom)
        r = clamp(r, 0.0, 1.0)
        if clutch_strong:
            r *= 0.5
        return r

    def player_score(pid: str) -> float:
        return (
            w_talent * talent_norm(pid)
            + w_urgency * urgency_norm(pid)
            + w_fatigue * fatigue(pid)
            - w_foul * foul_risk(pid)
        )

    # -------------------------
    # 4) Lineup score + current score (Planned Change Threshold option A)
    # -------------------------
    def lineup_score(lineup: Sequence[str]) -> float:
        base = 0.0
        for pid in lineup:
            base += player_score(pid)

        overlap = sum(1 for pid in lineup if pid in current_set)
        changes = 5 - overlap

        return base + cont_bonus * overlap - swap_penalty * changes

    # Always compute and expose current_score for callers.
    current_score = lineup_score(current_on_court)

    # -------------------------
    # 5) Brute-force search (with core requirement relaxation)
    # -------------------------
    best_lineup: Optional[List[str]] = None
    best_score: float = -1e18

    topk: List[Tuple[float, List[str], int]] = []  # (score, lineup, core_count)

    if slots == 0:
        fixed = list(keep)
        if len(fixed) == 5 and lineup_shape_ok(fixed):
            best_lineup = fixed
            best_score = lineup_score(fixed)
        else:
            fallback = list(current_on_court)
            return fallback, {
                "reason": "slots=0 but invalid shape/size; fallback current",
                "best_lineup": fallback,
                "current_score": float(current_score),
                "best_score": float(current_score),
                "improvement": 0.0,
                }
    else:
        for req_core in req_core_try_list:
            found_any = False

            for combo in combinations(cand, slots):
                lineup = list(keep) + list(combo)
                if len(lineup) != 5:
                    continue

                if not lineup_shape_ok(lineup):
                    continue

                core_count = len(set(lineup) & core)
                if core_count < req_core:
                    continue

                found_any = True
                sc = lineup_score(lineup)

                if sc > best_score:
                    best_score = sc
                    best_lineup = list(lineup)

                if debug_topk and debug_topk > 0:
                    topk.append((sc, list(lineup), core_count))

            if found_any:
                break

    if not best_lineup:
        fallback = list(current_on_court)
        return fallback, {
            "reason": "no feasible lineup under hard shape constraint; fallback current",
            "best_lineup": fallback,
            "current_score": float(current_score),
            "best_score": float(current_score),
            "improvement": 0.0,
        }

    # -------------------------
    # 6) Debug payload (optional) + always expose scores
    # -------------------------
    best_score = float(best_score)
    improvement = best_score - float(current_score)

    if debug_topk and debug_topk > 0:
        topk.sort(key=lambda x: x[0], reverse=True)
        topk = topk[: int(debug_topk)]
        debug["topk"] = [
            {
                "score": float(sc),
                "lineup": lu,
                "core_count": int(core_count),
                "overlap": int(sum(1 for pid in lu if pid in current_set)),
            }
            for sc, lu, core_count in topk
        ]

    debug.update(
        {
            "best_lineup": list(best_lineup),
            "current_score": float(current_score),
            "best_score": best_score,
            "improvement": float(improvement),
            "score_scale_hint": {
                "player_score_max_approx": float(w_talent + w_urgency + w_fatigue),
                "lineup_base_max_approx": float(5.0 * (w_talent + w_urgency + w_fatigue)),
                "cont_bonus": float(cont_bonus),
                "swap_penalty": float(swap_penalty),
            },
            "keep": sorted(list(keep)),
            "slots": int(slots),
            "eligible_n": int(len(eligible)),
            "cand_n": int(len(cand)),
            "mode": str(mode),
            "level": str(level),
            "index": float(index),
            "lock_mult": float(lock_mult),
            "min_core_required_initial": int(min_core_required),
            "core_available": int(len(core)),
            "quarter": int(q),
            "clutch_strong_foul_halved": bool(clutch_strong),
        }
    )

    return list(best_lineup), debug


def _init_targets(team: TeamState, rules: Dict[str, Any]) -> Dict[str, int]:
    """Initialize per-player minutes targets (seconds).

    Priority:
    1) TeamState.rotation_target_sec_by_pid (if provided by UI/config)
    2) tactics.context:
       - ROTATION_TARGET_SEC_BY_PID: {pid: seconds}
       - ROTATION_TARGET_MIN_BY_PID: {pid: minutes}
       - (aliases) TARGET_SEC_BY_PID / TARGET_MIN_BY_PID
    3) Fallback: index-bucket targets from rules["fatigue_targets"] (starter/rotation/bench)
    """
    # 1) TeamState fields (preferred)
    team_user_sec = _coerce_pid_to_int_map(getattr(team, "rotation_target_sec_by_pid", None))
    user_sec: Dict[str, int] = dict(team_user_sec)

    # 2) tactics.context fields
    if not user_sec:
        ctx = _get_tactics_context(team)

        # user targets (seconds)
        user_sec = _coerce_pid_to_int_map(
            ctx.get("ROTATION_TARGET_SEC_BY_PID") or ctx.get("TARGET_SEC_BY_PID")
        )

        # user targets (minutes) -> seconds
        if not user_sec:
            user_min = _coerce_pid_to_int_map(
                ctx.get("ROTATION_TARGET_MIN_BY_PID") or ctx.get("TARGET_MIN_BY_PID")
            )
            if user_min:
                user_sec = {pid: int(m * 60) for pid, m in user_min.items()}

    # 3) fallback bucket targets
    tcfg = rules.get("fatigue_targets", {})
    starter_sec = int(tcfg.get("starter_sec", 32 * 60))
    rotation_sec = int(tcfg.get("rotation_sec", 16 * 60))
    bench_sec = int(tcfg.get("bench_sec", 8 * 60))

    targets: Dict[str, int] = {}
    for idx, p in enumerate(team.lineup):
        if p.pid in user_sec:
            targets[p.pid] = int(user_sec[p.pid])
            continue
        if idx < 5:
            targets[p.pid] = starter_sec
        elif idx < 8:
            targets[p.pid] = rotation_sec
        else:
            targets[p.pid] = bench_sec
    return targets



def _get_on_court(game_state: GameState, team: TeamState, home: TeamState) -> List[str]:
    return game_state.on_court_home if team is home else game_state.on_court_away


def _set_on_court(game_state: GameState, team: TeamState, home: TeamState, players: List[str]) -> None:
    team.set_on_court(list(players))
    if team is home:
        game_state.on_court_home = list(team.on_court_pids)
    else:
        game_state.on_court_away = list(team.on_court_pids)


def _update_minutes(
    game_state: GameState,
    pids: List[str],
    delta_sec: float,
    team: TeamState,
    home: TeamState,
) -> None:
    inc = int(max(delta_sec, 0))
    key = team_key(team, home)
    mins_map = game_state.minutes_played_sec.setdefault(key, {})
    for pid in pids:
        mins_map[pid] = mins_map.get(pid, 0) + inc
        

# -------------------------
# Rotation v1.0 (checkpoint-based, dead-ball only)
# -------------------------

# Checkpoints per quarter clock (sec remaining)
_ROTATION_CHECKPOINTS: Tuple[int, ...] = (540, 360, 180, 60)
_ROTATION_CHECKPOINT_WINDOW_SEC: int = 20

# Stability controls
_ROTATION_PLANNED_COOLDOWN_SEC: int = 60
_ROTATION_MIN_STINT_SEC: int = 90
_ROTATION_MIN_STINT_RELAX_AT_CLOCK_SEC: int = 30
_ROTATION_PLANNED_MAX_SWAPS: int = 3

# Fatigue safety
_ROTATION_FATIGUE_EMERGENCY: float = 0.18
_ROTATION_FATIGUE_CAUTION: float = 0.28  # (soft; currently handled via scoring)


_COACH_PRESETS: Dict[str, Dict[str, float]] = {
    "Balanced": {
        "depth_n": 9,
        "planned_change_threshold": 0.30,
        "clutch_lock_mult": 1.00,
        "garbage_lock_mult": 1.00,
    },
    "Playoff Tight": {
        "depth_n": 8,
        "planned_change_threshold": 0.45,
        "clutch_lock_mult": 1.10,
        "garbage_lock_mult": 0.70,
    },
    "Regular Deep": {
        "depth_n": 10,
        "planned_change_threshold": 0.20,
        "clutch_lock_mult": 0.90,
        "garbage_lock_mult": 1.10,
    },
    "Defense First": {
        "depth_n": 9,
        "planned_change_threshold": 0.25,
        "clutch_lock_mult": 1.00,
        "garbage_lock_mult": 0.90,
    },
    "Offense First": {
        "depth_n": 9,
        "planned_change_threshold": 0.25,
        "clutch_lock_mult": 1.00,
        "garbage_lock_mult": 0.85,
    },
    "Small Ball": {
        "depth_n": 9,
        "planned_change_threshold": 0.25,
        "clutch_lock_mult": 0.95,
        "garbage_lock_mult": 0.90,
    },
    "Development": {
        "depth_n": 11,
        "planned_change_threshold": 0.18,
        "clutch_lock_mult": 0.80,
        "garbage_lock_mult": 1.15,
    },
    "Flow/Reactive": {
        "depth_n": 9,
        "planned_change_threshold": 0.15,
        "clutch_lock_mult": 0.95,
        "garbage_lock_mult": 1.00,
    },
}


def _game_elapsed_sec(game_state: GameState, rules: Mapping[str, Any]) -> int:
    """Monotonic elapsed game time in seconds (regulation + OT)."""
    reg_quarters = int(rules.get("quarters", 4))
    quarter_len = int(float(rules.get("quarter_length", 720)))
    ot_len = int(float(rules.get("overtime_length", 300)))

    q = int(getattr(game_state, "quarter", 1))
    clock = float(getattr(game_state, "clock_sec", 0))

    if q <= reg_quarters:
        # elapsed within regulation
        return int(max(0, (q - 1) * quarter_len + (quarter_len - clock)))

    # elapsed includes completed regulation + completed OTs + current OT elapsed
    reg_total = reg_quarters * quarter_len
    ot_index = max(0, q - reg_quarters - 1)
    return int(max(0, reg_total + ot_index * ot_len + (ot_len - clock)))


def _ensure_rotation_team_state(game_state: GameState, key: str, *, quarter: int) -> None:
    # These attributes exist after models.py patch; keep safe if older saves exist.
    if not hasattr(game_state, "rotation_last_sub_game_sec"):
        game_state.rotation_last_sub_game_sec = {}
    if not hasattr(game_state, "rotation_last_in_game_sec"):
        game_state.rotation_last_in_game_sec = {}
    if not hasattr(game_state, "rotation_checkpoint_mask"):
        game_state.rotation_checkpoint_mask = {}
    if not hasattr(game_state, "rotation_checkpoint_quarter"):
        game_state.rotation_checkpoint_quarter = {}

    game_state.rotation_last_sub_game_sec.setdefault(key, -10**9)
    game_state.rotation_last_in_game_sec.setdefault(key, {})
    game_state.rotation_checkpoint_mask.setdefault(key, 0)
    game_state.rotation_checkpoint_quarter.setdefault(key, int(quarter))

    # Reset checkpoint mask when quarter changes
    if int(game_state.rotation_checkpoint_quarter.get(key, quarter)) != int(quarter):
        game_state.rotation_checkpoint_quarter[key] = int(quarter)
        game_state.rotation_checkpoint_mask[key] = 0


def _rotation_checkpoint_due_and_mark(
    game_state: GameState,
    rules: Mapping[str, Any],
    *,
    team_key_str: str,
) -> bool:
    """Return True if a planned-sub checkpoint should be evaluated on this dead-ball.

    - In-window: abs(clock - cp) <= window
    - Missed: once the time is passed, first dead-ball after triggers catch-up once
    - Marks processed checkpoints immediately (even if later logic decides to skip sub).
    """
    reg_quarters = int(rules.get("quarters", 4))
    q = int(getattr(game_state, "quarter", 1))
    if q > reg_quarters:
        return False  # no planned checkpoints in OT

    clock = int(float(getattr(game_state, "clock_sec", 0)))
    window = int(_ROTATION_CHECKPOINT_WINDOW_SEC)

    mask = int(getattr(game_state, "rotation_checkpoint_mask", {}).get(team_key_str, 0))

    in_window: List[Tuple[int, int]] = []
    passed: List[Tuple[int, int]] = []
    for idx, cp in enumerate(_ROTATION_CHECKPOINTS):
        bit = 1 << idx
        if mask & bit:
            continue
        if abs(clock - int(cp)) <= window:
            in_window.append((abs(clock - int(cp)), idx))
        elif clock < int(cp) - window:
            # passed without a dead-ball in the window -> catch-up at first dead-ball after passing
            passed.append((int(cp) - clock, idx))

    if in_window:
        _, idx = min(in_window, key=lambda x: x[0])
        mask |= (1 << idx)
        game_state.rotation_checkpoint_mask[team_key_str] = int(mask)
        return True

    if passed:
        # catch-up: process the most recently passed checkpoint (closest), and mark all earlier ones too
        _, idx = min(passed, key=lambda x: x[0])
        for j in range(idx + 1):
            mask |= (1 << j)
        game_state.rotation_checkpoint_mask[team_key_str] = int(mask)
        return True

    return False


def _get_preset_for_team(team: TeamState) -> Dict[str, float]:
    ctx = _get_tactics_context(team)
    is_user = bool(ctx.get("USER_COACH", False))
    if is_user:
        return dict(_COACH_PRESETS["Balanced"])

    name = str(ctx.get("COACH_PRESET") or "Balanced")
    if name not in _COACH_PRESETS:
        name = "Balanced"
    return dict(_COACH_PRESETS[name])


def _coerce_roster_pid_set(team: TeamState) -> Set[str]:
    return {p.pid for p in (team.lineup or [])}


def _template_lineup_for_mode(ctx: Dict[str, Any], mode: str) -> List[str]:
    if mode == "CLUTCH":
        return _coerce_pid_list(ctx.get("CLUTCH_LINEUP_PIDS"))
    if mode == "GARBAGE":
        return _coerce_pid_list(ctx.get("GARBAGE_LINEUP_PIDS"))
    return []


def _explicit_lock_pids(team: TeamState, ctx: Dict[str, Any]) -> Set[str]:
    # Optional legacy support: explicit "do not sub" locks (planned only).
    locks = set(_coerce_pid_list(getattr(team, "rotation_lock_pids", None)))
    if not locks:
        locks = set(_coerce_pid_list(ctx.get("ROTATION_LOCK_PIDS") or ctx.get("LOCK_PIDS")))
    return locks


def _player_score_v1(
    pid: str,
    *,
    team: TeamState,
    home: TeamState,
    game_state: GameState,
    rules: Mapping[str, Any],
    mode: str,
    level: str,
    w_talent: float = 1.00,
    w_urgency: float = 0.80,
    w_fatigue: float = 0.60,
    w_foul: float = 0.70,
) -> float:
    """Same core scoring ingredients as pick_desired_five_bruteforce (for swap-cap selection)."""
    key = team_key(team, home)
    foul_out = int(rules.get("foul_out", 6))
    q = int(getattr(game_state, "quarter", 1))

    pf_map = game_state.player_fouls.get(key, {})
    fat_map = game_state.fatigue.get(key, {})
    mins_map = game_state.minutes_played_sec.get(key, {})
    targets = game_state.targets_sec_home if team is home else game_state.targets_sec_away

    player_by_pid = {p.pid: p for p in team.lineup}

    remaining_est = max(1, int(_estimate_remaining_game_sec(game_state, rules)))

    trouble_by_q = {1: 2, 2: 3, 3: 4, 4: 5}
    t = trouble_by_q.get(min(q, 4), 5)

    clutch_strong = (mode == "CLUTCH" and level == "STRONG")

    def talent_proxy() -> float:
        p = player_by_pid.get(pid)
        d = getattr(p, "derived", None) if p else None
        if isinstance(d, dict) and d:
            vals = [float(v) for v in d.values()]
            return sum(vals) / max(1, len(vals))
        return 50.0

    def talent_norm() -> float:
        return clamp(talent_proxy() / 100.0, 0.0, 1.0)

    def fatigue() -> float:
        return clamp(float(fat_map.get(pid, 1.0)), 0.0, 1.0)

    def played() -> int:
        return int(mins_map.get(pid, 0))

    def target() -> int:
        return int(targets.get(pid, 0))

    def urgency_norm() -> float:
        need = max(0, target() - played())
        urg = need / float(remaining_est)
        urg = clamp(urg, 0.0, 1.5)
        return clamp(urg / 1.5, 0.0, 1.0)

    def foul_risk() -> float:
        fouls = int(pf_map.get(pid, 0))
        if fouls < t:
            return 0.0
        denom = max(1, foul_out - (t - 1))
        r = (fouls - (t - 1)) / float(denom)
        r = clamp(r, 0.0, 1.0)
        if clutch_strong:
            r *= 0.5
        return r

    return (
        w_talent * talent_norm()
        + w_urgency * urgency_norm()
        + w_fatigue * fatigue()
        - w_foul * foul_risk()
    )


def ensure_rotation_v1_state(game_state: GameState, home: TeamState, away: TeamState, rules: Mapping[str, Any]) -> None:
    """Initialize rotation v1.0 state containers (safe no-op if already initialized)."""
    q = int(getattr(game_state, "quarter", 1))
    now = _game_elapsed_sec(game_state, rules)

    for team in (home, away):
        k = team_key(team, home)
        _ensure_rotation_team_state(game_state, k, quarter=q)

        # If on-court already populated, initialize last-in timestamps for those pids.
        last_in = game_state.rotation_last_in_game_sec.setdefault(k, {})
        on_court = list(_get_on_court(game_state, team, home))
        for pid in on_court:
            last_in.setdefault(str(pid), int(now))


def maybe_substitute_deadball_v1(
    rng: random.Random,
    team: TeamState,
    home: TeamState,
    game_state: GameState,
    rules: Mapping[str, Any],
    *,
    q_index: int,
    pos_start: str,
    pressure_index: float,
    garbage_index: float,
) -> bool:
    """Dead-ball substitution entry for Rotation v1.0.

    Returns True if the on-court lineup changed.
    """
    del rng  # deterministic for now; kept for future randomness knobs

    ctx = _get_tactics_context(team)
    is_user = bool(ctx.get("USER_COACH", False))
    preset = _get_preset_for_team(team)

    q = int(getattr(game_state, "quarter", 1))
    now = _game_elapsed_sec(game_state, rules)

    key = team_key(team, home)
    _ensure_rotation_team_state(game_state, key, quarter=q)

    foul_out = int(rules.get("foul_out", 6))
    pf_map = game_state.player_fouls.get(key, {})
    fat_map = game_state.fatigue.get(key, {})

    on_court = list(_get_on_court(game_state, team, home))
    current_set = set(on_court)
    roster_set = _coerce_roster_pid_set(team)

    # Determine mode/level/index/template/lock_mult from smoothed context (computed in sim_game).
    mode = str(getattr(game_state, "dominant_mode", "NEUTRAL") or "NEUTRAL")
    if mode == "CLUTCH":
        level = str(getattr(game_state, "clutch_level", "OFF") or "OFF")
        index = float(pressure_index)
        lock_mult = float(preset.get("clutch_lock_mult", 1.0))
    elif mode == "GARBAGE":
        level = str(getattr(game_state, "garbage_level", "OFF") or "OFF")
        index = float(garbage_index)
        lock_mult = float(preset.get("garbage_lock_mult", 1.0))
    else:
        mode = "NEUTRAL"
        level = "OFF"
        index = 0.0
        lock_mult = 1.0

    template_lineup = _template_lineup_for_mode(ctx, mode)
    # normalize templates to roster pids
    template_lineup = [pid for pid in template_lineup if pid in roster_set]

    # Rotation pool selection
    eligible_pool: List[str] = []
    if is_user:
        user_pool = _coerce_pid_list(ctx.get("ROTATION_POOL_PIDS"))
        if user_pool:
            eligible_pool = [pid for pid in user_pool if pid in roster_set]
        else:
            # Balanced fallback construction
            n = int(_COACH_PRESETS["Balanced"]["depth_n"])
            eligible_pool = [p.pid for p in (team.lineup or [])[: max(5, n)]]
    else:
        n = int(preset.get("depth_n", _COACH_PRESETS["Balanced"]["depth_n"]))
        eligible_pool = [p.pid for p in (team.lineup or [])[: max(5, n)]]

    # Always include current on-court to keep caller-safe
    for pid in on_court:
        if pid in roster_set and pid not in eligible_pool:
            eligible_pool.append(pid)

    # Garbage exception: allow template pids even if outside rotation pool
    if mode == "GARBAGE":
        for pid in template_lineup:
            if pid in roster_set and pid not in eligible_pool:
                eligible_pool.append(pid)

    # -------------------------
    # 1) Forced substitution: foul-out
    # -------------------------
    forced_out = [pid for pid in on_court if int(pf_map.get(pid, 0)) >= foul_out]
    if forced_out:
        must_keep = {pid for pid in on_court if pid not in forced_out and int(pf_map.get(pid, 0)) < foul_out}
        desired5, _dbg = pick_desired_five_bruteforce(
            team,
            home,
            game_state,
            rules,
            current_on_court=on_court,
            eligible_pool=eligible_pool,
            must_keep=must_keep,
            mode=mode,
            level=level,
            index=index,
            template_lineup=template_lineup,
            lock_mult=lock_mult,
        )
        desired_set = set(desired5)
        if desired_set != current_set:
            _set_on_court(game_state, team, home, list(desired5))
            game_state.rotation_last_sub_game_sec[key] = int(now)
            last_in = game_state.rotation_last_in_game_sec.setdefault(key, {})
            for pid in desired5:
                if pid not in current_set:
                    last_in[str(pid)] = int(now)
            return True

    # -------------------------
    # 2) Emergency fatigue substitution (<= 0.18): immediate
    # -------------------------
    emergency_out = [pid for pid in on_court if float(fat_map.get(pid, 1.0)) <= float(_ROTATION_FATIGUE_EMERGENCY)]
    if emergency_out:
        must_keep = {pid for pid in on_court if pid not in emergency_out and int(pf_map.get(pid, 0)) < foul_out}
        desired5, _dbg = pick_desired_five_bruteforce(
            team,
            home,
            game_state,
            rules,
            current_on_court=on_court,
            eligible_pool=eligible_pool,
            must_keep=must_keep,
            mode=mode,
            level=level,
            index=index,
            template_lineup=template_lineup,
            lock_mult=lock_mult,
        )
        desired_set = set(desired5)
        if desired_set != current_set:
            _set_on_court(game_state, team, home, list(desired5))
            game_state.rotation_last_sub_game_sec[key] = int(now)
            last_in = game_state.rotation_last_in_game_sec.setdefault(key, {})
            for pid in desired5:
                if pid not in current_set:
                    last_in[str(pid)] = int(now)
            return True

    # -------------------------
    # 3) Planned substitution (checkpoint only)
    # -------------------------
    # Planned subs are only evaluated at checkpoints (with catch-up) and are blocked by cooldown.
    planned_due = _rotation_checkpoint_due_and_mark(game_state, rules, team_key_str=key)
    if not planned_due:
        return False

    last_sub = int(game_state.rotation_last_sub_game_sec.get(key, -10**9))
    if int(now) - int(last_sub) < int(_ROTATION_PLANNED_COOLDOWN_SEC):
        return False

    # Build must_keep for planned (min stint + optional explicit locks)
    must_keep: Set[str] = set()

    clock = int(float(getattr(game_state, "clock_sec", 0)))
    relax_min_stint = (clock <= int(_ROTATION_MIN_STINT_RELAX_AT_CLOCK_SEC))
    last_in_map = game_state.rotation_last_in_game_sec.setdefault(key, {})
    if not relax_min_stint:
        for pid in on_court:
            entered = int(last_in_map.get(pid, 0))
            if int(now) - int(entered) < int(_ROTATION_MIN_STINT_SEC):
                must_keep.add(pid)

    # Optional legacy explicit locks apply to planned only
    locks = _explicit_lock_pids(team, ctx)
    for pid in locks:
        if pid in current_set:
            must_keep.add(pid)

    desired5, dbg = pick_desired_five_bruteforce(
        team,
        home,
        game_state,
        rules,
        current_on_court=on_court,
        eligible_pool=eligible_pool,
        must_keep=must_keep,
        mode=mode,
        level=level,
        index=index,
        template_lineup=template_lineup,
        lock_mult=lock_mult,
    )

    desired_set = set(desired5)
    if desired_set == current_set:
        return False

    # Planned Change Threshold: applied only in neutral/weak (OFF) situations.
    # (In MID/STRONG clutch/garbage, lineup enforcement is the point.)
    improvement = float(dbg.get("improvement", 0.0))
    threshold = float(preset.get("planned_change_threshold", _COACH_PRESETS["Balanced"]["planned_change_threshold"]))
    if mode == "NEUTRAL" and level == "OFF" and improvement < threshold:
        return False

    # Planned swap cap: apply up to 3 swaps, preserving Handler>=1 and Big>=1.
    out_list = [pid for pid in on_court if pid not in desired_set]
    in_list = [pid for pid in desired5 if pid not in current_set]
    if not out_list or not in_list:
        return False

    # Shape helpers (same as pick)
    role_by_pid: Dict[str, str] = {}
    team_roles = getattr(team, "rotation_offense_role_by_pid", None)
    if isinstance(team_roles, dict) and team_roles:
        role_by_pid = {str(pid): str(role) for pid, role in team_roles.items()}
    else:
        raw_roles = ctx.get("ROTATION_OFFENSE_ROLE_BY_PID") or ctx.get("OFFENSE_ROLE_BY_PID")
        if isinstance(raw_roles, dict):
            role_by_pid = {str(pid): str(role) for pid, role in raw_roles.items()}

    player_by_pid = {p.pid: p for p in team.lineup}

    def groups_for(pid: str) -> Tuple[str, ...]:
        role = role_by_pid.get(pid)
        if role and role in ROLE_TO_GROUPS:
            return ROLE_TO_GROUPS[role]
        pos = getattr(player_by_pid.get(pid), "pos", "G")
        return _fallback_groups_from_pos(pos)

    def is_handler(pid: str) -> bool:
        return "Handler" in groups_for(pid)

    def is_big(pid: str) -> bool:
        return "Big" in groups_for(pid)

    def lineup_shape_ok(pids: Sequence[str]) -> bool:
        return any(is_handler(pid) for pid in pids) and any(is_big(pid) for pid in pids)

    # Sort outs by low score (more removable), ins by high score (more desirable)
    out_sorted = sorted(
        out_list,
        key=lambda pid: _player_score_v1(pid, team=team, home=home, game_state=game_state, rules=rules, mode=mode, level=level),
    )
    remaining_in: Set[str] = set(in_list)

    new_on_court = list(on_court)
    swaps_done = 0

    for pid_out in out_sorted:
        if swaps_done >= int(_ROTATION_PLANNED_MAX_SWAPS):
            break
        if pid_out not in new_on_court:
            continue

        best_in: Optional[str] = None
        best_benefit: float = -1e18
        out_sc = _player_score_v1(
            pid_out, team=team, home=home, game_state=game_state, rules=rules, mode=mode, level=level
        )
        for pid_in in list(remaining_in):
            if pid_in in new_on_court:
                continue
            # test swap for shape
            tmp = list(new_on_court)
            tmp[tmp.index(pid_out)] = pid_in
            if not lineup_shape_ok(tmp):
                continue
            in_sc = _player_score_v1(
                pid_in, team=team, home=home, game_state=game_state, rules=rules, mode=mode, level=level
            )
            benefit = float(in_sc) - float(out_sc)
            if benefit > best_benefit:
                best_benefit = benefit
                best_in = pid_in

        if best_in is None:
            continue

        # apply swap
        idx_out = new_on_court.index(pid_out)
        new_on_court[idx_out] = best_in
        remaining_in.remove(best_in)
        swaps_done += 1

    if swaps_done <= 0:
        return False

    # Apply + update trackers
    prev_set = set(on_court)
    _set_on_court(game_state, team, home, list(new_on_court)[:5])
    game_state.rotation_last_sub_game_sec[key] = int(now)

    last_in = game_state.rotation_last_in_game_sec.setdefault(key, {})
    for pid in new_on_court[:5]:
        if pid not in prev_set:
            last_in[str(pid)] = int(now)

    return True

