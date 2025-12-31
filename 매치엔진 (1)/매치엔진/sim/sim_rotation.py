from __future__ import annotations

"""Rotation utilities (on-court tracking, minutes accounting, auto-sub logic).

This module controls:
- per-player minutes tracking
- auto-substitution / rotation decisions
- on-court pid lists in GameState

NOTE: Split from sim.py on 2025-12-27.
"""

import random
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

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


def _perform_rotation(
    rng: random.Random,
    team: TeamState,
    home: TeamState,
    game_state: GameState,
    rules: Dict[str, Any],
    is_garbage: bool,
) -> None:
    """Auto-sub logic with priorities:
    1) converge to user-specified per-player minutes targets (soft constraint; best-effort)
    2) maintain Initiator_Primary exactly once on-court (hard constraint if configured)
    3) prefer swaps within same role group (Handler/Wing/Big), otherwise allow cross-group
    4) use fatigue as a performance-preservation tie-breaker within the above constraints

    Notes:
    - fatigue is NOT a hard in/out gate (no sub_in threshold). This avoids "no available subs" stalls.
    - players with a regulation-length target (e.g., 48:00) are treated as locked and will not be subbed
      due to fatigue/target logic (only foul-out can force removal).
    """
    ctx = _get_tactics_context(team)

    # role assignment (pid -> role_name)
    role_by_pid: Dict[str, str] = {}
    # 1) TeamState fields (preferred)
    team_roles = getattr(team, "rotation_offense_role_by_pid", None)
    if isinstance(team_roles, dict) and team_roles:
        role_by_pid = {str(pid): str(role) for pid, role in team_roles.items()}
    else:
        # 2) tactics.context fields
        raw_roles = ctx.get("ROTATION_OFFENSE_ROLE_BY_PID") or ctx.get("OFFENSE_ROLE_BY_PID")
        if isinstance(raw_roles, dict):
            role_by_pid = {str(pid): str(role) for pid, role in raw_roles.items()}

    # explicit locks (optional)
    explicit_locks: Set[str] = set(_coerce_pid_list(getattr(team, "rotation_lock_pids", None)))
    if not explicit_locks:
        explicit_locks = set(_coerce_pid_list(ctx.get("ROTATION_LOCK_PIDS") or ctx.get("LOCK_PIDS")))

    foul_out = int(rules.get("foul_out", 6))
    targets = game_state.targets_sec_home if team is home else game_state.targets_sec_away
    on_court = list(_get_on_court(game_state, team, home))
    key = team_key(team, home)
    pf_map = game_state.player_fouls.get(key, {})
    fat_map = game_state.fatigue.get(key, {})
    mins_map = game_state.minutes_played_sec.get(key, {})

    # bench candidates: any non-fouled-out player not currently on court
    bench = [
        p.pid
        for p in team.lineup
        if p.pid not in on_court and pf_map.get(p.pid, 0) < foul_out
    ]

    # helpers
    def fatigue(pid: str) -> float:
        return float(fat_map.get(pid, 1.0))

    def minutes(pid: str) -> int:
        return int(mins_map.get(pid, 0))

    def target(pid: str) -> int:
        return int(targets.get(pid, 0))

    def deficit(pid: str) -> int:
        # positive => needs minutes; negative => over target
        return target(pid) - minutes(pid)

    remaining_est = _estimate_remaining_game_sec(game_state, rules)
    reg_total = _regulation_total_sec(rules)

    def is_locked(pid: str) -> bool:
        # Explicit lock OR "48:00 target" style lock
        # (If user sets beyond regulation, treat as locked as well.)
        return pid in explicit_locks or target(pid) >= reg_total

    def role_name(pid: str) -> str:
        return role_by_pid.get(pid, "")

    def is_initiator_primary(pid: str) -> bool:
        return role_name(pid) == "Initiator_Primary"

    def groups(pid: str) -> Tuple[str, ...]:
        rn = role_name(pid)
        if rn in ROLE_TO_GROUPS:
            return ROLE_TO_GROUPS[rn]
        # fallback by player position
        pl = team.find_player(pid) if hasattr(team, "find_player") else None
        pos = getattr(pl, "pos", "F") if pl is not None else "F"
        return _fallback_groups_from_pos(pos)

    def group_match_bonus(pid_out: str, pid_in: str) -> float:
        go = set(groups(pid_out))
        gi = set(groups(pid_in))
        if not go or not gi:
            return 0.0
        if go == gi:
            return 2.0
        if go.intersection(gi):
            return 1.0
        return 0.0

    # Initiator_Primary constraint is enabled only if user assigns at least one eligible pid.
    eligible_initiators: Set[str] = {pid for pid, rn in role_by_pid.items() if rn == "Initiator_Primary"}
    enforce_initiator = bool(eligible_initiators)

    def initiator_count(pids: Sequence[str]) -> int:
        return sum(1 for pid in pids if is_initiator_primary(pid))

    # "Rest budget": can a player afford to sit and still hit target?
    # played + remaining_est - target > 0 means they can miss some time and still hit target.
    def rest_budget(pid: str) -> int:
        return minutes(pid) + remaining_est - target(pid)

    # Score bench players: prioritize hitting targets first; use fatigue/garbage as tie-breakers
    def in_score(pid_in: str) -> float:
        # deficit is primary: more minutes needed => higher score
        d = deficit(pid_in)
        # fatigue is secondary: prefer fresher players when targets allow
        f = fatigue(pid_in)
        # small penalty if already over target (lets still be chosen if forced by constraints)
        over_pen = -min(0, d) / 120.0
        score = (max(d, 0) / 60.0) + (0.6 * f) + over_pen

        if is_garbage:
            # slight preference for lower-target players during garbage time
            score -= (target(pid_in) / 60.0) * 0.05
        return score

    # Score on-court players for removal: prefer those who can afford rest and are not needed for targets
    def out_score(pid_out: str) -> float:
        if pf_map.get(pid_out, 0) >= foul_out:
            return 1e9  # forced
        if is_locked(pid_out):
            return -1e9  # do not sub out (unless foul-out)
        rb = rest_budget(pid_out)
        if rb <= 0:
            # If they can't afford rest, strongly discourage subbing out
            return -1e6

        d = deficit(pid_out)
        f = fatigue(pid_out)

        # More over-target => more removable
        over = max(-d, 0) / 60.0  # minutes over target
        # More rest budget => more removable
        rest = rb / 120.0
        # More tired => more removable (but not dominant)
        tired = (1.0 - f) * 1.5

        score = over + rest + tired
        if is_garbage:
            # prefer to rest high-target players in garbage time
            score += (target(pid_out) / 60.0) * 0.05
        return score

    # Build candidate lists
    out_candidates = sorted(on_court, key=out_score, reverse=True)
    in_candidates = sorted(bench, key=in_score, reverse=True)

    swaps = 0

    # Attempt up to 2 swaps (keeps stability; caller runs this frequently).
    for pid_out in out_candidates:
        if swaps >= 2:
            break

        # If no one on bench, stop
        if not in_candidates:
            break

        # If pid_out is a forced foul-out, we must replace regardless.
        forced_out = pf_map.get(pid_out, 0) >= foul_out

        # Select the best pid_in subject to constraints.
        best_pid_in: Optional[str] = None
        best_score: float = -1e18

        current_initiators = initiator_count(on_court) if enforce_initiator else 0

        for pid_in in in_candidates:
            # Do not sub in a fouled-out player (already filtered, but keep safe)
            if pf_map.get(pid_in, 0) >= foul_out:
                continue

            # Initiator constraint:
            # - must end with exactly 1 initiator on court (if enforce_initiator)
            if enforce_initiator:
                out_is_init = is_initiator_primary(pid_out)
                in_is_init = is_initiator_primary(pid_in)

                new_count = current_initiators - (1 if out_is_init else 0) + (1 if in_is_init else 0)

                # Disallow 2 initiators
                if new_count > 1:
                    continue

                # Disallow 0 initiators unless truly impossible in this moment
                # (e.g., no eligible initiator available on bench and none currently on court after out)
                if new_count < 1:
                    # If we already have 1 initiator and we're trying to remove it, only allow if no
                    # initiator exists to put in.
                    bench_has_init = any(is_initiator_primary(b) for b in in_candidates)
                    if bench_has_init:
                        continue

            s = in_score(pid_in) + group_match_bonus(pid_out, pid_in)

            # If this out is not very removable (e.g., negative score), only allow swap if forced
            if (out_score(pid_out) < -1e5) and not forced_out:
                continue

            if s > best_score:
                best_score = s
                best_pid_in = pid_in

        if best_pid_in is None:
            # No valid in found for this out; try next out candidate.
            continue

        # Execute swap
        in_candidates.remove(best_pid_in)
        if pid_out in on_court:
            on_court[on_court.index(pid_out)] = best_pid_in
            swaps += 1

    # Enforce Initiator_Primary exactly once if enabled (post-fix)
    if enforce_initiator:
        cur = initiator_count(on_court)

        # If we have 0 initiator, try to bring one in from bench
        if cur == 0:
            bench_inits = [pid for pid in bench if is_initiator_primary(pid)]
            if bench_inits:
                pid_in = max(bench_inits, key=in_score)
                # choose someone to sub out (not locked) with worst "need to be on court"
                non_locked_on = [pid for pid in on_court if not is_locked(pid) and not is_initiator_primary(pid)]
                if non_locked_on:
                    pid_out = max(non_locked_on, key=out_score)
                    on_court[on_court.index(pid_out)] = pid_in

        # If we have 2+ initiators, remove extras
        elif cur > 1:
            # keep the initiator with highest need (deficit), remove others
            inits = [pid for pid in on_court if is_initiator_primary(pid)]
            keep = max(inits, key=lambda pid: (deficit(pid), fatigue(pid)))
            extras = [pid for pid in inits if pid != keep]

            non_initiator_bench = [pid for pid in bench if not is_initiator_primary(pid)]
            for pid_out in extras:
                if not non_initiator_bench:
                    break
                pid_in = max(non_initiator_bench, key=in_score)
                non_initiator_bench.remove(pid_in)
                if pid_out in on_court and not is_locked(pid_out):
                    on_court[on_court.index(pid_out)] = pid_in

    _set_on_court(game_state, team, home, on_court[:5])
