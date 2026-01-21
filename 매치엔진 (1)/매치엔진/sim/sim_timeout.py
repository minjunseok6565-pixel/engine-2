from __future__ import annotations

import random
from typing import Any, Dict, Optional, Tuple

from .core import clamp
from .team_keys import HOME, AWAY


_DEADBALL_STARTS = {"start_q", "after_score", "after_tov_dead", "after_foul"}


def ensure_timeout_state(game_state: Any, rules: Dict[str, Any]) -> None:
    """Initialize timeout-related state on GameState (idempotent)."""
    to_rules = rules.get("timeouts", {}) if isinstance(rules, dict) else {}
    per_team = int(to_rules.get("per_team", 7))

    if not isinstance(getattr(game_state, "timeouts_remaining", None), dict) or not game_state.timeouts_remaining:
        game_state.timeouts_remaining = {HOME: per_team, AWAY: per_team}
    else:
        game_state.timeouts_remaining.setdefault(HOME, per_team)
        game_state.timeouts_remaining.setdefault(AWAY, per_team)

    if not isinstance(getattr(game_state, "timeouts_used", None), dict) or not game_state.timeouts_used:
        game_state.timeouts_used = {HOME: 0, AWAY: 0}
    else:
        game_state.timeouts_used.setdefault(HOME, 0)
        game_state.timeouts_used.setdefault(AWAY, 0)

    if not isinstance(getattr(game_state, "timeout_last_possession", None), dict) or not game_state.timeout_last_possession:
        game_state.timeout_last_possession = {HOME: -999999, AWAY: -999999}
    else:
        game_state.timeout_last_possession.setdefault(HOME, -999999)
        game_state.timeout_last_possession.setdefault(AWAY, -999999)

    if not isinstance(getattr(game_state, "timeout_log", None), list):
        game_state.timeout_log = []

    if not isinstance(getattr(game_state, "run_pts_by_scoring_side", None), dict) or not game_state.run_pts_by_scoring_side:
        game_state.run_pts_by_scoring_side = {HOME: 0, AWAY: 0}
    else:
        game_state.run_pts_by_scoring_side.setdefault(HOME, 0)
        game_state.run_pts_by_scoring_side.setdefault(AWAY, 0)

    if not isinstance(getattr(game_state, "consecutive_team_tos", None), dict) or not game_state.consecutive_team_tos:
        game_state.consecutive_team_tos = {HOME: 0, AWAY: 0}
    else:
        game_state.consecutive_team_tos.setdefault(HOME, 0)
        game_state.consecutive_team_tos.setdefault(AWAY, 0)

    # last_scoring_side is optional; leave as-is if already present


def is_deadball_window(pos_start: str) -> bool:
    return str(pos_start) in _DEADBALL_STARTS


def update_timeout_trackers(game_state: Any, offense_side: str, pos_res: Dict[str, Any]) -> None:
    """Update run / consecutive-TOV trackers. Call ONLY on true possession ends."""
    if not isinstance(pos_res, dict):
        return
    end_reason = str(pos_res.get("end_reason") or "")
    if end_reason in ("", "DEADBALL_STOP", "PERIOD_END"):
        return

    ensure_timeout_state(game_state, {})  # safe initialization if needed

    side = str(offense_side)
    if side not in (HOME, AWAY):
        return

    # Points scored (assumes offense is the scoring side when points_scored > 0)
    pts = int(pos_res.get("points_scored", 0) or 0)
    if pts > 0:
        scoring_side = side
        last = getattr(game_state, "last_scoring_side", None)
        if last == scoring_side:
            game_state.run_pts_by_scoring_side[scoring_side] = int(game_state.run_pts_by_scoring_side.get(scoring_side, 0)) + pts
        else:
            other = AWAY if scoring_side == HOME else HOME
            game_state.run_pts_by_scoring_side[scoring_side] = pts
            game_state.run_pts_by_scoring_side[other] = 0
            game_state.last_scoring_side = scoring_side

    # Consecutive turnovers (team possessions)
    if end_reason in ("TURNOVER", "SHOTCLOCK"):
        game_state.consecutive_team_tos[side] = int(game_state.consecutive_team_tos.get(side, 0)) + 1
    else:
        # any non-turnover possession by that team breaks its TO streak
        game_state.consecutive_team_tos[side] = 0


def maybe_timeout_deadball(
    rng: random.Random,
    game_state: Any,
    rules: Dict[str, Any],
    pos_start: str,
    next_offense_side: str,
    pressure_index: float,
    avg_energy_home: float,
    avg_energy_away: float,
    home_team_id: Optional[str] = None,
    away_team_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Attempt a dead-ball timeout (v1). Returns event dict if fired, else None."""
    if not isinstance(rules, dict):
        return None

    ai = rules.get("timeout_ai", {})
    if not isinstance(ai, dict) or not bool(ai.get("enabled", True)):
        return None
    if bool(ai.get("deadball_only", True)) and not is_deadball_window(pos_start):
        return None

    ensure_timeout_state(game_state, rules)

    allow_both = bool(ai.get("allow_both_teams_deadball", True))
    cand_sides = [HOME, AWAY] if allow_both else [str(next_offense_side)]

    poss_idx = int(getattr(game_state, "possession", 0) or 0)
    per_team = int((rules.get("timeouts", {}) or {}).get("per_team", 7))
    cooldown = int(ai.get("cooldown_possessions", 3))

    scorediff = int(getattr(game_state, "score_home", 0) or 0) - int(getattr(game_state, "score_away", 0) or 0)
    trailing_side = HOME if scorediff < 0 else AWAY if scorediff > 0 else None

    side_info = []
    for side in cand_sides:
        if side not in (HOME, AWAY):
            continue
        remaining = int(game_state.timeouts_remaining.get(side, per_team))
        if remaining <= 0:
            continue
        last_pos = int(game_state.timeout_last_possession.get(side, -999999))
        if poss_idx - last_pos < cooldown:
            continue

        p, reason = _compute_timeout_probability(
            side=side,
            remaining=remaining,
            per_team=per_team,
            game_state=game_state,
            rules=rules,
            pressure_index=float(pressure_index),
            avg_energy_home=float(avg_energy_home),
            avg_energy_away=float(avg_energy_away),
        )
        if p > 0:
            side_info.append((side, float(p), str(reason)))

    if not side_info:
        return None

    fired = []
    for side, p, reason in side_info:
        if float(rng.random()) < float(p):
            fired.append((side, p, reason))

    if not fired:
        return None

    # Choose at most one timeout per dead-ball window:
    # Prefer trailing side if it fired; otherwise pick highest p.
    if len(fired) == 1:
        chosen = fired[0]
    else:
        if trailing_side is not None:
            ts = [x for x in fired if x[0] == trailing_side]
            if ts:
                chosen = max(ts, key=lambda x: x[1])
            else:
                chosen = max(fired, key=lambda x: x[1])
        else:
            chosen = max(fired, key=lambda x: x[1])

    side, p, reason = chosen

    # Consume timeout
    game_state.timeouts_remaining[side] = int(game_state.timeouts_remaining.get(side, per_team)) - 1
    game_state.timeouts_used[side] = int(game_state.timeouts_used.get(side, 0)) + 1
    game_state.timeout_last_possession[side] = poss_idx

    team_id = (home_team_id if side == HOME else away_team_id) if (home_team_id or away_team_id) else None

    event = {
        "event_type": "TIMEOUT",
        "by_side": side,
        "by_team_id": team_id,
        "quarter": int(getattr(game_state, "quarter", 0) or 0),
        "clock_sec": float(getattr(game_state, "clock_sec", 0.0) or 0.0),
        "shot_clock_sec": float(getattr(game_state, "shot_clock_sec", 0.0) or 0.0),
        "pos_start": str(pos_start),
        "possession_index": poss_idx,
        "reason": str(reason),
        "timeouts_remaining_after": int(game_state.timeouts_remaining.get(side, 0)),
        "p": float(p),
    }

    game_state.timeout_log.append(dict(event))

    # Recovery (optional; default disabled)
    rec = rules.get("timeout_recovery", {})
    if isinstance(rec, dict) and bool(rec.get("enabled", False)):
        # v1: implement later if desired (kept as hook; no-op here by default)
        pass

    return event


def _compute_timeout_probability(
    side: str,
    remaining: int,
    per_team: int,
    game_state: Any,
    rules: Dict[str, Any],
    pressure_index: float,
    avg_energy_home: float,
    avg_energy_away: float,
) -> Tuple[float, str]:
    ai = rules.get("timeout_ai", {}) or {}
    val = rules.get("timeout_value", {}) or {}

    # --- Trigger G: run stop (opponent consecutive scoring points) ---
    opponent = AWAY if side == HOME else HOME
    run_pts = 0
    if getattr(game_state, "last_scoring_side", None) == opponent:
        run_pts = int((getattr(game_state, "run_pts_by_scoring_side", {}) or {}).get(opponent, 0))

    run_thr = int(ai.get("run_pts_threshold", 8))
    run_hard = int(ai.get("run_pts_hard", max(run_thr + 1, 12)))
    p_run = float(ai.get("p_run", 0.0))
    p_run_term = 0.0
    if run_pts >= run_thr and p_run > 0:
        s = _soft_hard_scale(float(run_pts), float(run_thr), float(run_hard), at_thr=0.60)
        p_run_term = p_run * s

    # --- Trigger G: ugly streak (same team consecutive turnovers) ---
    to_streak = int((getattr(game_state, "consecutive_team_tos", {}) or {}).get(side, 0))
    to_thr = int(ai.get("to_streak_threshold", 3))
    to_hard = int(ai.get("to_streak_hard", max(to_thr + 1, 4)))
    p_to = float(ai.get("p_to", 0.0))
    p_to_term = 0.0
    if to_streak >= to_thr and p_to > 0:
        s = _soft_hard_scale(float(to_streak), float(to_thr), float(to_hard), at_thr=0.65)
        p_to_term = p_to * s

    # --- Secondary triggers ---
    # Pressure-driven timeouts (continuous 0..1; replaces legacy is_clutch boolean)
    p_pressure = float(ai.get("p_pressure", 0.0))
    p_pressure_term = 0.0
    if p_pressure > 0:
        pr = clamp(float(pressure_index), 0.0, 1.0)
        if pr > 0:
            # Linear scaling by default; keep simple and predictable.
            p_pressure_term = p_pressure * pr

    fatigue_thr = float(ai.get("fatigue_threshold", 0.55))
    p_fatigue = float(ai.get("p_fatigue", 0.0))
    energy = float(avg_energy_home if side == HOME else avg_energy_away)
    p_fatigue_term = 0.0
    if p_fatigue > 0 and energy < fatigue_thr:
        # scale with how far below threshold we are
        s = clamp((fatigue_thr - energy) / max(fatigue_thr, 1e-6), 0.0, 1.0)
        p_fatigue_term = p_fatigue * (0.50 + 0.50 * s)

    p_base = float(ai.get("p_base", 0.0))

    base_p = p_base
    reason = "base"
    if p_run_term > base_p:
        base_p, reason = p_run_term, "run"
    if p_to_term > base_p:
        base_p, reason = p_to_term, "to_streak"
    if p_pressure_term > base_p:
        base_p, reason = p_pressure_term, "pressure"
    if p_fatigue_term > base_p:
        base_p, reason = p_fatigue_term, "fatigue"

    if base_p <= 0:
        return 0.0, reason

    # --- H: value multipliers ---
    # remaining-value (more remaining -> more willing)
    alpha = float(val.get("remaining_alpha", 0.70))
    if per_team <= 0:
        m_remaining = 1.0
    else:
        m_remaining = float((max(remaining, 0) / float(per_team)) ** max(alpha, 0.0))

    # blowout suppression
    soft = float(val.get("blowout_soft", 10.0))
    hard = float(val.get("blowout_hard", 18.0))
    floor = float(val.get("blowout_floor", 0.30))
    score_home = int(getattr(game_state, "score_home", 0) or 0)
    score_away = int(getattr(game_state, "score_away", 0) or 0)
    absdiff = float(abs(score_home - score_away))
    m_blowout = _linear_drop(absdiff, soft, hard, floor)

    # losing team calls more, winning team calls less
    diff_from_side = float((score_home - score_away) if side == HOME else (score_away - score_home))
    trail_scale = float(val.get("trail_scale", 12.0))
    trail_k = float(val.get("trail_k", 0.35))
    lead_scale = float(val.get("lead_scale", 12.0))
    lead_k = float(val.get("lead_k", 0.35))
    lead_floor = float(val.get("lead_floor", 0.55))

    if diff_from_side < 0:  # trailing
        t = clamp((-diff_from_side) / max(trail_scale, 1e-6), 0.0, 1.0)
        m_score = 1.0 + trail_k * t
    elif diff_from_side > 0:  # leading
        t = clamp((diff_from_side) / max(lead_scale, 1e-6), 0.0, 1.0)
        m_score = max(lead_floor, 1.0 - lead_k * t)
    else:
        m_score = 1.0

    # late-game conservatism (regulation progress only; OT treated as 1.0)
    late_beta = float(val.get("late_beta", 0.50))
    late_floor = float(val.get("late_floor", 0.60))
    progress = _regulation_progress(game_state, rules)
    m_late = max(late_floor, 1.0 - late_beta * progress)

    p = base_p * m_remaining * m_blowout * m_score * m_late
    p_cap = float(ai.get("p_cap", 0.85))
    p = clamp(p, 0.0, p_cap)

    return float(p), reason


def _soft_hard_scale(x: float, thr: float, hard: float, at_thr: float = 0.60) -> float:
    """0 below thr, then at_thr at thr, ramps to 1.0 at hard."""
    if x < thr:
        return 0.0
    if hard <= thr:
        return 1.0
    t = clamp((x - thr) / (hard - thr), 0.0, 1.0)
    return clamp(at_thr + (1.0 - at_thr) * t, 0.0, 1.0)


def _linear_drop(x: float, soft: float, hard: float, floor: float) -> float:
    """1.0 up to soft, linear down to floor at hard, then floor."""
    if x <= soft:
        return 1.0
    if hard <= soft:
        return max(floor, 0.0)
    if x >= hard:
        return max(floor, 0.0)
    t = (x - soft) / (hard - soft)
    return clamp(1.0 - t * (1.0 - max(floor, 0.0)), max(floor, 0.0), 1.0)


def _regulation_progress(game_state: Any, rules: Dict[str, Any]) -> float:
    """0..1 progress through regulation only (OT treated as 1.0)."""
    try:
        reg_q = int(rules.get("quarters", 4))
        qlen = float(rules.get("quarter_length", 720.0))
        q = int(getattr(game_state, "quarter", 1) or 1)
        clock = float(getattr(game_state, "clock_sec", 0.0) or 0.0)
        if reg_q <= 0 or qlen <= 0:
            return 0.0
        if q > reg_q:
            return 1.0
        elapsed = (q - 1) * qlen + (qlen - clock)
        total = reg_q * qlen
        return clamp(elapsed / max(total, 1e-6), 0.0, 1.0)
    except Exception:
        return 0.0
