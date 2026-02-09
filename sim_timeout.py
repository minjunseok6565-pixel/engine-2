from __future__ import annotations

import random
from typing import Any, Dict, Optional, Tuple

from .core import clamp
from .replay import emit_event


_DEADBALL_STARTS = {"start_q", "after_score", "after_tov_dead", "after_foul", "after_block_oob"}


def ensure_timeout_state(game_state: Any, rules: Dict[str, Any]) -> None:
    """Initialize timeout-related state on GameState (idempotent)."""
    to_rules = rules.get("timeouts", {}) if isinstance(rules, dict) else {}
    per_team = int(to_rules.get("per_team", 7))

    home_team_id = str(getattr(game_state, "home_team_id", "") or "").strip()
    away_team_id = str(getattr(game_state, "away_team_id", "") or "").strip()
    if not home_team_id or not away_team_id:
        raise ValueError(
            f"ensure_timeout_state(): GameState.home_team_id/away_team_id must be set (home={home_team_id!r}, away={away_team_id!r})"
        )
    if home_team_id == away_team_id:
        raise ValueError(f"ensure_timeout_state(): invalid game team ids (both {home_team_id!r})")

    def _ensure_team_dict_attr(attr: str, default_val: int) -> None:
        d = getattr(game_state, attr, None)
        if not isinstance(d, dict):
            d = {}
            setattr(game_state, attr, d)
        # Keyed ONLY by team_id. No side-key migration/normalization.
        if home_team_id not in d:
            d[home_team_id] = default_val
        if away_team_id not in d:
            d[away_team_id] = default_val

    _ensure_team_dict_attr("timeouts_remaining", per_team)
    _ensure_team_dict_attr("timeouts_used", 0)
    _ensure_team_dict_attr("timeout_last_possession", -999999)
    _ensure_team_dict_attr("run_pts", 0)
    _ensure_team_dict_attr("consecutive_team_tos", 0)

    # last_scoring_team_id is optional; leave as-is if already present


def is_deadball_window(pos_start: str) -> bool:
    return str(pos_start) in _DEADBALL_STARTS


def update_timeout_trackers(game_state: Any, offense_team_id: str, pos_res: Dict[str, Any]) -> None:
    """Update run / consecutive-TOV trackers. Call ONLY on true possession ends."""
    if not isinstance(pos_res, dict):
        return
    end_reason = str(pos_res.get("end_reason") or "")
    if end_reason in ("", "DEADBALL_STOP", "PERIOD_END"):
        return

    ensure_timeout_state(game_state, {})  # safe initialization if needed

    tid = str(offense_team_id).strip()
    home_team_id = str(getattr(game_state, "home_team_id", "") or "").strip()
    away_team_id = str(getattr(game_state, "away_team_id", "") or "").strip()
    if tid not in (home_team_id, away_team_id):
        raise ValueError(
            f"update_timeout_trackers(): offense_team_id not in this game: offense={tid!r}, home={home_team_id!r}, away={away_team_id!r}"
        )
    opp_tid = away_team_id if tid == home_team_id else home_team_id

    # Points scored (assumes offense is the scoring side when points_scored > 0)
    pts = int(pos_res.get("points_scored", 0) or 0)
    if pts > 0:
        last = getattr(game_state, "last_scoring_team_id", None)
        if last == tid:
            game_state.run_pts[tid] = int(game_state.run_pts.get(tid, 0)) + pts
        else:
            game_state.run_pts[tid] = pts
            game_state.run_pts[opp_tid] = 0
            game_state.last_scoring_team_id = tid

    # Consecutive turnovers (team possessions)
    if end_reason in ("TURNOVER", "SHOTCLOCK"):
        game_state.consecutive_team_tos[tid] = int(game_state.consecutive_team_tos.get(tid, 0)) + 1
    else:
        # any non-turnover possession by that team breaks its TO streak
        game_state.consecutive_team_tos[tid] = 0


def maybe_timeout_deadball(
    rng: random.Random,
    game_state: Any,
    rules: Dict[str, Any],
    pos_start: str,
    next_offense_team_id: str,
    pressure_index: float,
    avg_energy_home: float,
    avg_energy_away: float,
    # Preferred: pass actual TeamState objects so TIMEOUT can be logged into replay_events.
    home: Any = None,
    away: Any = None,
) -> Optional[Dict[str, Any]]:
    """Attempt a dead-ball timeout (v1). Returns replay_event dict if fired, else None.

    IMPORTANT: 기록(Source of Truth)은 replay_events 하나로 통일한다.
    TIMEOUT은 emit_event(...)로만 1회 기록하며, 별도 timeout_log는 유지하지 않는다.
    """
    if not isinstance(rules, dict):
        return None

    ai = rules.get("timeout_ai", {})
    if not isinstance(ai, dict) or not bool(ai.get("enabled", True)):
        return None
    if bool(ai.get("deadball_only", True)) and not is_deadball_window(pos_start):
        return None

    ensure_timeout_state(game_state, rules)

    home_team_id = str(getattr(game_state, "home_team_id", "") or "").strip()
    away_team_id = str(getattr(game_state, "away_team_id", "") or "").strip()
    if not home_team_id or not away_team_id:
        raise ValueError(
            f"maybe_timeout_deadball(): GameState.home_team_id/away_team_id must be set (home={home_team_id!r}, away={away_team_id!r})"
        )
    if home_team_id == away_team_id:
        raise ValueError(f"maybe_timeout_deadball(): invalid game team ids (both {home_team_id!r})")

    # Timeout logic relies on authoritative TeamState objects for score snapshots + replay emission.
    if home is None or away is None:
        raise ValueError(
            f"maybe_timeout_deadball(): home/away TeamState objects must be provided for replay emission (game home={home_team_id!r}, away={away_team_id!r})"
        )
    if str(getattr(home, "team_id", "") or "").strip() != home_team_id:
        raise ValueError(
            f"maybe_timeout_deadball(): home TeamState.team_id mismatch: game_state.home_team_id={home_team_id!r}, home.team_id={getattr(home, 'team_id', None)!r}"
        )
    if str(getattr(away, "team_id", "") or "").strip() != away_team_id:
        raise ValueError(
            f"maybe_timeout_deadball(): away TeamState.team_id mismatch: game_state.away_team_id={away_team_id!r}, away.team_id={getattr(away, 'team_id', None)!r}"
        )
    
    allow_both = bool(ai.get("allow_both_teams_deadball", True))
    next_tid = str(next_offense_team_id).strip()
    if next_tid not in (home_team_id, away_team_id):
        raise ValueError(
            f"maybe_timeout_deadball(): next_offense_team_id not in this game: next={next_tid!r}, home={home_team_id!r}, away={away_team_id!r}"
        )
    cand_team_ids = [home_team_id, away_team_id] if allow_both else [next_tid]

    poss_idx = int(getattr(game_state, "possession", 0) or 0)
    per_team = int((rules.get("timeouts", {}) or {}).get("per_team", 7))
    cooldown = int(ai.get("cooldown_possessions", 3))

    score_home = int(getattr(home, "pts", 0) or 0)
    score_away = int(getattr(away, "pts", 0) or 0)
    scorediff = score_home - score_away
    trailing_team_id = home_team_id if scorediff < 0 else away_team_id if scorediff > 0 else None

    side_info = []
    for tid in cand_team_ids:
        remaining = int(game_state.timeouts_remaining.get(tid, per_team))
        if remaining <= 0:
            continue
        last_pos = int(game_state.timeout_last_possession.get(tid, -999999))
        if poss_idx - last_pos < cooldown:
            continue

        p, reason = _compute_timeout_probability(
            team_id=tid,
            remaining=remaining,
            per_team=per_team,
            game_state=game_state,
            rules=rules,
            pressure_index=float(pressure_index),
            avg_energy_home=float(avg_energy_home),
            avg_energy_away=float(avg_energy_away),
            score_home=score_home,
            score_away=score_away,
        )
        if p > 0:
            side_info.append((tid, float(p), str(reason)))

    if not side_info:
        return None

    fired = []
    for tid, p, reason in side_info:
        if float(rng.random()) < float(p):
            fired.append((tid, p, reason))

    if not fired:
        return None

    # Choose at most one timeout per dead-ball window:
    # Prefer trailing side if it fired; otherwise pick highest p.
    if len(fired) == 1:
        chosen = fired[0]
    else:
        if trailing_team_id is not None:
            ts = [x for x in fired if x[0] == trailing_team_id]
            if ts:
                chosen = max(ts, key=lambda x: x[1])
            else:
                chosen = max(fired, key=lambda x: x[1])
        else:
            chosen = max(fired, key=lambda x: x[1])

    team_id, p, reason = chosen
    opp_team_id = away_team_id if team_id == home_team_id else home_team_id

    # Consume timeout
    game_state.timeouts_remaining[team_id] = int(game_state.timeouts_remaining.get(team_id, per_team)) - 1
    game_state.timeouts_used[team_id] = int(game_state.timeouts_used.get(team_id, 0)) + 1
    game_state.timeout_last_possession[team_id] = poss_idx

    # 기록은 replay_events로만 1회 남긴다.
    event = emit_event(
        game_state,
        event_type="TIMEOUT",
        home=home,
        away=away,
        rules=rules,
        include_lineups=True,
        team_id=str(team_id),
        opp_team_id=str(opp_team_id),
        pos_start=str(pos_start),
        reason=str(reason),
        timeouts_remaining_after=int(game_state.timeouts_remaining.get(team_id, 0)),
        p=float(p),
        # Optional snapshots (이미 존재하는 상태값을 이벤트 시점 스냅샷으로 담음)
        timeouts_remaining=dict(getattr(game_state, "timeouts_remaining", {}) or {}),
        timeouts_used=dict(getattr(game_state, "timeouts_used", {}) or {}),
        run_pts=dict(getattr(game_state, "run_pts", {}) or {}),
        consecutive_team_tos=dict(getattr(game_state, "consecutive_team_tos", {}) or {}),
        last_scoring_team_id=getattr(game_state, "last_scoring_team_id", None),
    )

    return event


def _compute_timeout_probability(
    team_id: str,
    remaining: int,
    per_team: int,
    game_state: Any,
    rules: Dict[str, Any],
    pressure_index: float,
    avg_energy_home: float,
    avg_energy_away: float,
    score_home: int,
    score_away: int,
) -> Tuple[float, str]:
    ai = rules.get("timeout_ai", {}) or {}
    val = rules.get("timeout_value", {}) or {}

    # --- Trigger G: run stop (opponent consecutive scoring points) ---
    tid = str(team_id).strip()
    home_team_id = str(getattr(game_state, "home_team_id", "") or "").strip()
    away_team_id = str(getattr(game_state, "away_team_id", "") or "").strip()
    if tid not in (home_team_id, away_team_id):
        raise ValueError(
            f"_compute_timeout_probability(): team_id not in this game: team_id={tid!r}, home={home_team_id!r}, away={away_team_id!r}"
        )
    opp_tid = away_team_id if tid == home_team_id else home_team_id

    run_pts = 0
    if getattr(game_state, "last_scoring_team_id", None) == opp_tid:
        run_pts = int((getattr(game_state, "run_pts", {}) or {}).get(opp_tid, 0))

    run_thr = int(ai.get("run_pts_threshold", 8))
    run_hard = int(ai.get("run_pts_hard", max(run_thr + 1, 12)))
    p_run = float(ai.get("p_run", 0.0))
    p_run_term = 0.0
    if run_pts >= run_thr and p_run > 0:
        s = _soft_hard_scale(float(run_pts), float(run_thr), float(run_hard), at_thr=0.60)
        p_run_term = p_run * s

    # --- Trigger G: ugly streak (same team consecutive turnovers) ---
    to_streak = int((getattr(game_state, "consecutive_team_tos", {}) or {}).get(tid, 0))
    to_thr = int(ai.get("to_streak_threshold", 3))
    to_hard = int(ai.get("to_streak_hard", max(to_thr + 1, 4)))
    p_to = float(ai.get("p_to", 0.0))
    p_to_term = 0.0
    if to_streak >= to_thr and p_to > 0:
        s = _soft_hard_scale(float(to_streak), float(to_thr), float(to_hard), at_thr=0.65)
        p_to_term = p_to * s

    # --- Secondary triggers ---
    # Pressure-driven timeouts (continuous 0..1).
    p_pressure = float(ai.get("p_pressure", 0.0))
    p_pressure_term = 0.0
    if p_pressure > 0:
        pr = clamp(float(pressure_index), 0.0, 1.0)
        if pr > 0:
            # Linear scaling by default; keep simple and predictable.
            p_pressure_term = p_pressure * pr

    fatigue_thr = float(ai.get("fatigue_threshold", 0.55))
    p_fatigue = float(ai.get("p_fatigue", 0.0))
    if tid == home_team_id:
        energy = float(avg_energy_home)
    else:
        energy = float(avg_energy_away)
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
    absdiff = float(abs(score_home - score_away))
    m_blowout = _linear_drop(absdiff, soft, hard, floor)

    # losing team calls more, winning team calls less
    # diff_from_team: team_score - opp_score (positive when leading)
    team_score = float(score_home if tid == home_team_id else score_away)
    opp_score = float(score_away if tid == home_team_id else score_home)
    diff_from_team = team_score - opp_score
    trail_scale = float(val.get("trail_scale", 12.0))
    trail_k = float(val.get("trail_k", 0.35))
    lead_scale = float(val.get("lead_scale", 12.0))
    lead_k = float(val.get("lead_k", 0.35))
    lead_floor = float(val.get("lead_floor", 0.55))

    if diff_from_team < 0:  # trailing
        t = clamp((-diff_from_team) / max(trail_scale, 1e-6), 0.0, 1.0)
        m_score = 1.0 + trail_k * t
    elif diff_from_team > 0:  # leading
        t = clamp((diff_from_team) / max(lead_scale, 1e-6), 0.0, 1.0)
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
