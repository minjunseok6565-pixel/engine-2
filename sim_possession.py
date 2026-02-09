from __future__ import annotations

"""Possession simulation (team style biasing, priors, resolve loop).

NOTE: Split from sim.py on 2025-12-27.
"""

import random
import math
import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

from .builders import (
    build_offense_action_probs,
    build_outcome_priors,
    get_action_base,
)
from . import shot_diet
from . import quality
from . import matchups
from .def_role_players import get_or_build_def_role_players, engine_get_stat
from .core import weighted_choice, clamp
from .models import GameState, TeamState
from .resolve import (
    choose_drb_rebounder,
    choose_orb_rebounder,
    rebound_orb_probability,
    resolve_outcome,
    commit_pending_pass_event,
)
from .sim_rotation import maybe_substitute_deadball_v1
from .role_fit import apply_role_fit_to_priors_and_tags

from .replay import emit_event
from .sim_clock import (
    apply_time_cost,
    apply_dead_ball_cost,
    simulate_inbound,
    commit_shot_clock_turnover,
)

if TYPE_CHECKING:
    from .game_config import GameConfig

#
# Turnover deadball/liveball classification
# ---------------------------------------
# These are the ONLY turnover outcome strings emitted by resolve_outcome()/sim_clock in this engine:
#   TO_HANDLE_LOSS, TO_BAD_PASS, TO_CHARGE, TO_INBOUND, TO_SHOT_CLOCK
#
# Policy:
#   - Deadball turnovers: charge / inbound / shot clock
#   - Liveball turnovers: handle loss / bad pass
#
# NOTE:
#   resolve_outcome() may attach payload flags that override this classification:
#     - payload['deadball_override'] -> force deadball (e.g., bad-pass lineout)
#     - payload['pos_start_next_override'] / payload['steal'] -> force after_steal start
#

from .possession.turnover_policy import _normalize_turnover_outcome, _turnover_is_deadball
from .possession.validate import _validate_possession_team_ids
from .possession.replay_payload import _clean_replay_payload
from .possession.quality_bias import apply_quality_to_turnover_priors
from .possession.stats import _player_stat
from .possession.team_style import (
    ensure_team_style,
    apply_team_style_to_action_probs,
    apply_team_style_to_outcome_priors,
)
from .possession.tactics_ctx import make_possession_tactics_ctx
from .possession.late_clock import build_late_clock_guardrails
from .possession.priors_bias import (
    apply_help_to_priors,
    apply_double_to_priors,
    apply_rotation_advantage_to_priors,
)

def simulate_possession(
    rng: random.Random,
    offense: TeamState,
    defense: TeamState,
    game_state: GameState,
    rules: Dict[str, Any],
    ctx: Dict[str, Any],
    game_cfg: Optional["GameConfig"] = None,
    max_steps: int = 7,
) -> Dict[str, Any]:
    """Simulate a single possession.

    Returns a dict describing how the possession ended so the game loop can be event-based.
    """

    if ctx is None:
        ctx = {}
    if game_cfg is None:
        raise ValueError("simulate_possession requires game_cfg")

    # Replay logging: infer which TeamState corresponds to home/away.
    # (We do this once per simulate_possession call; it's cheap and keeps logging consistent.)
    home_team, away_team, off_team_id, def_team_id = _validate_possession_team_ids(offense, defense, game_state, ctx or {})

    # Possession-continuation support:
    # Some dead-ball events (e.g. no-shot foul) can stop play and restart with the same offense.
    # In those cases, the game loop will call simulate_possession again with ctx['_pos_continuation']=True.
    # We must avoid double-counting possessions and must preserve possession-scope aggregates.
    is_continuation = bool(ctx.get("_pos_continuation", False))
    if not is_continuation:
        offense.possessions += 1
        before_pts = int(offense.pts)
        # Allow exactly one MATCHUP_SET per possession.
        # Continuation calls (e.g. no-shot foul restart) should not re-emit the initial 5v5 map.
        ctx.pop("_matchup_set_emitted", None)
        # Clear possession-scoped tactical controls so they never leak across possessions.
        ctx.pop("_hunt_plan_applied", None)
        ctx.pop("hunt_action_mult_by_base", None)
        ctx.pop("double_active", None)
        ctx.pop("matchups_temp_locks", None)
        ctx.pop("team_help_level", None)
        ctx.pop("team_help_delta", None)
        ctx.pop("help_level_by_pid", None)
        ctx.pop("def_pressure", None)
        ctx.pop("rotation_adv", None)
    else:
        before_pts = int(ctx.get("_pos_before_pts", int(offense.pts)))

    # Current segment start vs. possession-origin start (for attribution like fastbreak/points_off_tov).
    pos_start = str(ctx.get("pos_start", ""))
    pos_origin = str(ctx.get("_pos_origin_start", pos_start))

    def _record_ctx_error(where: str, exc: BaseException) -> None:
        try:
            errs = ctx.setdefault("errors", [])
            errs.append(
                {
                    "where": where,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        except Exception:
            return

    def _top_up_shot_clock_after_def_no_shot_foul() -> None:
        """Top up shot clock to foul_reset (e.g., 14) if remaining is below it.

        This is used for defensive no-shot fouls where the offense retains the ball and play
        will restart with an inbound in a *separate* continuation segment (pos_start='after_foul').
        """
        try:
            foul_reset = float(rules.get("foul_reset", 14))
        except Exception:
            foul_reset = 14.0
        try:
            full_sc = float(rules.get("shot_clock", 24))
        except Exception:
            full_sc = 24.0

        if foul_reset <= 0 or full_sc <= 0:
            return
        foul_reset = min(foul_reset, full_sc)
        try:
            if float(game_state.shot_clock_sec) < foul_reset:
                game_state.shot_clock_sec = foul_reset
        except Exception:
            # If shot_clock_sec is missing/invalid, fall back to foul_reset.
            game_state.shot_clock_sec = foul_reset
            

    tempo_mult = float(ctx.get("tempo_mult", 1.0))
    time_costs = rules.get("time_costs", {})
    had_orb = bool(ctx.get("_pos_had_orb", False))

    # per-team style profile (persistent; increases team diversity)
    team_style = ensure_team_style(rng, offense, rules)
    if team_style:
        tempo_mult *= float(team_style.get("tempo_mult", 1.0))
        # mutate ctx in-place (continuation must preserve possession-scope flags)
        ctx["tempo_mult"] = tempo_mult
        ctx["team_style"] = team_style

    # Dead-ball start can trigger inbound (score, quarter start, dead-ball TO, no-shot foul restart, etc.)
    dead_ball_starts = {"start_q", "after_score", "after_tov_dead", "after_foul", "after_block_oob"}
    if pos_start in dead_ball_starts:
        # dead-ball inbound attempt
        if simulate_inbound(rng, offense, defense, rules):
            # IMPORTANT:
            # Inbound turnovers are dead-ball turnovers. Next possession should start as dead-ball inbound.
            # Also, inbound turnover consumes 0 action-time, so repeated inbound turnovers could freeze the
            # game clock in the outer loop unless we apply a small dead-ball admin cost.
            time_costs = rules.get("time_costs", {}) or {}
            try:
                inbound_tov_cost = float(time_costs.get("InboundTurnover", 1.0))
            except Exception:
                inbound_tov_cost = 1.0
            if inbound_tov_cost > 0:
                apply_dead_ball_cost(game_state, inbound_tov_cost, tempo_mult)
                if game_state.clock_sec <= 0:
                    game_state.clock_sec = 0
                    return {
                        "end_reason": "PERIOD_END",
                        "pos_start_next": pos_start,
                        "points_scored": int(offense.pts) - before_pts,
                        "had_orb": had_orb,
                        "pos_start": pos_origin,
                        "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
                    }
                    
                    
            return {
                "end_reason": "TURNOVER",
                "pos_start_next": "after_tov_dead",
                "points_scored": int(offense.pts) - before_pts,
                "had_orb": had_orb,
                "pos_start": pos_origin,
                "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
                "turnover_outcome": "TO_INBOUND",
                "turnover_deadball": True,
            }

    # shot_diet wiring
    style = shot_diet.compute_shot_diet_style(offense, defense, game_state=game_state, ctx=ctx)
    tactic_name = None
    try:
        tactic_name = offense.tactics.offense_scheme
    except Exception as exc:
        _record_ctx_error("tactic_name_access", exc)
        tactic_name = None
    ctx["shot_diet_style"] = style
    ctx["tactic_name"] = tactic_name

    # --- Matchups (Plan-1 MVP) ---
    # Build and maintain a 5v5 OFF_PID -> DEF_PID matchup map for the current on-court units.
    # This mapping is used by resolve.py to pick a primary defender and blend defensive values.
    _ensure_matchups, _cache_help_levels, _maybe_apply_hunt_plan, _maybe_inject_matchup_force, _update_def_pressure_for_step = make_possession_tactics_ctx(
        offense=offense,
        defense=defense,
        game_state=game_state,
        ctx=ctx,
        rng=rng,
        home_team=home_team,
        away_team=away_team,
        rules=rules,
        off_team_id=off_team_id,
        def_team_id=def_team_id,
        pos_origin=pos_origin,
        is_continuation=is_continuation,
        emit_event=emit_event,
        clamp=clamp,
        player_stat=_player_stat,
        record_ctx_error=_record_ctx_error,
    )

    # Build initial matchup map for this possession/segment.
    _ensure_matchups(reason="pos_start")
    _cache_help_levels()
    _maybe_apply_hunt_plan()

    def _apply_contextual_action_weights(probs: Dict[str, float]) -> Dict[str, float]:
        """Soft-bias action weights by possession context (no per-team fixed style)."""
        if not probs:
            return probs
        if bool(ctx.get("dead_ball_inbound", False)):
            return probs
        out = dict(probs)
        changed = False

        # Transition bias (existing behavior).
        pstart = str(ctx.get("pos_start", pos_start))
        if pstart in ("after_drb", "after_tov", "after_steal", "after_block"):
            mult_tbl = rules.get("transition_weight_mult", {}) or {}
            try:
                mult = float(mult_tbl.get(pstart, mult_tbl.get("default", 1.0)))
            except Exception:
                mult = 1.0
            if mult > 1.0:
                for k, v in list(out.items()):
                    if get_action_base(k, game_cfg) == "TransitionEarly":
                        out[k] = float(v) * mult
                        changed = True

        # Hunt bias (base-action multipliers).
        hunt_mult = ctx.get("hunt_action_mult_by_base")
        if isinstance(hunt_mult, dict) and hunt_mult:
            for k, v in list(out.items()):
                base = get_action_base(k, game_cfg)
                if base in hunt_mult:
                    try:
                        m = float(hunt_mult.get(base, 1.0))
                    except Exception:
                        m = 1.0
                    if m != 1.0:
                        out[k] = float(v) * clamp(m, 0.25, 2.50)
                        changed = True

        if not changed:
            return probs
        s = sum(float(x) for x in out.values())
        if s <= 0:
            return probs
        for k in out:
            out[k] = float(out[k]) / s
        return out



    # -------------------------------------------------------------------------
    # Late-clock action selection guardrails
    # -------------------------------------------------------------------------
    # Problem 1/2 fix: prevent "no attempt" period ends and excessive shotclock
    # violations by selecting only feasible actions given the remaining time.

    _late_clock = build_late_clock_guardrails(game_state, rules, tempo_mult, game_cfg)

    # Rebind locals for backward compatibility with the existing loop code below.
    time_costs = _late_clock.time_costs
    timing = _late_clock.timing

    min_release_window = _late_clock.min_release_window
    urgent_budget_sec = _late_clock.urgent_budget_sec
    quickshot_cost_sec = _late_clock.quickshot_cost_sec
    soft_slack_span = _late_clock.soft_slack_span
    soft_slack_floor = _late_clock.soft_slack_floor
    quickshot_inject_base = _late_clock.quickshot_inject_base
    quickshot_inject_urgency_mult = _late_clock.quickshot_inject_urgency_mult
    pass_reset_suppress_urgency = _late_clock.pass_reset_suppress_urgency

    _budget_sec = _late_clock.budget_sec
    _estimate_action_cost_sec = _late_clock.estimate_action_cost_sec
    _is_nonterminal_base = _late_clock.is_nonterminal_base
    _normalize_prob_map = _late_clock.normalize_prob_map
    choose_action_with_budget = _late_clock.choose_action_with_budget
    _apply_urgent_outcome_constraints = _late_clock.apply_urgent_outcome_constraints

    off_probs = build_offense_action_probs(offense.tactics, defense.tactics, ctx=ctx, game_cfg=game_cfg)
    off_probs = _apply_contextual_action_weights(off_probs)
    off_probs = apply_team_style_to_action_probs(off_probs, team_style, game_cfg)

    action = choose_action_with_budget(rng, off_probs)
    offense.off_action_counts[action] = offense.off_action_counts.get(action, 0) + 1

    # --- Fatigue intensity tracking (per time-segment, visible to sim_game via ctx) ---
    # Track a possession-level play family so follow-up time (kickouts/extra passes) is attributed
    # to the initiating action family (e.g., PnR or TransitionEarly).
    fatigue_family_base = get_action_base(action, game_cfg)

    def _accumulate_fatigue_usage(delta_sec: float) -> None:
        """Accumulate intensity usage time into ctx['_fatigue_seg_usage'].

        Values are seconds spent in TransitionEarly or PnR/PnP/DHO play families within the
        current time-segment. sim_game converts this into proportional fatigue multipliers.
        """
        try:
            d = float(delta_sec)
        except Exception:
            return
        if d <= 0:
            return
        u = ctx.get('_fatigue_seg_usage')
        if not isinstance(u, dict):
            return
        base = str(fatigue_family_base or '')
        if base == 'TransitionEarly':
            u['transition_sec'] = float(u.get('transition_sec', 0.0) or 0.0) + d
        if base in ('PnR', 'PnP', 'DHO'):
            u['pnr_sec'] = float(u.get('pnr_sec', 0.0) or 0.0) + d

    tags = {
        "in_transition": (get_action_base(action, game_cfg) == "TransitionEarly"),
        "is_side_pnr": (action == "SideAnglePnR"),
        "avg_fatigue_off": ctx.get("avg_fatigue_off"),
        "fatigue_bad_mult_max": ctx.get("fatigue_bad_mult_max"),
        "fatigue_bad_critical": ctx.get("fatigue_bad_critical"),
        "fatigue_bad_bonus": ctx.get("fatigue_bad_bonus"),
        "fatigue_bad_cap": ctx.get("fatigue_bad_cap"),
    }

    # --- ADD: action-dependent tags refresh helper ---
    def _refresh_action_tags(_action: str, _tags: dict) -> None:
        _tags["in_transition"] = (get_action_base(_action, game_cfg) == "TransitionEarly")
        _tags["is_side_pnr"] = (_action == "SideAnglePnR")

    # ensure initial consistency (safe even if already set above)
    _refresh_action_tags(action, tags)


    # `max_steps` is used as a safety against "no-time-progress" loops (e.g. sequences of 0-cost actions/passes).
    # When we observe `max_steps` consecutive iterations with no change to either the shot clock or game clock,
    # we force a real action (a quick SpotUp) so the possession ends naturally instead of producing an
    # artificial SHOTCLOCK turnover.
    stall_steps = 0
    pass_chain = 0

    def _bump_stall(_stall: int, _sc0: float, _gc0: float) -> int:
        """Increment stall counter if no time progressed this iteration, else reset to 0."""
        try:
            if float(game_state.shot_clock_sec) == float(_sc0) and float(game_state.clock_sec) == float(_gc0):
                return _stall + 1
        except Exception:
            # If clocks are in an unexpected state, prefer forcing progress sooner.
            return _stall + 1
        return 0

    while game_state.clock_sec > 0:
        sc0 = float(game_state.shot_clock_sec)
        gc0 = float(game_state.clock_sec)

        forced_due_to_stall = False
        if stall_steps >= max_steps:
            forced_due_to_stall = True
            stall_steps = 0
            action = "QuickShot"
            tags["forced_max_steps"] = True
            _refresh_action_tags(action, tags)
            fatigue_family_base = get_action_base(action, game_cfg)

        action_cost = float(_estimate_action_cost_sec(action))
        # Clamp cost so we never consume more time than remains.
        tm = float(tempo_mult) if float(tempo_mult) > 0 else 1.0
        max_base_cost = max(0.0, min(float(sc0), float(gc0)) / tm)
        if action_cost > max_base_cost:
            action_cost = max_base_cost

        clock_expired = False
        shotclock_expired = False

        if action_cost > 0:
            apply_time_cost(game_state, action_cost, tempo_mult)
        elif forced_due_to_stall:
            # When forcing a bailout due to stalling, ensure clocks advance a bit.
            forced_cost = min(0.75, max_base_cost)
            if forced_cost > 0:
                apply_time_cost(game_state, forced_cost, tempo_mult)

        # Normalize negative clocks to 0 for stability.
        if game_state.clock_sec < 0:
            game_state.clock_sec = 0
        if game_state.shot_clock_sec < 0:
            game_state.shot_clock_sec = 0

        # Track how much game-clock time elapsed in this iteration for fatigue-intensity attribution.
        delta_gc = max(float(gc0) - float(game_state.clock_sec), 0.0)
        if delta_gc > 0:
            _accumulate_fatigue_usage(delta_gc)

        clock_expired = (game_state.clock_sec <= 0)
        shotclock_expired = (game_state.shot_clock_sec <= 0)

        base_action_now = get_action_base(action, game_cfg)

        # Update per-step defensive pressure context (help/double) BEFORE building priors.
        _update_def_pressure_for_step(action=action, base_action=base_action_now, tags=tags)

        # If time expires during a non-terminal action (pass/reset), end immediately.
        if shotclock_expired and _is_nonterminal_base(base_action_now):
            commit_shot_clock_turnover(offense)
            # Replay log: shot-clock violation -> turnover (deadball)
            try:
                emit_event(
                    game_state,
                    event_type="TURNOVER",
                    home=home_team,
                    away=away_team,
                    rules=rules,
                    team_id=off_team_id,
                    opp_team_id=def_team_id,
                    pos_start=str(pos_origin),
                    pos_start_next="after_tov_dead",
                    outcome="TO_SHOT_CLOCK",
                    deadball_override=True,
                    tov_deadball_reason="SHOT_CLOCK",
                )
            except Exception:
                pass
            return {
                "end_reason": "SHOTCLOCK",
                "pos_start_next": "after_tov_dead",
                "points_scored": int(offense.pts) - before_pts,
                "had_orb": had_orb,
                "pos_start": pos_origin,
                "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
            }
        if clock_expired and _is_nonterminal_base(base_action_now):
            game_state.clock_sec = 0
            return {
                "end_reason": "PERIOD_END",
                "pos_start_next": pos_start,
                "points_scored": int(offense.pts) - before_pts,
                "had_orb": had_orb,
                "pos_start": pos_origin,
                "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
            }


        # shot_diet: pass ctx so outcome multipliers can apply
        pri = build_outcome_priors(action, offense.tactics, defense.tactics, tags, ctx=ctx, game_cfg=game_cfg)
        pri = apply_team_style_to_outcome_priors(pri, team_style)
        pri = apply_role_fit_to_priors_and_tags(pri, get_action_base(action, game_cfg), offense, tags, game_cfg=game_cfg)
        pri = apply_quality_to_turnover_priors(pri, get_action_base(action, game_cfg), offense, defense, tags, ctx)
        pri = apply_help_to_priors(pri, ctx)
        pri = apply_double_to_priors(pri, ctx)
        pri = apply_rotation_advantage_to_priors(pri, ctx)
        pri = _apply_urgent_outcome_constraints(pri)
        if clock_expired or shotclock_expired:
            pri_term = {k: v for k, v in pri.items() if (not k.startswith("PASS_") and not k.startswith("RESET_"))}
            if pri_term:
                pri = _normalize_prob_map(pri_term)
        outcome = weighted_choice(rng, pri)

        _ensure_matchups(reason="pre_resolve")
        _maybe_inject_matchup_force()

        term, payload = resolve_outcome(
            rng,
            outcome,
            action,
            offense,
            defense,
            tags,
            pass_chain,
            ctx=ctx,
            game_state=game_state,
            game_cfg=game_cfg,
        )

        if term == "SCORE":
            # Replay log: made shot (resolve payload contains pid/points/assist/outcome etc.)
            try:
                rp = _clean_replay_payload(payload)
                emit_event(
                    game_state,
                    event_type="SCORE",
                    home=home_team,
                    away=away_team,
                    rules=rules,
                    team_id=off_team_id,
                    opp_team_id=def_team_id,
                    pos_start=str(pos_origin),
                    pos_start_next="after_score",
                    **rp,
                )
            except Exception:
                pass
            return {
                "end_reason": "SCORE",
                "pos_start_next": "after_score",
                "points_scored": int(offense.pts) - before_pts,
                "had_orb": had_orb,
                "pos_start": pos_origin,
                "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
            }

        if term == "TURNOVER":
            tov_outcome = _normalize_turnover_outcome(payload.get("outcome") if isinstance(payload, dict) else "")
            is_dead = _turnover_is_deadball(tov_outcome)
            pos_start_next = ("after_tov_dead" if is_dead else "after_tov")
            tov_deadball_reason = None
            tov_is_steal = False
            tov_stealer_pid = None
            pstart_override_for_log = None

            if isinstance(payload, dict):
                # Allow resolve layer to override live/dead classification (e.g., bad-pass lineout).
                if payload.get("deadball_override") is True:
                    is_dead = True
                elif payload.get("deadball_override") is False:
                    is_dead = False
                if payload.get("tov_deadball_reason") is not None:
                    try:
                        tov_deadball_reason = str(payload.get("tov_deadball_reason"))
                    except Exception:
                        tov_deadball_reason = None

                tov_is_steal = bool(payload.get("steal", False))
                tov_stealer_pid = payload.get("stealer_pid")

                # Allow explicit next-start override (preferred) and fallback to 'steal' flag.
                pstart_override = payload.get("pos_start_next_override")
                if pstart_override:
                    try:
                        pstart_override_for_log = str(pstart_override)
                        pos_start_next = pstart_override_for_log
                    except Exception:
                        pos_start_next = pos_start_next
                elif tov_is_steal and not is_dead:
                    pos_start_next = "after_steal"

            # Re-derive default after override, so deadball flags remain consistent.
            if pos_start_next == "after_tov_dead":
                is_dead = True
            elif pos_start_next == "after_tov":
                is_dead = False
            # Replay log: turnover (resolve payload contains pid/outcome/type/steal/...).
            try:
                # Charge is both a turnover and an offensive foul; log it as an offensive foul whistle.
                event_type = "TURNOVER"
                if isinstance(payload, dict) and (payload.get("offensive_foul") or tov_outcome == "TO_CHARGE"):
                    event_type = "OFFENSIVE_FOUL"
                rp = _clean_replay_payload(payload, drop={"pos_start_next_override"})
                emit_event(
                    game_state,
                    event_type=event_type,
                    home=home_team,
                    away=away_team,
                    rules=rules,
                    team_id=off_team_id,
                    opp_team_id=def_team_id,
                    pos_start=str(pos_origin),
                    pos_start_next=str(pos_start_next),
                    pos_start_next_override=pstart_override_for_log,
                    **rp,
                )
            except Exception:
                pass
            return {
                "end_reason": "TURNOVER",
                "pos_start_next": (pos_start_next if pos_start_next else ("after_tov_dead" if is_dead else "after_tov")),
                "points_scored": int(offense.pts) - before_pts,
                "had_orb": had_orb,
                "pos_start": pos_origin,
                "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
                "turnover_outcome": tov_outcome,
                "turnover_deadball": bool(is_dead),
                "turnover_deadball_reason": tov_deadball_reason,
                "turnover_is_steal": bool(tov_is_steal),
                "turnover_stealer_pid": tov_stealer_pid,
            }

        if term == "FOUL_NO_SHOTS":
            # Dead-ball stop, offense retains ball.
            _top_up_shot_clock_after_def_no_shot_foul()
            # NOTE: We intentionally do NOT run the inbound here.
            # The game loop may want to do substitutions / timeouts / UI stops between the whistle and inbound.
            # Replay log: foul (no shots). team_side is the fouling team (defense).
            try:
                rp = _clean_replay_payload(payload)
                emit_event(
                    game_state,
                    event_type="FOUL_NO_SHOTS",
                    home=home_team,
                    away=away_team,
                    rules=rules,
                    team_id=def_team_id,
                    opp_team_id=off_team_id,
                    pos_start=str(pos_origin),
                    pos_start_next="after_foul",
                    **rp,
                )
            except Exception:
                pass
            return {
                "end_reason": "DEADBALL_STOP",
                "deadball_reason": "FOUL_NO_SHOTS",
                "pos_start_next": "after_foul",
                "points_scored": int(offense.pts) - before_pts,
                "had_orb": had_orb,
                "pos_start": pos_origin,
                "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
            }



        if term == "FOUL_FT":
            # Replay log: foul + FT trip result. team_side is the fouling team (defense).
            # (Log once here, regardless of whether last FT was made; rebound (if any) is logged separately.)
            try:
                rp = _clean_replay_payload(payload)
                emit_event(
                    game_state,
                    event_type="FOUL_FT",
                    home=home_team,
                    away=away_team,
                    rules=rules,
                    team_id=def_team_id,
                    opp_team_id=off_team_id,
                    pos_start=str(pos_origin),
                    pos_start_next=("after_score" if bool(getattr(payload, "get", lambda *_: False)("last_made", False)) else None),
                    **rp,
                )
            except Exception:
                pass
            # If last FT made -> dead-ball score, possession ends.
            if bool(payload.get("last_made", False)):
                return {
                    "end_reason": "SCORE",
                    "pos_start_next": "after_score",
                    "points_scored": int(offense.pts) - before_pts,
                    "had_orb": had_orb,
                    "pos_start": pos_origin,
                    "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
                    "ended_with_ft_trip": True,
                }
                

            # last FT missed -> live rebound
            # Before we select a rebounder, force foul-out substitutions NOW so a fouled-out player
            # cannot remain in the rebounder candidate pool.
            try:
                foul_out = int(ctx.get("foul_out", rules.get("foul_out", 6)))
                pf_map = (game_state.player_fouls.get(def_team_id, {}) or {})

                forced_out = [
                    pid for pid in list(getattr(defense, "on_court_pids", []) or [])
                    if int(pf_map.get(pid, 0)) >= foul_out
                ]

                if forced_out:

                    # --- segment close (minutes/fatigue accounting in sim_game) ---
                    # NOTE: ctx is mutated in-place (no shallow copy); keep shared objects in-place for sim_game bookkeeping.
                    # This ensures sim_game can observe segment/flags across continuation.
                    try:
                        segments = ctx.get("_time_segments")
                        last_ref = ctx.get("_seg_last_clock_sec")
                        if isinstance(segments, list) and isinstance(last_ref, dict):
                            now_clock = float(getattr(game_state, "clock_sec", 0.0))
                            last_clock = float(last_ref.get("v", now_clock))
                            seg_elapsed = max(last_clock - now_clock, 0.0)

                            if seg_elapsed > 0:
                                segments.append(
                                    {
                                        "elapsed": float(seg_elapsed),
                                        "off": list(ctx.get("_seg_off_on_court") or ctx.get("off_on_court") or []),
                                        "def": list(ctx.get("_seg_def_on_court") or ctx.get("def_on_court") or []),
                                        "fatigue_usage": dict((ctx.get('_fatigue_seg_usage') or {}) if isinstance(ctx, dict) else {}),
                                    }
                                )

                            # advance segment cursor (in-place so sim_game sees it)
                            last_ref["v"] = now_clock
                            # Reset per-segment intensity usage so the next segment starts clean.
                            uref = ctx.get('_fatigue_seg_usage')
                            if isinstance(uref, dict):
                                uref['transition_sec'] = 0.0
                                uref['pnr_sec'] = 0.0
                    except Exception as exc:
                        _record_ctx_error("forced_sub.segment_close_pre", exc)

                    changed = maybe_substitute_deadball_v1(
                        rng,
                        defense,
                        home_team,
                        away_team,
                        game_state,
                        rules,
                        q_index=max(0, int(getattr(game_state, "quarter", 1)) - 1),
                        pos_start="after_foul",
                        pressure_index=float(ctx.get("pressure_index", 0.0)),
                        garbage_index=float(ctx.get("garbage_index", 0.0)),
                    )

                    # --- sync ctx lineup snapshots + invalidate lineup-based caches (remainder of possession) ---
                    if changed:
                        try:
                            new_off = list(getattr(offense, "on_court_pids", []) or [])
                            new_def = list(getattr(defense, "on_court_pids", []) or [])

                            # resolve.py prefers ctx['*_on_court'] if present
                            ctx["off_on_court"] = list(new_off)
                            ctx["def_on_court"] = list(new_def)

                            # segment state must be updated IN-PLACE (ctx is shallow-copied earlier)
                            seg_off = ctx.get("_seg_off_on_court")
                            if isinstance(seg_off, list):
                                seg_off.clear()
                                seg_off.extend(new_off)

                            seg_def = ctx.get("_seg_def_on_court")
                            if isinstance(seg_def, list):
                                seg_def.clear()
                                seg_def.extend(new_def)

                            # invalidate role assignment cache (lineup-dependent)
                            ctx.pop("def_role_players", None)
                            ctx.pop("def_role_players_detail", None)

                            # invalidate + recompute shot diet style immediately (lineup-dependent)
                            ctx.pop("shot_diet_style", None)
                            ctx["shot_diet_style"] = shot_diet.compute_shot_diet_style(
                                offense, defense, game_state=game_state, ctx=ctx
                            )

                            # Rebuild matchup map (lineup-dependent).
                            _ensure_matchups(reason="forced_sub")

                            del changed  # not used beyond this point; keep explicit for clarity
                        except Exception as exc:
                            _record_ctx_error("forced_sub.ctx_sync_post", exc)

            except ValueError:
                raise
            except Exception:
                # Forced-sub logic should not break simulation, but SSOT violations must crash.
                pass

            orb_mult = float(offense.tactics.context.get("ORB_MULT", 1.0)) * float(rules.get("ft_orb_mult", 0.75))
            drb_mult = float(defense.tactics.context.get("DRB_MULT", 1.0))
            p_orb = rebound_orb_probability(offense, defense, orb_mult, drb_mult, game_cfg=game_cfg)
            if rng.random() < p_orb:
                offense.orb += 1
                rbd = choose_orb_rebounder(rng, offense)
                offense.add_player_stat(rbd.pid, "ORB", 1)
                # Replay log: offensive rebound after missed FT
                try:
                    emit_event(
                        game_state,
                        event_type="REB",
                        home=home_team,
                        away=away_team,
                        rules=rules,
                        team_id=off_team_id,
                        opp_team_id=def_team_id,
                        pos_start=str(pos_origin),
                        pid=getattr(rbd, "pid", None),
                        outcome="ORB",
                    )
                except Exception:
                    pass
                game_state.shot_clock_sec = float(rules.get("foul_reset", rules.get("orb_reset", game_state.shot_clock_sec)))

                # ORB -> immediate Putback (one-shot action): only available after offensive rebound.
                # If chosen, we also force the rebounder as the shot actor via ctx['force_actor_pid'].
                pm = getattr(game_cfg, "prob_model", {}) or {}
                try:
                    p_base = float(pm.get("putback_try_base_ft", 0.30))
                except Exception:
                    p_base = 0.30

                reb_or = _player_stat(rbd, "REB_OR", 50.0)
                fin_rim = _player_stat(rbd, "FIN_RIM", 50.0)
                fin_dunk = _player_stat(rbd, "FIN_DUNK", 50.0)
                physical = _player_stat(rbd, "PHYSICAL", 50.0)

                fin_raw = 0.65 * fin_rim + 0.35 * fin_dunk
                orb01 = clamp((reb_or - 70.0) / 45.0, 0.0, 1.0)
                fin01 = clamp((fin_raw - 70.0) / 45.0, 0.0, 1.0)
                phy01 = clamp((physical - 60.0) / 45.0, 0.0, 1.0)

                try:
                    w_reb = float(pm.get("putback_try_w_reb_or", 0.55))
                    w_fin = float(pm.get("putback_try_w_fin", 0.35))
                    w_phy = float(pm.get("putback_try_w_phy", 0.20))
                except Exception:
                    w_reb, w_fin, w_phy = 0.55, 0.35, 0.20

                try:
                    mult_min = float(pm.get("putback_try_skill_mult_min", 0.70))
                    mult_max = float(pm.get("putback_try_skill_mult_max", 1.35))
                except Exception:
                    mult_min, mult_max = 0.70, 1.35

                skill_mult = 1.0 + w_reb * (orb01 - 0.5) + w_fin * (fin01 - 0.5) + w_phy * (phy01 - 0.5)
                skill_mult = clamp(skill_mult, mult_min, mult_max)

                try:
                    p_min = float(pm.get("putback_try_clamp_min", 0.02))
                    p_max = float(pm.get("putback_try_clamp_max", 0.45))
                except Exception:
                    p_min, p_max = 0.02, 0.45

                p_putback = clamp(float(p_base) * float(skill_mult), p_min, p_max)
                if rng.random() < p_putback:
                    action = "Putback"
                    offense.off_action_counts[action] = offense.off_action_counts.get(action, 0) + 1
                    ctx["force_actor_pid"] = getattr(rbd, "pid", None)
                    _refresh_action_tags(action, tags)
                    fatigue_family_base = get_action_base(action, game_cfg)
                    pass_chain = 0
                    had_orb = True
                    stall_steps = _bump_stall(stall_steps, sc0, gc0)
                    continue

                r2 = rng.random()
                if r2 < 0.45:
                    action = "Kickout"
                elif r2 < 0.60:
                    action = "ExtraPass"
                else:
                    action = "Drive"
                _refresh_action_tags(action, tags)
                fatigue_family_base = get_action_base(action, game_cfg)
                pass_chain = 0
                had_orb = True
                stall_steps = _bump_stall(stall_steps, sc0, gc0)
                continue


            defense.drb += 1
            rbd = choose_drb_rebounder(rng, defense)
            defense.add_player_stat(rbd.pid, "DRB", 1)
            # Replay log: defensive rebound after missed FT
            try:
                emit_event(
                    game_state,
                    event_type="REB",
                    home=home_team,
                    away=away_team,
                    rules=rules,
                    team_id=def_team_id,
                    opp_team_id=off_team_id,
                    pos_start=str(pos_origin),
                    pid=getattr(rbd, "pid", None),
                    outcome="DRB",
                )
            except Exception:
                pass
            return {
                "end_reason": "DRB",
                "pos_start_next": "after_drb",
                "points_scored": int(offense.pts) - before_pts,
                "had_orb": had_orb,
                "pos_start": pos_origin,
                "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
                "ended_with_ft_trip": True,
            }

        if term == "MISS":
            blocked = bool(payload.get("blocked", False)) if isinstance(payload, dict) else False
            block_kind = str(payload.get("block_kind", "")) if (blocked and isinstance(payload, dict)) else ""
            blocker_pid = payload.get("blocker_pid") if (blocked and isinstance(payload, dict)) else None

            # If the miss was blocked, sometimes the block goes out-of-bounds -> dead-ball inbound,
            # offense retains (continuation). This is a key "game-feel" lever for rim protection.
            if blocked:
                pm = getattr(game_cfg, "prob_model", {}) or {}
                bk = (block_kind or "").lower()
                if ("rim" in bk) or (bk == "shot_rim"):
                    k = "rim"
                elif ("post" in bk) or (bk == "shot_post"):
                    k = "post"
                elif ("mid" in bk) or (bk == "shot_mid"):
                    k = "mid"
                else:
                    k = "3"

                try:
                    p_oob = float(pm.get(f"block_oob_base_{k}", 0.22))
                except Exception:
                    p_oob = 0.22

                if rng.random() < clamp(p_oob, 0.0, 0.95):
                    # BLOCK_OOB: defense last touched -> out of bounds, offense retains.
                    # NBA-style: keep the remaining (unexpired) shot clock.
                    # Replay log: miss (blocked) that ends in deadball stop / inbound retain
                    try:
                        rp = _clean_replay_payload(payload)
                        emit_event(
                            game_state,
                            event_type="MISS",
                            home=home_team,
                            away=away_team,
                            rules=rules,
                            team_id=off_team_id,
                            opp_team_id=def_team_id,
                            pos_start=str(pos_origin),
                            pos_start_next="after_block_oob",
                            deadball_reason="BLOCK_OOB",
                            **rp,
                        )
                    except Exception:
                        pass

                    return {
                        "end_reason": "DEADBALL_STOP",
                        "deadball_reason": "BLOCK_OOB",
                        "pos_start_next": "after_block_oob",
                        "points_scored": int(offense.pts) - before_pts,
                        "had_orb": had_orb,
                        "pos_start": pos_origin,
                        "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
                        "was_blocked": True,
                        "blocker_pid": blocker_pid,
                        "block_kind": block_kind,
                    }

            # Replay log: regular miss (may lead to rebound)
            try:
                rp = _clean_replay_payload(payload)
                emit_event(
                    game_state,
                    event_type="MISS",
                    home=home_team,
                    away=away_team,
                    rules=rules,
                    team_id=off_team_id,
                    opp_team_id=def_team_id,
                    pos_start=str(pos_origin),
                    **rp,
                )
            except Exception:
                pass

            orb_mult = float(offense.tactics.context.get("ORB_MULT", 1.0))
            drb_mult = float(defense.tactics.context.get("DRB_MULT", 1.0))
            p_orb = rebound_orb_probability(offense, defense, orb_mult, drb_mult, game_cfg=game_cfg)

            # Blocked misses that stay in play are harder for the offense to recover.
            if blocked:
                pm = getattr(game_cfg, "prob_model", {}) or {}
                bk = (block_kind or "").lower()
                if ("rim" in bk) or (bk == "shot_rim"):
                    k = "rim"
                elif ("post" in bk) or (bk == "shot_post"):
                    k = "post"
                elif ("mid" in bk) or (bk == "shot_mid"):
                    k = "mid"
                else:
                    k = "3"
                try:
                    mult = float(pm.get(f"blocked_orb_mult_{k}", 0.82))
                except Exception:
                    mult = 0.82
                p_orb = clamp(float(p_orb) * clamp(mult, 0.10, 1.20), 0.02, 0.60)
                
            if rng.random() < p_orb:
                offense.orb += 1
                rbd = choose_orb_rebounder(rng, offense)
                offense.add_player_stat(rbd.pid, "ORB", 1)
                # Replay log: offensive rebound after miss
                try:
                    emit_event(
                        game_state,
                        event_type="REB",
                        home=home_team,
                        away=away_team,
                        rules=rules,
                        team_id=off_team_id,
                        opp_team_id=def_team_id,
                        pos_start=str(pos_origin),
                        pid=getattr(rbd, "pid", None),
                        outcome="ORB",
                    )
                except Exception:
                    pass
                game_state.shot_clock_sec = float(rules.get("orb_reset", game_state.shot_clock_sec))

                # ORB -> immediate Putback (one-shot action): only available after offensive rebound.
                # If chosen, we also force the rebounder as the shot actor via ctx['force_actor_pid'].
                pm = getattr(game_cfg, "prob_model", {}) or {}

                o = str(outcome or "")
                if o in ("SHOT_RIM_LAYUP", "SHOT_RIM_DUNK", "SHOT_RIM_CONTACT", "SHOT_TOUCH_FLOATER"):
                    bucket = "rim"
                elif o == "SHOT_POST":
                    bucket = "post"
                elif o in ("SHOT_MID_CS", "SHOT_MID_PU"):
                    bucket = "mid"
                elif o in ("SHOT_3_CS", "SHOT_3_OD"):
                    bucket = "3"
                else:
                    bucket = "rim"

                pb_defaults = {"rim": 0.27, "post": 0.24, "mid": 0.14, "3": 0.08}
                try:
                    p_base = float(pm.get(f"putback_try_base_{bucket}", pb_defaults.get(bucket, 0.27)))
                except Exception:
                    p_base = float(pb_defaults.get(bucket, 0.27))

                # If the miss was blocked but stayed in play, putback tries are less frequent.
                if blocked:
                    try:
                        p_base = float(p_base) * float(pm.get("putback_try_mult_blocked", 0.70))
                    except Exception:
                        p_base = float(p_base) * 0.70

                reb_or = _player_stat(rbd, "REB_OR", 50.0)
                fin_rim = _player_stat(rbd, "FIN_RIM", 50.0)
                fin_dunk = _player_stat(rbd, "FIN_DUNK", 50.0)
                physical = _player_stat(rbd, "PHYSICAL", 50.0)

                fin_raw = 0.65 * fin_rim + 0.35 * fin_dunk
                orb01 = clamp((reb_or - 70.0) / 45.0, 0.0, 1.0)
                fin01 = clamp((fin_raw - 70.0) / 45.0, 0.0, 1.0)
                phy01 = clamp((physical - 60.0) / 45.0, 0.0, 1.0)

                try:
                    w_reb = float(pm.get("putback_try_w_reb_or", 0.55))
                    w_fin = float(pm.get("putback_try_w_fin", 0.35))
                    w_phy = float(pm.get("putback_try_w_phy", 0.20))
                except Exception:
                    w_reb, w_fin, w_phy = 0.55, 0.35, 0.20

                try:
                    mult_min = float(pm.get("putback_try_skill_mult_min", 0.70))
                    mult_max = float(pm.get("putback_try_skill_mult_max", 1.35))
                except Exception:
                    mult_min, mult_max = 0.70, 1.35

                skill_mult = 1.0 + w_reb * (orb01 - 0.5) + w_fin * (fin01 - 0.5) + w_phy * (phy01 - 0.5)
                skill_mult = clamp(skill_mult, mult_min, mult_max)

                try:
                    p_min = float(pm.get("putback_try_clamp_min", 0.02))
                    p_max = float(pm.get("putback_try_clamp_max", 0.45))
                except Exception:
                    p_min, p_max = 0.02, 0.45

                p_putback = clamp(float(p_base) * float(skill_mult), p_min, p_max)
                if rng.random() < p_putback:
                    action = "Putback"
                    offense.off_action_counts[action] = offense.off_action_counts.get(action, 0) + 1
                    ctx["force_actor_pid"] = getattr(rbd, "pid", None)
                    _refresh_action_tags(action, tags)
                    fatigue_family_base = get_action_base(action, game_cfg)
                    pass_chain = 0
                    had_orb = True
                    stall_steps = _bump_stall(stall_steps, sc0, gc0)
                    continue

                
                r2 = rng.random()
                if r2 < 0.45:
                    action = "Kickout"
                elif r2 < 0.60:
                    action = "ExtraPass"
                else:
                    action = "Drive"
                _refresh_action_tags(action, tags)
                fatigue_family_base = get_action_base(action, game_cfg)
                pass_chain = 0
                had_orb = True
                stall_steps = _bump_stall(stall_steps, sc0, gc0)
                continue

            defense.drb += 1
            rbd = choose_drb_rebounder(rng, defense)
            defense.add_player_stat(rbd.pid, "DRB", 1)
            # Replay log: defensive rebound after miss
            try:
                emit_event(
                    game_state,
                    event_type="REB",
                    home=home_team,
                    away=away_team,
                    rules=rules,
                    team_id=def_team_id,
                    opp_team_id=off_team_id,
                    pos_start=str(pos_origin),
                    pid=getattr(rbd, "pid", None),
                    outcome="DRB",
                )
            except Exception:
                pass
            return {
                "end_reason": "DRB",
                "pos_start_next": ("after_block" if blocked else "after_drb"),
                "points_scored": int(offense.pts) - before_pts,
                "had_orb": had_orb,
                "pos_start": pos_origin,
                "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
                "was_blocked": bool(blocked),
                "blocker_pid": blocker_pid,
                "block_kind": block_kind,
            }

        if term == "RESET":
            reset_cost = float(time_costs.get("Reset", 0.0))
            tm = float(tempo_mult) if float(tempo_mult) > 0 else 1.0
            budget_now = _budget_sec()

            # If time is tight, skip the reset and force a quick attempt.
            if budget_now <= (min_release_window + 0.05) or (reset_cost * tm) >= max(0.0, budget_now - min_release_window):
                action = "QuickShot"
                _refresh_action_tags(action, tags)
                fatigue_family_base = get_action_base(action, game_cfg)
                pass_chain = 0
                stall_steps = _bump_stall(stall_steps, sc0, gc0)
                continue
                
            if reset_cost > 0:
                # Reset itself should not inherit the prior PnR/Transition family.
                fatigue_family_base = 'Reset'
                gc_before = float(game_state.clock_sec)
                apply_time_cost(game_state, reset_cost, tempo_mult)
                if game_state.clock_sec < 0:
                    game_state.clock_sec = 0
                if game_state.shot_clock_sec < 0:
                    game_state.shot_clock_sec = 0
                delta_reset = max(gc_before - float(game_state.clock_sec), 0.0)
                if delta_reset > 0:
                    _accumulate_fatigue_usage(delta_reset)
                    
                if game_state.shot_clock_sec <= 0:
                    commit_shot_clock_turnover(offense)
                    # Replay log: shot-clock violation via RESET path -> turnover (deadball)
                    try:
                        emit_event(
                            game_state,
                            event_type="TURNOVER",
                            home=home_team,
                            away=away_team,
                            rules=rules,
                            team_id=off_team_id,
                            opp_team_id=def_team_id,
                            pos_start=str(pos_origin),
                            pos_start_next="after_tov_dead",
                            outcome="TO_SHOT_CLOCK",
                            deadball_override=True,
                            tov_deadball_reason="SHOT_CLOCK",
                        )
                    except Exception:
                        pass
                    return {
                        "end_reason": "SHOTCLOCK",
                        "pos_start_next": "after_tov_dead",
                        "points_scored": int(offense.pts) - before_pts,
                        "had_orb": had_orb,
                        "pos_start": pos_origin,
                        "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
                    }
                if game_state.clock_sec <= 0:
                    game_state.clock_sec = 0
                    return {
                        "end_reason": "PERIOD_END",
                        "pos_start_next": pos_start,
                        "points_scored": int(offense.pts) - before_pts,
                        "had_orb": had_orb,
                        "pos_start": pos_origin,
                        "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
                    }
            off_probs = build_offense_action_probs(offense.tactics, defense.tactics, ctx=ctx, game_cfg=game_cfg)
            off_probs = _apply_contextual_action_weights(off_probs)
            off_probs = apply_team_style_to_action_probs(off_probs, team_style, game_cfg)
            action = choose_action_with_budget(rng, off_probs)
            offense.off_action_counts[action] = offense.off_action_counts.get(action, 0) + 1
            _refresh_action_tags(action, tags)
            fatigue_family_base = get_action_base(action, game_cfg)
            pass_chain = 0
            stall_steps = _bump_stall(stall_steps, sc0, gc0)
            continue


        if term == "CONTINUE":
            pass_chain = payload.get("pass_chain", pass_chain + 1)
            pass_cost_nominal = 0.0
            if outcome in ("PASS_KICKOUT", "PASS_SKIP"):
                pass_cost_nominal = float(time_costs.get("Kickout", 0.0))
            elif outcome == "PASS_EXTRA":
                pass_cost_nominal = float(time_costs.get("ExtraPass", 0.0))
            elif outcome == "PASS_SHORTROLL":
                # Reuse ExtraPass cost for shortroll outlets if no dedicated key exists.
                pass_cost_nominal = float(time_costs.get("ExtraPass", 0.0))

            tm = float(tempo_mult) if float(tempo_mult) > 0 else 1.0
            budget_now = _budget_sec()

            # Apply as much pass time-cost as feasible, then force QuickShot if the full pass
            # would leave insufficient release window. This preserves late-clock realism while
            # still committing the pass event for assist tracking.
            pass_cost_apply = 0.0
            force_quick_after_pass = False
            if pass_cost_nominal > 0.0:
                max_cost = max(0.0, (budget_now - min_release_window) / tm) if tm > 0 else 0.0
                pass_cost_apply = min(pass_cost_nominal, max_cost)
                if pass_cost_nominal > (max_cost + 1e-9):
                    force_quick_after_pass = True

            if pass_cost_apply > 0.0:
                gc_before = float(game_state.clock_sec)
                apply_time_cost(game_state, pass_cost_apply, tempo_mult)
                if game_state.clock_sec < 0:
                    game_state.clock_sec = 0
                if game_state.shot_clock_sec < 0:
                    game_state.shot_clock_sec = 0

                delta_pass = max(gc_before - float(game_state.clock_sec), 0.0)
                if delta_pass > 0:
                    _accumulate_fatigue_usage(delta_pass)
                    
                if game_state.shot_clock_sec <= 0:
                    # Drop any staged pass event to avoid leaking into the next possession.
                    ctx.pop("_pending_pass_event", None)
                    commit_shot_clock_turnover(offense)
                    return {
                        "end_reason": "SHOTCLOCK",
                        "pos_start_next": "after_tov_dead",
                        "points_scored": int(offense.pts) - before_pts,
                        "had_orb": had_orb,
                        "pos_start": pos_origin,
                        "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
                    }
                if game_state.clock_sec <= 0:
                    ctx.pop("_pending_pass_event", None)
                    game_state.clock_sec = 0
                    return {
                        "end_reason": "PERIOD_END",
                        "pos_start_next": pos_start,
                        "points_scored": int(offense.pts) - before_pts,
                        "had_orb": had_orb,
                        "pos_start": pos_origin,
                        "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
                    }

            # Commit staged pass event AFTER time cost has been applied so the recorded
            # shot-clock timestamp is accurate for assist-window logic.
            commit_pending_pass_event(ctx, game_state)

            if force_quick_after_pass:
                action = "QuickShot"
            elif outcome in ("PASS_KICKOUT", "PASS_SKIP", "PASS_EXTRA"):
                # Avoid chaining extra passes late: choose a budget-feasible catch-and-shoot.
                action = choose_action_with_budget(rng, {"SpotUp": 0.72, "ExtraPass": 0.28})
            elif outcome == "PASS_SHORTROLL":
                action = choose_action_with_budget(rng, {"Drive": 0.40, "Kickout": 0.60})
            else:
                action = choose_action_with_budget(rng, off_probs)

            # If we newly enter a high-intensity family mid-possession, switch the family so
            # subsequent time is attributed correctly (prevents missing late PnR/transition).
            try:
                new_base = get_action_base(action, game_cfg)
            except Exception:
                new_base = ''
            if str(fatigue_family_base or '') not in ('TransitionEarly', 'PnR', 'PnP', 'DHO') and new_base in ('TransitionEarly', 'PnR', 'PnP', 'DHO'):
                fatigue_family_base = new_base

            if (not force_quick_after_pass) and pass_chain >= 3:
                # After a long pass chain, bias toward a shot attempt.
                action = choose_action_with_budget(rng, {"SpotUp": 1.0})

            _refresh_action_tags(action, tags)
            stall_steps = _bump_stall(stall_steps, sc0, gc0)
            continue


    # If we exit the loop here, the only expected reason is the period/game clock reaching 0.
    game_state.clock_sec = 0
    return {
        "end_reason": "PERIOD_END",
        "pos_start_next": pos_start,
        "points_scored": int(offense.pts) - before_pts,
        "had_orb": had_orb,
        "pos_start": pos_origin,
        "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
    }



# -------------------------
# Game simulation

# -------------------------
