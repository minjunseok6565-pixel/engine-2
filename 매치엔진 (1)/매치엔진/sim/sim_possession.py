from __future__ import annotations

"""Possession simulation (team style biasing, priors, resolve loop).

NOTE: Split from sim.py on 2025-12-27.
"""

import random
import math
import warnings
from typing import Any, Dict, Optional, TYPE_CHECKING

from .builders import (
    build_defense_action_probs,
    build_offense_action_probs,
    build_outcome_priors,
    get_action_base,
)
from . import shot_diet
from . import quality
from .def_role_players import get_or_build_def_role_players, engine_get_stat
from .core import weighted_choice, clamp
from .models import GameState, TeamState
from .resolve import (
    choose_drb_rebounder,
    choose_orb_rebounder,
    rebound_orb_probability,
    resolve_outcome,
)
from .role_fit import apply_role_fit_to_priors_and_tags

from .sim_clock import (
    apply_time_cost,
    apply_dead_ball_cost,
    simulate_inbound,
    commit_shot_clock_turnover,
)

if TYPE_CHECKING:
    from config.game_config import GameConfig

def apply_quality_to_turnover_priors(
    pri: Dict[str, float],
    base_action: str,
    offense: TeamState,
    defense: TeamState,
    tags: Dict[str, Any],
    ctx: Dict[str, Any],
) -> Dict[str, float]:
    """Adjust TO_HANDLE_LOSS prior weight using quality-driven 'pressure'.

    quality.compute_quality_score returns an offense-perspective quality score:
      + higher => more open / better for offense
      - lower  => tougher / worse for offense (better defense)

    For turnovers, we want better defense => higher TO probability, so we invert:
        pressure = -quality_score

    We apply an exponential multiplier to pri['TO_HANDLE_LOSS']:

        pri['TO_HANDLE_LOSS'] *= exp(clamp(pressure * K_TO_QUALITY, -CLAMP, +CLAMP))

    Tuning knobs (defense.tactics.context):
      - K_TO_QUALITY (default 0.25)
      - TO_QUALITY_LOG_CLAMP (default 1.0)
    """
    if "TO_HANDLE_LOSS" not in pri:
        return pri

    scheme = getattr(defense.tactics, "defense_scheme", "")
    role_players = get_or_build_def_role_players(ctx, defense, scheme=scheme)

    debug_q = bool(ctx.get("debug_quality", False))
    q_res = quality.compute_quality_score(
        scheme=str(scheme),
        base_action=str(base_action),
        outcome="TO_HANDLE_LOSS",
        role_players=role_players,
        get_stat=engine_get_stat,
        return_detail=debug_q,
    )
    q_score = float(q_res.score) if (debug_q and hasattr(q_res, "score")) else float(q_res)

    pressure = -q_score

    tctx = getattr(defense.tactics, "context", {}) or {}
    k_to = float(tctx.get("K_TO_QUALITY", 0.25))
    log_clamp = float(tctx.get("TO_QUALITY_LOG_CLAMP", 1.0))
    log_mult = clamp(pressure * k_to, -log_clamp, log_clamp)

    pri["TO_HANDLE_LOSS"] = float(pri.get("TO_HANDLE_LOSS", 0.0)) * math.exp(log_mult)

    if debug_q:
        tags["to_quality_score"] = q_score
        tags["to_pressure"] = pressure
        tags["to_log_mult"] = log_mult
        tags["to_weight_after"] = float(pri["TO_HANDLE_LOSS"])

    return pri

def _draw_style_mult(
    rng: random.Random,
    std: float,
    lo: float,
    hi: float,
) -> float:
    return clamp(rng.gauss(1.0, float(std)), float(lo), float(hi))

def ensure_team_style(rng: random.Random, team: TeamState, rules: Dict[str, Any]) -> Dict[str, float]:
    """Attach a persistent style profile to a team to increase between-team diversity.

    This intentionally adds team-to-team dispersion even when rosters are homogeneous
    (useful for calibration targets like stddev(pace/3PAr/FTr/TOV%)).
    """
    try:
        ctx = team.tactics.context
    except Exception as exc:
        warnings.warn(
            f"ensure_team_style: failed to access tactics.context for team '{getattr(team, 'name', 'unknown')}' "
            f"({type(exc).__name__}: {exc})"
        )
        ctx = None
    if not isinstance(ctx, dict):
        return {}
    if isinstance(ctx.get("TEAM_STYLE"), dict):
        return ctx["TEAM_STYLE"]

    cfg = rules.get("team_style", {}) or {}
    style = {
        # pace / transition tendency (affects time cost through tempo_mult)
        "tempo_mult": _draw_style_mult(rng, std=float(cfg.get("tempo_std", 0.032)), lo=0.92, hi=1.08),
        # shot diet (3PT vs rim)
        "three_bias": _draw_style_mult(rng, std=float(cfg.get("three_std", 0.12)), lo=0.70, hi=1.35),
        "rim_bias": _draw_style_mult(rng, std=float(cfg.get("rim_std", 0.10)), lo=0.75, hi=1.30),
        # turnovers / fouls diversity
        "tov_bias": _draw_style_mult(rng, std=float(cfg.get("tov_std", 0.14)), lo=0.70, hi=1.40),
        "ftr_bias": _draw_style_mult(rng, std=float(cfg.get("ftr_std", 0.18)), lo=0.60, hi=1.50),
    }

    ctx["TEAM_STYLE"] = style
    return style

def _renorm(d: Dict[str, float]) -> Dict[str, float]:
    s = sum(float(v) for v in (d or {}).values())
    if s <= 0:
        return d
    return {k: float(v) / s for k, v in d.items()}

def apply_team_style_to_action_probs(
    probs: Dict[str, float],
    style: Dict[str, float],
    game_cfg: "GameConfig",
) -> Dict[str, float]:
    if not probs or not style:
        return probs
    out = dict(probs)
    three_bias = float(style.get("three_bias", 1.0))
    rim_bias = float(style.get("rim_bias", 1.0))
    tempo_mult = float(style.get("tempo_mult", 1.0))

    for k, v in list(out.items()):
        base = get_action_base(k, game_cfg)
        mult = 1.0
        if base == "TransitionEarly":
            mult *= tempo_mult ** 0.85
        if base in ("Kickout", "ExtraPass", "SpotUp"):
            mult *= three_bias
        if base in ("Drive", "Cut"):
            mult *= rim_bias
        if base in ("PnR", "DHO"):
            mult *= (0.55 * three_bias + 0.45 * rim_bias)
        out[k] = float(v) * float(mult)

    return _renorm(out)

def apply_team_style_to_outcome_priors(pri: Dict[str, float], style: Dict[str, float]) -> Dict[str, float]:
    if not pri or not style:
        return pri
    out = dict(pri)
    three_bias = float(style.get("three_bias", 1.0))
    rim_bias = float(style.get("rim_bias", 1.0))
    tov_bias = float(style.get("tov_bias", 1.0))
    ftr_bias = float(style.get("ftr_bias", 1.0))

    for k, v in list(out.items()):
        vv = float(v)
        if k.startswith("TO_"):
            vv *= tov_bias
        elif k.startswith("FOUL_DRAW_") or k == "FOUL_REACH_TRAP":
            vv *= ftr_bias
        elif k.startswith("SHOT_3_"):
            vv *= three_bias
        elif k.startswith("SHOT_RIM_"):
            vv *= rim_bias
        out[k] = vv

    return _renorm(out)


# -------------------------
# Possession simulation
# -------------------------

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
    offense.possessions += 1
    before_pts = int(offense.pts)

    if ctx is None:
        ctx = {}
    if game_cfg is None:
        raise ValueError("simulate_possession requires game_cfg")

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

    tempo_mult = float(ctx.get("tempo_mult", 1.0))
    time_costs = rules.get("time_costs", {})
    had_orb = False

    # per-team style profile (persistent; increases team diversity)
    team_style = ensure_team_style(rng, offense, rules)
    if team_style:
        tempo_mult *= float(team_style.get("tempo_mult", 1.0))
        # keep ctx immutable-ish
        ctx = dict(ctx)
        ctx["tempo_mult"] = tempo_mult
        ctx["team_style"] = team_style

    # Dead-ball start can trigger inbound (score, quarter start, dead-ball TO, etc.)
    pos_start = str(ctx.get("pos_start", ""))
    dead_ball_starts = {"start_q", "after_score", "after_tov_dead"}
    if pos_start in dead_ball_starts:
        # dead-ball inbound attempt
        if simulate_inbound(rng, offense, defense, rules):
            return {
                "end_reason": "TURNOVER",
                "pos_start_next": "after_tov",
                "points_scored": int(offense.pts) - before_pts,
                "had_orb": had_orb,
                "pos_start": pos_start,
                "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
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

    def _apply_contextual_action_weights(probs: Dict[str, float]) -> Dict[str, float]:
        """Soft-bias action weights by possession context (no per-team fixed style)."""
        if not probs:
            return probs
        if bool(ctx.get("dead_ball_inbound", False)):
            return probs
        pstart = str(ctx.get("pos_start", pos_start))
        if pstart not in ("after_drb", "after_tov"):
            return probs
        mult_tbl = rules.get("transition_weight_mult", {}) or {}
        try:
            mult = float(mult_tbl.get(pstart, mult_tbl.get("default", 1.0)))
        except Exception:
            mult = 1.0
        if mult <= 1.0:
            return probs

        out = dict(probs)
        changed = False
        for k, v in list(out.items()):
            if get_action_base(k, game_cfg) == "TransitionEarly":
                out[k] = float(v) * mult
                changed = True
        if not changed:
            return probs
        s = sum(out.values())
        if s <= 0:
            return probs
        for k in out:
            out[k] /= s
        return out

    off_probs = build_offense_action_probs(offense.tactics, defense.tactics, ctx=ctx, game_cfg=game_cfg)
    off_probs = _apply_contextual_action_weights(off_probs)
    off_probs = apply_team_style_to_action_probs(off_probs, team_style, game_cfg)
    def_probs = build_defense_action_probs(defense.tactics, game_cfg=game_cfg)

    action = weighted_choice(rng, off_probs)
    offense.off_action_counts[action] = offense.off_action_counts.get(action, 0) + 1

    def_action = weighted_choice(rng, def_probs)
    defense.def_action_counts[def_action] = defense.def_action_counts.get(def_action, 0) + 1

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
            action = "SpotUp"
            tags["forced_max_steps"] = True
            _refresh_action_tags(action, tags)

        action_cost = float(time_costs.get(get_action_base(action, game_cfg), 0.0))
        if action_cost > 0:
            apply_time_cost(game_state, action_cost, tempo_mult)
            if game_state.shot_clock_sec <= 0:
                commit_shot_clock_turnover(offense)
                return {
                    "end_reason": "SHOTCLOCK",
                    "pos_start_next": "after_tov_dead",
                    "points_scored": int(offense.pts) - before_pts,
                    "had_orb": had_orb,
                    "pos_start": pos_start,
                    "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
                }
            if game_state.clock_sec <= 0:
                game_state.clock_sec = 0
                return {
                    "end_reason": "PERIOD_END",
                    "pos_start_next": pos_start,
                    "points_scored": int(offense.pts) - before_pts,
                    "had_orb": had_orb,
                    "pos_start": pos_start,
                    "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
                }
        elif forced_due_to_stall:
            # If the rules table provides 0-cost actions, the clocks may not move.
            # When we are forcing a bailout action due to stalling, make sure time advances.
            forced_cost = 0.75
            apply_time_cost(game_state, forced_cost, tempo_mult)
            if game_state.shot_clock_sec <= 0:
                commit_shot_clock_turnover(offense)
                return {
                    "end_reason": "SHOTCLOCK",
                    "pos_start_next": "after_tov_dead",
                    "points_scored": int(offense.pts) - before_pts,
                    "had_orb": had_orb,
                    "pos_start": pos_start,
                    "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
                }
            if game_state.clock_sec <= 0:
                game_state.clock_sec = 0
                return {
                    "end_reason": "PERIOD_END",
                    "pos_start_next": pos_start,
                    "points_scored": int(offense.pts) - before_pts,
                    "had_orb": had_orb,
                    "pos_start": pos_start,
                    "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
                }

        # shot_diet: pass ctx so outcome multipliers can apply
        pri = build_outcome_priors(action, offense.tactics, defense.tactics, tags, ctx=ctx, game_cfg=game_cfg)
        pri = apply_team_style_to_outcome_priors(pri, team_style)
        pri = apply_role_fit_to_priors_and_tags(pri, get_action_base(action, game_cfg), offense, tags, game_cfg=game_cfg)
        pri = apply_quality_to_turnover_priors(pri, get_action_base(action, game_cfg), offense, defense, tags, ctx)
        outcome = weighted_choice(rng, pri)

        term, payload = resolve_outcome(
            rng,
            outcome,
            action,
            offense,
            defense,
            tags,
            pass_chain,
            def_action=def_action,
            ctx=ctx,
            game_state=game_state,
            game_cfg=game_cfg,
        )

        if term == "SCORE":
            return {
                "end_reason": "SCORE",
                "pos_start_next": "after_score",
                "points_scored": int(offense.pts) - before_pts,
                "had_orb": had_orb,
                "pos_start": pos_start,
                "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
            }

        if term == "TURNOVER":
            return {
                "end_reason": "TURNOVER",
                "pos_start_next": "after_tov",
                "points_scored": int(offense.pts) - before_pts,
                "had_orb": had_orb,
                "pos_start": pos_start,
                "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
            }

        if term == "FOUL_NO_SHOTS":
            # dead-ball stop, offense retains ball
            stop_cost = float(time_costs.get("FoulStop", 0.0))
            if stop_cost > 0:
                apply_dead_ball_cost(game_state, stop_cost, tempo_mult)
                if game_state.clock_sec <= 0:
                    game_state.clock_sec = 0
                    return {
                        "end_reason": "PERIOD_END",
                        "pos_start_next": "after_foul",
                        "points_scored": int(offense.pts) - before_pts,
                        "had_orb": had_orb,
                        "pos_start": pos_start,
                        "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
                    }

            # 14s reset rule (if < 14 then reset up)
            foul_reset = float(rules.get("foul_reset", 14))
            if game_state.shot_clock_sec < foul_reset:
                game_state.shot_clock_sec = foul_reset

            # inbound restart (can turnover)
            if simulate_inbound(rng, offense, defense, rules):
                return {
                    "end_reason": "TURNOVER",
                    "pos_start_next": "after_tov",
                    "points_scored": int(offense.pts) - before_pts,
                    "had_orb": had_orb,
                    "pos_start": pos_start,
                    "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
                }

            # restart with set-play bias
            ctx = dict(ctx)
            ctx["pos_start"] = "after_foul"
            ctx["dead_ball_inbound"] = True
            off_probs = build_offense_action_probs(offense.tactics, defense.tactics, ctx=ctx, game_cfg=game_cfg)
            off_probs = _apply_contextual_action_weights(off_probs)
            off_probs = apply_team_style_to_action_probs(off_probs, team_style, game_cfg)
            action = weighted_choice(rng, off_probs)
            offense.off_action_counts[action] = offense.off_action_counts.get(action, 0) + 1
            _refresh_action_tags(action, tags)
            pass_chain = 0
            stall_steps = _bump_stall(stall_steps, sc0, gc0)
            continue


        if term == "FOUL_FT":
            # If last FT made -> dead-ball score, possession ends.
            if bool(payload.get("last_made", False)):
                return {
                    "end_reason": "SCORE",
                    "pos_start_next": "after_score",
                    "points_scored": int(offense.pts) - before_pts,
                    "had_orb": had_orb,
                    "pos_start": pos_start,
                    "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
                    "ended_with_ft_trip": True,
                }

            # last FT missed -> live rebound
            orb_mult = float(offense.tactics.context.get("ORB_MULT", 1.0)) * float(rules.get("ft_orb_mult", 0.75))
            drb_mult = float(defense.tactics.context.get("DRB_MULT", 1.0))
            p_orb = rebound_orb_probability(offense, defense, orb_mult, drb_mult, game_cfg=game_cfg)
            if rng.random() < p_orb:
                offense.orb += 1
                rbd = choose_orb_rebounder(rng, offense)
                offense.add_player_stat(rbd.pid, "ORB", 1)
                game_state.shot_clock_sec = float(rules.get("foul_reset", rules.get("orb_reset", game_state.shot_clock_sec)))
                r2 = rng.random()
                if r2 < 0.45:
                    action = "Kickout"
                elif r2 < 0.60:
                    action = "ExtraPass"
                else:
                    action = "Drive"
                _refresh_action_tags(action, tags)
                pass_chain = 0
                had_orb = True
                stall_steps = _bump_stall(stall_steps, sc0, gc0)
                continue


            defense.drb += 1
            rbd = choose_drb_rebounder(rng, defense)
            defense.add_player_stat(rbd.pid, "DRB", 1)
            return {
                "end_reason": "DRB",
                "pos_start_next": "after_drb",
                "points_scored": int(offense.pts) - before_pts,
                "had_orb": had_orb,
                "pos_start": pos_start,
                "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
                "ended_with_ft_trip": True,
            }

        if term == "MISS":
            orb_mult = float(offense.tactics.context.get("ORB_MULT", 1.0))
            drb_mult = float(defense.tactics.context.get("DRB_MULT", 1.0))
            p_orb = rebound_orb_probability(offense, defense, orb_mult, drb_mult, game_cfg=game_cfg)
            if rng.random() < p_orb:
                offense.orb += 1
                rbd = choose_orb_rebounder(rng, offense)
                offense.add_player_stat(rbd.pid, "ORB", 1)
                game_state.shot_clock_sec = float(rules.get("orb_reset", game_state.shot_clock_sec))
                r2 = rng.random()
                if r2 < 0.45:
                    action = "Kickout"
                elif r2 < 0.60:
                    action = "ExtraPass"
                else:
                    action = "Drive"
                _refresh_action_tags(action, tags)
                pass_chain = 0
                had_orb = True
                stall_steps = _bump_stall(stall_steps, sc0, gc0)
                continue

            defense.drb += 1
            rbd = choose_drb_rebounder(rng, defense)
            defense.add_player_stat(rbd.pid, "DRB", 1)
            return {
                "end_reason": "DRB",
                "pos_start_next": "after_drb",
                "points_scored": int(offense.pts) - before_pts,
                "had_orb": had_orb,
                "pos_start": pos_start,
                "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
            }

        if term == "RESET":
            reset_cost = float(time_costs.get("Reset", 0.0))
            if reset_cost > 0:
                apply_time_cost(game_state, reset_cost, tempo_mult)
                if game_state.shot_clock_sec <= 0:
                    commit_shot_clock_turnover(offense)
                    return {
                        "end_reason": "SHOTCLOCK",
                        "pos_start_next": "after_tov_dead",
                        "points_scored": int(offense.pts) - before_pts,
                        "had_orb": had_orb,
                        "pos_start": pos_start,
                        "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
                    }
                if game_state.clock_sec <= 0:
                    game_state.clock_sec = 0
                    return {
                        "end_reason": "PERIOD_END",
                        "pos_start_next": pos_start,
                        "points_scored": int(offense.pts) - before_pts,
                        "had_orb": had_orb,
                        "pos_start": pos_start,
                        "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
                    }
            off_probs = build_offense_action_probs(offense.tactics, defense.tactics, ctx=ctx, game_cfg=game_cfg)
            off_probs = _apply_contextual_action_weights(off_probs)
            off_probs = apply_team_style_to_action_probs(off_probs, team_style, game_cfg)
            action = weighted_choice(rng, off_probs)
            offense.off_action_counts[action] = offense.off_action_counts.get(action, 0) + 1
            _refresh_action_tags(action, tags)
            pass_chain = 0
            stall_steps = _bump_stall(stall_steps, sc0, gc0)
            continue


        if term == "CONTINUE":
            pass_chain = payload.get("pass_chain", pass_chain + 1)
            pass_cost = 0.0
            if outcome in ("PASS_KICKOUT", "PASS_SKIP"):
                pass_cost = float(time_costs.get("Kickout", 0.0))
            elif outcome == "PASS_EXTRA":
                pass_cost = float(time_costs.get("ExtraPass", 0.0))
            if pass_cost > 0:
                apply_time_cost(game_state, pass_cost, tempo_mult)
                if game_state.shot_clock_sec <= 0:
                    commit_shot_clock_turnover(offense)
                    return {
                        "end_reason": "SHOTCLOCK",
                        "pos_start_next": "after_tov_dead",
                        "points_scored": int(offense.pts) - before_pts,
                        "had_orb": had_orb,
                        "pos_start": pos_start,
                        "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
                    }
                if game_state.clock_sec <= 0:
                    game_state.clock_sec = 0
                    return {
                        "end_reason": "PERIOD_END",
                        "pos_start_next": pos_start,
                        "points_scored": int(offense.pts) - before_pts,
                        "had_orb": had_orb,
                        "pos_start": pos_start,
                        "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
                    }

            if outcome in ("PASS_KICKOUT", "PASS_SKIP", "PASS_EXTRA"):
                action = "SpotUp" if rng.random() < 0.72 else "ExtraPass"
            elif outcome == "PASS_SHORTROLL":
                action = "Drive" if rng.random() < 0.40 else "Kickout"
            else:
                action = weighted_choice(rng, off_probs)

            if pass_chain >= 3:
                action = "SpotUp"

            _refresh_action_tags(action, tags)
            stall_steps = _bump_stall(stall_steps, sc0, gc0)
            continue


    # If we exit the loop here, the only expected reason is the period/game clock reaching 0.
    # (SHOTCLOCK is handled immediately when time costs are applied.)
    if game_state.shot_clock_sec <= 0:
        commit_shot_clock_turnover(offense)
        return {
            "end_reason": "SHOTCLOCK",
            "pos_start_next": "after_tov_dead",
            "points_scored": int(offense.pts) - before_pts,
            "had_orb": had_orb,
            "pos_start": pos_start,
            "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
        }

    game_state.clock_sec = 0
    return {
        "end_reason": "PERIOD_END",
        "pos_start_next": pos_start,
        "points_scored": int(offense.pts) - before_pts,
        "had_orb": had_orb,
        "pos_start": pos_start,
        "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
    }



# -------------------------
# Game simulation

# -------------------------
