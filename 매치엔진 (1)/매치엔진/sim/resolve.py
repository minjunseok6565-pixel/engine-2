from __future__ import annotations

import random
import math
from collections.abc import Mapping
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from .builders import get_action_base
from .core import clamp, dot_profile, sigmoid
from .defense import team_def_snapshot
from .era import DEFAULT_PROB_MODEL
from .participants import (
    choose_assister_deterministic,
    choose_creator_for_pulloff,
    choose_finisher_rim,
    choose_post_target,
    choose_passer,
    choose_shooter_for_mid,
    choose_shooter_for_three,
    choose_weighted_player,
    choose_default_actor,
    choose_fouler_pid,
    choose_orb_rebounder as _choose_orb_rebounder,
    choose_drb_rebounder as _choose_drb_rebounder,
)
from .prob import (
    _shot_kind_from_outcome,
    _team_variance_mult,
    prob_from_scores,
)
from .profiles import OUTCOME_PROFILES, CORNER3_PROB_BY_ACTION_BASE
from .models import GameState, Player, TeamState

if TYPE_CHECKING:
    from config.game_config import GameConfig

def _pick_default_actor(offense: TeamState) -> Player:
    """12-role first, then best passer. Used when an outcome has no specific participant chooser."""
    return choose_default_actor(offense)

from . import quality
from .def_role_players import get_or_build_def_role_players, engine_get_stat

def _knob_mult(game_cfg: "GameConfig", key: str, default: float = 1.0) -> float:
    knobs = game_cfg.knobs if isinstance(game_cfg.knobs, Mapping) else {}
    try:
        return float(knobs.get(key, default))
    except Exception:
        return float(default)

# ------------------------------------------------------------
# Fouled-shot contact penalty (reduces and-ones, increases 2FT trips)
#
# Bucketed defaults (can override via ctx or prob_model):
#   ctx["foul_contact_pmake_mult_hard"] / ["_normal"] / ["_soft"]
#   prob_model["foul_contact_pmake_mult_hard"] / ...
# ------------------------------------------------------------
CONTACT_PENALTY_MULT = {
    "hard":   0.22,  # SHOT_RIM_CONTACT, SHOT_POST
    "normal": 0.30,  # SHOT_RIM_LAYUP (rim but weaker contact)
    "soft":   0.40,  # SHOT_MID_PU, SHOT_3_OD (jumper fouls)
}

# How to bucket each FOUL_DRAW "would-be shot"
FOUL_DRAW_CONTACT_BUCKET = {
    "SHOT_RIM_CONTACT": "hard",
    "SHOT_POST": "hard",
    "SHOT_RIM_LAYUP": "normal",
    "SHOT_MID_PU": "soft",
    "SHOT_3_OD": "soft",
}


# -------------------------
# Rebound / Free throws
# -------------------------

def resolve_free_throws(
    rng: random.Random,
    shooter: Player,
    n: int,
    team: TeamState,
    game_cfg: "GameConfig",
) -> Dict[str, Any]:
    pm = game_cfg.prob_model if isinstance(game_cfg.prob_model, Mapping) else DEFAULT_PROB_MODEL
    ft = shooter.get("SHOT_FT")
    p = clamp(
        float(pm.get("ft_base", 0.45)) + (ft / 100.0) * float(pm.get("ft_range", 0.47)),
        float(pm.get("ft_min", 0.40)),
        float(pm.get("ft_max", 0.95)),
    )
    fta = 0
    ftm = 0
    last_made = False
    for _ in range(int(n)):
        team.fta += 1
        team.add_player_stat(shooter.pid, "FTA", 1)
        fta += 1
        made = rng.random() < p
        last_made = bool(made)
        if made:
            team.ftm += 1
            team.pts += 1
            team.add_player_stat(shooter.pid, "FTM", 1)
            team.add_player_stat(shooter.pid, "PTS", 1)
            ftm += 1
    return {"fta": fta, "ftm": ftm, "last_made": last_made, "p_ft": float(p)}

def rebound_orb_probability(
    offense: TeamState,
    defense: TeamState,
    orb_mult: float,
    drb_mult: float,
    game_cfg: "GameConfig",
) -> float:
    off_players = offense.on_court_players()
    def_players = defense.on_court_players()
    off_orb = sum(p.get("REB_OR") for p in off_players) / max(len(off_players), 1)
    def_drb = sum(p.get("REB_DR") for p in def_players) / max(len(def_players), 1)
    off_orb *= orb_mult
    def_drb *= drb_mult
    pm = game_cfg.prob_model if isinstance(game_cfg.prob_model, Mapping) else DEFAULT_PROB_MODEL
    base = float(pm.get("orb_base", 0.26)) * _knob_mult(game_cfg, "orb_base_mult", 1.0)
    return prob_from_scores(
        None,
        base,
        off_orb,
        def_drb,
        kind="rebound",
        variance_mult=1.0,
        game_cfg=game_cfg,
    )

def choose_orb_rebounder(rng: random.Random, offense: TeamState) -> Player:
    """Compatibility wrapper: rebounder selection lives in participants."""
    return _choose_orb_rebounder(rng, offense)


def choose_drb_rebounder(rng: random.Random, defense: TeamState) -> Player:
    """Compatibility wrapper: rebounder selection lives in participants."""
    return _choose_drb_rebounder(rng, defense)



# -------------------------
# Outcome helpers
# -------------------------

def is_shot(o: str) -> bool: return o.startswith("SHOT_")
def is_pass(o: str) -> bool: return o.startswith("PASS_")
def is_to(o: str) -> bool: return o.startswith("TO_")
def is_foul(o: str) -> bool: return o.startswith("FOUL_")
def is_reset(o: str) -> bool: return o.startswith("RESET_")


def shot_zone_from_outcome(outcome: str) -> Optional[str]:
    if outcome in ("SHOT_RIM_LAYUP", "SHOT_RIM_DUNK", "SHOT_RIM_CONTACT", "SHOT_TOUCH_FLOATER"):
        return "rim"
    if outcome in ("SHOT_MID_CS", "SHOT_MID_PU"):
        return "mid"
    if outcome in ("SHOT_3_CS", "SHOT_3_OD"):
        return "3"
    return None


def shot_zone_detail_from_outcome(
    outcome: str,
    action: str,
    game_cfg: "GameConfig",
    rng: Optional[random.Random] = None,
) -> Optional[str]:
    """Map outcome -> NBA shot-chart zone (detail).

    For 3PA, we sample corner vs ATB using a *base-action* probability table so
    we don't deterministically over-produce corner 3s.
    """
    base_action = get_action_base(action, game_cfg)

    if outcome in ("SHOT_RIM_LAYUP", "SHOT_RIM_DUNK", "SHOT_RIM_CONTACT"):
        return "Restricted_Area"
    if outcome in ("SHOT_TOUCH_FLOATER", "SHOT_POST"):
        return "Paint_Non_RA"
    if outcome in ("SHOT_MID_CS", "SHOT_MID_PU"):
        return "Mid_Range"

    if outcome in ("SHOT_3_CS", "SHOT_3_OD"):
        p = float(CORNER3_PROB_BY_ACTION_BASE.get(base_action, CORNER3_PROB_BY_ACTION_BASE.get("default", 0.12)))
        r = (rng.random() if rng is not None else random.random())
        return "Corner_3" if r < p else "ATB_3"

    return None

def outcome_points(o: str) -> int:
    return 3 if o in ("SHOT_3_CS","SHOT_3_OD") else 2 if o.startswith("SHOT_") else 0


# -------------------------
# Resolve sampled outcome into events
# -------------------------

def resolve_outcome(
    rng: random.Random,
    outcome: str,
    action: str,
    offense: TeamState,
    defense: TeamState,
    tags: Dict[str, Any],
    pass_chain: int,
    def_action: str,
    ctx: Optional[Dict[str, Any]] = None,
    game_state: Optional[GameState] = None,
    game_cfg: Optional["GameConfig"] = None,
) -> Tuple[str, Dict[str, Any]]:
    # count outcome
    offense.outcome_counts[outcome] = offense.outcome_counts.get(outcome, 0) + 1

    if ctx is None:
        ctx = {}
    if game_cfg is None:
        raise ValueError("resolve_outcome requires game_cfg")

    off_team_key = str(ctx.get("off_team_key") or "")
    def_team_key = str(ctx.get("def_team_key") or "")

    def _with_team(payload: Dict[str, Any], include_fouler: bool = False) -> Dict[str, Any]:
        if off_team_key:
            payload.setdefault("team", off_team_key)
        if include_fouler and def_team_key:
            payload.setdefault("fouler_team", def_team_key)
        return payload

    if outcome == "TO_SHOT_CLOCK":
        actor = _pick_default_actor(offense)
        offense.tov += 1
        offense.add_player_stat(actor.pid, "TOV", 1)
        return "TURNOVER", _with_team({"outcome": outcome, "pid": actor.pid})

    def _record_exception(where: str, exc: BaseException) -> None:
        """Record exceptions into ctx for debugging without breaking sim flow."""
        try:
            errs = ctx.setdefault("errors", [])
            errs.append(
                {
                    "where": where,
                    "outcome": outcome,
                    "action": action,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        except Exception:
            # Never allow debug recording to crash the sim.
            return

    # role-fit bad outcome logging (internal; only when role-fit was applied on this step)
    try:
        if bool(tags.get("role_fit_applied", False)):
            g = str(tags.get("role_fit_grade", "B"))
            if is_to(outcome):
                offense.role_fit_bad_totals["TO"] = offense.role_fit_bad_totals.get("TO", 0) + 1
                offense.role_fit_bad_by_grade.setdefault(g, {}).setdefault("TO", 0)
                offense.role_fit_bad_by_grade[g]["TO"] += 1
            elif is_reset(outcome):
                offense.role_fit_bad_totals["RESET"] = offense.role_fit_bad_totals.get("RESET", 0) + 1
                offense.role_fit_bad_by_grade.setdefault(g, {}).setdefault("RESET", 0)
                offense.role_fit_bad_by_grade[g]["RESET"] += 1
    except Exception as e:
        _record_exception("role_fit_bad_logging", e)
        pass

    # Prob model / tuning knobs (ctx can override per-run)
    pm = ctx.get("prob_model")
    if not isinstance(pm, Mapping):
        pm = game_cfg.prob_model if isinstance(game_cfg.prob_model, Mapping) else DEFAULT_PROB_MODEL

    # shot_diet participant bias (optional)
    style = ctx.get("shot_diet_style")

    base_action = get_action_base(action, game_cfg)
    def_snap = team_def_snapshot(defense)
    prof = OUTCOME_PROFILES.get(outcome)
    if not prof:
        return "RESET", {"outcome": outcome}

    # choose participants
    if is_shot(outcome):
        if outcome in ("SHOT_3_CS",):
            actor = choose_shooter_for_three(rng, offense, style=style)
        elif outcome in ("SHOT_MID_CS",):
            actor = choose_shooter_for_mid(rng, offense, style=style)
        elif outcome in ("SHOT_3_OD","SHOT_MID_PU"):
            actor = choose_creator_for_pulloff(rng, offense, outcome, style=style)
        elif outcome == "SHOT_POST":
            actor = choose_post_target(offense)
        elif outcome in ("SHOT_RIM_DUNK",):
            actor = choose_finisher_rim(rng, offense, dunk_bias=True, style=style, base_action=base_action)
        else:
            actor = choose_finisher_rim(rng, offense, dunk_bias=False, style=style, base_action=base_action)
    elif is_pass(outcome):
        actor = choose_passer(rng, offense, base_action, outcome, style=style)
    elif is_foul(outcome):
        # foul draw actor: tie to most likely attempt type
        if outcome == "FOUL_DRAW_POST":
            actor = choose_post_target(offense)
        elif outcome == "FOUL_DRAW_JUMPER":
            actor = choose_creator_for_pulloff(rng, offense, "SHOT_3_OD", style=style)
        else:
            actor = choose_finisher_rim(rng, offense, dunk_bias=False, style=style, base_action=base_action)
    else:
        actor = _pick_default_actor(offense)

    variance_mult = _team_variance_mult(offense, game_cfg) * float(ctx.get("variance_mult", 1.0))

    # compute scores
    off_vals = {k: actor.get(k) for k in prof["offense"].keys()}
    off_score = dot_profile(off_vals, prof["offense"])
    def_vals = {k: float(def_snap.get(k, 50.0)) for k in prof["defense"].keys()}
    def_score = dot_profile(def_vals, prof["defense"])
    def_score *= float(ctx.get("def_eff_mult", 1.0))

    fatigue_map = ctx.get("fatigue_map", {}) or {}
    fatigue_logit_max = float(ctx.get("fatigue_logit_max", -0.25))
    fatigue_val = float(fatigue_map.get(actor.pid, 1.0))
    fatigue_logit_delta = (1.0 - fatigue_val) * fatigue_logit_max

    # PASS-carry: applied once to the *next* shot/pass (and optionally shooting-foul) and then consumed.
    carry_in = 0.0
    if is_shot(outcome) or is_pass(outcome) or (is_foul(outcome) and outcome.startswith("FOUL_DRAW_")):
        try:
            carry_in = float(ctx.pop("carry_logit_delta", 0.0) or 0.0)
        except Exception as e:
            _record_exception("carry_logit_delta_pop", e)
            carry_in = 0.0

    # resolve by type
    if is_shot(outcome):
        # QUALITY: scheme structure + defensive role stats -> logit delta (shot).
        scheme = getattr(defense.tactics, "defense_scheme", "")
        debug_q = bool(ctx.get("debug_quality", False))
        role_players = get_or_build_def_role_players(
            ctx,
            defense,
            scheme=scheme,
            debug_detail_key=("def_role_players_detail" if debug_q else None),
        )
        q_detail = None
        try:
            if debug_q:
                q_detail = quality.compute_quality_score(
                    scheme=scheme,
                    base_action=base_action,
                    outcome=outcome,
                    role_players=role_players,
                    get_stat=engine_get_stat,
                    return_detail=True,
                )
                q_score = float(q_detail.score)
            else:
                q_score = float(quality.compute_quality_score(
                    scheme=scheme,
                    base_action=base_action,
                    outcome=outcome,
                    role_players=role_players,
                    get_stat=engine_get_stat,
                ))
        except Exception as e:
            _record_exception("quality_compute_shot", e)
            q_score = 0.0
        q_delta = float(quality.score_to_logit_delta(outcome, q_score))
        # Reduce existing def_score impact on SHOT to avoid double counting.
        def_score = float(quality.mix_def_score_for_shot(float(def_score)))
        shot_dbg = {}
        if debug_q:
            shot_dbg = {"q_score": float(q_score), "q_delta": float(q_delta), "q_detail": q_detail, "carry_in": float(carry_in)}
        shot_base = game_cfg.shot_base if isinstance(game_cfg.shot_base, Mapping) else {}
        base_p = shot_base.get(outcome, 0.45)
        kind = _shot_kind_from_outcome(outcome)
        if kind == "shot_rim":
            base_p *= _knob_mult(game_cfg, "shot_base_rim_mult", 1.0)
        elif kind == "shot_mid":
            base_p *= _knob_mult(game_cfg, "shot_base_mid_mult", 1.0)
        else:
            base_p *= _knob_mult(game_cfg, "shot_base_3_mult", 1.0)
        p_make = prob_from_scores(
            rng,
            base_p,
            off_score,
            def_score,
            kind=kind,
            variance_mult=variance_mult,
            logit_delta=float(tags.get('role_logit_delta', 0.0)) + float(carry_in) + float(q_delta),
            fatigue_logit_delta=fatigue_logit_delta,
            game_cfg=game_cfg,
        )

        pts = outcome_points(outcome)

        offense.fga += 1
        zone = shot_zone_from_outcome(outcome)
        if zone:
            offense.shot_zones[zone] = offense.shot_zones.get(zone, 0) + 1
        zone_detail = shot_zone_detail_from_outcome(outcome, action, game_cfg, rng)
        if zone_detail:
            offense.shot_zone_detail.setdefault(zone_detail, {"FGA": 0, "FGM": 0, "AST_FGM": 0})
            offense.shot_zone_detail[zone_detail]["FGA"] += 1
        if game_state is not None and "first_fga_shotclock_sec" not in ctx:
            ctx["first_fga_shotclock_sec"] = float(game_state.shot_clock_sec)
        offense.add_player_stat(actor.pid, "FGA", 1)
        if pts == 3:
            offense.tpa += 1
            offense.add_player_stat(actor.pid, "3PA", 1)

        if rng.random() < p_make:
            offense.fgm += 1
            offense.add_player_stat(actor.pid, "FGM", 1)
            if pts == 3:
                offense.tpm += 1
                offense.add_player_stat(actor.pid, "3PM", 1)
            offense.pts += pts
            offense.add_player_stat(actor.pid, "PTS", pts)
            if zone_detail:
                offense.shot_zone_detail[zone_detail]["FGM"] += 1

            assisted = False
            assister_pid = None
            pass_chain_val = ctx.get("pass_chain", pass_chain)
            base_action = get_action_base(action, game_cfg)

            if "_CS" in outcome:
                assisted = True
            elif outcome in ("SHOT_RIM_LAYUP", "SHOT_RIM_DUNK", "SHOT_RIM_CONTACT"):
                # Rim finishes: strongly assisted off movement/advantage actions.
                if pass_chain_val and float(pass_chain_val) > 0:
                    assisted = True
                else:
                    # 컷/롤/핸드오프 계열은 패스 동반 가능성이 높음 (PnR 세부액션 포함)
                    if base_action in ("Cut", "PnR", "DHO") and rng.random() < 0.90:
                        assisted = True
                    elif base_action in ("Kickout", "ExtraPass") and rng.random() < 0.70:
                        assisted = True
                    elif base_action == "Drive" and rng.random() < 0.7:
                        assisted = True
            elif outcome == "SHOT_TOUCH_FLOATER":
                # Touch/floater: reduce assisted credit to pull down Paint_Non_RA AST share.
                if pass_chain_val and float(pass_chain_val) >= 2:
                    assisted = True
                else:
                    if base_action in ("Cut", "PnR", "DHO") and rng.random() < 0.55:
                        assisted = True
                    elif base_action in ("Kickout", "ExtraPass") and rng.random() < 0.40:
                        assisted = True
                    elif base_action == "Drive" and rng.random() < 0.18:
                        assisted = True
            elif outcome == "SHOT_3_OD":
                # OD 3도 2+패스 연쇄에서는 일부 assist로 잡히는 편이 자연스럽다
                if pass_chain_val and float(pass_chain_val) >= 2 and base_action in ("PnR", "DHO", "Kickout", "ExtraPass") and rng.random() < 0.28:
                    assisted = True
            # "_PU" 계열은 기본적으로 unassisted로 둔다

            if assisted:
                assister = choose_assister_deterministic(offense, actor.pid)
                if assister:
                    assister_pid = assister.pid
                    offense.ast += 1
                    offense.add_player_stat(assister.pid, "AST", 1)
                    if zone_detail:
                        offense.shot_zone_detail[zone_detail]["AST_FGM"] += 1

            if zone_detail in ("Restricted_Area", "Paint_Non_RA"):
                offense.pitp += 2

            return "SCORE", _with_team({
                "outcome": outcome,
                "pid": actor.pid,
                "points": pts,
                "shot_zone_detail": zone_detail,
                "assisted": assisted,
                "assister_pid": assister_pid,
                **shot_dbg,
            })
        else:
            return "MISS", _with_team({
                "outcome": outcome,
                "pid": actor.pid,
                "points": pts,
                "shot_zone_detail": zone_detail,
                "assisted": False,
                "assister_pid": None,
                **shot_dbg,
            })

    if is_pass(outcome):
        pass_base = game_cfg.pass_base_success if isinstance(game_cfg.pass_base_success, Mapping) else {}
        base_s = pass_base.get(outcome, 0.90) * _knob_mult(game_cfg, "pass_base_success_mult", 1.0)

        # PASS completion (offense vs defense) - this preserves passer skill influence.
        p_ok = prob_from_scores(
            rng,
            base_s,
            off_score,
            def_score,
            kind="pass",
            variance_mult=variance_mult,
            logit_delta=float(tags.get('role_logit_delta', 0.0)) + float(carry_in),
            game_cfg=game_cfg,
        )

        # PASS quality (defensive scheme structure + defensive role stats)
        scheme = getattr(defense.tactics, "defense_scheme", "")
        debug_q = bool(ctx.get("debug_quality", False))
        role_players = get_or_build_def_role_players(
            ctx,
            defense,
            scheme=scheme,
            debug_detail_key=("def_role_players_detail" if debug_q else None),
        )

        q_detail = None
        try:
            if debug_q:
                q_detail = quality.compute_quality_score(
                    scheme=scheme,
                    base_action=base_action,
                    outcome=outcome,
                    role_players=role_players,
                    get_stat=engine_get_stat,
                    return_detail=True,
                )
                q_score = float(q_detail.score)
            else:
                q_score = float(
                    quality.compute_quality_score(
                        scheme=scheme,
                        base_action=base_action,
                        outcome=outcome,
                        role_players=role_players,
                        get_stat=engine_get_stat,
                    )
                )
        except Exception as e:
            _record_exception("quality_compute_pass", e)
            q_score = 0.0

                # Threshold buckets (score in [-2.5, +2.5])
        t_to = float(ctx.get("pass_q_to", -1.5))
        t_reset = float(ctx.get("pass_q_reset", -0.7))
        t_neg = float(ctx.get("pass_q_neg", -0.2))
        t_pos = float(ctx.get("pass_q_pos", 0.2))

        # Smooth/continuous PASS quality buckets.
        # - Old behavior: hard cutoffs (<= t_to => TO, <= t_reset => RESET, carry bucket by <= t_neg / >= t_pos)
        # - New behavior: the same thresholds define the *midpoints* (p=0.5) of sigmoid transitions.
        #   Larger slopes => closer to the old step-function behavior.
        s_to = float(ctx.get("pass_q_to_slope", 6.0))
        s_reset = float(ctx.get("pass_q_reset_slope", 6.0))
        s_carry = float(ctx.get("pass_q_carry_slope", 5.0))

        # Probabilistic bucket 1: turnover chance increases as q_score drops below t_to.
        p_to = float(sigmoid(s_to * (t_to - q_score)))
        if rng.random() < p_to:
            offense.outcome_counts["TO_BAD_PASS"] = offense.outcome_counts.get("TO_BAD_PASS", 0) + 1
            offense.tov += 1
            offense.add_player_stat(actor.pid, "TOV", 1)
            payload = {"outcome": "TO_BAD_PASS", "pid": actor.pid, "type": "PASS_QUALITY_TO"}
            if debug_q:
                payload.update(
                    {
                        "q_score": q_score,
                        "q_detail": q_detail,
                        "thresholds": {"to": t_to, "reset": t_reset, "neg": t_neg, "pos": t_pos},
                        "probs": {"p_to": float(p_to)},
                        "slopes": {"to": float(s_to), "reset": float(s_reset), "carry": float(s_carry)},
                        "carry_in": float(carry_in),
                    }
                )
            return "TURNOVER", _with_team(payload)

        # Probabilistic bucket 2: reset chance increases as q_score drops below t_reset.
        p_reset = float(sigmoid(s_reset * (t_reset - q_score)))
        if rng.random() < p_reset:
            payload = {"outcome": outcome, "type": "PASS_QUALITY_RESET"}
            if debug_q:
                payload.update(
                    {
                        "q_score": q_score,
                        "q_detail": q_detail,
                        "thresholds": {"to": t_to, "reset": t_reset, "neg": t_neg, "pos": t_pos},
                        "probs": {"p_reset": float(p_reset)},
                        "slopes": {"to": float(s_to), "reset": float(s_reset), "carry": float(s_carry)},
                        "carry_in": float(carry_in),
                    }
                )
            return "RESET", payload

        # For normal quality passes: sample completion. On success, store carry bucket.
        if rng.random() < p_ok:
            carry_out = 0.0
            carry_bucket = "neutral"

            # Probabilistic carry bucket: negative / neutral / positive (softmax-like).
            # We clamp logits to avoid exp overflow.
            logit_neg = float(clamp(s_carry * (t_neg - q_score), -12.0, 12.0))
            logit_pos = float(clamp(s_carry * (q_score - t_pos), -12.0, 12.0))
            w_neg = math.exp(logit_neg)
            w_pos = math.exp(logit_pos)
            w_neu = 1.0
            denom = w_neg + w_neu + w_pos
            p_neg = w_neg / denom
            p_pos = w_pos / denom
            p_neu = w_neu / denom

            r = rng.random()
            if r < p_neg:
                carry_bucket = "negative"
                carry_out = float(quality.score_to_logit_delta(outcome, q_score))
            elif r < (p_neg + p_pos):
                carry_bucket = "positive"
                carry_out = float(quality.score_to_logit_delta(outcome, q_score))
            else:
                carry_bucket = "neutral"
                carry_out = 0.0

            if carry_out != 0.0:
                try:
                    prev = float(ctx.get("carry_logit_delta", 0.0) or 0.0)
                except Exception as e:
                    _record_exception("carry_logit_delta_prev_parse", e)
                    prev = 0.0
                ctx["carry_logit_delta"] = float(quality.apply_pass_carry(prev + carry_out, next_outcome="*"))

            payload = {"outcome": outcome, "pass_chain": pass_chain + 1}
            if debug_q:
                payload.update(
                    {
                        "q_score": q_score,
                        "q_detail": q_detail,
                        "thresholds": {"to": t_to, "reset": t_reset, "neg": t_neg, "pos": t_pos},
                        "carry_bucket": carry_bucket,
                        "carry_out": float(carry_out),
                        "carry_in": float(carry_in),
                        "probs": {
                            "p_to": float(p_to),
                            "p_reset": float(p_reset),
                            "carry": {"neg": float(p_neg), "neu": float(p_neu), "pos": float(p_pos)},
                        },
                        "slopes": {"to": float(s_to), "reset": float(s_reset), "carry": float(s_carry)},
                        "p_ok": float(p_ok),
                    }
                )
            return "CONTINUE", payload

        # PASS failed (but not catastrophic enough to be a bad-pass turnover)
        payload = {"outcome": outcome, "type": "PASS_FAIL"}
        if debug_q:
            payload.update(
                {"q_score": q_score, "q_detail": q_detail, "carry_in": float(carry_in), "p_ok": float(p_ok)}
            )
        return "RESET", payload

    if is_to(outcome):
        offense.tov += 1
        offense.add_player_stat(actor.pid, "TOV", 1)
        return "TURNOVER", _with_team({"outcome": outcome, "pid": actor.pid})
    if is_foul(outcome):
        fouler_pid = None
        team_fouls = ctx.get("team_fouls") or {}
        player_fouls_by_team = ctx.get("player_fouls_by_team") or {}
        pf = player_fouls_by_team.setdefault(def_team_key, {})
        foul_out_limit = int(ctx.get("foul_out", 6))
        bonus_threshold = int(ctx.get("bonus_threshold", 5))
        def_on_court = ctx.get("def_on_court") or [p.pid for p in defense.on_court_players()]

        # assign a random fouler from on-court defenders (MVP)
        if def_on_court:
            fouler_pid = choose_fouler_pid(rng, defense, list(def_on_court), pf, foul_out_limit)
            if fouler_pid:
                pf[fouler_pid] = pf.get(fouler_pid, 0) + 1
                if game_state is not None:
                    game_state.player_fouls.setdefault(def_team_key, {})[fouler_pid] = pf[fouler_pid]

        # update team fouls
        team_fouls[def_team_key] = team_fouls.get(def_team_key, 0) + 1
        if game_state is not None:
            game_state.team_fouls[def_team_key] = team_fouls[def_team_key]

        in_bonus = bool(team_fouls.get(def_team_key, 0) >= bonus_threshold)

        # Non-shooting foul (reach/trap) becomes dead-ball unless in bonus.
        if outcome == "FOUL_REACH_TRAP" and not in_bonus:
            if fouler_pid and pf.get(fouler_pid, 0) >= foul_out_limit:
                if game_state is not None:
                    game_state.fatigue.setdefault(def_team_key, {})[fouler_pid] = 0.0
            return "FOUL_NO_SHOTS", _with_team(
                {"outcome": outcome, "pid": actor.pid, "fouler": fouler_pid, "bonus": False},
                include_fouler=True,
            )

        # Otherwise: free throws (bonus or shooting)
        shot_made = False
        pts = 0
        shot_key = None
        and_one = False

        if outcome.startswith("FOUL_DRAW_"):
            # treat as a shooting foul tied to shot type
            # Choose which "would-be" shot was fouled (affects shot-chart + and-1 mix)
            if outcome == "FOUL_DRAW_JUMPER":
                # most shooting fouls on jumpers are 2s; 3PT fouls are rarer
                shot_key = "SHOT_3_OD" if rng.random() < 0.08 else "SHOT_MID_PU"
            elif outcome == "FOUL_DRAW_POST":
                # post-ups draw both contact finishes and true post shots
                shot_key = "SHOT_POST" if rng.random() < 0.55 else "SHOT_RIM_CONTACT"
            else:  # FOUL_DRAW_RIM
                shot_key = "SHOT_RIM_CONTACT" if rng.random() < 0.40 else "SHOT_RIM_LAYUP"

            pts = 3 if shot_key == "SHOT_3_OD" else 2

            # QUALITY: apply scheme/role quality delta to FOUL_DRAW make-prob (shot-like).
            scheme = getattr(defense.tactics, "defense_scheme", "")
            debug_q = bool(ctx.get("debug_quality", False))
            role_players = get_or_build_def_role_players(
                ctx,
                defense,
                scheme=scheme,
                debug_detail_key=("def_role_players_detail" if debug_q else None),
            )
            q_detail = None
            try:
                if debug_q:
                    q_detail = quality.compute_quality_score(
                        scheme=scheme,
                        base_action=base_action,
                        outcome=outcome,
                        role_players=role_players,
                        get_stat=engine_get_stat,
                        return_detail=True,
                    )
                    q_score = float(q_detail.score)
                else:
                    q_score = float(quality.compute_quality_score(
                        scheme=scheme,
                        base_action=base_action,
                        outcome=outcome,
                        role_players=role_players,
                        get_stat=engine_get_stat,
                    ))
            except Exception as e:
                _record_exception("quality_compute_foul_draw", e)
                q_score = 0.0
            q_delta = float(quality.score_to_logit_delta(outcome, q_score))
            foul_dbg = {}
            if debug_q:
                foul_dbg = {"q_score": float(q_score), "q_delta": float(q_delta), "q_detail": q_detail, "carry_in": float(carry_in)}

            shot_base = game_cfg.shot_base if isinstance(game_cfg.shot_base, Mapping) else {}
            base_p = shot_base.get(shot_key, 0.45)
            kind = _shot_kind_from_outcome(shot_key)
            if kind == "shot_rim":
                base_p *= _knob_mult(game_cfg, "shot_base_rim_mult", 1.0)
            elif kind == "shot_mid":
                base_p *= _knob_mult(game_cfg, "shot_base_mid_mult", 1.0)
            else:
                base_p *= _knob_mult(game_cfg, "shot_base_3_mult", 1.0)

            p_make = prob_from_scores(
                rng,
                base_p,
                off_score,
                def_score,
                kind=kind,
                variance_mult=variance_mult,
                logit_delta=float(tags.get('role_logit_delta', 0.0)) + float(carry_in) + float(q_delta),
                fatigue_logit_delta=fatigue_logit_delta,
                game_cfg=game_cfg,
            )

            # Apply contact penalty ONLY for fouled shots.
            # This reduces and-ones (shot_made -> nfts=1) and shifts mix toward 2FT trips.
            bucket = FOUL_DRAW_CONTACT_BUCKET.get(shot_key, "normal")
            default_mult = float(CONTACT_PENALTY_MULT.get(bucket, 1.0))
            mult = float(
                ctx.get(
                    f"foul_contact_pmake_mult_{bucket}",
                    pm.get(f"foul_contact_pmake_mult_{bucket}", default_mult),
                )
            )
            if mult != 1.0:
                pmin = float(ctx.get("foul_contact_pmake_min", pm.get("foul_contact_pmake_min", 0.01)))
                pmax = float(ctx.get("foul_contact_pmake_max", pm.get("foul_contact_pmake_max", 0.99)))
                p_make = clamp(p_make * mult, pmin, pmax)

            # Boxscore convention for shooting fouls:
            # - MISSED fouled shot -> no FGA/3PA is recorded.
            # - MADE fouled shot   -> counts as FGA (+3PA if it was a 3), and can be an and-one.
            shot_zone = shot_zone_from_outcome(shot_key)
            zone_detail = shot_zone_detail_from_outcome(shot_key, action, game_cfg, rng)

            shot_made = rng.random() < p_make
            if shot_made:
                # count the attempt only when it goes in (and-one / 4-pt play)
                offense.fga += 1
                offense.add_player_stat(actor.pid, "FGA", 1)
                if shot_zone:
                    offense.shot_zones[shot_zone] = offense.shot_zones.get(shot_zone, 0) + 1
                if zone_detail:
                    offense.shot_zone_detail.setdefault(zone_detail, {"FGA": 0, "FGM": 0, "AST_FGM": 0})
                    offense.shot_zone_detail[zone_detail]["FGA"] += 1
                if game_state is not None and "first_fga_shotclock_sec" not in ctx:
                    ctx["first_fga_shotclock_sec"] = float(game_state.shot_clock_sec)
                if pts == 3:
                    offense.tpa += 1
                    offense.add_player_stat(actor.pid, "3PA", 1)

                offense.fgm += 1
                offense.add_player_stat(actor.pid, "FGM", 1)
                if pts == 3:
                    offense.tpm += 1
                    offense.add_player_stat(actor.pid, "3PM", 1)
                offense.pts += pts
                offense.add_player_stat(actor.pid, "PTS", pts)
                and_one = True
                if zone_detail:
                    offense.shot_zone_detail[zone_detail]["FGM"] += 1

                # minimal assist treatment on rim fouls (jumper fouls remain unassisted)
                assisted = False
                assister_pid = None
                if shot_key != "SHOT_3_OD":
                    try:
                        assisted = bool(ctx.get("pass_chain", pass_chain)) and float(
                            ctx.get("pass_chain", pass_chain)
                        ) > 0
                    except Exception as e:
                        _record_exception("assist_flag_parse", e)
                        assisted = False
                if assisted:
                    assister = choose_assister_deterministic(offense, actor.pid)
                    if assister:
                        assister_pid = assister.pid
                        offense.ast += 1
                        offense.add_player_stat(assister.pid, "AST", 1)
                        if zone_detail:
                            offense.shot_zone_detail[zone_detail]["AST_FGM"] += 1

                if zone_detail in ("Restricted_Area", "Paint_Non_RA"):
                    offense.pitp += 2

            nfts = 1 if shot_made else (3 if pts == 3 else 2)
        else:
            # bonus free throws, no shot attempt
            nfts = 2

        ft_res = resolve_free_throws(rng, actor, nfts, offense, game_cfg=game_cfg)

        if fouler_pid and pf.get(fouler_pid, 0) >= foul_out_limit:
            if game_state is not None:
                game_state.fatigue.setdefault(def_team_key, {})[fouler_pid] = 0.0

        payload = {
            "outcome": outcome,
            "pid": actor.pid,
            "fouler": fouler_pid,
            "bonus": in_bonus and not outcome.startswith("FOUL_DRAW_"),
            "shot_key": shot_key,
            "shot_made": shot_made,
            "and_one": and_one,
            "nfts": int(nfts),
        }
        if isinstance(ft_res, Mapping):
            payload.update(ft_res)
        if isinstance(foul_dbg, Mapping) and foul_dbg:
            payload.update(foul_dbg)
        return "FOUL_FT", _with_team(payload, include_fouler=True)


    if is_reset(outcome):
        return "RESET", {"outcome": outcome}

    return "RESET", {"outcome": outcome}
