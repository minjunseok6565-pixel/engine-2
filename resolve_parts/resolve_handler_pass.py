from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Dict, Tuple

from ..core import clamp, dot_profile, sigmoid
from ..def_role_players import get_or_build_def_role_players, engine_get_stat
from ..profiles import OUTCOME_PROFILES
from ..prob import _team_variance_mult, prob_from_scores
from .. import quality
from .resolve_context import ResolveContext, _knob_mult
from .resolve_pass_tracking import clear_pass_tracking
from ..participants import choose_stealer_pid  # used in steal split

def handle_pass(
    rc: ResolveContext,
    _with_matchup,
    _record_exception,
) -> Tuple[str, Dict[str, Any]]:
    rng = rc.rng
    outcome = rc.outcome
    action = rc.action
    offense = rc.offense
    defense = rc.defense
    tags = rc.tags
    pass_chain = rc.pass_chain
    ctx = rc.ctx
    game_state = rc.game_state
    game_cfg = rc.game_cfg
    pm = rc.pm
    style = rc.style
    base_action = rc.base_action
    variance_mult = rc.variance_mult
    off_score = rc.off_score
    def_score = rc.def_score
    fatigue_logit_delta = rc.fatigue_logit_delta
    carry_in = rc.carry_in
    actor = rc.actor
    help_level = rc.help_level
    double_strength = rc.double_strength
    double_doubler_pid = rc.double_doubler_pid
    double_source = getattr(rc, "double_source", None)

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
        logit_delta=float(tags.get('role_logit_delta', 0.0)) + float(carry_in) + (-0.04 * float(help_level)) + (-0.16 * float(double_strength)),
        fatigue_logit_delta=fatigue_logit_delta,
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
    t_to = float(ctx.get("pass_q_to", -0.47))
    t_reset = float(ctx.get("pass_q_reset", -0.3))
    t_neg = float(ctx.get("pass_q_neg", -0.2))
    t_pos = float(ctx.get("pass_q_pos", 0.2))

    # Smooth/continuous PASS quality buckets.
    # - Old behavior: hard cutoffs (<= t_to => TO, <= t_reset => RESET, carry bucket by <= t_neg / >= t_pos)
    # - New behavior: the same thresholds define the *midpoints* (p=0.5) of sigmoid transitions.
    #   Larger slopes => closer to the old step-function behavior.
    s_to = float(ctx.get("pass_q_to_slope", 6.5))
    s_reset = float(ctx.get("pass_q_reset_slope", 6.0))
    s_carry = float(ctx.get("pass_q_carry_slope", 5.0))

    # Probabilistic bucket 1: turnover chance increases as q_score drops below t_to.
    # Apply passer skill adjustment directly to the turnover logit (higher passer skill => fewer bad-pass TOs).
    passer_span = float(ctx.get("pass_q_to_passer_logit_span", 0.6))
    compute_passer = bool(debug_q) or abs(passer_span) > 1e-12

    passer_bp = 50.0
    bp_norm = 0.0
    if compute_passer:
        try:
            prof = OUTCOME_PROFILES.get(outcome, {}).get("offense")
            if not isinstance(prof, dict) or not prof:
                prof = OUTCOME_PROFILES.get("TO_BAD_PASS", {}).get("offense")
            if isinstance(prof, dict) and prof:
                vals = {k: float(actor.get(k, fatigue_sensitive=True)) for k in prof.keys()}
                passer_bp = float(dot_profile(vals, prof, missing_default=50.0))
                bp_norm = float(clamp((passer_bp - 50.0) / 50.0, -1.0, 1.0))
        except Exception as e:
            _record_exception("pass_q_passer_bp", e)
            passer_bp = 50.0
            bp_norm = 0.0

    to_logit_raw = float(s_to * (t_to - q_score))
    to_logit_eff = float(to_logit_raw - (bp_norm * passer_span) + (0.12 * float(help_level)) + (0.20 * float(double_strength)))
    p_to = float(sigmoid(to_logit_eff))
    p_to_raw = float(sigmoid(to_logit_raw)) if debug_q else None
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
                    "passer_bp": float(passer_bp),
                    "bp_norm": float(bp_norm),
                    "passer_span": float(passer_span),
                    "to_logit_raw": float(to_logit_raw),
                    "to_logit_eff": float(to_logit_eff),
                    "probs": {"p_to_raw": float(p_to_raw), "p_to": float(p_to)},
                    "slopes": {"to": float(s_to), "reset": float(s_reset), "carry": float(s_carry)},
                    "carry_in": float(carry_in),
                }
            )
        # STEAL / LINEOUT split for TO_BAD_PASS:
        # - If intercepted, credit STL and mark as a live-ball steal (strong transition start).
        # - Otherwise, allow some bad passes to become dead-ball lineouts to reduce "all live-ball" feel.
        p_steal = 0.0
        p_lineout = 0.0
        try:
            steal_base = float(pm.get("steal_bad_pass_base", 0.60))
            steal_mult = {
                "PASS_SKIP": 1.13,
                "PASS_EXTRA": 1.03,
                "PASS_KICKOUT": 0.97,
                "PASS_SHORTROLL": 0.92,
            }.get(outcome, 1.00)
            base_steal = steal_base * float(steal_mult)

            def_feat = getattr(style, "def_features", {}) if style is not None else {}
            d_press = float(def_feat.get("D_STEAL_PRESS", 0.5))
            steal_logit_delta = (-float(q_score)) * 0.40 + (d_press - 0.5) * 1.10 + (0.18 * float(help_level)) + (0.22 * float(double_strength))

            steal_var = _team_variance_mult(defense, game_cfg) * float(ctx.get("variance_mult", 1.0))
            p_steal = prob_from_scores(
                rng,
                base_steal,
                def_score,
                off_score,
                kind="steal",
                variance_mult=steal_var,
                logit_delta=float(steal_logit_delta),
                game_cfg=game_cfg,
            )

            if rng.random() < p_steal:
                stealer_pid = None
                try:
                    if double_doubler_pid and defense.is_on_court(double_doubler_pid):
                        stealer_pid = double_doubler_pid
                except Exception:
                    stealer_pid = None
                if not stealer_pid:
                    stealer_pid = choose_stealer_pid(rng, defense)
                if stealer_pid:
                    defense.add_player_stat(stealer_pid, "STL", 1)
                payload.update({"steal": True, "stealer_pid": stealer_pid, "pos_start_next_override": "after_steal"})
            else:
                lineout_base = float(pm.get("bad_pass_lineout_base", 0.30))
                p_lineout = clamp(lineout_base + max(0.0, -float(q_score)) * 0.06, 0.05, 0.55)
                if rng.random() < p_lineout:
                    payload.update({"deadball_override": True, "tov_deadball_reason": "LINEOUT_BAD_PASS"})

            if debug_q:
                payload.setdefault("probs", {}).update({"p_steal": float(p_steal), "p_lineout": float(p_lineout)})
        except Exception as e:
            _record_exception("steal_split_bad_pass", e)

        clear_pass_tracking(ctx)

        return "TURNOVER", _with_matchup(payload)


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
                    "passer_bp": float(passer_bp),
                    "bp_norm": float(bp_norm),
                    "passer_span": float(passer_span),
                    "to_logit_raw": float(to_logit_raw),
                    "to_logit_eff": float(to_logit_eff),
                    "probs": {"p_to": float(p_to), "p_to_raw": float(p_to_raw), "p_reset": float(p_reset)},
                    "slopes": {"to": float(s_to), "reset": float(s_reset), "carry": float(s_carry)},
                    "carry_in": float(carry_in),
                }
            )
        clear_pass_tracking(ctx)

        return "RESET", _with_matchup(payload)

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

        ctx["_pending_pass_event"] = {"pid": actor.pid, "outcome": outcome, "base_action": base_action}

        # If we successfully pass out of a double, create a one-step rotation advantage
        # (4v3 / scramble defense) and terminate the on-ball double for the next step.
        if float(double_strength) > 1e-9:
            try:
                ctx["rotation_adv"] = {
                    "ttl": 1,
                    "adv": float(clamp(float(double_strength), 0.0, 1.0)),
                    "source": (str(double_source or "DOUBLE") or "DOUBLE"),
                }
            except Exception:
                pass
            ctx.pop("double_active", None)

        payload = {"outcome": outcome, "pass_chain": pass_chain + 1}
        if debug_q:
            payload.update(
                {
                    "q_score": q_score,
                    "q_detail": q_detail,
                    "passer_bp": float(passer_bp),
                    "bp_norm": float(bp_norm),
                    "passer_span": float(passer_span),
                    "to_logit_raw": float(to_logit_raw),
                    "to_logit_eff": float(to_logit_eff),
                    "thresholds": {"to": t_to, "reset": t_reset, "neg": t_neg, "pos": t_pos},
                    "carry_bucket": carry_bucket,
                    "carry_out": float(carry_out),
                    "carry_in": float(carry_in),
                    "probs": {
                        "p_to": float(p_to),
                        "p_to_raw": float(p_to_raw),
                        "p_reset": float(p_reset),
                        "carry": {"neg": float(p_neg), "neu": float(p_neu), "pos": float(p_pos)},
                    },
                    "slopes": {"to": float(s_to), "reset": float(s_reset), "carry": float(s_carry)},
                    "p_ok": float(p_ok),
                }
            )
        return "CONTINUE", _with_matchup(payload)

    # PASS failed (but not catastrophic enough to be a bad-pass turnover)
    payload = {"outcome": outcome, "type": "PASS_FAIL"}
    if debug_q:
        payload.update(
            {"q_score": q_score, "q_detail": q_detail, "carry_in": float(carry_in), "p_ok": float(p_ok)}
        )
    clear_pass_tracking(ctx)

    return "RESET", _with_matchup(payload)
