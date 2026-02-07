from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Tuple

from ..builders import get_action_base
from ..core import clamp
from ..def_role_players import get_or_build_def_role_players, engine_get_stat
from ..prob import _shot_kind_from_outcome, _team_variance_mult, prob_from_scores
from ..participants import choose_assister_weighted, choose_blocker_pid
from .. import quality
from .resolve_context import ResolveContext, _knob_mult
from .resolve_pass_tracking import pick_assister_from_history, clear_pass_tracking
from .resolve_shot_chart import (
    shot_zone_from_outcome,
    shot_zone_detail_from_outcome,
    outcome_points,
    _should_award_fastbreak_fg,
)

def handle_shot(
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
    def_score_raw = float(def_score)
    def_score = float(quality.mix_def_score_for_shot(float(def_score_raw)))
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

    # Putback: fixed make penalty (logit space). Keeps putbacks "tight" regardless of quality table.
    putback_pen = 0.0
    if base_action == "Putback":
        try:
            putback_pen = float(pm.get("putback_make_logit_penalty", -0.30))
        except Exception as e:
            _record_exception("putback_make_logit_penalty", e)
            putback_pen = -0.30
    if debug_q:
        shot_dbg["putback_pen"] = float(putback_pen)

    # Help defense / double pressure adjustments (logit space).
    help_shot_ld = 0.0
    if abs(float(help_level)) > 1e-9:
        if kind == "shot_rim":
            help_shot_ld = -0.10 * float(help_level)
        elif kind == "shot_post":
            help_shot_ld = -0.08 * float(help_level)
        elif kind == "shot_mid":
            help_shot_ld = -0.03 * float(help_level)
        else:  # shot_3
            help_shot_ld = 0.09 * float(help_level)

    double_shot_pen = (-0.22 * float(double_strength)) if float(double_strength) > 1e-9 else 0.0
    if debug_q:
        shot_dbg["help_shot_ld"] = float(help_shot_ld)
        shot_dbg["double_shot_pen"] = float(double_shot_pen)

    p_make = prob_from_scores(
        rng,
        base_p,
        off_score,
        def_score,
        kind=kind,
        variance_mult=variance_mult,
        logit_delta=float(tags.get('role_logit_delta', 0.0)) + float(carry_in) + float(q_delta) + float(putback_pen) + float(help_shot_ld) + float(double_shot_pen),
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
    if game_state is not None and ctx.get("first_fga_shotclock_sec") is None:
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
        # Fastbreak points: credit on the scoring FG event (exclude FT points).
        try:
            if _should_award_fastbreak_fg(ctx, ctx.get("first_fga_shotclock_sec")):
                offense.fastbreak_pts += int(pts)
        except Exception as e:
            _record_exception("fastbreak_pts_award_fg", e)
        if zone_detail:
            offense.shot_zone_detail[zone_detail]["FGM"] += 1

        assisted_heur = False
        assister_pid = None
        pass_chain_val = ctx.get("pass_chain", pass_chain)
        base_action = get_action_base(action, game_cfg)

        if "_CS" in outcome:
            assisted_heur = True
        elif outcome in ("SHOT_RIM_LAYUP", "SHOT_RIM_DUNK", "SHOT_RIM_CONTACT"):
            # Rim finishes: strongly assisted off movement/advantage actions.
            if pass_chain_val and float(pass_chain_val) > 0:
                assisted_heur = True
            else:
                # 컷/롤/핸드오프 계열은 패스 동반 가능성이 높음 (PnR 세부액션 포함)
                if base_action in ("Cut", "PnR", "DHO") and rng.random() < 0.90:
                    assisted_heur = True
                elif base_action in ("Kickout", "ExtraPass") and rng.random() < 0.70:
                    assisted_heur = True
                elif base_action == "Drive" and rng.random() < 0.7:
                    assisted_heur = True
        elif outcome == "SHOT_TOUCH_FLOATER":
            # Touch/floater: reduce assisted credit to pull down Paint_Non_RA AST share.
            if pass_chain_val and float(pass_chain_val) >= 2:
                assisted_heur = True
            else:
                if base_action in ("Cut", "PnR", "DHO") and rng.random() < 0.55:
                    assisted_heur = True
                elif base_action in ("Kickout", "ExtraPass") and rng.random() < 0.40:
                    assisted_heur = True
                elif base_action == "Drive" and rng.random() < 0.18:
                    assisted_heur = True
        elif outcome == "SHOT_3_OD":
            # OD 3도 2+패스 연쇄에서는 일부 assist로 잡히는 편이 자연스럽다
            if pass_chain_val and float(pass_chain_val) >= 2 and base_action in ("PnR", "DHO", "Kickout", "ExtraPass") and rng.random() < 0.28:
                assisted_heur = True
        # "_PU" 계열은 기본적으로 unassisted로 둔다

        # Prefer the true last passer if we have a committed pass event in the assist window.
        assister_pid = pick_assister_from_history(ctx, offense, actor.pid, game_state, outcome)

        assisted = False
        if assister_pid is not None:
            assisted = True
        else:
            assisted = bool(assisted_heur)
            if assisted:
                assister = choose_assister_weighted(rng, offense, actor.pid, base_action, outcome, style=style)
                if assister:
                    assister_pid = assister.pid
                else:
                    assisted = False
                    assister_pid = None

        if assister_pid is not None:
            offense.ast += 1
            offense.add_player_stat(assister_pid, "AST", 1)
            if zone_detail:
                offense.shot_zone_detail[zone_detail]["AST_FGM"] += 1

        if zone_detail in ("Restricted_Area", "Paint_Non_RA"):
            offense.pitp += 2

        clear_pass_tracking(ctx)

        return "SCORE", _with_matchup({
            "outcome": outcome,
            "pid": actor.pid,
            "points": pts,
            "shot_zone_detail": zone_detail,
            "assisted": assisted,
            "assister_pid": assister_pid,
            **shot_dbg,
        })
    else:
        payload = {
            "outcome": outcome,
            "pid": actor.pid,
            "points": pts,
            "shot_zone_detail": zone_detail,
            "assisted": False,
            "assister_pid": None,
            **shot_dbg,
        }

        # BLOCK: on missed shots only, sample whether the miss was a block and credit a defender.
        # This is intentionally a "miss subtype" so it doesn't distort make rates.
        try:
            if kind == "shot_rim":
                base_block = float(pm.get("block_base_rim", 0.085))
            elif kind == "shot_post":
                base_block = float(pm.get("block_base_post", 0.065))
            elif kind == "shot_mid":
                base_block = float(pm.get("block_base_mid", 0.022))
            else:  # shot_3
                base_block = float(pm.get("block_base_3", 0.012))

            def_feat = getattr(style, "def_features", {}) if style is not None else {}
            d_rim = float(def_feat.get("D_RIM_PROTECT", 0.5))
            d_poa = float(def_feat.get("D_POA", 0.5))
            d_help = float(def_feat.get("D_HELP_CLOSEOUT", 0.5))

            if kind in ("shot_rim", "shot_post"):
                block_logit_delta = (-float(q_score)) * 0.30 + (d_rim - 0.5) * 1.00 + (d_help - 0.5) * 0.35
            else:
                block_logit_delta = (-float(q_score)) * 0.25 + (d_poa - 0.5) * 0.70 + (d_help - 0.5) * 0.25

            block_var = _team_variance_mult(defense, game_cfg) * float(ctx.get("variance_mult", 1.0))
            p_block = prob_from_scores(
                rng,
                base_block,
                def_score_raw,
                off_score,
                kind="block",
                variance_mult=block_var,
                logit_delta=float(block_logit_delta),
                game_cfg=game_cfg,
            )

            if rng.random() < p_block:
                blocker_pid = choose_blocker_pid(rng, defense, kind)
                if blocker_pid:
                    defense.add_player_stat(blocker_pid, "BLK", 1)
                payload.update({"blocked": True, "blocker_pid": blocker_pid, "block_kind": kind})

            if debug_q:
                payload["p_block"] = float(p_block)
                payload["block_logit_delta"] = float(block_logit_delta)
        except Exception as e:
            _record_exception("block_model", e)

        clear_pass_tracking(ctx)
        return "MISS", _with_matchup(payload)

