from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Tuple

from ..core import clamp
from ..def_role_players import get_or_build_def_role_players, engine_get_stat
from ..participants import choose_assister_weighted, choose_fouler_pid
from ..prob import _shot_kind_from_outcome, prob_from_scores
from .. import quality
from .resolve_context import ResolveContext, _knob_mult
from .resolve_ft_rebound import resolve_free_throws
from .resolve_pass_tracking import pick_assister_from_history, clear_pass_tracking
from .resolve_shot_chart import (
    shot_zone_from_outcome,
    shot_zone_detail_from_outcome,
    _should_award_fastbreak_fg,
)

# ------------------------------------------------------------
# Fouled-shot contact penalty (reduces and-ones, increases 2FT trips)
# ------------------------------------------------------------
CONTACT_PENALTY_MULT = {
    "hard":   0.22,  # SHOT_RIM_CONTACT, SHOT_POST
    "normal": 0.30,  # SHOT_RIM_LAYUP (rim but weaker contact)
    "soft":   0.40,  # SHOT_MID_PU, SHOT_3_OD (jumper fouls)
}

FOUL_DRAW_CONTACT_BUCKET = {
    "SHOT_RIM_CONTACT": "hard",
    "SHOT_POST": "hard",
    "SHOT_RIM_LAYUP": "normal",
    "SHOT_MID_PU": "soft",
    "SHOT_3_OD": "soft",
}

def handle_foul(
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

    game_id = rc.game_id
    off_team_id = rc.off_team_id
    def_team_id = rc.def_team_id

    fouler_pid = None
    team_fouls = game_state.team_fouls
    player_fouls_by_team = game_state.player_fouls
    if def_team_id not in team_fouls:
        raise ValueError(
            "resolve_outcome(): missing team_fouls bucket for def_team_id "
            f"(game_id={game_id!r}, def_team_id={def_team_id!r}, keys={list(team_fouls.keys())!r})"
        )
    if def_team_id not in player_fouls_by_team:
        raise ValueError(
            "resolve_outcome(): missing player_fouls bucket for def_team_id "
            f"(game_id={game_id!r}, def_team_id={def_team_id!r}, keys={list(player_fouls_by_team.keys())!r})"
        )
    if def_team_id not in game_state.fatigue:
        raise ValueError(
            "resolve_outcome(): missing fatigue bucket for def_team_id "
            f"(game_id={game_id!r}, def_team_id={def_team_id!r}, keys={list(game_state.fatigue.keys())!r})"
        )
    pf = player_fouls_by_team[def_team_id]
    foul_out_limit = int(ctx.get("foul_out", 6))
    bonus_threshold = int(ctx.get("bonus_threshold", 5))
    def_on_court = ctx.get("def_on_court") or [p.pid for p in defense.on_court_players()]

    # assign a random fouler from on-court defenders (MVP)
    if def_on_court:
        fouler_pid = choose_fouler_pid(rng, defense, list(def_on_court), pf, foul_out_limit, outcome)
        if fouler_pid:
            pf[fouler_pid] = pf.get(fouler_pid, 0) + 1

    # update team fouls
    team_fouls[def_team_id] = int(team_fouls[def_team_id]) + 1

    in_bonus = bool(int(team_fouls[def_team_id]) >= bonus_threshold)

    # Non-shooting foul (reach/trap) becomes dead-ball unless in bonus.
    if outcome == "FOUL_REACH_TRAP" and not in_bonus:
        if fouler_pid and pf.get(fouler_pid, 0) >= foul_out_limit:
            game_state.fatigue[def_team_id][fouler_pid] = 0.0
        clear_pass_tracking(ctx)
        return "FOUL_NO_SHOTS", _with_matchup(
            {"outcome": outcome, "pid": actor.pid, "fouler": fouler_pid, "bonus": False}
        )

    # Otherwise: free throws (bonus or shooting)
    shot_made = False
    pts = 0
    shot_key = None
    and_one = False
    foul_dbg = {}
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
            logit_delta=float(tags.get('role_logit_delta', 0.0)) + float(carry_in) + float(q_delta) + (
                (-0.10 * float(help_level)) if kind == "shot_rim" else
                (-0.08 * float(help_level)) if kind == "shot_post" else
                (-0.03 * float(help_level)) if kind == "shot_mid" else
                (0.09 * float(help_level))
            ) + ((-0.22 * float(double_strength)) if float(double_strength) > 1e-9 else 0.0),
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
            if game_state is not None and ctx.get("first_fga_shotclock_sec") is None:
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
            # Fastbreak points: credit only the made FG (exclude subsequent FT).
            try:
                if _should_award_fastbreak_fg(ctx, ctx.get("first_fga_shotclock_sec")):
                    offense.fastbreak_pts += int(pts)
            except Exception as e:
                _record_exception("fastbreak_pts_award_fg", e)
            and_one = True
            if zone_detail:
                offense.shot_zone_detail[zone_detail]["FGM"] += 1

            # minimal assist treatment on rim fouls (jumper fouls remain unassisted)
            assisted_heur = False
            assister_pid = None
            if shot_key != "SHOT_3_OD":
                try:
                    assisted_heur = bool(ctx.get("pass_chain", pass_chain)) and float(
                        ctx.get("pass_chain", pass_chain)
                    ) > 0
                except Exception as e:
                    _record_exception("assist_flag_parse", e)
                    assisted_heur = False

            # Prefer true last passer within the assist window (based on shot_key).
            assister_pid = pick_assister_from_history(ctx, offense, actor.pid, game_state, shot_key)

            assisted = False
            if assister_pid is not None:
                assisted = True
            else:
                assisted = bool(assisted_heur)
                if assisted:
                    assister = choose_assister_weighted(rng, offense, actor.pid, base_action, shot_key, style=style)
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

        nfts = 1 if shot_made else (3 if pts == 3 else 2)
    else:
        # bonus free throws, no shot attempt
        nfts = 2

    ft_res = resolve_free_throws(rng, actor, nfts, offense, game_cfg=game_cfg)

    if fouler_pid and pf.get(fouler_pid, 0) >= foul_out_limit:
        game_state.fatigue[def_team_id][fouler_pid] = 0.0

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
    clear_pass_tracking(ctx)
    return "FOUL_FT", _with_matchup(payload)
