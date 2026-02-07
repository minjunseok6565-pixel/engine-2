from __future__ import annotations

from typing import Any, Dict, Tuple

from ..models import Player
from ..participants import choose_default_actor, choose_stealer_pid
from ..def_role_players import get_or_build_def_role_players, engine_get_stat
from ..prob import _team_variance_mult, prob_from_scores
from ..core import clamp
from .. import quality, matchups
from .resolve_context import ResolveContext
from .resolve_pass_tracking import clear_pass_tracking

def _pick_default_actor(offense) -> Player:
    """Local copy (keeps resolve.py slimmer)."""
    return choose_default_actor(offense)

def handle_shot_clock_turnover(
    rng,
    outcome: str,
    action: str,
    offense,
    defense,
    tags: Dict[str, Any],
    pass_chain: int,
    ctx: Dict[str, Any],
    game_state,
    game_cfg,
) -> Tuple[str, Dict[str, Any]]:
    # This block is copied verbatim from resolve.py's TO_SHOT_CLOCK special-case.
    clear_pass_tracking(ctx)
    actor = _pick_default_actor(offense)
    offense.tov += 1
    offense.add_player_stat(actor.pid, "TOV", 1)
    # Matchup info (best-effort; also consume any one-shot matchup_force)
    defender_pid, matchup_source, matchup_event = matchups.get_primary_defender_pid(
        actor.pid, defense, ctx, off_player=actor
    )
    try:
        force = ctx.get("matchup_force")
        if isinstance(force, dict):
            f_off = str(force.get("off_pid") or "").strip()
            f_def = str(force.get("def_pid") or "").strip()
            if f_off and f_def and f_off == actor.pid and defense.is_on_court(f_def):
                ttl = int(force.get("ttl", 1) or 1) - 1
                if ttl <= 0:
                    ctx.pop("matchup_force", None)
                else:
                    force["ttl"] = ttl
    except Exception:
        ctx.pop("matchup_force", None)

    payload_sc = {
        "outcome": outcome,
        "pid": actor.pid,
        "defender_pid": defender_pid,
        "matchup_source": matchup_source,
        "matchup_event": matchup_event,
        "matchups_version": int(ctx.get("matchups_version", 0) or 0),
    }
    try:
        ctx["matchup_last"] = {
            "off_pid": actor.pid,
            "def_pid": defender_pid,
            "source": matchup_source,
            "event": matchup_event,
            "version": int(ctx.get("matchups_version", 0) or 0),
        }
    except Exception:
        pass
    return "TURNOVER", payload_sc

def handle_turnover(
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

    clear_pass_tracking(ctx)
    offense.tov += 1
    offense.add_player_stat(actor.pid, "TOV", 1)

    payload: Dict[str, Any] = {"outcome": outcome, "pid": actor.pid}

    if outcome == "TO_CHARGE":
        # Offensive foul (charge): count as a turnover AND an offensive personal/team foul.
        team_fouls = game_state.team_fouls
        player_fouls_by_team = game_state.player_fouls
        if off_team_id not in team_fouls:
            raise ValueError(
                "resolve_outcome(): missing team_fouls bucket for off_team_id "
                f"(game_id={game_id!r}, off_team_id={off_team_id!r}, keys={list(team_fouls.keys())!r})"
            )
        if off_team_id not in player_fouls_by_team:
            raise ValueError(
                "resolve_outcome(): missing player_fouls bucket for off_team_id "
                f"(game_id={game_id!r}, off_team_id={off_team_id!r}, keys={list(player_fouls_by_team.keys())!r})"
            )
        if off_team_id not in game_state.fatigue:
            raise ValueError(
                "resolve_outcome(): missing fatigue bucket for off_team_id "
                f"(game_id={game_id!r}, off_team_id={off_team_id!r}, keys={list(game_state.fatigue.keys())!r})"
            )

        pf = player_fouls_by_team[off_team_id]
        foul_out_limit = int(ctx.get("foul_out", 6))
        pf[actor.pid] = pf.get(actor.pid, 0) + 1
        team_fouls[off_team_id] = int(team_fouls[off_team_id]) + 1

        if pf.get(actor.pid, 0) >= foul_out_limit:
            game_state.fatigue[off_team_id][actor.pid] = 0.0

        # Hint flags for replay/UI (sim_possession may log this as an offensive foul whistle).
        payload.update(
            {
                "offensive_foul": True,
                "foul_type": "CHARGE",
                "fouler": actor.pid,
                "team_foul": True,
                "player_fouls": int(pf.get(actor.pid, 0)),
                "team_fouls": int(team_fouls[off_team_id]),
                "deadball_override": True,
                "tov_deadball_reason": "CHARGE",
            }
        )


    # For select live-ball turnovers, split into (steal vs non-steal) so defensive playmakers
    # are credited and downstream possession context can reflect stronger transition starts.
    if outcome in ("TO_BAD_PASS", "TO_HANDLE_LOSS"):
        scheme = getattr(defense.tactics, "defense_scheme", "")
        debug_q = bool(ctx.get("debug_quality", False))
        role_players = get_or_build_def_role_players(
            ctx,
            defense,
            scheme=scheme,
            debug_detail_key=("def_role_players_detail" if debug_q else None),
        )

        q_detail = None
        q_score = 0.0
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
            _record_exception("quality_compute_to", e)
            q_score = 0.0

        def_feat = getattr(style, "def_features", {}) if style is not None else {}
        d_press = float(def_feat.get("D_STEAL_PRESS", 0.5))

        p_steal = 0.0
        p_lineout = 0.0
        steal_logit_delta = 0.0
        try:
            if outcome == "TO_BAD_PASS":
                base_steal = float(pm.get("steal_bad_pass_base", 0.60))
                steal_logit_delta = (-float(q_score)) * 0.40 + (d_press - 0.5) * 1.10 + (0.18 * float(help_level)) + (0.22 * float(double_strength))
                lineout_base = float(pm.get("bad_pass_lineout_base", 0.30))
                p_lineout = clamp(lineout_base + max(0.0, -float(q_score)) * 0.06, 0.05, 0.55)
            else:  # TO_HANDLE_LOSS
                base_steal = float(pm.get("steal_handle_loss_base", 0.72))
                steal_logit_delta = (-float(q_score)) * 0.35 + (d_press - 0.5) * 1.00 + (0.18 * float(help_level)) + (0.22 * float(double_strength))
                p_lineout = clamp(0.06 + max(0.0, -float(q_score)) * 0.04, 0.02, 0.25)

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
                if rng.random() < p_lineout:
                    payload.update(
                        {
                            "deadball_override": True,
                            "tov_deadball_reason": ("LINEOUT_BAD_PASS" if outcome == "TO_BAD_PASS" else "LINEOUT_LOOSE"),
                        }
                    )

            if debug_q:
                payload.update(
                    {
                        "q_score": float(q_score),
                        "q_detail": q_detail,
                        "p_steal": float(p_steal),
                        "p_lineout": float(p_lineout),
                        "steal_logit_delta": float(steal_logit_delta),
                    }
                )
        except Exception as e:
            _record_exception("steal_split_to", e)

    return "TURNOVER", _with_matchup(payload)
