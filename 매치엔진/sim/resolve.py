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
    choose_assister_weighted,
    choose_creator_for_pulloff,
    choose_finisher_rim,
    choose_post_target,
    choose_passer,
    choose_shooter_for_mid,
    choose_shooter_for_three,
    choose_weighted_player,
    choose_default_actor,
    choose_fouler_pid,
    choose_stealer_pid,
    choose_blocker_pid,
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
    from .game_config import GameConfig

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


# -------------------------
# Assist attribution tracking (pass history)
# -------------------------
#
# We attribute assists to the *actual last successful passer* when possible.
# Pass events are staged in ctx["_pending_pass_event"] during resolve_outcome()
# and then committed in sim_possession.py *after* time cost has been applied,
# so the recorded shot-clock timestamp matches the true timing window.

_ASSIST_WINDOW_SEC = {
    "SHOT_3_CS": 2.00,
    "SHOT_MID_CS": 2.00,

    "SHOT_RIM_LAYUP": 3.10,
    "SHOT_RIM_DUNK": 3.10,
    "SHOT_RIM_CONTACT": 3.10,

    "SHOT_TOUCH_FLOATER": 2.40,
    "SHOT_POST": 2.70,

    "SHOT_3_OD": 0.95,
    "SHOT_MID_PU": 0.80,
}
_DEFAULT_ASSIST_WINDOW_SEC = 2.40
_PASS_HISTORY_MAXLEN = 3


def clear_pass_tracking(ctx: Dict[str, Any]) -> None:
    """Clear pass-history + staged pass event for the current possession sequence."""
    if not isinstance(ctx, dict):
        return
    ctx.pop("_pending_pass_event", None)
    ctx.pop("pass_history", None)
    ctx.pop("_pass_seq", None)


def commit_pending_pass_event(ctx: Dict[str, Any], game_state: Optional[GameState]) -> None:
    """Commit staged pass event into pass_history using *post-time-cost* clocks.

    Called by sim_possession.py immediately after apply_time_cost for a pass.
    """
    if not isinstance(ctx, dict):
        return
    ev = ctx.pop("_pending_pass_event", None)
    if not isinstance(ev, dict):
        return

    pid = str(ev.get("pid") or "")
    if not pid:
        return

    seq = int(ctx.get("_pass_seq", 0)) + 1
    ctx["_pass_seq"] = seq

    hist = ctx.get("pass_history")
    if not isinstance(hist, list):
        hist = []
        ctx["pass_history"] = hist

    sc = float(getattr(game_state, "shot_clock_sec", 0.0)) if game_state is not None else 0.0
    gc = float(getattr(game_state, "clock_sec", 0.0)) if game_state is not None else 0.0

    hist.append({
        "seq": seq,
        "pid": pid,
        "outcome": str(ev.get("outcome") or ""),
        "base_action": str(ev.get("base_action") or ""),
        "shot_clock_sec": sc,
        "game_clock_sec": gc,
    })

    if len(hist) > _PASS_HISTORY_MAXLEN:
        del hist[:-_PASS_HISTORY_MAXLEN]


def _assist_window_sec(shot_outcome: str) -> float:
    return float(_ASSIST_WINDOW_SEC.get(str(shot_outcome), _DEFAULT_ASSIST_WINDOW_SEC))


def pick_assister_from_history(
    ctx: Dict[str, Any],
    offense: TeamState,
    shooter_pid: str,
    game_state: Optional[GameState],
    shot_outcome: str,
) -> Optional[str]:
    """Pick assister from pass_history if the last pass is within the assist window."""
    if game_state is None:
        return None
    hist = ctx.get("pass_history")
    if not isinstance(hist, list) or not hist:
        return None

    win = _assist_window_sec(shot_outcome)
    shot_sc = float(getattr(game_state, "shot_clock_sec", 0.0))

    for ev in reversed(hist):
        if not isinstance(ev, dict):
            continue
        pid = str(ev.get("pid") or "")
        if not pid or pid == str(shooter_pid or ""):
            continue
        if not offense.is_on_court(pid):
            continue

        pass_sc = float(ev.get("shot_clock_sec", 0.0))
        dt = pass_sc - shot_sc
        if 0.0 <= dt <= win:
            return pid

    return None



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


def _should_award_fastbreak_fg(ctx: dict, first_fga_sc) -> bool:
    """Fastbreak points should be credited on the *scoring FG event*, not at possession end.

    Rules (v1):
    - Only possessions that *originated* from a live-ball transition (after DRB / after TOV).
    - Never credit during possession-continuation (dead-ball stop -> inbound -> set offense).
    - Only credit within the early clock window, using the first FGA shot-clock snapshot.
    - Free throws are intentionally excluded (handled elsewhere).
    """
    try:
        origin = str(ctx.get("_pos_origin_start") or ctx.get("pos_start") or "")
    except Exception:
        origin = ""
    if origin not in ("after_tov", "after_drb"):
        return False
    # Any dead-ball continuation segment implies defense is set -> not a fastbreak score.
    if bool(ctx.get("_pos_continuation", False)):
        return False
    # Additional guardrail: explicit dead-ball inbound segments should never count as fastbreak.
    if bool(ctx.get("dead_ball_inbound", False)):
        return False
    if first_fga_sc is None:
        return False
    try:
        return float(first_fga_sc) >= 16.0
    except Exception:
        return False


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
    # Strong SSOT contract: resolve layer never infers home/away and never accepts side-keyed ctx.
    game_id = str(ctx.get("game_id", "") or "").strip()

    # Prefer object SSOT (TeamState.team_id). ctx may redundantly include off_team_id/def_team_id for validation.
    off_team_id = str(getattr(offense, "team_id", "") or "").strip()
    def_team_id = str(getattr(defense, "team_id", "") or "").strip()
    if not off_team_id or not def_team_id:
        raise ValueError(
            f"resolve_outcome(): offense/defense must have non-empty team_id (game_id={game_id!r}, off={off_team_id!r}, def={def_team_id!r})"
        )
    if off_team_id == def_team_id:
        raise ValueError(
            f"resolve_outcome(): offense.team_id == defense.team_id == {off_team_id!r} (game_id={game_id!r})"
        )

    ctx_off = str(ctx.get("off_team_id", "") or "").strip()
    ctx_def = str(ctx.get("def_team_id", "") or "").strip()
    if ctx_off and ctx_off != off_team_id:
        raise ValueError(
            f"resolve_outcome(): ctx.off_team_id mismatch (game_id={game_id!r}, ctx={ctx_off!r}, offense.team_id={off_team_id!r})"
        )
    if ctx_def and ctx_def != def_team_id:
        raise ValueError(
            f"resolve_outcome(): ctx.def_team_id mismatch (game_id={game_id!r}, ctx={ctx_def!r}, defense.team_id={def_team_id!r})"
        )

    if outcome == "TO_SHOT_CLOCK":
        clear_pass_tracking(ctx)
        actor = _pick_default_actor(offense)
        offense.tov += 1
        offense.add_player_stat(actor.pid, "TOV", 1)
        return "TURNOVER", {"outcome": outcome, "pid": actor.pid}

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
        clear_pass_tracking(ctx)
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

            return "SCORE", {
                "outcome": outcome,
                "pid": actor.pid,
                "points": pts,
                "shot_zone_detail": zone_detail,
                "assisted": assisted,
                "assister_pid": assister_pid,
                **shot_dbg,
            }
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
            return "MISS", payload


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
        t_to = float(ctx.get("pass_q_to", -0.9))
        t_reset = float(ctx.get("pass_q_reset", -0.3))
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
                steal_logit_delta = (-float(q_score)) * 0.40 + (d_press - 0.5) * 1.10

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

            return "TURNOVER", payload


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
            clear_pass_tracking(ctx)
                
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
                
            ctx["_pending_pass_event"] = {"pid": actor.pid, "outcome": outcome, "base_action": base_action}
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
        clear_pass_tracking(ctx)
        
        return "RESET", payload

    if is_to(outcome):
        clear_pass_tracking(ctx)
        offense.tov += 1
        offense.add_player_stat(actor.pid, "TOV", 1)
        
        payload: Dict[str, Any] = {"outcome": outcome, "pid": actor.pid}

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
                    steal_logit_delta = (-float(q_score)) * 0.40 + (d_press - 0.5) * 1.10
                    lineout_base = float(pm.get("bad_pass_lineout_base", 0.30))
                    p_lineout = clamp(lineout_base + max(0.0, -float(q_score)) * 0.06, 0.05, 0.55)
                else:  # TO_HANDLE_LOSS
                    base_steal = float(pm.get("steal_handle_loss_base", 0.72))
                    steal_logit_delta = (-float(q_score)) * 0.35 + (d_press - 0.5) * 1.00
                    p_lineout = clamp(0.10 + max(0.0, -float(q_score)) * 0.04, 0.02, 0.25)

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

        return "TURNOVER", payload

    if is_foul(outcome):
        # Fouls mutate team-scoped state (team_fouls / player_fouls / fatigue). game_state is required.
        if game_state is None:
            raise ValueError(
                f"resolve_outcome(): game_state is required for foul outcomes (game_id={game_id!r}, outcome={outcome!r})"
            )

        foul_out_limit = int(ctx.get("foul_out", 6))
        bonus_threshold = int(ctx.get("bonus_threshold", 5))
        # On-court defenders snapshot (prefer explicit ctx snapshot from sim_game; fallback to TeamState SSOT).
        def_on_court = ctx.get("def_on_court")
        if not isinstance(def_on_court, list) or not def_on_court:
            def_on_court = [p.pid for p in defense.on_court_players()]

        # Strict team-id keyed containers (must already be initialized by sim_game).
        if def_team_id not in game_state.player_fouls:
            raise ValueError(
                f"resolve_outcome(): GameState.player_fouls missing defense team_id (game_id={game_id!r}, def_team_id={def_team_id!r})"
            )
        if def_team_id not in game_state.team_fouls:
            raise ValueError(
                f"resolve_outcome(): GameState.team_fouls missing defense team_id (game_id={game_id!r}, def_team_id={def_team_id!r})"
            )
        if def_team_id not in game_state.fatigue:
            raise ValueError(
                f"resolve_outcome(): GameState.fatigue missing defense team_id (game_id={game_id!r}, def_team_id={def_team_id!r})"
            )

        pf = game_state.player_fouls[def_team_id]
        team_fouls = game_state.team_fouls
        fatigue = game_state.fatigue[def_team_id]

        fouler_pid = None

        # Assign a fouler from on-court defenders.
        if def_on_court:
            fouler_pid = choose_fouler_pid(rng, defense, list(def_on_court), pf, foul_out_limit, outcome)
            if fouler_pid:
                pf[fouler_pid] = pf.get(fouler_pid, 0) + 1
        # Update team fouls (defense committed the foul).
        team_fouls[def_team_id] = int(team_fouls[def_team_id]) + 1
        # Non-shooting foul becomes dead-ball unless in bonus.

        # Non-shooting foul (reach/trap) becomes dead-ball unless in bonus.
        if outcome == "FOUL_REACH_TRAP" and not in_bonus:
            if fouler_pid and pf.get(fouler_pid, 0) >= foul_out_limit:
                # Foul-out sentinel: clamp energy to 0.0 so rotation can force substitution.
                fatigue[fouler_pid] = 0.0
            clear_pass_tracking(ctx)
            return "FOUL_NO_SHOTS", {
                "outcome": outcome,
                "pid": actor.pid,
                "fouler": fouler_pid,
                "bonus": False,
            }

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
                _record_exception("foul_draw_quality", e)
                q_score = 0.0
            # Determine make-prob for the "would-be" shot (and-1 logic).
            try:
                base_make = float(pm.get("foul_draw_make_base", 0.46))
                make_var = _team_variance_mult(offense, game_cfg) * float(ctx.get("variance_mult", 1.0))
                p_make = prob_from_scores(
                    rng,
                    base_make,
                    off_score,
                    def_score,
                    kind="foul_draw_make",
                    variance_mult=make_var,
                    logit_delta=float(q_score) * 0.35,
                    game_cfg=game_cfg,
                )
                shot_made = bool(rng.random() < p_make)
                if debug_q:
                    foul_dbg.update({"q_score": float(q_score), "q_detail": q_detail, "p_make": float(p_make)})
            except Exception as e:
                _record_exception("foul_draw_make_model", e)
                shot_made = False

            # If shot made: credit FGA/FGM + points + potential assist.
            if shot_made:
                offense.fga += 1

                offense.fgm += 1
                offense.pts += pts

                actor.add_player_stat("FGA", 1)
                actor.add_player_stat("FGM", 1)
                actor.add_player_stat("PTS", pts)

                and_one = True

                zone_detail = None
                try:
                    zone_detail = shot_zone_detail(shot_key)
                    if zone_detail:
                        offense.shot_zone_detail.setdefault(zone_detail, {"FGA": 0, "FGM": 0, "AST_FGM": 0})
                        offense.shot_zone_detail[zone_detail]["FGA"] += 1
                        offense.shot_zone_detail[zone_detail]["FGM"] += 1
                except Exception as e:
                    _record_exception("foul_draw_zone_detail", e)

                # Prefer true last passer within the assist window for rim/post fouls.
                assister_pid = pick_assister_from_history(ctx, offense, actor.pid, game_state, shot_key)
                if assister_pid is None and shot_key != "SHOT_3_OD":
                    try:
                        assisted_heur = bool(ctx.get("pass_chain", pass_chain)) and float(ctx.get("pass_chain", pass_chain)) > 0
                    except Exception as e:
                        _record_exception("assist_flag_parse", e)
                        assisted_heur = False

                if assisted_heur:
                        assister = choose_assister_weighted(rng, offense, actor.pid, base_action, shot_key, style=style)
                        assister_pid = assister.pid if assister else None

                if assister_pid is not None:
                    offense.ast += 1
                    offense.add_player_stat(assister_pid, "AST", 1)
                    try:
                        if zone_detail:
                            offense.shot_zone_detail[zone_detail]["AST_FGM"] += 1
                    except Exception:
                        pass

                if zone_detail in ("Restricted_Area", "Paint_Non_RA"):
                    offense.pitp += 2

            nfts = 1 if shot_made else (3 if pts == 3 else 2)
        else:
            # bonus free throws, no shot attempt
            nfts = 2

        ft_res = resolve_free_throws(rng, actor, nfts, offense, game_cfg=game_cfg)

        # Foul-out sentinel after the trip.  
        if fouler_pid and pf.get(fouler_pid, 0) >= foul_out_limit:
            fatigue[fouler_pid] = 0.0

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
        if isinstance(ft_res, dict):
            payload.update(ft_res)
        if isinstance(foul_dbg, dict) and foul_dbg:
            payload.update(foul_dbg)
        clear_pass_tracking(ctx)
        return "FOUL_FT", payload


    if is_reset(outcome):
        clear_pass_tracking(ctx)
        return "RESET", {"outcome": outcome}

    clear_pass_tracking(ctx)
    return "RESET", {"outcome": outcome}
