from __future__ import annotations

import random
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from ..builders import get_action_base
from ..core import clamp, dot_profile
from ..defense import team_def_snapshot
from ..era import DEFAULT_PROB_MODEL
from ..models import GameState, Player, TeamState
from ..profiles import OUTCOME_PROFILES
from ..prob import _team_variance_mult
from ..def_role_players import engine_get_stat

from ..participants import (
    choose_assister_weighted,  # not used here but kept for parity if needed later
    choose_creator_for_pulloff,
    choose_finisher_rim,
    choose_post_target,
    choose_passer,
    choose_shooter_for_mid,
    choose_shooter_for_three,
    choose_weighted_player,
    choose_default_actor,
)
from .. import matchups
from .resolve_pass_tracking import clear_pass_tracking

if TYPE_CHECKING:
    from ..game_config import GameConfig

def is_shot(o: str) -> bool: return str(o).startswith("SHOT_")
def is_pass(o: str) -> bool: return str(o).startswith("PASS_")
def is_to(o: str) -> bool: return str(o).startswith("TO_")
def is_foul(o: str) -> bool: return str(o).startswith("FOUL_")
def is_reset(o: str) -> bool: return str(o).startswith("RESET_")

def _pick_default_actor(offense: TeamState) -> Player:
    """12-role first, then best passer. Used when an outcome has no specific participant chooser."""
    return choose_default_actor(offense)

def _knob_mult(game_cfg: "GameConfig", key: str, default: float = 1.0) -> float:
    knobs = game_cfg.knobs if isinstance(game_cfg.knobs, Mapping) else {}
    try:
        return float(knobs.get(key, default))
    except Exception:
        return float(default)

@dataclass(frozen=True)
class ResolveContext:
    rng: random.Random
    outcome: str
    action: str
    offense: TeamState
    defense: TeamState
    tags: Dict[str, Any]
    pass_chain: int
    ctx: Dict[str, Any]
    game_state: GameState
    game_cfg: "GameConfig"

    game_id: Any
    off_team_id: str
    def_team_id: str

    pm: Mapping[str, Any]
    style: Any
    base_action: str
    def_snap: Dict[str, Any]
    prof: Dict[str, Any]

    actor: Player
    defender_pid: Optional[str]
    matchup_source: str
    matchup_event: Optional[str]

    help_level: float
    double_strength: float
    double_doubler_pid: Optional[str]
    double_source: Optional[str]
    double_label: Optional[str]

    variance_mult: float
    off_score: float
    def_score: float
    fatigue_logit_delta: float
    carry_in: float

def build_resolve_context(
    rng: random.Random,
    outcome: str,
    action: str,
    offense: TeamState,
    defense: TeamState,
    tags: Dict[str, Any],
    pass_chain: int,
    ctx: Dict[str, Any],
    game_state: GameState,
    game_cfg: "GameConfig",
    *,
    game_id: Any,
    off_team_id: str,
    def_team_id: str,
    _record_exception,
) -> Tuple[Optional[ResolveContext], Optional[Tuple[str, Dict[str, Any]]]]:
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
        return None, ("RESET", {
            "outcome": outcome,
            "defender_pid": None,
            "matchup_source": "fallback",
            "matchup_event": None,
            "matchups_version": int(ctx.get("matchups_version", 0) or 0),
        })

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

    # Optional forced actor (e.g., ORB -> immediate Putback):
    # If provided, override the chosen shooter/foul-draw/turnover actor once and then consume.
    force_pid = ctx.get("force_actor_pid")
    if force_pid and (is_shot(outcome) or is_foul(outcome) or is_to(outcome)):
        try:
            fp = str(force_pid)
            if fp and offense.is_on_court(fp):
                fp_player = offense.find_player(fp)
                if fp_player is not None:
                    actor = fp_player
        except Exception as e:
            _record_exception("force_actor_pid", e)
        finally:
            # consume (one-shot) to avoid leaking into subsequent steps
            ctx.pop("force_actor_pid", None)

    # --- Matchups (Plan-1 MVP) ---
    defender_pid: Optional[str] = None
    matchup_source: str = "fallback"
    matchup_event: Optional[str] = None

    # Apply and consume one-shot forced matchup (ttl) if it targets this actor.
    # IMPORTANT: ttl is only consumed when the forced matchup is *actually applied* on a terminal on-ball event.
    try:
        force = ctx.get("matchup_force")
        force_def_pid: Optional[str] = None
        force_event: Optional[str] = None
        if isinstance(force, dict):
            f_off = str(force.get("off_pid") or "").strip()
            f_def = str(force.get("def_pid") or "").strip()
            if f_off and f_def and f_off == actor.pid and defense.is_on_court(f_def):
                force_def_pid = f_def
                force_event = str(force.get("event") or "") or None

                is_terminal = bool(is_shot(outcome) or is_to(outcome) or (is_foul(outcome) and outcome.startswith("FOUL_DRAW_")))
                if is_terminal:
                    try:
                        ttl = int(force.get("ttl", 1) or 1) - 1
                        if ttl <= 0:
                            ctx.pop("matchup_force", None)
                        else:
                            force["ttl"] = ttl
                    except Exception:
                        ctx.pop("matchup_force", None)

        if force_def_pid:
            defender_pid = force_def_pid
            matchup_source = "force"
            matchup_event = force_event
        else:
            defender_pid, matchup_source, matchup_event = matchups.get_primary_defender_pid(
                actor.pid, defense, ctx, off_player=actor
            )
    except Exception as e:
        _record_exception("matchup_primary_defender", e)
        defender_pid, matchup_source, matchup_event = None, "fallback", None

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

    # Tactical context (read-only defaults): defensive help + double/trap.
    # Help SSOT: ctx['def_pressure']['help']['eff_resolve'] (scheme baseline + team delta)
    # Additional adjustment: helper skill increases help impact; leaving a good shooter reduces how much help can be applied safely.
    base_help = 0.0
    helper_pid: Optional[str] = None
    leave_cost_norm = 0.0
    try:
        dp = ctx.get("def_pressure") if isinstance(ctx.get("def_pressure"), dict) else {}
        hp = dp.get("help") if isinstance(dp.get("help"), dict) else {}
        base_help = float(hp.get("eff_resolve", 0.0) or 0.0)
        helper_pid = str(hp.get("helper_pid") or "").strip() or None
        leave_cost_norm = float(hp.get("leave_cost_norm", 0.0) or 0.0)
    except Exception:
        base_help = 0.0
        helper_pid = None
        leave_cost_norm = 0.0

    base_help = clamp(base_help, -1.0, 1.0)
    leave_cost_norm = clamp(leave_cost_norm, -1.0, 1.0)

    helper_skill_norm = 0.0
    if helper_pid and defense.is_on_court(helper_pid):
        try:
            hp_player = defense.find_player(helper_pid)
        except Exception:
            hp_player = None
        try:
            h_help = float(engine_get_stat(hp_player, "DEF_HELP", 50.0)) if hp_player is not None else 50.0
            helper_skill_norm = float(clamp((h_help - 50.0) / 50.0, -1.0, 1.0))
        except Exception:
            helper_skill_norm = 0.0

    help_level = float(clamp(base_help + (0.20 * helper_skill_norm) - (0.25 * leave_cost_norm), -1.0, 1.0))

    double_strength: float = 0.0
    double_doubler_pid: Optional[str] = None
    double_source: Optional[str] = None
    double_label: Optional[str] = None


    variance_mult = _team_variance_mult(offense, game_cfg) * float(ctx.get("variance_mult", 1.0))

    # compute scores
    off_vals = {k: actor.get(k) for k in prof["offense"].keys()}
    off_score = dot_profile(off_vals, prof["offense"])
    # Blend team defense snapshot with the primary defender stats (per-key weights).
    blend = ctx.get("matchup_def_blend", {}) or {}
    if not isinstance(blend, Mapping):
        blend = {}

    defender_player = None
    try:
        if defender_pid:
            defender_player = defense.find_player(defender_pid)
            if defender_player is not None and not defense.is_on_court(defender_pid):
                defender_player = None
    except Exception as e:
        _record_exception("matchup_defender_lookup", e)
        defender_player = None

    def_vals = {}
    for k in prof["defense"].keys():
        t_val = float(def_snap.get(k, 50.0))
        d_val = float(engine_get_stat(defender_player, k, 50.0)) if defender_player is not None else 50.0
        try:
            w = float(blend.get(k, 0.5))
        except Exception:
            w = 0.5
        w = clamp(w, 0.0, 1.0)
        def_vals[k] = (w * d_val) + ((1.0 - w) * t_val)
    def_score = dot_profile(def_vals, prof["defense"])

    fatigue_map = ctx.get("fatigue_map", {}) or {}
    fatigue_logit_max = float(ctx.get("fatigue_logit_max", -0.25))

    # Prefer actor.energy when available; fallback to ctx fatigue_map for back-compat.
    try:
        fatigue_val = float(getattr(actor, "energy", fatigue_map.get(actor.pid, 1.0)))
    except Exception:
        fatigue_val = float(fatigue_map.get(actor.pid, 1.0))
    fatigue_val = clamp(fatigue_val, 0.0, 1.0)

    # Base linear penalty (existing behavior)
    fatigue_logit_delta = (1.0 - fatigue_val) * fatigue_logit_max

    # Red-zone extra penalty (new): only applies when energy < crit.
    crit = float(ctx.get("fatigue_logit_red_crit", 0.0) or 0.0)
    red_max = float(ctx.get("fatigue_logit_red_max", 0.0) or 0.0)
    red_pow = float(ctx.get("fatigue_logit_red_pow", 1.0) or 1.0)

    if crit > 1e-9 and red_max != 0.0 and fatigue_val < crit:
        t = (crit - fatigue_val) / crit  # 0..1
        fatigue_logit_delta += (t ** red_pow) * red_max

    # PASS-carry: applied once to the *next* shot/pass (and optionally shooting-foul) and then consumed.
    carry_in = 0.0
    if is_shot(outcome) or is_pass(outcome) or (is_foul(outcome) and outcome.startswith("FOUL_DRAW_")):
        try:
            carry_in = float(ctx.pop("carry_logit_delta", 0.0) or 0.0)
        except Exception as e:
            _record_exception("carry_logit_delta_pop", e)
            carry_in = 0.0


    # --- Double / trap resolution (possession-scoped) ---
    # We keep this lightweight: apply small, clear tradeoffs without changing the flow-control shape
    # (i.e., we do not convert a SHOT outcome into a PASS outcome here).
    def _best_threat_pid() -> Optional[str]:
        best_pid = None
        best_val = float("-inf")
        try:
            for p in offense.on_court_players():
                pid = str(getattr(p, "pid", "") or "").strip()
                if not pid:
                    continue
                try:
                    val = float(matchups._threat_score(p))
                except Exception:
                    val = float(engine_get_stat(p, "PASS_CREATE", 50.0))
                if val > best_val + 1e-9:
                    best_val = val
                    best_pid = pid
        except Exception:
            return None
        return best_pid

    def _choose_def_by_tag(tag: str, exclude: set[str]) -> Optional[str]:
        t = str(tag or "").strip().upper()
        best_pid = None
        best_val = float("-inf")
        for p in defense.on_court_players():
            pid = str(getattr(p, "pid", "") or "").strip()
            if not pid or pid in exclude:
                continue
            if t == "BEST_HELP":
                val = float(engine_get_stat(p, "DEF_HELP", 50.0))
            elif t == "BEST_POA":
                val = float(engine_get_stat(p, "DEF_POA", 50.0))
            elif t == "BEST_POST":
                val = float(engine_get_stat(p, "DEF_POST", 50.0))
            else:  # BEST_STEAL (default)
                val = float(engine_get_stat(p, "DEF_STEAL", 50.0))
            if val > best_val + 1e-9:
                best_val = val
                best_pid = pid
        return best_pid

    def _resolve_doubler_pid(spec: Any, exclude: set[str]) -> Optional[str]:
        # spec may be dict ({def_pid, tag}) or a direct pid string.
        if isinstance(spec, Mapping):
            dpid = str(spec.get("def_pid") or "").strip()
            if dpid and defense.is_on_court(dpid) and dpid not in exclude:
                return dpid
            tag = str(spec.get("tag") or "").strip().upper()
            if tag:
                return _choose_def_by_tag(tag, exclude)
            return None
        s = str(spec or "").strip()
        if s and defense.is_on_court(s) and s not in exclude:
            return s
        return None

    def _resolve_double_from_ctx(consume: bool) -> Optional[Tuple[float, Optional[str], Optional[str], Optional[str]]]:
        spec = ctx.get("double_active")
        if not isinstance(spec, dict):
            return None
        off_pid = str(spec.get("off_pid") or "").strip()
        try:
            ttl = int(spec.get("ttl", 1) or 1)
        except Exception:
            ttl = 1
        if ttl <= 0:
            # Clean up invalid spec to avoid lingering states.
            try:
                ctx.pop("double_active", None)
            except Exception:
                pass
            return None
        # If the target is not on the floor anymore, drop the spec.
        if not off_pid or (not offense.is_on_court(off_pid)):
            try:
                ctx.pop("double_active", None)
            except Exception:
                pass
            return None

        # IMPORTANT (ghost-double fix):
        # Even if the planned double does not apply to the current outcome actor, a "step" has passed.
        # Consume TTL so a pre-prior double plan doesn't keep biasing priors across multiple steps.
        if consume:
            try:
                ttl2 = ttl - 1
                if ttl2 <= 0:
                    ctx.pop("double_active", None)
                else:
                    spec["ttl"] = ttl2
            except Exception:
                try:
                    ctx.pop("double_active", None)
                except Exception:
                    pass

        # Apply the double only when it targets this outcome's actor (on-ball assumption).
        if off_pid != actor.pid:
            return None

        try:
            strength = float(spec.get("strength", 0.0) or 0.0)
        except Exception:
            strength = 0.0
        strength = clamp(strength, 0.0, 1.0)

        primary_mismatch = False
        exclude = set([str(defender_pid or "").strip()]) if defender_pid else set()
        doubler = str(spec.get("doubler_pid") or "").strip()
        if doubler and (not defense.is_on_court(doubler) or doubler in exclude):
            doubler = ""
        if not doubler:
            doubler = _resolve_doubler_pid(spec.get("doubler"), exclude) or ""
        if not doubler:
            tag = str(spec.get("doubler_tag") or "").strip().upper()
            if tag:
                doubler = _choose_def_by_tag(tag, exclude) or ""
        doubler_pid = doubler if doubler else None

        # Validate that the double spec still corresponds to the current primary defender.
        # If not, dampen the double (screens/switches invalidate the initial plan).
        try:
            spec_primary = str(spec.get("primary_def_pid") or "").strip()
            cur_primary = str(defender_pid or "").strip()
            if spec_primary and cur_primary and spec_primary != cur_primary:
                strength = float(clamp(strength * 0.5, 0.0, 1.0))
                primary_mismatch = True
        except Exception:
            primary_mismatch = False

        source = str(spec.get("source") or "CTX") or "CTX"
        if primary_mismatch:
            source = f"{source}_PRIMARY_MISMATCH"
        label = str(spec.get("label") or "").strip() or None


        if strength <= 1e-9:
            return None
        return (float(strength), doubler_pid, source, label)

    def _resolve_double_from_rules() -> Optional[Tuple[float, Optional[str], Optional[str], Optional[str]]]:
        dctx = getattr(getattr(defense, "tactics", None), "context", None)
        if not isinstance(dctx, dict):
            return None
        rules = dctx.get("DOUBLE_RULES")
        if not isinstance(rules, list) or not rules:
            return None

        exclude = set([str(defender_pid or "").strip()]) if defender_pid else set()
        best_pid = None

        for rule in rules:
            if not isinstance(rule, dict):
                continue

            # Base-action gating.
            wba = rule.get("when_base_actions")
            if isinstance(wba, list) and wba:
                if base_action not in set(str(x) for x in wba if str(x)):
                    continue

            target = rule.get("target")
            if not isinstance(target, Mapping):
                target = rule

            off_pid = str(target.get("off_pid") or "").strip()
            off_role = str(target.get("off_role") or "").strip()
            tag = str(target.get("tag") or "").strip().upper()

            matched = False
            if off_pid and off_pid == actor.pid:
                matched = True
            elif off_role:
                try:
                    roles = getattr(offense, "roles", None)
                    if isinstance(roles, Mapping):
                        rp = str(roles.get(off_role) or "").strip()
                        if rp and rp == actor.pid:
                            matched = True
                except Exception:
                    matched = False
            elif tag == "BEST_THREAT":
                if best_pid is None:
                    best_pid = _best_threat_pid()
                if best_pid and best_pid == actor.pid:
                    matched = True

            if not matched:
                continue

            try:
                strength = float(rule.get("strength", 0.65) or 0.65)
            except Exception:
                strength = 0.65
            strength = clamp(strength, 0.0, 1.0)
            if strength <= 1e-9:
                continue

            doubler_spec = rule.get("doubler")
            if doubler_spec is None:
                doubler_spec = rule.get("doubler_pid")
            if doubler_spec is None:
                tag2 = str(rule.get("doubler_tag") or "").strip().upper()
                if tag2:
                    doubler_spec = {"tag": tag2}

            doubler_pid = _resolve_doubler_pid(doubler_spec, exclude)
            label = str(rule.get("label") or "").strip() or None

            return (float(strength), doubler_pid, "RULE", label)

        return None

    # Resolve a double spec if applicable to this (actor, outcome).
    double_can_apply = bool(is_shot(outcome) or is_pass(outcome) or is_to(outcome) or (is_foul(outcome) and outcome.startswith("FOUL_DRAW_")))
    if double_can_apply:
        try:
            resolved = _resolve_double_from_ctx(consume=True)
            if resolved is None:
                # If priors already evaluated a double plan this step (def_pressure flag),
                # do NOT re-evaluate DOUBLE_RULES here only when a double was actually planned/active
                # (keeps priors/resolve consistent). If no planned double was active, allow rules
                # to fire at resolve-time (restores legacy behavior).
                plan_eval = False
                planned_active = False
                try:
                    dp = ctx.get("def_pressure") if isinstance(ctx.get("def_pressure"), dict) else {}
                    plan_eval = (dp.get("double_plan_evaluated") is True)
                    dbl = dp.get("double") if isinstance(dp.get("double"), dict) else {}
                    planned_active = bool(dbl.get("active", False))
                except Exception:
                    plan_eval = False
                    planned_active = False

                # Legacy/edge safety: if def_pressure is missing but a ctx double spec exists, treat it as planned.
                if not planned_active:
                    try:
                        spec2 = ctx.get("double_active")
                        if isinstance(spec2, dict):
                            op2 = str(spec2.get("off_pid") or "").strip()
                            try:
                                ttl2 = int(spec2.get("ttl", 0) or 0)
                            except Exception:
                                ttl2 = 0
                            if ttl2 > 0 and op2 and offense.is_on_court(op2):
                                planned_active = True
                    except Exception:
                        planned_active = planned_active

                # Skip rules ONLY when priors had an active planned double.
                if not (plan_eval and planned_active):
                    resolved = _resolve_double_from_rules()
            if resolved is not None:
                ds, dp, src, lbl = resolved
                double_strength = float(clamp(ds, 0.0, 1.0))
                double_doubler_pid = dp
                double_source = src
                double_label = lbl
        except Exception as e:
            _record_exception("double_resolve", e)
            double_strength = 0.0
            double_doubler_pid = None
            double_source = None
            double_label = None

    # resolve by type

    # Return computed context (no behavior changes; just structured output).
    rc = ResolveContext(
        rng=rng,
        outcome=outcome,
        action=action,
        offense=offense,
        defense=defense,
        tags=tags,
        pass_chain=pass_chain,
        ctx=ctx,
        game_state=game_state,
        game_cfg=game_cfg,
        game_id=game_id,
        off_team_id=str(off_team_id),
        def_team_id=str(def_team_id),
        pm=pm,
        style=style,
        base_action=base_action,
        def_snap=def_snap,
        prof=prof,
        actor=actor,
        defender_pid=defender_pid,
        matchup_source=matchup_source,
        matchup_event=matchup_event,
        help_level=float(help_level),
        double_strength=float(double_strength),
        double_doubler_pid=double_doubler_pid,
        double_source=double_source,
        double_label=double_label,
        variance_mult=float(variance_mult),
        off_score=float(off_score),
        def_score=float(def_score),
        fatigue_logit_delta=float(fatigue_logit_delta),
        carry_in=float(carry_in),
    )
    return rc, None
