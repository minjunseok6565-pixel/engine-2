from __future__ import annotations

from . import shot_diet

from collections.abc import Mapping
from typing import Any, Dict, Optional, TYPE_CHECKING

from .core import apply_min_floor, apply_multipliers, apply_temperature, clamp, normalize_weights
from .era import get_defense_meta_params
from .tactics import TacticsConfig

if TYPE_CHECKING:
    from config.game_config import GameConfig


# -------------------------
# Builders
# -------------------------

def _fallback_scheme(weights: Mapping[str, Any], fallback: str) -> Dict[str, float]:
    if fallback in weights and isinstance(weights.get(fallback), Mapping):
        return dict(weights[fallback])
    for val in weights.values():
        if isinstance(val, Mapping):
            return dict(val)
    return {}


def get_action_base(action: str, game_cfg: "GameConfig") -> str:
    aliases = game_cfg.action_aliases if isinstance(game_cfg.action_aliases, Mapping) else {}
    return aliases.get(action, action)

def build_offense_action_probs(
    off_tac: TacticsConfig,
    def_tac: Optional[TacticsConfig] = None,
    ctx: Optional[Dict[str, Any]] = None,
    game_cfg: Optional["GameConfig"] = None,
) -> Dict[str, float]:
    """Build offense action distribution.

    UI rule (fixed): normalize((W_scheme[action] ^ sharpness) * off_action_mult[action] * def_opp_action_mult[action]).
    """
    if game_cfg is None:
        raise ValueError("build_offense_action_probs requires game_cfg")
    scheme_weights = game_cfg.off_scheme_action_weights if isinstance(game_cfg.off_scheme_action_weights, Mapping) else {}
    base = dict(
        scheme_weights.get(
            off_tac.offense_scheme,
            _fallback_scheme(scheme_weights, "Spread_HeavyPnR"),
        )
    )
    sharp = clamp(off_tac.scheme_weight_sharpness, 0.70, 1.40)
    # 1) scheme sharpening first
    base = {a: (max(w, 0.0) ** sharp) for a, w in base.items()}
    # 2) offense UI multipliers
    for a, m in off_tac.action_weight_mult.items():
        base[a] = base.get(a, 0.5) * float(m)
    # 3) defense can distort opponent action choice (e.g., transition defense priority)
    if def_tac is not None:
        for a, m in getattr(def_tac, 'opp_action_weight_mult', {}).items():
            base[a] = base.get(a, 0.5) * float(m)

    context = ctx or {}
    if context.get("is_clutch"):
        base["PnR"] = base.get("PnR", 0.5) * 1.05
        base["Drive"] = base.get("Drive", 0.5) * 1.05
        base["TransitionEarly"] = base.get("TransitionEarly", 0.5) * 0.90

    # possession-start context tweaks (event-based possession start)
    pos_start = str(context.get("pos_start", ""))
    if pos_start == "after_drb":
        base["TransitionEarly"] = base.get("TransitionEarly", 0.5) * 0.9
    elif pos_start == "after_score":
        base["TransitionEarly"] = base.get("TransitionEarly", 0.5) * 0.85
    elif pos_start == "after_tov":
        base["TransitionEarly"] = base.get("TransitionEarly", 0.5) * 0.98
    elif pos_start == "after_tov_dead":
        base["TransitionEarly"] = base.get("TransitionEarly", 0.5) * 0.90
    elif pos_start == "start_q":
        base["TransitionEarly"] = base.get("TransitionEarly", 0.5) * 0.90

    # dead-ball inbound restart tends to produce set play rather than early transition
    if bool(context.get("dead_ball_inbound", False)):
        base["TransitionEarly"] = base.get("TransitionEarly", 0.5) * 0.70
        base["HornsSet"] = base.get("HornsSet", 0.5) * 1.05
        base["ElbowHub"] = base.get("ElbowHub", 0.5) * 1.04
        base["PnR"] = base.get("PnR", 0.5) * 1.03
        base["DHO"] = base.get("DHO", 0.5) * 1.03
        base["PostUp"] = base.get("PostUp", 0.5) * 1.02

    if def_tac is None:
        return normalize_weights(base)

    meta = get_defense_meta_params()
    tables = meta.get("defense_meta_action_mult_tables", {})
    strength = float(meta.get("defense_meta_strength", 0.45))
    lo = float(meta.get("defense_meta_clamp_lo", 0.80))
    hi = float(meta.get("defense_meta_clamp_hi", 1.20))
    temp = float(meta.get("defense_meta_temperature", 1.10))
    floor = float(meta.get("defense_meta_floor", 0.03))

    scheme_map = {
        "Switch": "Switch_Everything",
        "SwitchEverything": "Switch_Everything",
        "Switch_Everything": "Switch_Everything",
        "Drop": "Drop",
        "Hedge_ShowRecover": "Hedge_ShowRecover",
        "Hedge": "Hedge_ShowRecover",
        "Blitz_TrapPnR": "Blitz_TrapPnR",
        "ICE_SidePnR": "ICE_SidePnR",
        "ICE": "ICE_SidePnR",
        "Zone": "Zone",
        "Matchup_Zone": "Zone",
        "PackLine_GapHelp": "PackLine_GapHelp",
    }
    scheme = scheme_map.get(getattr(def_tac, "defense_scheme", ""), getattr(def_tac, "defense_scheme", ""))
    meta_mults = tables.get(scheme, {})
    for a, mult in meta_mults.items():
        mult_final = clamp(1.0 + (float(mult) - 1.0) * strength, lo, hi)
        base[a] = base.get(a, 0.5) * mult_final

    probs = apply_temperature(base, temp)
    probs = apply_min_floor(probs, floor)
    # shot_diet wiring
    if ctx is not None:
        style = ctx.get("shot_diet_style")
        tactic_name = ctx.get("tactic_name")
        if style is not None and tactic_name is not None:
            mult_by_base = shot_diet.get_action_multipliers(style, tactic_name)
            for act in list(probs.keys()):
                base_action = get_action_base(act, game_cfg)
                probs[act] = max(probs.get(act, 0.0) * mult_by_base.get(base_action, 1.0), 1e-6)
    return normalize_weights(probs)

def build_defense_action_probs(tac: TacticsConfig, game_cfg: Optional["GameConfig"] = None) -> Dict[str, float]:
    """Build defense 'action' distribution (mostly for logging/feel).

    UI rule (fixed): normalize((Wdef_scheme[action] ^ sharpness) * def_action_mult[action]).
    """
    if game_cfg is None:
        raise ValueError("build_defense_action_probs requires game_cfg")
    scheme_weights = game_cfg.def_scheme_action_weights if isinstance(game_cfg.def_scheme_action_weights, Mapping) else {}
    base = dict(
        scheme_weights.get(
            tac.defense_scheme,
            _fallback_scheme(scheme_weights, "Drop"),
        )
    )
    sharp = clamp(tac.def_scheme_weight_sharpness, 0.70, 1.40)
    base = {a: (max(w, 0.0) ** sharp) for a, w in base.items()}
    for a, m in tac.def_action_weight_mult.items():
        base[a] = base.get(a, 0.5) * float(m)
    return normalize_weights(base)

def effective_scheme_multiplier(base_mult: float, strength: float) -> float:
    s = clamp(strength, 0.70, 1.40)
    return 1.0 + (float(base_mult) - 1.0) * s

def _knob_mult(game_cfg: "GameConfig", key: str, default: float = 1.0) -> float:
    knobs = game_cfg.knobs if isinstance(game_cfg.knobs, Mapping) else {}
    try:
        return float(knobs.get(key, default))
    except Exception:
        return float(default)

def build_outcome_priors(
    action: str,
    off_tac: TacticsConfig,
    def_tac: TacticsConfig,
    tags: Dict[str, Any],
    ctx: Optional[Dict[str, Any]] = None,
    game_cfg: Optional["GameConfig"] = None,
) -> Dict[str, float]:
    if game_cfg is None:
        raise ValueError("build_outcome_priors requires game_cfg")
    base_action = get_action_base(action, game_cfg)
    priors = game_cfg.action_outcome_priors if isinstance(game_cfg.action_outcome_priors, Mapping) else {}
    default_priors = priors.get("SpotUp") if "SpotUp" in priors else _fallback_scheme(priors, "")
    pri = dict(priors.get(base_action, default_priors))

    # offense global
    pri = apply_multipliers(pri, off_tac.outcome_global_mult)

    # offense per-action
    pri = apply_multipliers_typesafe(pri, off_tac.outcome_by_action_mult.get(action, {}))
    pri = apply_multipliers_typesafe(pri, off_tac.outcome_by_action_mult.get(base_action, {}))

    # offense scheme
    off_mult = game_cfg.offense_scheme_mult if isinstance(game_cfg.offense_scheme_mult, Mapping) else {}
    sm = off_mult.get(off_tac.offense_scheme, {}).get(action) or off_mult.get(off_tac.offense_scheme, {}).get(base_action) or {}
    for o, m in sm.items():
        if o in pri:
            pri[o] *= effective_scheme_multiplier(m, off_tac.scheme_outcome_strength)

    # defense knobs on opponent priors
    pri = apply_multipliers(pri, def_tac.opp_outcome_global_mult)
    pri = apply_multipliers_typesafe(pri, def_tac.opp_outcome_by_action_mult.get(action, {}))
    pri = apply_multipliers_typesafe(pri, def_tac.opp_outcome_by_action_mult.get(base_action, {}))

    # defense scheme
    def_mult = game_cfg.defense_scheme_mult if isinstance(game_cfg.defense_scheme_mult, Mapping) else {}
    dm = def_mult.get(def_tac.defense_scheme, {}).get(action) or def_mult.get(def_tac.defense_scheme, {}).get(base_action) or {}
    for o, m in dm.items():
        if o in pri:
            pri[o] *= effective_scheme_multiplier(m, def_tac.def_scheme_outcome_strength)

    # conditional (MVP subset)
    if def_tac.defense_scheme == "ICE_SidePnR" and tags.get("is_side_pnr", False):
        for o in ("RESET_RESREEN","PASS_KICKOUT"):
            if o in pri:
                pri[o] *= 1.03

    if tags.get("in_transition", False):
        for o in ("TO_HANDLE_LOSS","TO_CHARGE","RESET_HUB","RESET_RESREEN"):
            if o in pri:
                pri[o] *= 0.92

    avg_fatigue_off = tags.get("avg_fatigue_off")
    if isinstance(avg_fatigue_off, (int, float)):
        mult = 1.0 + (1.0 - float(avg_fatigue_off)) * (float(tags.get("fatigue_bad_mult_max", 1.12)) - 1.0)
        if avg_fatigue_off < float(tags.get("fatigue_bad_critical", 0.25)):
            mult += float(tags.get("fatigue_bad_bonus", 0.08))
        mult = clamp(mult, 1.0, float(tags.get("fatigue_bad_cap", 1.20)))
        for o in list(pri.keys()):
            if o.startswith("TO_") or o.startswith("RESET_"):
                pri[o] = pri.get(o, 0.0) * mult

    to_base = _knob_mult(game_cfg, "to_base_mult", 1.0)
    foul_base = _knob_mult(game_cfg, "foul_base_mult", 1.0)
    if to_base != 1.0:
        for o in list(pri.keys()):
            if o.startswith("TO_"):
                pri[o] = pri.get(o, 0.0) * to_base
    if foul_base != 1.0:
        for o in list(pri.keys()):
            if o.startswith("FOUL_"):
                pri[o] = pri.get(o, 0.0) * foul_base

    meta = get_defense_meta_params()
    rules = meta.get("defense_meta_priors_rules", {})
    scheme_map = {
        "Switch": "Switch_Everything",
        "SwitchEverything": "Switch_Everything",
        "Switch_Everything": "Switch_Everything",
        "Drop": "Drop",
        "Hedge_ShowRecover": "Hedge_ShowRecover",
        "Hedge": "Hedge_ShowRecover",
        "Blitz_TrapPnR": "Blitz_TrapPnR",
        "ICE_SidePnR": "ICE_SidePnR",
        "ICE": "ICE_SidePnR",
        "Zone": "Zone",
        "Matchup_Zone": "Zone",
        "PackLine_GapHelp": "PackLine_GapHelp",
    }
    scheme = scheme_map.get(getattr(def_tac, "defense_scheme", ""), getattr(def_tac, "defense_scheme", ""))
    for rule in rules.get(scheme, []):
        target = rule.get("key")
        if not target:
            continue
        if rule.get("require_base_action") and rule.get("require_base_action") != base_action:
            continue
        if "mult" in rule:
            if target in pri:
                pri[target] = pri.get(target, 0.0) * float(rule.get("mult", 1.0))
        if "add" in rule:
            pri[target] = pri.get(target, 0.0) + float(rule.get("add", 0.0))
        if "min" in rule:
            if target in pri:
                pri[target] = max(pri.get(target, 0.0), float(rule.get("min", 0.0)))

    # shot_diet wiring
    context = ctx if ctx is not None else tags
    style = context.get("shot_diet_style") if isinstance(context, Mapping) else None
    tactic_name = context.get("tactic_name") if isinstance(context, Mapping) else None
    if style is not None and tactic_name is not None:
        out_mult = shot_diet.get_outcome_multipliers(style, tactic_name, base_action)
        for outcome in list(pri.keys()):
            pri[outcome] = max(pri.get(outcome, 0.0) * out_mult.get(outcome, 1.0), 1e-6)

    return normalize_weights(pri)

def apply_multipliers_typesafe(pri: Dict[str, float], mults: Dict[str, float]) -> Dict[str, float]:
    out = dict(pri)
    for o, m in mults.items():
        if o in out:
            out[o] *= float(m)
    return out
