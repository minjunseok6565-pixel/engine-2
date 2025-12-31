from __future__ import annotations

from .profiles import OUTCOME_PROFILES

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .core import clamp
from .models import DERIVED_DEFAULT, Player, TeamState, ROLE_FALLBACK_RANK
from .profiles import (
    ACTION_ALIASES,
    ACTION_OUTCOME_PRIORS,
    DEFENSE_SCHEME_MULT,
    DEF_SCHEME_ACTION_WEIGHTS,
    OFFENSE_SCHEME_MULT,
    OFF_SCHEME_ACTION_WEIGHTS,
    PASS_BASE_SUCCESS,
    SHOT_BASE,
)
from .tactics import TacticsConfig

SHOT_DIET_SUPPORTED_OFFENSE_SCHEMES = {
    "Spread_HeavyPnR",
    "FiveOut",
    "Drive_Kick",
    "Motion_SplitCut",
    "DHO_Chicago",
    "Post_InsideOut",
    "Horns_Elbow",
    "Transition_Early",
}

# -------------------------
# Validation / Sanitization (Commercial-ready input safety)
# -------------------------

@dataclass
class ValidationConfig:
    """Controls how strictly we validate and sanitize user inputs."""
    strict: bool = True  # True: raise on critical issues (missing derived keys, invalid schemes, invalid lineup)
    mult_lo: float = 0.70
    mult_hi: float = 1.40
    derived_lo: float = 0.0
    derived_hi: float = 100.0
    missing_derived_policy: str = "error"  # "error" or "fill"
    default_derived_value: float = DERIVED_DEFAULT
    # If True, we will clamp out-of-range numbers instead of erroring (still logs warnings).
    clamp_out_of_range: bool = True


@dataclass
class ValidationReport:
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def to_dict(self) -> Dict[str, Any]:
        return {"warnings": list(self.warnings), "errors": list(self.errors), "ok": (len(self.errors) == 0)}


def _is_finite_number(x: Any) -> bool:
    try:
        v = float(x)
    except Exception:
        return False
    return math.isfinite(v)


def _collect_required_derived_keys() -> List[str]:
    keys: set[str] = set()

    # Anything that affects outcome resolution
    for _, sides in OUTCOME_PROFILES.items():
        keys.update(sides.get("offense", {}).keys())
        keys.update(sides.get("defense", {}).keys())

    # Role fallbacks and other selectors
    keys.update(ROLE_FALLBACK_RANK.values())
    keys.update([
        "SHOT_FT",
        "REB_OR", "REB_DR",
        "PHYSICAL", "ENDURANCE", "FAT_CAPACITY",
        "DEF_POA", "DEF_HELP", "DEF_STEAL", "DEF_RIM", "DEF_POST",
        "FIN_RIM", "FIN_DUNK", "FIN_CONTACT",
        "SHOT_3_CS", "SHOT_3_OD", "SHOT_MID_CS", "SHOT_MID_PU", "SHOT_TOUCH",
        "POST_SCORE", "POST_CONTROL",
        "DRIVE_CREATE", "HANDLE_SAFE", "FIRST_STEP",
        "PASS_SAFE", "PASS_CREATE", "PNR_READ", "SHORTROLL_PLAY",
    ])
    keys.update({
        "PNR_READ", "DRIVE_CREATE", "PASS_CREATE", "HANDLE_SAFE",
        "PHYSICAL", "FIN_CONTACT", "SHORTROLL_PLAY", "FIN_RIM", "FIN_DUNK",
        "FIRST_STEP", "SHOT_3_OD", "SHOT_MID_PU", "PASS_SAFE", "SHOT_FT",
        "SHOT_3_CS", "SHOT_MID_CS", "REB_OR", "ENDURANCE",
        "POST_SCORE", "POST_CONTROL",
        "DEF_RIM", "DEF_HELP", "DEF_POA", "DEF_STEAL", "DEF_POST", "REB_DR", "FAT_CAPACITY"
    })
    return sorted(keys)


REQUIRED_DERIVED_KEYS: List[str] = _collect_required_derived_keys()


def _collect_allowed_off_actions() -> set[str]:
    s: set[str] = set()
    for d in OFF_SCHEME_ACTION_WEIGHTS.values():
        s.update(d.keys())
    # base actions and aliases should also be allowed
    s.update(ACTION_OUTCOME_PRIORS.keys())
    s.update(ACTION_ALIASES.keys())
    s.update(ACTION_ALIASES.values())
    return s


def _collect_allowed_def_actions() -> set[str]:
    s: set[str] = set()
    for d in DEF_SCHEME_ACTION_WEIGHTS.values():
        s.update(d.keys())
    return s


def _collect_allowed_outcomes() -> set[str]:
    s: set[str] = set()
    s.update(OUTCOME_PROFILES.keys())
    s.update(SHOT_BASE.keys())
    s.update(PASS_BASE_SUCCESS.keys())
    for pri in ACTION_OUTCOME_PRIORS.values():
        s.update(pri.keys())
    return s


ALLOWED_OFF_ACTIONS: set[str] = set()
ALLOWED_DEF_ACTIONS: set[str] = set()
ALLOWED_OUTCOMES: set[str] = set()

def refresh_allowed_sets() -> None:
    """Recompute allowed keys after era parameters change."""
    global ALLOWED_OFF_ACTIONS, ALLOWED_DEF_ACTIONS, ALLOWED_OUTCOMES
    ALLOWED_OFF_ACTIONS = _collect_allowed_off_actions()
    ALLOWED_DEF_ACTIONS = _collect_allowed_def_actions()
    ALLOWED_OUTCOMES = _collect_allowed_outcomes()

refresh_allowed_sets()


def _clamp_mult(v: float, cfg: ValidationConfig) -> float:
    return clamp(v, cfg.mult_lo, cfg.mult_hi)


def _sanitize_mult_dict(
    mults: Dict[str, Any],
    allowed_keys: set[str],
    cfg: ValidationConfig,
    report: ValidationReport,
    path: str,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, raw in (mults or {}).items():
                # outcome key aliases (backward compatibility)
        if k == "TO_SHOTCLOCK":
            k = "TO_SHOT_CLOCK"
            
        if k not in allowed_keys:
            report.warn(f"{path}: unknown key '{k}' ignored")
            continue
        if not _is_finite_number(raw):
            msg = f"{path}.{k}: non-numeric multiplier '{raw}'"
            if cfg.strict:
                report.error(msg)
            else:
                report.warn(msg + " (ignored)")
            continue
        v = float(raw)
        vv = _clamp_mult(v, cfg)
        if abs(vv - v) > 1e-9:
            report.warn(f"{path}.{k}: clamped {v:.3f} -> {vv:.3f}")
        out[k] = vv
    return out


def _sanitize_outcome_mult_dict(
    mults: Dict[str, Any],
    cfg: ValidationConfig,
    report: ValidationReport,
    path: str,
) -> Dict[str, float]:
    return _sanitize_mult_dict(mults, ALLOWED_OUTCOMES, cfg, report, path)


def _sanitize_nested_outcome_by_action(
    nested: Dict[str, Any],
    cfg: ValidationConfig,
    report: ValidationReport,
    path: str,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for act, sub in (nested or {}).items():
        if act not in ALLOWED_OFF_ACTIONS:
            report.warn(f"{path}: unknown action '{act}' ignored")
            continue
        if not isinstance(sub, dict):
            msg = f"{path}.{act}: expected dict, got {type(sub).__name__}"
            if cfg.strict:
                report.error(msg)
            else:
                report.warn(msg + " (ignored)")
            continue
        clean = _sanitize_outcome_mult_dict(sub, cfg, report, f"{path}.{act}")
        if clean:
            out[act] = clean
    return out


def sanitize_tactics_config(tac: TacticsConfig, cfg: ValidationConfig, report: ValidationReport, label: str) -> None:
    """Mutates tactics in-place: clamps all UI knobs and ignores unknown keys."""

    if tac.offense_scheme not in OFF_SCHEME_ACTION_WEIGHTS:
        msg = f"{label}.offense_scheme: unknown scheme '{tac.offense_scheme}'"
        if cfg.strict:
            report.error(msg)
        else:
            report.warn(msg + " (fallback to Spread_HeavyPnR)")
            tac.offense_scheme = "Spread_HeavyPnR"
    if tac.offense_scheme not in SHOT_DIET_SUPPORTED_OFFENSE_SCHEMES:
        msg = f"{label}.offense_scheme: unsupported for shot diet '{tac.offense_scheme}'"
        if cfg.strict:
            report.error(msg)
        else:
            report.warn(msg + " (fallback to Spread_HeavyPnR)")
            tac.offense_scheme = "Spread_HeavyPnR"

    if tac.defense_scheme not in DEF_SCHEME_ACTION_WEIGHTS:
        msg = f"{label}.defense_scheme: unknown scheme '{tac.defense_scheme}'"
        if cfg.strict:
            report.error(msg)
        else:
            report.warn(msg + " (fallback to Drop)")
            tac.defense_scheme = "Drop"

    # Scalar knobs
    for attr in ("scheme_weight_sharpness", "scheme_outcome_strength", "def_scheme_weight_sharpness", "def_scheme_outcome_strength"):
        raw = getattr(tac, attr, 1.0)
        if not _is_finite_number(raw):
            msg = f"{label}.{attr}: non-numeric '{raw}'"
            if cfg.strict:
                report.error(msg)
            else:
                report.warn(msg + " (set to 1.0)")
                setattr(tac, attr, 1.0)
            continue
        v = float(raw)
        vv = _clamp_mult(v, cfg)
        if abs(vv - v) > 1e-9:
            report.warn(f"{label}.{attr}: clamped {v:.3f} -> {vv:.3f}")
        setattr(tac, attr, vv)

    # Offense multipliers
    tac.action_weight_mult = _sanitize_mult_dict(tac.action_weight_mult, ALLOWED_OFF_ACTIONS, cfg, report, f"{label}.action_weight_mult")
    tac.outcome_global_mult = _sanitize_outcome_mult_dict(tac.outcome_global_mult, cfg, report, f"{label}.outcome_global_mult")
    tac.outcome_by_action_mult = _sanitize_nested_outcome_by_action(tac.outcome_by_action_mult, cfg, report, f"{label}.outcome_by_action_mult")

    # Defense multipliers
    tac.def_action_weight_mult = _sanitize_mult_dict(tac.def_action_weight_mult, ALLOWED_DEF_ACTIONS, cfg, report, f"{label}.def_action_weight_mult")
    tac.opp_action_weight_mult = _sanitize_mult_dict(getattr(tac, "opp_action_weight_mult", {}), ALLOWED_OFF_ACTIONS, cfg, report, f"{label}.opp_action_weight_mult")
    tac.opp_outcome_global_mult = _sanitize_outcome_mult_dict(tac.opp_outcome_global_mult, cfg, report, f"{label}.opp_outcome_global_mult")
    tac.opp_outcome_by_action_mult = _sanitize_nested_outcome_by_action(tac.opp_outcome_by_action_mult, cfg, report, f"{label}.opp_outcome_by_action_mult")

    # Context values (some are multipliers, some are special knobs)
    if tac.context is None:
        tac.context = {}
    clean_ctx: Dict[str, Any] = {}
    for k, v in tac.context.items():
        if k.endswith("_MULT"):
            if not _is_finite_number(v):
                msg = f"{label}.context.{k}: non-numeric '{v}'"
                if cfg.strict:
                    report.error(msg)
                    continue
                report.warn(msg + " (set to 1.0)")
                clean_ctx[k] = 1.0
                continue
            fv = float(v)
            fvv = _clamp_mult(fv, cfg)
            if abs(fvv - fv) > 1e-9:
                report.warn(f"{label}.context.{k}: clamped {fv:.3f} -> {fvv:.3f}")
            clean_ctx[k] = fvv

        elif k == "ROLE_FIT_STRENGTH":
            # 0..1 scalar (separate from multiplier clamp range)
            if not _is_finite_number(v):
                msg = f"{label}.context.{k}: non-numeric '{v}'"
                if cfg.strict:
                    report.error(msg)
                    continue
                report.warn(msg + " (set to 0.65)")
                clean_ctx[k] = 0.65
                continue
            fv = float(v)
            fvv = clamp(fv, 0.0, 1.0)
            if abs(fvv - fv) > 1e-9:
                report.warn(f"{label}.context.{k}: clamped {fv:.3f} -> {fvv:.3f}")
            clean_ctx[k] = fvv

        else:
            clean_ctx[k] = v
    tac.context = clean_ctx


def sanitize_player_derived(p: Player, cfg: ValidationConfig, report: ValidationReport, label: str) -> None:
    """Ensures derived stats exist, are numeric, and contain required keys."""
    if p.derived is None:
        report.warn(f"{label}.{p.pid}: derived is None (set to empty)")
        p.derived = {}

    # Coerce numeric & clamp
    clean: Dict[str, float] = {}
    for k, raw in p.derived.items():
        if not _is_finite_number(raw):
            msg = f"{label}.{p.pid}.derived.{k}: non-numeric '{raw}'"
            if cfg.strict:
                report.error(msg)
                continue
            report.warn(msg + " (dropped)")
            continue
        v = float(raw)
        if cfg.clamp_out_of_range:
            vv = clamp(v, cfg.derived_lo, cfg.derived_hi)
            if abs(vv - v) > 1e-9:
                report.warn(f"{label}.{p.pid}.derived.{k}: clamped {v:.2f} -> {vv:.2f}")
            v = vv
        clean[k] = v
    p.derived = clean

    # Required keys
    missing = [k for k in REQUIRED_DERIVED_KEYS if k not in p.derived]
    if missing:
        msg = f"{label}.{p.pid}: missing derived keys ({len(missing)}): {', '.join(missing[:8])}{'...' if len(missing)>8 else ''}"
        if cfg.missing_derived_policy == "fill":
            report.warn(msg + f" (filled with {cfg.default_derived_value})")
            for k in missing:
                p.derived[k] = float(cfg.default_derived_value)
        else:
            report.error(msg)


def validate_and_sanitize_team(team: TeamState, cfg: ValidationConfig, report: ValidationReport, label: str) -> None:
    # Lineup sanity
    if not isinstance(team.lineup, list) or len(team.lineup) == 0:
        report.error(f"{label}: lineup missing")
        return

    if len(team.lineup) < 5:
        msg = f"{label}: lineup size is {len(team.lineup)} (expected at least 5)"
        if cfg.strict:
            report.error(msg)
        else:
            report.warn(msg + " (engine will use available players)")

    # Unique PIDs
    pids = [p.pid for p in team.lineup]
    if len(set(pids)) != len(pids):
        report.error(f"{label}: duplicate player pid in lineup")
    if any((not isinstance(pid, str)) or (pid.strip() == "") for pid in pids):
        report.error(f"{label}: invalid empty pid in lineup")

    # Player derived validation
    for p in team.lineup:
        sanitize_player_derived(p, cfg, report, label)

    # Roles sanity (warn-only; engine already has fallbacks)
    if team.roles is None:
        team.roles = {}
        report.warn(f"{label}: roles missing (empty roles)")
    lineup_pid_set = set(pids)
    for role, pid in list(team.roles.items()):
        if pid not in lineup_pid_set:
            msg = f"{label}.roles.{role}: pid '{pid}' not in lineup"
            if cfg.strict:
                report.error(msg)
            else:
                report.warn(msg + " (will fallback automatically)")

    # Tactics sanity + clamp
    if team.tactics is None:
        report.error(f"{label}: tactics missing")
        return
    sanitize_tactics_config(team.tactics, cfg, report, f"{label}.tactics")
