from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import importlib
import warnings

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
# -------------------------
# Tunable update safety
# -------------------------
# apply_tunable_updates() has a special hook that scales shot-related outcome priors.
# If applied repeatedly by multiplying in-place, the effect accumulates unintentionally.
# We keep an unscaled base snapshot and re-apply the *current* multiplier idempotently.

_BASE_ACTION_OUTCOME_PRIORS: Dict[str, Any] = copy.deepcopy(ACTION_OUTCOME_PRIORS)
_SHOT_PRIOR_SCALE: float = 1.0


def _refresh_base_action_outcome_priors_snapshot() -> None:
    global _BASE_ACTION_OUTCOME_PRIORS
    _BASE_ACTION_OUTCOME_PRIORS = copy.deepcopy(ACTION_OUTCOME_PRIORS)


def _restore_action_outcome_priors_to_base() -> None:
    # Do NOT rebind dict globals; keep references stable.
    ACTION_OUTCOME_PRIORS.clear()
    ACTION_OUTCOME_PRIORS.update(copy.deepcopy(_BASE_ACTION_OUTCOME_PRIORS))


def _apply_shot_prior_scale(mult: float) -> None:
    # Scale only shot outcomes inside each action's outcome prior dict.
    for act, pri in ACTION_OUTCOME_PRIORS.items():
        if not isinstance(pri, dict):
            continue
        for o in list(pri.keys()):
            if isinstance(o, str) and o.startswith("SHOT_"):
                try:
                    pri[o] = float(pri.get(o, 0.0)) * float(mult)
                except Exception:
                    pri[o] = 0.0



# -------------------------
# Era / Parameter externalization (0-1)
# -------------------------
# Commercial goal: make tuning possible WITHOUT touching code.
# We externalize priors, scheme weights/multipliers, shot/pass bases, and prob model parameters into a JSON "era" file.

DEFAULT_PROB_MODEL: Dict[str, float] = {
    # Generic success-prob model clamps
    "base_p_min": 0.02,
    "base_p_max": 0.98,
    "prob_min": 0.03,
    "prob_max": 0.97,

    # OffScore-DefScore scaling (bigger = less sensitive)
    "shot_scale": 18.0,
    "pass_scale": 20.0,
    "rebound_scale": 22.0,

    # ORB baseline used in rebound_orb_probability()
    "orb_base": 0.33,

    # FT model used in resolve_free_throws()
    "ft_base": 0.45,
    "ft_range": 0.47,
    "ft_min": 0.40,
    "ft_max": 0.95,
}

ERA_PROB_MODEL: Dict[str, float] = dict(DEFAULT_PROB_MODEL)

# Logistic parameters by outcome kind (2-1, 2-2)
# NOTE: 'scale' and 'sensitivity' are redundant (sensitivity ~= 1/scale). We keep both for readability.
DEFAULT_LOGISTIC_PARAMS: Dict[str, Dict[str, float]] = {
    "default": {"scale": 18.0, "sensitivity": 1.0 / 18.0},

    # 2-2 table (user-provided)
    "shot_3":   {"scale": 30.0, "sensitivity": 1.0 / 30.0},   # 3PT make
    "shot_mid": {"scale": 24.0, "sensitivity": 1.0 / 24.0},   # midrange make
    "shot_rim": {"scale": 18.0, "sensitivity": 1.0 / 18.0},   # rim finishes
    "shot_post":{"scale": 20.0, "sensitivity": 1.0 / 20.0},   # post shots
    "pass":     {"scale": 28.0, "sensitivity": 1.0 / 28.0},   # pass success
    "rebound":  {"scale": 22.0, "sensitivity": 1.0 / 22.0},   # ORB% model (legacy)
    "turnover": {"scale": 24.0, "sensitivity": 1.0 / 24.0},   # reserved (TO is prior-only)
}

# Variance knob (2-3): logit-space Gaussian noise, so mean stays roughly stable.
DEFAULT_VARIANCE_PARAMS: Dict[str, Any] = {
    "logit_noise_std": 0.20,  # global volatility
    "kind_mult": {
        "shot_3": 1.15,
        "shot_mid": 1.05,
        "shot_rim": 0.95,
        "shot_post": 1.00,
        "pass": 0.85,
        "rebound": 0.60,
    },
    # optional per-team multiplier range (clamped)
    "team_mult_lo": 0.60,
    "team_mult_hi": 1.55,
}

ERA_LOGISTIC_PARAMS: Dict[str, Dict[str, float]] = dict(DEFAULT_LOGISTIC_PARAMS)
ERA_VARIANCE_PARAMS: Dict[str, Any] = copy.deepcopy(DEFAULT_VARIANCE_PARAMS)

DEFAULT_ROLE_FIT = {"default_strength": 0.65}
ERA_ROLE_FIT: Dict[str, Any] = copy.deepcopy(DEFAULT_ROLE_FIT)

MVP_RULES = {
    "quarters": 4,
    "quarter_length": 720,
    # --- Overtime rules ---
    "overtime_length": 300,
    "overtime_bonus_threshold": 4,  # 기존 2 -> 4 (NBA 스타일 기본값)

    # --- Break / rest modeling (does NOT consume game clock) ---
    "break_sec_between_periods": 130,  # Q1->Q2, Q2->Q3, Q3->Q4
    "break_sec_before_ot": 130,        # Regulation -> OT1, and between OTs

    # --- OT start possession ---
    "ot_start_possession_mode": "jumpball",  # "jumpball" or "random"
    "ot_jumpball": {"scale": 12.0},          # 점프볼 승률 민감도 (클수록 50:50에 가까움)

    # --- Recovery during breaks (fatigue only; no minutes/clock) ---
    "break_recovery": {
        "on_court_per_sec": 0.0010,  # 코트 위에 있던 선수의 휴식 회복(초당)
        "bench_per_sec": 0.0016,     # 벤치 선수의 휴식 회복(초당)
    },
    "shot_clock": 24,
    "orb_reset": 14,
    "foul_reset": 14,
    "ft_orb_mult": 0.75,
    "foul_out": 6,
    "bonus_threshold": 5,
    "inbound": {
        "tov_base": 0.010,
        "tov_min": 0.003,
        "tov_max": 0.060,
        "def_scale": 0.00035,
        "off_scale": 0.00030,
    },
    "fatigue_loss": {
        "handler": 0.012,
        "wing": 0.010,
        "big": 0.009,
        "transition_emphasis": 0.001,
        "heavy_pnr": 0.001,
    },
    "fatigue_thresholds": {"sub_out": 0.35, "sub_in": 0.70},
    "fatigue_targets": {
        "starter_sec": 32 * 60,
        "rotation_sec": 16 * 60,
        "bench_sec": 8 * 60,
    },
    "fatigue_effects": {
        "logit_delta_max": -0.25,
        "bad_mult_max": 1.12,
        "bad_critical": 0.25,
        "bad_bonus": 0.08,
        "bad_cap": 1.20,
        "def_mult_min": 0.90,
    },
    "time_costs": {
        "possession_setup": 3.2,
        "setup_start_q": 2.6,
        "setup_after_score": 3.5,
        "setup_after_drb": 3.1,
        "setup_after_tov": 2.1,
        "FoulStop": 2.6,
        "PnR": 8.3,
        "DHO": 7.1,
        "Drive": 6.2,
        "PostUp": 8.2,
        "HornsSet": 7.0,
        "SpotUp": 4.4,
        "Cut": 4.6,
        "TransitionEarly": 4.0,
        "Kickout": 2.8,
        "ExtraPass": 3.0,
        "Reset": 4.5,
    },
    
    "transition_weight_mult": {
        "default": 1.0,
        "after_drb": 4.5,
        "after_tov": 6.0,
    },
}

DEFENSE_META_PARAMS = {
    "defense_meta_strength": 0.45,
    "defense_meta_clamp_lo": 0.80,
    "defense_meta_clamp_hi": 1.20,
    "defense_meta_temperature": 1.10,
    "defense_meta_floor": 0.03,
    "defense_meta_action_mult_tables": {
        "Drop": {
            "PnR": 0.92,
            "Drive": 0.95,
            "PostUp": 1.05,
            "HornsSet": 1.02,
            "Cut": 1.03,
            "Kickout": 1.02,
            "ExtraPass": 1.02,
        },
        "Switch_Everything": {
            "PnR": 0.85,
            "DHO": 0.92,
            "Drive": 0.95,
            "PostUp": 1.10,
            "Cut": 1.08,
            "SpotUp": 1.02,
            "HornsSet": 1.05,
            "ExtraPass": 1.02,
        },
        "Hedge_ShowRecover": {
            "PnR": 0.90,
            "Drive": 0.92,
            "Kickout": 1.05,
            "ExtraPass": 1.05,
            "SpotUp": 1.04,
            "DHO": 0.95,
        },
        "Blitz_TrapPnR": {
            "PnR": 0.82,
            "Drive": 0.90,
            "ExtraPass": 1.08,
            "Kickout": 1.08,
            "SpotUp": 1.06,
            "Cut": 1.03,
            "HornsSet": 1.02,
        },
        "ICE_SidePnR": {
            "PnR": 0.92,
            "Drive": 0.90,
            "SpotUp": 1.03,
            "Kickout": 1.05,
            "ExtraPass": 1.03,
            "DHO": 1.02,
            "Cut": 1.02,
        },
        "Zone": {
            "Drive": 0.85,
            "PostUp": 0.90,
            "SpotUp": 1.06,
            "ExtraPass": 1.08,
            "Kickout": 1.06,
            "DHO": 0.95,
            "Cut": 0.92,
            "HornsSet": 1.02,
        },
        "PackLine_GapHelp": {
            "Drive": 0.82,
            "PnR": 0.95,
            "SpotUp": 1.04,
            "Kickout": 1.06,
            "ExtraPass": 1.05,
            "PostUp": 1.02,
            "Cut": 0.95,
            "DHO": 0.98,
        },
    },
    "defense_meta_priors_rules": {
        "Drop": [
            {"key": "SHOT_MID_PU", "mult": 1.08},
            {"key": "SHOT_3_OD", "mult": 1.03},
            {"key": "SHOT_RIM_LAYUP", "mult": 0.96},
            {"key": "SHOT_RIM_DUNK", "mult": 0.96},
            {"key": "SHOT_RIM_CONTACT", "mult": 0.96},
        ],
        "Hedge_ShowRecover": [
            {"key": "PASS_KICKOUT", "mult": 1.06},
            {"key": "PASS_EXTRA", "mult": 1.05},
        ],
        "Blitz_TrapPnR": [
            {"key": "PASS_SHORTROLL", "min": 0.10, "require_base_action": "PnR"},
            {"key": "FOUL_REACH_TRAP", "add": 0.02},
        ],
        "Zone": [
            {"key": "SHOT_3_CS", "mult": 1.06},
            {"key": "PASS_EXTRA", "mult": 1.06},
        ],
        "PackLine_GapHelp": [
            {"key": "SHOT_3_CS", "mult": 1.05},
            {"key": "PASS_KICKOUT", "mult": 1.06},
            {"key": "TO_CHARGE", "mult": 1.04},
            {"key": "SHOT_RIM_LAYUP", "mult": 0.95},
            {"key": "SHOT_RIM_DUNK", "mult": 0.95},
            {"key": "SHOT_RIM_CONTACT", "mult": 0.95},
        ],
        "Switch_Everything": [
            {"key": "SHOT_POST", "mult": 1.08},
            {"key": "TO_HANDLE_LOSS", "mult": 1.04},
        ],
    },
}

ERA_TARGETS: Dict[str, Dict[str, Any]] = {
    "era_modern_nbaish_v1": {
        "targets": {
            "pace": 99.0,
            "ortg": 115.0,
            "tov_pct": 0.135,
            "three_rate": 0.40,
            "ftr": 0.24,
            "orb_pct": 0.28,
            "shot_share_rim": 0.33,
            "shot_share_mid": 0.12,
            "shot_share_three": 0.55,
            "corner3_share": 0.17,
        },
        "tolerances": {
            "pace": 3.0,
            "ortg": 4.0,
            "tov_pct": 0.010,
            "three_rate": 0.04,
            "ftr": 0.04,
            "orb_pct": 0.03,
            "shot_share_rim": 0.04,
            "shot_share_mid": 0.03,
            "shot_share_three": 0.05,
            "corner3_share": 0.04,
        },
        "op_thresholds": {
            "ortg_hi": 127.0,
            "tov_pct_hi": 0.20,
            "pace_lo": 89.0,
            "pace_hi": 109.0,
        },
    }
}

TUNABLE_REGISTRY: Dict[str, Tuple[str, str]] = {
    "SHOT_BASE_RIM": ("match_engine.prob", "SHOT_BASE_RIM"),
    "SHOT_BASE_MID": ("match_engine.prob", "SHOT_BASE_MID"),
    "SHOT_BASE_3": ("match_engine.prob", "SHOT_BASE_3"),
    "PASS_BASE_SUCCESS_MULT": ("match_engine.prob", "PASS_BASE_SUCCESS_MULT"),
    "ORB_BASE": ("match_engine.resolve", "ORB_BASE"),
    "TO_BASE": ("match_engine.resolve", "TO_BASE"),
    "FOUL_BASE": ("match_engine.resolve", "FOUL_BASE"),
}


# Snapshot built-in defaults (used as fallback if era json is missing keys)
DEFAULT_ERA: Dict[str, Any] = {
    "name": "builtin_default",
    "version": "1.0",
    "knobs": {"mult_lo": 0.70, "mult_hi": 1.40},
    "prob_model": dict(DEFAULT_PROB_MODEL),

    "logistic_params": copy.deepcopy(DEFAULT_LOGISTIC_PARAMS),
    "variance_params": copy.deepcopy(DEFAULT_VARIANCE_PARAMS),

    "role_fit": {"default_strength": 0.65},

    "shot_base": dict(SHOT_BASE),
    "pass_base_success": dict(PASS_BASE_SUCCESS),

    "action_outcome_priors": copy.deepcopy(ACTION_OUTCOME_PRIORS),
    "action_aliases": dict(ACTION_ALIASES),

    "off_scheme_action_weights": copy.deepcopy(OFF_SCHEME_ACTION_WEIGHTS),
    "def_scheme_action_weights": copy.deepcopy(DEF_SCHEME_ACTION_WEIGHTS),

    "offense_scheme_mult": copy.deepcopy(OFFENSE_SCHEME_MULT),
    "defense_scheme_mult": copy.deepcopy(DEFENSE_SCHEME_MULT),
}

_ERA_CACHE: Dict[str, Dict[str, Any]] = {}
_ACTIVE_ERA_NAME: str = "builtin_default"
_ACTIVE_ERA_VERSION: str = "1.0"


def get_mvp_rules() -> Dict[str, Any]:
    return copy.deepcopy(MVP_RULES)


def get_defense_meta_params() -> Dict[str, Any]:
    return copy.deepcopy(DEFENSE_META_PARAMS)


def get_era_targets(name: str) -> Dict[str, Any]:
    return copy.deepcopy(ERA_TARGETS.get(name, ERA_TARGETS.get("era_modern_nbaish_v1", {})))


def _import_attr(mod_path: str, attr: str):
    mod = importlib.import_module(mod_path)
    return mod, getattr(mod, attr)


def snapshot_tunables() -> Dict[str, Any]:
    snap: Dict[str, Any] = {}
    for key, (mod_path, attr) in TUNABLE_REGISTRY.items():
        try:
            _, val = _import_attr(mod_path, attr)
            snap[key] = copy.deepcopy(val)
        except Exception:
            snap[key] = None
    return snap


def restore_tunables(snapshot: Dict[str, Any]) -> None:
    for key, val in (snapshot or {}).items():
        target = TUNABLE_REGISTRY.get(key)
        if not target:
            continue
        mod_path, attr = target
        try:
            mod = importlib.import_module(mod_path)
            setattr(mod, attr, copy.deepcopy(val))
        except Exception:
            continue


def apply_tunable_updates(updates: Dict[str, Any]) -> None:
    updates = updates or {}
    for key, delta in updates.items():
        target = TUNABLE_REGISTRY.get(key)
        if not target:
            continue
        mod_path, attr = target
        try:
            mod = importlib.import_module(mod_path)
            cur = getattr(mod, attr)
            setattr(mod, attr, delta if not isinstance(cur, (int, float)) else float(delta))
        except Exception:
            continue
    
    # special hook: allow global shot prior scaling (idempotent; fatigue can change but priors should not accumulate)
    if "ACTION_PRIOR_SHOT_SCALE" in updates:
        global _SHOT_PRIOR_SCALE
        try:
            mult = float(updates.get("ACTION_PRIOR_SHOT_SCALE", 1.0))
            _SHOT_PRIOR_SCALE = mult
            _restore_action_outcome_priors_to_base()
            _apply_shot_prior_scale(_SHOT_PRIOR_SCALE)
        except Exception as e:
            warnings.warn(
                f"apply_tunable_updates: failed to apply ACTION_PRIOR_SHOT_SCALE ({updates.get('ACTION_PRIOR_SHOT_SCALE')}); {type(e).__name__}: {e}",
                RuntimeWarning,
            )


def _resolve_era_path(era_name: str) -> Optional[str]:
    """Resolve an era name into an on-disk JSON file path, if it exists."""
    if not isinstance(era_name, str) or not era_name:
        return None
    # direct path
    if era_name.endswith(".json") or "/" in era_name or "\\" in era_name:
        return era_name if os.path.exists(era_name) else None

    here = Path(__file__).resolve().parent
    candidates = [
        here / f"era_{era_name}.json",
        here / f"era_{era_name.lower()}.json",
        here / "eras" / f"era_{era_name}.json",
        here / "eras" / f"era_{era_name.lower()}.json",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def _merge_dict(dst: Dict[str, Any], src2: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(dst)
    for k, v in (src2 or {}).items():
        out[k] = v
    return out


def load_era_config(era: Any) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Load an era config (dict) + return (config, warnings, errors)."""
    warnings: List[str] = []
    errors: List[str] = []

    if isinstance(era, dict):
        raw = era
        era_name = str(raw.get("name") or "custom")
    else:
        era_name = str(era or "default")
        if era_name in _ERA_CACHE:
            return _ERA_CACHE[era_name], [], []

        path = _resolve_era_path("default" if era_name == "default" else era_name)
        if path is None:
            warnings.append(f"era file not found for '{era_name}', using built-in defaults")
            cfg = copy.deepcopy(DEFAULT_ERA)
            cfg["name"] = era_name
            _ERA_CACHE[era_name] = cfg
            return cfg, warnings, errors

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            errors.append(f"failed to read era json ({path}): {e}")
            cfg = copy.deepcopy(DEFAULT_ERA)
            cfg["name"] = era_name
            _ERA_CACHE[era_name] = cfg
            return cfg, warnings, errors

        if not isinstance(raw, dict):
            errors.append(f"era json root must be an object/dict (got {type(raw).__name__})")
            cfg = copy.deepcopy(DEFAULT_ERA)
            cfg["name"] = era_name
            _ERA_CACHE[era_name] = cfg
            return cfg, warnings, errors

    cfg, w2, e2 = validate_and_fill_era_dict(raw)
    warnings.extend(w2)
    errors.extend(e2)

    cfg["name"] = str(raw.get("name") or era_name)
    cfg["version"] = str(raw.get("version") or cfg.get("version") or "1.0")

    _ERA_CACHE[cfg["name"]] = cfg
    return cfg, warnings, errors


def validate_and_fill_era_dict(raw: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Validate an era dict and fill missing keys from DEFAULT_ERA."""
    warnings: List[str] = []
    errors: List[str] = []

    cfg = copy.deepcopy(DEFAULT_ERA)
    for k, v in raw.items():
        cfg[k] = v

    required_blocks = [
        "shot_base", "pass_base_success",
        "action_outcome_priors", "action_aliases",
        "off_scheme_action_weights", "def_scheme_action_weights",
        "offense_scheme_mult", "defense_scheme_mult",
        "prob_model", "knobs",
        "logistic_params", "variance_params",
    ]
    for k in required_blocks:
        if k not in cfg or cfg[k] is None:
            warnings.append(f"missing key '{k}' (filled from defaults)")
            cfg[k] = copy.deepcopy(DEFAULT_ERA.get(k))

    dict_blocks = list(required_blocks)
    for k in dict_blocks:
        if not isinstance(cfg.get(k), dict):
            errors.append(f"'{k}' must be an object/dict (got {type(cfg.get(k)).__name__}); using defaults")
            cfg[k] = copy.deepcopy(DEFAULT_ERA.get(k))

    # Light sanity warnings
    for kk, vv in (cfg.get("prob_model") or {}).items():
        if not isinstance(vv, (int, float)) and vv is not None:
            warnings.append(f"prob_model.{kk}: expected number, got {type(vv).__name__}")
    for kk, vv in (cfg.get("knobs") or {}).items():
        if not isinstance(vv, (int, float)) and vv is not None:
            warnings.append(f"knobs.{kk}: expected number, got {type(vv).__name__}")

    return cfg, warnings, errors


def apply_era_config(era_cfg: Dict[str, Any]) -> None:
    """Apply an era config to global tuning parameters.

    IMPORTANT:
    - Do NOT rebind dict globals (e.g., SHOT_BASE = {...}) because other modules may already
      hold references imported from profiles/era.
    - Always mutate dicts in-place via clear()/update().
    """
    global _ACTIVE_ERA_NAME, _ACTIVE_ERA_VERSION

    if not isinstance(era_cfg, dict):
        return

    # ---- profiles.py에서 import된 dict들 (반드시 in-place로 갱신) ----
    sb = era_cfg.get("shot_base")
    if isinstance(sb, dict):
        SHOT_BASE.clear()
        SHOT_BASE.update(dict(sb))

    pb = era_cfg.get("pass_base_success")
    if isinstance(pb, dict):
        PASS_BASE_SUCCESS.clear()
        PASS_BASE_SUCCESS.update(dict(pb))

    aop = era_cfg.get("action_outcome_priors")
    if isinstance(aop, dict):
        ACTION_OUTCOME_PRIORS.clear()
        ACTION_OUTCOME_PRIORS.update(copy.deepcopy(aop))
        _refresh_base_action_outcome_priors_snapshot()
        if _SHOT_PRIOR_SCALE != 1.0:
            _restore_action_outcome_priors_to_base()
            _apply_shot_prior_scale(_SHOT_PRIOR_SCALE)

    aa = era_cfg.get("action_aliases")
    if isinstance(aa, dict):
        ACTION_ALIASES.clear()
        ACTION_ALIASES.update(dict(aa))

    offw = era_cfg.get("off_scheme_action_weights")
    if isinstance(offw, dict):
        OFF_SCHEME_ACTION_WEIGHTS.clear()
        OFF_SCHEME_ACTION_WEIGHTS.update(copy.deepcopy(offw))

    defw = era_cfg.get("def_scheme_action_weights")
    if isinstance(defw, dict):
        DEF_SCHEME_ACTION_WEIGHTS.clear()
        DEF_SCHEME_ACTION_WEIGHTS.update(copy.deepcopy(defw))

    osm = era_cfg.get("offense_scheme_mult")
    if isinstance(osm, dict):
        OFFENSE_SCHEME_MULT.clear()
        OFFENSE_SCHEME_MULT.update(copy.deepcopy(osm))

    dsm = era_cfg.get("defense_scheme_mult")
    if isinstance(dsm, dict):
        DEFENSE_SCHEME_MULT.clear()
        DEFENSE_SCHEME_MULT.update(copy.deepcopy(dsm))

    # ---- era.py 내부 dict들 (다른 모듈이 from .era import ... 로 잡고 있을 수 있으니 in-place) ----
    pm = era_cfg.get("prob_model")
    if isinstance(pm, dict):
        merged = _merge_dict(DEFAULT_PROB_MODEL, pm)
        ERA_PROB_MODEL.clear()
        ERA_PROB_MODEL.update(merged)

    lp = era_cfg.get("logistic_params")
    if isinstance(lp, dict):
        ERA_LOGISTIC_PARAMS.clear()
        ERA_LOGISTIC_PARAMS.update(copy.deepcopy(lp))

    vp = era_cfg.get("variance_params")
    if isinstance(vp, dict):
        ERA_VARIANCE_PARAMS.clear()
        ERA_VARIANCE_PARAMS.update(copy.deepcopy(vp))

    rf = era_cfg.get("role_fit")
    if isinstance(rf, dict):
        merged = _merge_dict(DEFAULT_ROLE_FIT, rf)
        ERA_ROLE_FIT.clear()
        ERA_ROLE_FIT.update(merged)

    _ACTIVE_ERA_NAME = str(era_cfg.get("name") or "unknown")
    _ACTIVE_ERA_VERSION = str(era_cfg.get("version") or "1.0")

    # NOTE: validation no longer depends on global allowed-set caches.
