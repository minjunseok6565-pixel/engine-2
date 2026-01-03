from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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


DEFAULT_ROLE_FIT = {"default_strength": 0.65}

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

def get_mvp_rules() -> Dict[str, Any]:
    return copy.deepcopy(MVP_RULES)


def get_defense_meta_params() -> Dict[str, Any]:
    return copy.deepcopy(DEFENSE_META_PARAMS)


def get_era_targets(name: str) -> Dict[str, Any]:
    return copy.deepcopy(ERA_TARGETS.get(name, ERA_TARGETS.get("era_modern_nbaish_v1", {})))


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


def load_era_config(era: Any) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Load an era config (dict) + return (config, warnings, errors)."""
    warnings: List[str] = []
    errors: List[str] = []

    if isinstance(era, dict):
        raw = era
        era_name = str(raw.get("name") or "custom")
    else:
        era_name = str(era or "default")
        path = _resolve_era_path("default" if era_name == "default" else era_name)
        if path is None:
            warnings.append(f"era file not found for '{era_name}', using built-in defaults")
            cfg = copy.deepcopy(DEFAULT_ERA)
            cfg["name"] = era_name
            return cfg, warnings, errors

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            errors.append(f"failed to read era json ({path}): {e}")
            cfg = copy.deepcopy(DEFAULT_ERA)
            cfg["name"] = era_name
            return cfg, warnings, errors

        if not isinstance(raw, dict):
            errors.append(f"era json root must be an object/dict (got {type(raw).__name__})")
            cfg = copy.deepcopy(DEFAULT_ERA)
            cfg["name"] = era_name
            return cfg, warnings, errors

    cfg, w2, e2 = validate_and_fill_era_dict(raw)
    warnings.extend(w2)
    errors.extend(e2)

    cfg["name"] = str(raw.get("name") or era_name)
    cfg["version"] = str(raw.get("version") or cfg.get("version") or "1.0")

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
