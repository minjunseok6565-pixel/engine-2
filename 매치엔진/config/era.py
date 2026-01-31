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
    "orb_base": 0.23,

    # FT model used in resolve_free_throws()
    "ft_base": 0.52,
    "ft_range": 0.47,
    "ft_min": 0.40,
    "ft_max": 0.95,

    # Steal / block event modeling (used by resolve.py; 0-1)
    # NOTE: These are event probabilities, not boxscore rates. They are intended to be
    # combined with score/quality modulation (prob_from_scores) and contextual deltas.
    "steal_bad_pass_base": 0.72,
    "steal_handle_loss_base": 0.84,
    "bad_pass_lineout_base": 0.20,

    "block_base_rim": 0.085,
    "block_base_post": 0.065,
    "block_base_mid": 0.022,
    "block_base_3": 0.012,

    # Block outcomes: out-of-bounds -> offense retains (dead-ball) vs live rebound.
    "block_oob_base_rim": 0.32,
    "block_oob_base_post": 0.30,
    "block_oob_base_mid": 0.18,
    "block_oob_base_3": 0.12,

    # If a blocked miss stays in-play, it is harder for the offense to recover the ball.
    "blocked_orb_mult_rim": 0.78,
    "blocked_orb_mult_post": 0.80,
    "blocked_orb_mult_mid": 0.86,
    "blocked_orb_mult_3": 0.88,


    # Putback modeling (ORB -> immediate try)
    # 'try' params are probabilities in [0..1] used at the ORB branch to choose a "Putback" action.
    "putback_try_base_rim": 0.20,
    "putback_try_base_post": 0.18,
    "putback_try_base_mid": 0.10,
    "putback_try_base_3": 0.06,
    "putback_try_base_ft": 0.22,

    # If the miss was blocked but stayed in-play, putback tries are less frequent (defender is attached).
    "putback_try_mult_blocked": 0.70,

    # Skill-based multiplier around 1.0 (computed from rebounder attributes; clamped below).
    "putback_try_w_reb_or": 0.55,
    "putback_try_w_fin": 0.35,
    "putback_try_w_phy": 0.20,
    "putback_try_skill_mult_min": 0.70,
    "putback_try_skill_mult_max": 1.35,

    # Final try probability clamp
    "putback_try_clamp_min": 0.02,
    "putback_try_clamp_max": 0.45,

    # Shot make penalty for Putback (logit-space delta added in resolve.py)
    "putback_make_logit_penalty": -0.30,
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
    "steal":    {"scale": 26.0, "sensitivity": 1.0 / 26.0},   # turnover->steal split
    "block":    {"scale": 28.0, "sensitivity": 1.0 / 28.0},   # miss->block split
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
        "steal": 0.75,
        "block": 0.80,
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
        "possession_setup": 3.3,
        "setup_start_q": 2.8,
        "setup_after_score": 3.6,
        "setup_after_drb": 3.3,
        "setup_after_tov": 2.2,
        "setup_after_steal": 1.7,
        "setup_after_block": 2.1,
        "FoulStop": 3.0,
        "BlockOOBStop": 2.2,
        "PnR": 8.3,
        "PnP": 8.2,
        "DHO": 7.3,
        "Drive": 6.5,
        "ISO": 7.8,
        "PostUp": 8.6,
        "HornsSet": 7.5,
        "SpotUp": 4.7,
        "QuickShot": 1.4,
        "Putback": 0.95,
        "Cut": 4.8,
        "TransitionEarly": 4.4,
        "Kickout": 3.2,
        "ExtraPass": 3.5,
        "Reset": 5.3,
    },

    "timing": {
        "min_release_window": 0.7,
        "urgent_budget_sec": 8.0,
        "quickshot_cost_sec": 1.2,
        "soft_slack_span": 4.0,
        "soft_slack_floor": 0.20,
        "quickshot_inject_base": 0.05,
        "quickshot_inject_urgency_mult": 0.35,
        "pass_reset_suppress_urgency": 0.85
    },

    # --- Context indices (continuous; used to compute pressure_index / garbage_index) ---
    # Used by sim_game.py to compute pressure_index / garbage_index in [0..1].
    # Tune to get the *feel* you want without hard mode switches.
    "context_indices": {
        # Points per possession estimate for converting score margin -> possession margin
        "ppp_estimate": 1.10,

        # Pressure (late-game, close): time window and "close" threshold in possessions
        "pressure_window_sec": 180.0,
        "pressure_poss_close": 3.0,

        # Garbage (late-game, blowout): time window and blowout band (start -> full) in possessions
        "garbage_window_sec": 360.0,
        "garbage_poss_start": 6.0,
        "garbage_poss_full": 9.0,
    },
    
    "transition_weight_mult": {
        "default": 1.0,
        "after_drb": 3.2,
        "after_tov": 4.0,
        "after_steal": 4.6,
        "after_block": 4.2,
    },
 
     # --- Timeouts (dead-ball only, v1) ---
     "timeouts": {"per_team": 7},
 
     "timeout_ai": {
         "enabled": True,
         "deadball_only": True,
         # v1: allow either team to call timeout during dead-ball (simplifies TO-streak trigger realism)
         "allow_both_teams_deadball": True,
 
         # Cooldown in possessions (easy & stable)
         "cooldown_possessions": 3,
 
         # base probability (usually 0; triggers drive decisions)
         "p_base": 0.00,
 
         # Trigger G-1: stop opponent run (consecutive scoring points)
         "run_pts_threshold": 8,
         "run_pts_hard": 12,
         "p_run": 0.30,
 
         # Trigger G-2: "this is ugly" streak (same team consecutive turnovers)
         "to_streak_threshold": 3,
         "to_streak_hard": 4,
         "p_to": 0.22,
 
         # Optional secondary triggers (tune freely)
         "p_pressure": 0.16,
         "p_fatigue": 0.10,
         "fatigue_threshold": 0.55,
 
         # hard cap
         "p_cap": 0.85,
     },
 
     "timeout_value": {
         # H: timeouts are "more spendable" when you have many, less when you have few
         "remaining_alpha": 0.70,
 
         # H: blowout suppression (applies to both sides)
         "blowout_soft": 10,
         "blowout_hard": 18,
         "blowout_floor": 0.30,
 
         # H: score-bias asymmetry (losing team calls more; winning team calls less)
         "trail_scale": 12.0,
         "trail_k": 0.35,      # max +35% when trailing by >= trail_scale
         "lead_scale": 12.0,
         "lead_k": 0.35,       # max -35% when leading by >= lead_scale
         "lead_floor": 0.55,   # don't go below this from lead bias alone
 
         # H: late-game is more conservative
         "late_beta": 0.50,    # linear downweight vs regulation progress
         "late_floor": 0.60,
     },
 
     # Recovery is optional; default off for v1 (can enable later)
     "timeout_recovery": {
         "enabled": False,
         # "equivalent break seconds" to apply as recovery effect
         "equiv_break_sec": 12.0,
         "on_court_mult": 1.0,
         "bench_mult": 1.0,
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
        "off_scheme_action_weights", 
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
