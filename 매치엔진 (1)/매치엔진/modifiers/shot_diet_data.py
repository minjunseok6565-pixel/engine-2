    # =============================================================================
    # [DATA FILE ONLY]  (자동 분리됨)
    # 이 파일은 로직이 아니라 '튜닝 테이블/상수'만 담는 **데이터 모듈**입니다.
    # LLM 컨텍스트에는 기본적으로 포함하지 말고, 테이블을 수정/튜닝할 때만 열어보세요.
    #
    # 포함 데이터(요약):
    #   - BASELINE/TAU_USAGE/USAGE_* / CLAMP_* / PROB_FLOOR: scalar config
#   - WEIGHTS_GLOBAL_OUTCOME: {base_action: {outcome: {feature_key: w}}}
#   - WEIGHTS_TACTIC_ACTION: {tactic: {base_action: {feature_key: w}}}
#   - WEIGHTS_TACTIC_OUTCOME_DELTA: {tactic: {base_action: {outcome: {feature_key: Δw}}}}
#   - TACTIC_ALPHA / SCREENER_ROLE_PRIORITY / SCHEME_ALIASES / ALPHA_*: presets
#   - 로직 파일: shot_diet.py
    # =============================================================================

"""
Data tables / presets for shot_diet.py.

This module is intentionally data-heavy.
Avoid including it in LLM context unless you are editing the tables.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# -------------------------
# Config (Spec v1)
# -------------------------

BASELINE: float = 0.50  # neutral point for features (0..1)
TAU_USAGE: float = 0.15  # usage softmax temperature
USAGE_MIN_PRIMARY: float = 0.55
USAGE_MAX_PRIMARY: float = 0.90

# Conservative starting clamps (Spec v1)
CLAMP_ACTION_MULT: Tuple[float, float] = (0.78, 1.28)
CLAMP_OUTCOME_MULT: Tuple[float, float] = (0.65, 1.45)

# Minimum multiplier floor in exp-domain already handled by clamps above.
# Optional probability floor (builders can apply as well).
PROB_FLOOR: float = 1e-6

# -------------------------
# Weights / presets (Work 2)
# -------------------------
# Structure:
# - Weights are log-space coefficients multiplied by (feature - BASELINE).
# - Missing keys imply 0 weight.
# - Outcome weights are defined per base action and per outcome.

def _w(**kwargs) -> Dict[str, float]:
    return dict(kwargs)


# Global outcome weights (base_action -> outcome -> feature weights)
WEIGHTS_GLOBAL_OUTCOME: Dict[str, Dict[str, Dict[str, float]]] = {
    "PnR": {
        "PASS_KICKOUT": _w(TEAM_SPACING=1.00, BH_PASS_CREATION=0.60, D_HELP_CLOSEOUT=-0.80),
        "PASS_SHORTROLL": _w(SC_SHORTROLL_PLAY=0.90, BH_PASS_CREATION=0.30, D_POA=0.40),
        "SHOT_RIM_LAYUP": _w(BH_DRIVE_PRESSURE=0.80, SC_ROLL_FINISH=0.60, D_RIM_PROTECT=-0.90, D_POA=-0.50),
        "SHOT_RIM_DUNK": _w(SC_ROLL_FINISH=0.70, BH_DRIVE_PRESSURE=0.40, D_RIM_PROTECT=-1.00),
        "SHOT_RIM_CONTACT": _w(BH_FOUL_PRESSURE=0.80, SC_ROLL_FINISH=0.50, D_RIM_PROTECT=0.40, D_POA=-0.30),
        "SHOT_3_CS": _w(TEAM_CATCH3_QUALITY=0.90, TEAM_SPACING=0.80, BH_PASS_CREATION=0.40, D_HELP_CLOSEOUT=-0.80),
        "SHOT_3_OD": _w(BH_PULLUP_THREAT=0.80, BH_PNR=0.40, D_POA=0.30),
        "SHOT_MID_PU": _w(BH_PULLUP_THREAT=0.70, D_RIM_PROTECT=0.40, D_POA=0.30, TEAM_SPACING=-0.20),
        "SHOT_TOUCH_FLOATER": _w(BH_DRIVE_PRESSURE=0.60, D_RIM_PROTECT=0.50),
        "TO_HANDLE_LOSS": _w(BH_BALL_SECURITY=-0.80, D_POA=0.60),
        "FOUL_DRAW_RIM": _w(BH_FOUL_PRESSURE=0.80, SC_ROLL_FINISH=0.50, D_RIM_PROTECT=0.30),
        "RESET_RESREEN": _w(SC_SCREEN_QUALITY=0.40, TEAM_EXTRA_PASS=0.30, D_POA=0.40, D_HELP_CLOSEOUT=0.40),
        "RESET_HUB": _w(TEAM_EXTRA_PASS=0.30, BH_PASS_CREATION=0.20, D_HELP_CLOSEOUT=0.40),
    },
    "Drive": {
        "SHOT_RIM_LAYUP": _w(BH_DRIVE_PRESSURE=1.10, D_RIM_PROTECT=-1.00, D_POA=-0.60, TEAM_SPACING=0.20),
        "SHOT_RIM_DUNK": _w(BH_DRIVE_PRESSURE=0.90, D_RIM_PROTECT=-1.10),
        "SHOT_RIM_CONTACT": _w(BH_FOUL_PRESSURE=0.90, D_RIM_PROTECT=0.40, D_POA=-0.40),
        "SHOT_TOUCH_FLOATER": _w(BH_DRIVE_PRESSURE=0.60, D_RIM_PROTECT=0.60),
        "PASS_KICKOUT": _w(TEAM_SPACING=0.90, BH_PASS_CREATION=0.60, D_HELP_CLOSEOUT=-0.80),
        "PASS_EXTRA": _w(TEAM_EXTRA_PASS=0.50, BH_PASS_CREATION=0.40, D_STEAL_PRESS=-0.50),
        "TO_HANDLE_LOSS": _w(BH_BALL_SECURITY=-0.80, D_POA=0.80),
        "TO_CHARGE": _w(BH_BALL_SECURITY=-0.60, D_POA=0.60, D_HELP_CLOSEOUT=0.40),
        "FOUL_DRAW_RIM": _w(BH_FOUL_PRESSURE=1.00, D_RIM_PROTECT=0.30),
        "RESET_HUB": _w(TEAM_EXTRA_PASS=0.40, BH_PASS_CREATION=0.30, D_HELP_CLOSEOUT=0.50),
    },
    "DHO": {
        "SHOT_3_CS": _w(TEAM_CATCH3_QUALITY=1.00, TEAM_SPACING=0.50, SC_SCREEN_QUALITY=0.50, D_HELP_CLOSEOUT=-0.80),
        "SHOT_3_OD": _w(BH_PULLUP_THREAT=0.80, SC_SCREEN_QUALITY=0.40, D_POA=0.30),
        "PASS_EXTRA": _w(TEAM_EXTRA_PASS=0.60, D_STEAL_PRESS=-0.50),
        "RESET_REDO_DHO": _w(SC_SCREEN_QUALITY=0.60, TEAM_EXTRA_PASS=0.40, D_HELP_CLOSEOUT=0.40),
        "TO_HANDLE_LOSS": _w(BH_BALL_SECURITY=-0.60, D_POA=0.50),
    },
    "SpotUp": {
        "SHOT_3_CS": _w(TEAM_CATCH3_QUALITY=1.20, TEAM_SPACING=0.80, D_HELP_CLOSEOUT=-0.90),
        "RESET_HUB": _w(TEAM_EXTRA_PASS=0.30, D_HELP_CLOSEOUT=0.40),
    },
    "Kickout": {
        "SHOT_3_CS": _w(TEAM_CATCH3_QUALITY=1.10, TEAM_SPACING=0.90, D_HELP_CLOSEOUT=-0.90),
        "PASS_EXTRA": _w(TEAM_EXTRA_PASS=0.50, D_STEAL_PRESS=-0.50),
        "RESET_HUB": _w(TEAM_EXTRA_PASS=0.20, D_HELP_CLOSEOUT=0.30),
    },
    "ExtraPass": {
        "PASS_EXTRA": _w(TEAM_EXTRA_PASS=1.00, D_STEAL_PRESS=-0.80),
        "PASS_SKIP": _w(TEAM_SPACING=0.60, BH_PASS_CREATION=0.50, D_STEAL_PRESS=-0.90),
        "SHOT_3_CS": _w(TEAM_CATCH3_QUALITY=0.80, TEAM_SPACING=0.80, D_HELP_CLOSEOUT=-0.70),
        "RESET_HUB": _w(TEAM_EXTRA_PASS=0.40, D_HELP_CLOSEOUT=0.50),
    },
    "Cut": {
        "SHOT_RIM_LAYUP": _w(TEAM_CUTTING=0.90, D_HELP_CLOSEOUT=-0.70, D_RIM_PROTECT=-0.60),
        "SHOT_RIM_DUNK": _w(TEAM_CUTTING=0.70, D_RIM_PROTECT=-0.80),
        "FOUL_DRAW_RIM": _w(TEAM_CUTTING=0.60, D_RIM_PROTECT=0.30),
        "PASS_EXTRA": _w(TEAM_EXTRA_PASS=0.40, D_STEAL_PRESS=-0.50),
        "RESET_HUB": _w(TEAM_EXTRA_PASS=0.30, D_HELP_CLOSEOUT=0.40),
    },
    "PostUp": {
        "SHOT_POST": _w(TEAM_POST_GRAVITY=1.20, D_POST=-1.00, D_HELP_CLOSEOUT=-0.50),
        "FOUL_DRAW_POST": _w(TEAM_POST_GRAVITY=0.80, D_POST=-0.90),
        "PASS_KICKOUT": _w(TEAM_SPACING=0.90, TEAM_EXTRA_PASS=0.50, D_HELP_CLOSEOUT=-0.80),
        "PASS_SKIP": _w(TEAM_SPACING=0.60, BH_PASS_CREATION=0.40, D_STEAL_PRESS=-0.90),
        "RESET_POST_OUT": _w(TEAM_EXTRA_PASS=0.60, D_HELP_CLOSEOUT=0.50),
        "TO_HANDLE_LOSS": _w(BH_BALL_SECURITY=-0.40, D_POST=0.40),
    },
    "HornsSet": {
        "PASS_EXTRA": _w(TEAM_EXTRA_PASS=0.70, BH_PASS_CREATION=0.40, D_STEAL_PRESS=-0.60),
        "SHOT_3_CS": _w(TEAM_SPACING=0.60, TEAM_CATCH3_QUALITY=0.60, D_HELP_CLOSEOUT=-0.70),
        "RESET_HUB": _w(TEAM_EXTRA_PASS=0.40, D_HELP_CLOSEOUT=0.50),
    },
    "TransitionEarly": {
        "SHOT_RIM_LAYUP": _w(TEAM_PACE=1.00, BH_DRIVE_PRESSURE=0.60, D_POA=-0.80, D_RIM_PROTECT=-0.50),
        "SHOT_RIM_DUNK": _w(TEAM_PACE=0.90, SC_ROLL_FINISH=0.60, D_RIM_PROTECT=-0.60),
        "SHOT_3_CS": _w(TEAM_SPACING=0.70, TEAM_CATCH3_QUALITY=0.70, D_HELP_CLOSEOUT=-0.50),
        "PASS_KICKOUT": _w(TEAM_SPACING=0.40, BH_PASS_CREATION=0.40, D_HELP_CLOSEOUT=-0.40),
        "TO_HANDLE_LOSS": _w(BH_BALL_SECURITY=-0.60, D_POA=0.40),
        "FOUL_DRAW_RIM": _w(BH_FOUL_PRESSURE=0.60),
    },
}

# Tactic-specific ACTION weights (tactic -> action -> feature weights)
WEIGHTS_TACTIC_ACTION: Dict[str, Dict[str, Dict[str, float]]] = {
    "Spread_HeavyPnR": {
        "PnR": _w(BH_PNR=1.40, TEAM_SPACING=0.60, SC_ROLL_FINISH=0.50, SC_SHORTROLL_PLAY=0.30, D_POA=-0.60),
        "Drive": _w(BH_DRIVE_PRESSURE=0.40, TEAM_SPACING=0.20, D_POA=-0.30),
        "Kickout": _w(TEAM_SPACING=0.40, BH_PASS_CREATION=0.30, D_HELP_CLOSEOUT=-0.40),
        "ExtraPass": _w(TEAM_EXTRA_PASS=0.20, TEAM_CATCH3_QUALITY=0.20, D_STEAL_PRESS=-0.30),
        "SpotUp": _w(TEAM_CATCH3_QUALITY=0.30, TEAM_SPACING=0.30, D_HELP_CLOSEOUT=-0.30),
        "PostUp": _w(TEAM_POST_GRAVITY=-0.40),
    },
    "FiveOut": {
        "SpotUp": _w(TEAM_SPACING=0.90, TEAM_CATCH3_QUALITY=0.60, D_HELP_CLOSEOUT=-0.40),
        "Drive": _w(BH_DRIVE_PRESSURE=0.80, TEAM_SPACING=0.40, D_RIM_PROTECT=-0.40, D_POA=-0.30),
        "Kickout": _w(TEAM_SPACING=0.50, BH_PASS_CREATION=0.40, D_HELP_CLOSEOUT=-0.40),
        "ExtraPass": _w(TEAM_EXTRA_PASS=0.50, TEAM_CATCH3_QUALITY=0.20, D_STEAL_PRESS=-0.30),
        "PnR": _w(BH_PNR=0.30, SC_POP_THREAT=0.20, TEAM_SPACING=0.20),
        "PostUp": _w(TEAM_POST_GRAVITY=-0.60),
    },
    "Drive_Kick": {
        "Drive": _w(BH_DRIVE_PRESSURE=1.30, BH_BALL_SECURITY=0.50, TEAM_SPACING=0.30, D_POA=-0.60, D_RIM_PROTECT=-0.40),
        "Kickout": _w(TEAM_SPACING=0.90, BH_PASS_CREATION=0.60, D_HELP_CLOSEOUT=-0.60),
        "ExtraPass": _w(TEAM_EXTRA_PASS=0.40, D_STEAL_PRESS=-0.30),
        "PnR": _w(BH_PNR=0.40, SC_ROLL_FINISH=0.20),
        "TransitionEarly": _w(TEAM_PACE=0.30, BH_DRIVE_PRESSURE=0.20),
    },
    "Motion_SplitCut": {
        "Cut": _w(TEAM_CUTTING=1.20, TEAM_SPACING=0.60, TEAM_EXTRA_PASS=0.40, D_HELP_CLOSEOUT=-0.50),
        "ExtraPass": _w(TEAM_EXTRA_PASS=0.80, BH_PASS_CREATION=0.40, D_STEAL_PRESS=-0.50),
        "SpotUp": _w(TEAM_SPACING=0.40, TEAM_CATCH3_QUALITY=0.40),
        "DHO": _w(TEAM_EXTRA_PASS=0.40, SC_SCREEN_QUALITY=0.30),
        "PnR": _w(BH_PNR=0.20),
    },
    "DHO_Chicago": {
        "DHO": _w(SC_SCREEN_QUALITY=1.20, TEAM_CATCH3_QUALITY=0.80, BH_PASS_CREATION=0.40, D_HELP_CLOSEOUT=-0.50, D_STEAL_PRESS=-0.40),
        "SpotUp": _w(TEAM_CATCH3_QUALITY=0.50, TEAM_SPACING=0.30),
        "ExtraPass": _w(TEAM_EXTRA_PASS=0.40, D_STEAL_PRESS=-0.20),
        "Cut": _w(TEAM_CUTTING=0.30),
    },
    "Post_InsideOut": {
        "PostUp": _w(TEAM_POST_GRAVITY=1.40, TEAM_SPACING=0.40, D_POST=-0.50, D_HELP_CLOSEOUT=-0.30),
        "ExtraPass": _w(TEAM_EXTRA_PASS=0.60, TEAM_SPACING=0.50, D_STEAL_PRESS=-0.40),
        "SpotUp": _w(TEAM_SPACING=0.40, TEAM_CATCH3_QUALITY=0.30),
        "Cut": _w(TEAM_CUTTING=0.30),
    },
    "Horns_Elbow": {
        "HornsSet": _w(BH_PASS_CREATION=1.10, SC_SHORTROLL_PLAY=0.60, BH_PULLUP_THREAT=0.40, TEAM_SPACING=0.30, D_POA=-0.40, D_HELP_CLOSEOUT=-0.40),
        "PnR": _w(BH_PNR=0.50, SC_SHORTROLL_PLAY=0.40),
        "DHO": _w(SC_SCREEN_QUALITY=0.30, TEAM_CATCH3_QUALITY=0.30),
        "PostUp": _w(TEAM_POST_GRAVITY=0.30),
    },
    "Transition_Early": {
        "TransitionEarly": _w(TEAM_PACE=1.60, BH_DRIVE_PRESSURE=0.60, TEAM_SPACING=0.40, D_DREB=-0.50, D_POA=-0.30),
        "Drive": _w(BH_DRIVE_PRESSURE=0.70, TEAM_SPACING=0.30, D_POA=-0.30),
        "SpotUp": _w(TEAM_CATCH3_QUALITY=0.40, TEAM_SPACING=0.20),
        "PnR": _w(BH_PNR=-0.30),
    },
}

# Tactic-specific OUTCOME deltas (tactic -> base_action -> outcome -> feature weights)
WEIGHTS_TACTIC_OUTCOME_DELTA: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {
    "Spread_HeavyPnR": {
        "PnR": {
            "PASS_KICKOUT": _w(TEAM_SPACING=0.30),
            "PASS_SHORTROLL": _w(SC_SHORTROLL_PLAY=0.30),
            "SHOT_3_CS": _w(TEAM_CATCH3_QUALITY=0.20),
            "SHOT_MID_PU": _w(BH_PULLUP_THREAT=-0.20),
        },
        "Drive": {
            "PASS_KICKOUT": _w(TEAM_SPACING=0.20),
            "TO_CHARGE": _w(BH_BALL_SECURITY=-0.20),
        },
    },
    "FiveOut": {
        "SpotUp": {
            "SHOT_3_CS": _w(TEAM_SPACING=0.20),
        },
        "Drive": {
            "PASS_KICKOUT": _w(TEAM_SPACING=0.25),
            "SHOT_RIM_LAYUP": _w(BH_DRIVE_PRESSURE=0.15),
        },
        "PnR": {
            "SHOT_3_CS": _w(TEAM_CATCH3_QUALITY=0.15),
            "SHOT_RIM_LAYUP": _w(D_RIM_PROTECT=-0.10),
        },
    },
    "Drive_Kick": {
        "Drive": {
            "PASS_KICKOUT": _w(TEAM_SPACING=0.30),
            "TO_HANDLE_LOSS": _w(BH_BALL_SECURITY=-0.25),
            "TO_CHARGE": _w(BH_BALL_SECURITY=-0.20),
            "FOUL_DRAW_RIM": _w(BH_FOUL_PRESSURE=0.15),
        },
        "Kickout": {
            "SHOT_3_CS": _w(TEAM_CATCH3_QUALITY=0.20),
        },
        "ExtraPass": {
            "PASS_EXTRA": _w(TEAM_EXTRA_PASS=0.15),
        },
    },
    "Motion_SplitCut": {
        "Cut": {
            "SHOT_RIM_LAYUP": _w(TEAM_CUTTING=0.25),
            "PASS_EXTRA": _w(TEAM_EXTRA_PASS=0.20),
        },
        "ExtraPass": {
            "PASS_EXTRA": _w(TEAM_EXTRA_PASS=0.20),
            "SHOT_3_CS": _w(TEAM_CATCH3_QUALITY=0.10),
        },
        "PnR": {
            "SHOT_3_OD": _w(BH_PULLUP_THREAT=-0.20),
        },
    },
    "DHO_Chicago": {
        "DHO": {
            "SHOT_3_CS": _w(TEAM_CATCH3_QUALITY=0.25),
            "RESET_REDO_DHO": _w(SC_SCREEN_QUALITY=0.25),
            "SHOT_3_OD": _w(BH_PULLUP_THREAT=0.10),
        },
        "SpotUp": {
            "SHOT_3_CS": _w(TEAM_SPACING=0.10),
        },
    },
    "Post_InsideOut": {
        "PostUp": {
            "PASS_KICKOUT": _w(TEAM_SPACING=0.25),
            "SHOT_POST": _w(TEAM_POST_GRAVITY=0.20),
            "RESET_POST_OUT": _w(TEAM_EXTRA_PASS=0.20),
        },
        "ExtraPass": {
            "SHOT_3_CS": _w(TEAM_CATCH3_QUALITY=0.15),
        },
    },
    "Horns_Elbow": {
        "HornsSet": {
            "PASS_SHORTROLL": _w(SC_SHORTROLL_PLAY=0.20),
            "SHOT_MID_PU": _w(BH_PULLUP_THREAT=0.20),
            "PASS_EXTRA": _w(TEAM_EXTRA_PASS=0.15),
        },
        "PnR": {
            "PASS_SHORTROLL": _w(SC_SHORTROLL_PLAY=0.15),
        },
    },
    "Transition_Early": {
        "TransitionEarly": {
            "SHOT_RIM_LAYUP": _w(TEAM_PACE=0.20),
            "SHOT_3_CS": _w(TEAM_CATCH3_QUALITY=0.15),
            "TO_HANDLE_LOSS": _w(BH_BALL_SECURITY=-0.10),
        },
        "Drive": {
            "FOUL_DRAW_RIM": _w(BH_FOUL_PRESSURE=0.10),
        },
    },
}

# Alphas per tactic (Spec v1; conservative, narrow ranges)
TACTIC_ALPHA: Dict[str, Tuple[float, float]] = {
    "Spread_HeavyPnR": (0.40, 0.70),
    "FiveOut": (0.40, 0.70),
    "Drive_Kick": (0.45, 0.75),
    "Motion_SplitCut": (0.45, 0.65),
    "DHO_Chicago": (0.45, 0.70),
    "Post_InsideOut": (0.40, 0.75),
    "Horns_Elbow": (0.40, 0.70),
    "Transition_Early": (0.55, 0.70),
}



# Screener role priorities per scheme (role_fit role names)
# NOTE: scheme name normalization is handled by _normalize_scheme_name().
SCREENER_ROLE_PRIORITY: Dict[str, List[str]] = {
    # Spread PnR: roll pressure -> short roll / pop as next layer
    "Spread_HeavyPnR": [
        "Roller_Finisher",
        "ShortRoll_Playmaker",
        "Pop_Spacer_Big",
        "Post_Hub",
        "Connector_Playmaker",
        "Spacer_Movement",
    ],
    # Drive & Kick: pop/ghost to clear lane -> roll to pin help
    "Drive_Kick": [
        "Pop_Spacer_Big",
        "Roller_Finisher",
        "Connector_Playmaker",
        "ShortRoll_Playmaker",
        "Spacer_Movement",
        "Post_Hub",
    ],
    # 5-out: pop/slip/riscreen to force switches & break help line
    "FiveOut": [
        "Pop_Spacer_Big",
        "ShortRoll_Playmaker",
        "Connector_Playmaker",
        "Spacer_Movement",
        "Roller_Finisher",
        "Post_Hub",
    ],
    # DHO + Chicago: handoff hub (delivery + decision + rescreen)
    "DHO_Chicago": [
        "Connector_Playmaker",
        "Pop_Spacer_Big",
        "Post_Hub",
        "Spacer_Movement",
        "ShortRoll_Playmaker",
        "Roller_Finisher",
    ],
    # Motion split-cut: off-ball screening by guards/wings
    "Motion_SplitCut": [
        "Connector_Playmaker",
        "Spacer_Movement",
        "Post_Hub",
        "Pop_Spacer_Big",
        "ShortRoll_Playmaker",
        "Roller_Finisher",
    ],
    # Post inside-out: anchor post hub, then free shooters
    "Post_InsideOut": [
        "Post_Hub",
        "Spacer_Movement",
        "Connector_Playmaker",
        "Roller_Finisher",
        "Pop_Spacer_Big",
        "ShortRoll_Playmaker",
    ],
    # Horns elbow: elbow short roll reads first
    "Horns_Elbow": [
        "ShortRoll_Playmaker",
        "Pop_Spacer_Big",
        "Post_Hub",
        "Roller_Finisher",
        "Connector_Playmaker",
        "Spacer_Movement",
    ],
    # Transition early/drag: speed & rim threat first
    "Transition_Early": [
        "Roller_Finisher",
        "Pop_Spacer_Big",
        "ShortRoll_Playmaker",
        "Spacer_Movement",
        "Connector_Playmaker",
        "Post_Hub",
    ],
}

# Common aliases -> canonical scheme keys used above
SCHEME_ALIASES: Dict[str, str] = {
    # 5-out
    "5-out": "FiveOut",
    "5out": "FiveOut",
    "fiveout": "FiveOut",
    "five_out": "FiveOut",
    # DHO chicago
    "dho-chicago": "DHO_Chicago",
    "dho_chicago": "DHO_Chicago",
    "dhochicago": "DHO_Chicago",
}

# Fallback alphas (used when tactic not found)
ALPHA_ACTION_FALLBACK: float = 0.35
ALPHA_OUTCOME_FALLBACK: float = 0.65
