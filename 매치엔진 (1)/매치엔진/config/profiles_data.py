    # =============================================================================
    # [DATA FILE ONLY]  (자동 분리됨)
    # 이 파일은 로직이 아니라 '튜닝 테이블/상수'만 담는 **데이터 모듈**입니다.
    # LLM 컨텍스트에는 기본적으로 포함하지 말고, 테이블을 수정/튜닝할 때만 열어보세요.
    #
    # 포함 데이터(요약):
    #   - OUTCOME_PROFILES: {outcome: {'offense':{stat:w}, 'defense':{stat:w}}}
#   - ACTION_OUTCOME_PRIORS: {base_action: {outcome: prior_prob}}
#   - OFF/DEF_SCHEME_ACTION_WEIGHTS: {scheme: {action: weight}}
#   - OFFENSE_SCHEME_MULT / DEFENSE_SCHEME_MULT: {scheme:{base_action:{outcome:mult}}}
#   - ACTION_ALIASES, SHOT_BASE, PASS_BASE_SUCCESS, CORNER3_PROB_BY_ACTION_BASE: misc tables
#   - 로직 파일: (주로 resolve.py / sim.py / shot_diet.py 등에서 참조)
    # =============================================================================

"""
Data tables for profiles.py.

This module is intentionally data-heavy (mostly large dict literals).
Avoid including it in LLM context unless you are editing/tuning these tables.
"""
# -------------------------
from __future__ import annotations

from typing import Dict

# Outcome resolution profiles (derived ability weights)
# -------------------------

OUTCOME_PROFILES: Dict[str, Dict[str, Dict[str, float]]] = {
    "SHOT_RIM_LAYUP": {
        "offense": {"FIN_RIM":0.55, "FIN_CONTACT":0.15, "SHOT_TOUCH":0.10, "HANDLE_SAFE":0.10, "ENDURANCE":0.10},
        "defense": {"DEF_RIM":0.45, "DEF_HELP":0.25, "PHYSICAL":0.15, "DEF_POA":0.10, "ENDURANCE":0.05},
    },
    "SHOT_RIM_DUNK": {
        "offense": {"FIN_DUNK":0.55, "FIN_CONTACT":0.20, "FIN_RIM":0.10, "HANDLE_SAFE":0.05, "ENDURANCE":0.10},
        "defense": {"DEF_RIM":0.50, "PHYSICAL":0.20, "DEF_HELP":0.20, "ENDURANCE":0.10},
    },
    "SHOT_RIM_CONTACT": {
        "offense": {"FIN_CONTACT":0.55, "FIN_RIM":0.20, "SHOT_TOUCH":0.10, "PHYSICAL":0.10, "ENDURANCE":0.05},
        "defense": {"DEF_RIM":0.40, "PHYSICAL":0.30, "DEF_HELP":0.20, "DEF_POST":0.10},
    },
    "SHOT_TOUCH_FLOATER": {
        "offense": {"SHOT_TOUCH":0.55, "FIN_RIM":0.15, "FIN_CONTACT":0.10, "DRIVE_CREATE":0.10, "ENDURANCE":0.10},
        "defense": {"DEF_RIM":0.30, "DEF_HELP":0.35, "DEF_POA":0.15, "PHYSICAL":0.10, "ENDURANCE":0.10},
    },
    "SHOT_MID_CS": {
        "offense": {"SHOT_MID_CS":0.85, "ENDURANCE":0.15},
        "defense": {"DEF_POA":0.35, "DEF_HELP":0.35, "ENDURANCE":0.20, "PHYSICAL":0.10},
    },
    "SHOT_3_CS": {
        "offense": {"SHOT_3_CS":0.85, "ENDURANCE":0.15},
        "defense": {"DEF_POA":0.35, "DEF_HELP":0.35, "ENDURANCE":0.25, "PHYSICAL":0.05},
    },
    "SHOT_MID_PU": {
        "offense": {"SHOT_MID_PU":0.65, "HANDLE_SAFE":0.15, "FIRST_STEP":0.10, "ENDURANCE":0.10},
        "defense": {"DEF_POA":0.50, "DEF_HELP":0.25, "ENDURANCE":0.15, "PHYSICAL":0.10},
    },
    "SHOT_3_OD": {
        "offense": {"SHOT_3_OD":0.60, "HANDLE_SAFE":0.20, "FIRST_STEP":0.10, "ENDURANCE":0.10},
        "defense": {"DEF_POA":0.55, "DEF_HELP":0.20, "ENDURANCE":0.20, "PHYSICAL":0.05},
    },
    "SHOT_POST": {
        "offense": {"POST_SCORE":0.40, "POST_CONTROL":0.20, "FIN_CONTACT":0.20, "SHOT_TOUCH":0.10, "PHYSICAL":0.10},
        "defense": {"DEF_POST":0.55, "DEF_HELP":0.20, "PHYSICAL":0.20, "DEF_RIM":0.05},
    },

    "PASS_KICKOUT": {
        "offense": {"PASS_CREATE":0.45, "PASS_SAFE":0.35, "PNR_READ":0.20},
        "defense": {"DEF_STEAL":0.55, "DEF_HELP":0.30, "DEF_POA":0.15},
    },
    "PASS_EXTRA": {
        "offense": {"PASS_SAFE":0.55, "PASS_CREATE":0.30, "PNR_READ":0.15},
        "defense": {"DEF_STEAL":0.50, "DEF_HELP":0.35, "ENDURANCE":0.15},
    },
    "PASS_SKIP": {
        "offense": {"PASS_CREATE":0.60, "PASS_SAFE":0.25, "PNR_READ":0.15},
        "defense": {"DEF_STEAL":0.55, "DEF_HELP":0.35, "DEF_POA":0.10},
    },
    "PASS_SHORTROLL": {
        "offense": {"SHORTROLL_PLAY":0.55, "PASS_SAFE":0.25, "PASS_CREATE":0.20},
        "defense": {"DEF_HELP":0.45, "DEF_STEAL":0.30, "ENDURANCE":0.25},
    },

    "TO_HANDLE_LOSS": {
        "offense": {"HANDLE_SAFE":0.60, "DRIVE_CREATE":0.20, "ENDURANCE":0.20},
        "defense": {"DEF_STEAL":0.50, "DEF_POA":0.30, "DEF_HELP":0.20}
    },
    "TO_BAD_PASS": {
        "offense": {"PASS_SAFE":0.55, "PASS_CREATE":0.25, "PNR_READ":0.20},
        "defense": {"DEF_STEAL":0.55, "DEF_HELP":0.30, "DEF_POA":0.15}
    },
    "TO_CHARGE": {
        "offense": {"DRIVE_CREATE":0.35, "PHYSICAL":0.35, "PNR_READ":0.15, "ENDURANCE":0.15},
        "defense": {"DEF_POA":0.40, "DEF_HELP":0.35, "PHYSICAL":0.25}
    },
    "TO_SHOT_CLOCK": {
        "offense": {"PNR_READ":0.35, "PASS_CREATE":0.25, "DRIVE_CREATE":0.20, "HANDLE_SAFE":0.10, "ENDURANCE":0.10},
        "defense": {"DEF_POA":0.35, "DEF_HELP":0.35, "ENDURANCE":0.20, "PHYSICAL":0.10}
    },

    "TO_INBOUND": {
    "offense": {"PASS_SAFE":0.55, "PASS_CREATE":0.20, "PNR_READ":0.10, "ENDURANCE":0.15},
    "defense": {"DEF_STEAL":0.55, "DEF_POA":0.20, "DEF_HELP":0.25},
    },

    "FOUL_DRAW_RIM": {
        "offense": {"FIN_CONTACT":0.60, "FIN_RIM":0.15, "PHYSICAL":0.15, "ENDURANCE":0.10},
        "defense": {"DEF_RIM":0.40, "PHYSICAL":0.25, "DEF_HELP":0.25, "ENDURANCE":0.10}
    },
    "FOUL_DRAW_POST": {
        "offense": {"FIN_CONTACT":0.40, "POST_SCORE":0.25, "PHYSICAL":0.20, "POST_CONTROL":0.15},
        "defense": {"DEF_POST":0.45, "PHYSICAL":0.35, "DEF_HELP":0.20}
    },
    "FOUL_DRAW_JUMPER": {
        "offense": {"SHOT_3_OD":0.30, "SHOT_MID_PU":0.30, "HANDLE_SAFE":0.20, "ENDURANCE":0.20},
        "defense": {"DEF_POA":0.45, "ENDURANCE":0.35, "PHYSICAL":0.20}
    },
    "FOUL_REACH_TRAP": {
        "offense": {"HANDLE_SAFE":0.35, "PASS_SAFE":0.35, "PNR_READ":0.20, "ENDURANCE":0.10},
        "defense": {"DEF_STEAL":0.45, "PHYSICAL":0.25, "ENDURANCE":0.30}
    },

    "RESET_HUB": {
        "offense": {"PASS_SAFE":0.55, "PNR_READ":0.25, "ENDURANCE":0.20},
        "defense": {"DEF_HELP":0.45, "DEF_STEAL":0.25, "ENDURANCE":0.30}
    },
    "RESET_RESREEN": {
        "offense": {"PNR_READ":0.35, "HANDLE_SAFE":0.20, "ENDURANCE":0.25, "PASS_SAFE":0.20},
        "defense": {"DEF_POA":0.35, "DEF_HELP":0.35, "ENDURANCE":0.30}
    },
    "RESET_REDO_DHO": {
        "offense": {"HANDLE_SAFE":0.30, "PASS_SAFE":0.30, "ENDURANCE":0.25, "PNR_READ":0.15},
        "defense": {"DEF_POA":0.40, "DEF_STEAL":0.20, "ENDURANCE":0.40}
    },
    "RESET_POST_OUT": {
        "offense": {"POST_CONTROL":0.35, "PASS_SAFE":0.40, "PASS_CREATE":0.15, "PHYSICAL":0.10},
        "defense": {"DEF_POST":0.40, "DEF_STEAL":0.30, "DEF_HELP":0.30}
    },
}

SHOT_BASE = {
    "SHOT_RIM_LAYUP": 0.55,
    "SHOT_RIM_DUNK": 0.72,
    "SHOT_RIM_CONTACT": 0.44,
    "SHOT_TOUCH_FLOATER": 0.37,
    "SHOT_MID_CS": 0.41,
    "SHOT_MID_PU": 0.38,
    "SHOT_3_CS": 0.333,
    "SHOT_3_OD": 0.33,
    "SHOT_POST": 0.44,
}

# Chance that a 3PA is a corner 3 instead of ATB 3 (keyed by *base* action)
CORNER3_PROB_BY_ACTION_BASE = {
    "default": 0.145,
    "Kickout": 0.23,
    "ExtraPass": 0.205,
    "SpotUp": 0.165,
    "TransitionEarly": 0.14,
    "PnR": 0.09,
    "DHO": 0.09,
}
PASS_BASE_SUCCESS = {
    "PASS_KICKOUT": 0.925,
    "PASS_EXTRA": 0.935,
    "PASS_SKIP": 0.905,
    "PASS_SHORTROLL": 0.885,
}



# -------------------------
# Scheme action weights
# -------------------------

OFF_SCHEME_ACTION_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Spread_HeavyPnR": {
        "PnR": 25,
        "TransitionEarly": 6,
        "Drive": 7,
        "Kickout": 8,
        "ExtraPass": 6,
        "SpotUp": 7,
        "Cut": 4,
    },
    "Drive_Kick": {
        "Drive": 30,
        "Kickout": 18,
        "ExtraPass": 12,
        "SpotUp": 12,
        "Cut": 6,
        "PnR": 3,
        "DHO": 2,
    },
    "FiveOut": {
        "Drive": 18,
        "SpotUp": 16,
        "Kickout": 14,
        "ExtraPass": 10,
        "Cut": 10,
        "DHO": 8,
        "PnR": 5,
    },
    "Motion_SplitCut": {
        "Cut": 18,
        "DHO": 8,
        "Drive": 10,
        "Kickout": 6,
        "ExtraPass": 6,
        "SpotUp": 6,
        "PnR": 4,
    },
    "DHO_Chicago": {
        "DHO": 16,
        "Drive": 12,
        "Kickout": 10,
        "ExtraPass": 6,
        "SpotUp": 10,
        "PnR": 6,
    },
    "Post_InsideOut": {
        "PostUp": 22,
        "Kickout": 14,
        "ExtraPass": 8,
        "SpotUp": 12,
        "Cut": 8,
        "Drive": 4,
        "DHO": 4,
    },
    "Horns_Elbow": {
        "HornsSet": 18,
        "PnR": 12,
        "DHO": 8,
        "Drive": 10,
        "Kickout": 8,
        "ExtraPass": 6,
        "SpotUp": 8,
        "Cut": 6,
    },
    "Transition_Early": {
        "TransitionEarly": 40,
        "Drive": 8,
        "Kickout": 8,
        "SpotUp": 8,
    },
}

# Defense scheme -> 'def_action' weights
# NOTE: def_action is currently used mainly for logging/feel and as a tuning hook.
# Keep the key-space small and stable; you can expand later if you start using def_action in resolve.
DEF_SCHEME_ACTION_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Drop": {
        "Contain": 40,
        "Help": 25,
        "Pressure": 10,
        "Switch": 15,
        "Zone_Shift": 10,
    },
    "Switch_Everything": {
        "Contain": 20,
        "Help": 20,
        "Pressure": 15,
        "Switch": 35,
        "Zone_Shift": 10,
    },
    "Hedge_ShowRecover": {
        "Contain": 25,
        "Help": 20,
        "Pressure": 20,
        "Switch": 20,
        "Zone_Shift": 15,
    },
    "Blitz_TrapPnR": {
        "Contain": 15,
        "Help": 15,
        "Pressure": 40,
        "Switch": 20,
        "Zone_Shift": 10,
    },
    "ICE_SidePnR": {
        "Contain": 35,
        "Help": 25,
        "Pressure": 10,
        "Switch": 15,
        "Zone_Shift": 15,
    },
    "Zone": {
        "Contain": 10,
        "Help": 20,
        "Pressure": 10,
        "Switch": 10,
        "Zone_Shift": 50,
    },
    "PackLine_GapHelp": {
        "Contain": 25,
        "Help": 35,
        "Pressure": 5,
        "Switch": 15,
        "Zone_Shift": 20,
    },
}

# -------------------------
# Action outcome priors (include fouls)
# -------------------------

ACTION_OUTCOME_PRIORS: Dict[str, Dict[str, float]] = {
    "PnR": {
        "SHOT_RIM_DUNK": 0.06,
        "SHOT_RIM_LAYUP": 0.21,
        "SHOT_RIM_CONTACT": 0.04,
        "SHOT_TOUCH_FLOATER": 0.17,
        "SHOT_MID_PU": 0.055,
        "SHOT_3_CS": 0.06,
        "SHOT_3_OD": 0.06,
        "PASS_KICKOUT": 0.19,
        "PASS_SHORTROLL": 0.155,
        "FOUL_DRAW_JUMPER": 0.01,
        "FOUL_DRAW_RIM": 0.11,
        "TO_HANDLE_LOSS": 0.02,
        "RESET_RESREEN": 0.059,
        "RESET_HUB": 0.03,
    },
    "DHO": {
        "SHOT_RIM_DUNK": 0.04,
        "SHOT_RIM_LAYUP": 0.26,
        "SHOT_TOUCH_FLOATER": 0.19,
        "SHOT_MID_PU": 0.047,
        "SHOT_3_CS": 0.09,
        "SHOT_3_OD": 0.033,
        "PASS_EXTRA": 0.145,
        "PASS_KICKOUT": 0.165,
        "FOUL_DRAW_JUMPER": 0.02,
        "FOUL_DRAW_RIM": 0.055,
        "TO_HANDLE_LOSS": 0.02,
        "RESET_REDO_DHO": 0.094,
    },
    "Drive": {
        "SHOT_RIM_CONTACT": 0.27,
        "SHOT_RIM_DUNK": 0.13,
        "SHOT_RIM_LAYUP": 0.43,
        "SHOT_TOUCH_FLOATER": 0.22,
        "SHOT_MID_PU": 0.019,
        "SHOT_3_OD": 0.03,
        "PASS_EXTRA": 0.12,
        "PASS_KICKOUT": 0.185,
        "FOUL_DRAW_RIM": 0.117,
        "TO_CHARGE": 0.015,
        "TO_HANDLE_LOSS": 0.024,
        "RESET_HUB": 0.049,
    },
    "Kickout": {
        "SHOT_MID_CS": 0.04,
        "SHOT_3_CS": 0.43,
        "PASS_EXTRA": 0.2,
        "PASS_SKIP": 0.105,
        "FOUL_DRAW_JUMPER": 0.022,
        "RESET_HUB": 0.078,
    },
    "ExtraPass": {
        "SHOT_MID_CS": 0.06,
        "SHOT_3_CS": 0.44,
        "PASS_EXTRA": 0.25,
        "PASS_SKIP": 0.18,
        "FOUL_DRAW_JUMPER": 0.02,
        "RESET_HUB": 0.215,
    },
    "PostUp": {
        "SHOT_RIM_CONTACT": 0.2,
        "SHOT_POST": 0.3,
        "PASS_EXTRA": 0.13,
        "PASS_KICKOUT": 0.175,
        "PASS_SKIP": 0.085,
        "FOUL_DRAW_POST": 0.15,
        "TO_HANDLE_LOSS": 0.02,
        "RESET_POST_OUT": 0.078,
    },
    "HornsSet": {
        "PASS_KICKOUT": 0.17,
        "SHOT_MID_CS": 0.06,
        "SHOT_3_CS": 0.16,
        "PASS_EXTRA": 0.2,
        "FOUL_DRAW_JUMPER": 0.01,
        "RESET_HUB": 0.345,
    },
    "SpotUp": {
        "SHOT_MID_CS": 0.13,
        "SHOT_3_CS": 0.56,
        "FOUL_DRAW_JUMPER": 0.02,
        "RESET_HUB": 0.189,
    },
    "Cut": {
        "SHOT_RIM_CONTACT": 0.09,
        "SHOT_RIM_DUNK": 0.1,
        "SHOT_RIM_LAYUP": 0.37,
        "SHOT_TOUCH_FLOATER": 0.16,
        "PASS_EXTRA": 0.06,
        "PASS_KICKOUT": 0.1,
        "FOUL_DRAW_RIM": 0.185,
        "TO_HANDLE_LOSS": 0.017,
        "RESET_HUB": 0.109,
    },
    "TransitionEarly": {
        "SHOT_RIM_CONTACT": 0.16,
        "SHOT_RIM_DUNK": 0.2,
        "SHOT_RIM_LAYUP": 0.28,
        "SHOT_TOUCH_FLOATER": 0.04,
        "SHOT_3_CS": 0.11,
        "SHOT_3_OD": 0.03,
        "PASS_EXTRA": 0.085,
        "PASS_KICKOUT": 0.145,
        "FOUL_DRAW_RIM": 0.115,
        "TO_HANDLE_LOSS": 0.022,
        "RESET_HUB": 0.085,
    },
}


ACTION_ALIASES = {
    "DragScreen": "PnR",
    "DoubleDrag": "PnR",
    "Rescreen": "PnR",
    "SideAnglePnR": "PnR",
    "SlipScreen": "PnR",
    "SpainPnR": "PnR",
    "ShortRollPlay": "PnR",
    "ZoomDHO": "DHO",
    "ReDHO_Handback": "DHO",
    "Chicago": "DHO",
    "Relocation": "SpotUp",
    "SkipPass": "ExtraPass",
    "Hammer": "Kickout",
    "PostEntry": "PostUp",
    "PostSplit": "Cut",
    "HighLow": "PostUp",
    "ElbowHub": "HornsSet",
    "OffBallScreen": "Cut",
    "ScreenTheScreener_STS": "Cut",
    "SecondaryBreak": "TransitionEarly",
    "QuickPost": "PostUp",
}


# -------------------------
# Distortion multipliers (schemes) - same as MVP v0
# -------------------------

OFFENSE_SCHEME_MULT: Dict[str, Dict[str, Dict[str, float]]] = {
    "Spread_HeavyPnR": {"PnR": {"PASS_SHORTROLL":1.10, "PASS_KICKOUT":1.05, "SHOT_3_OD":1.10, "SHOT_MID_PU":1.05, "RESET_RESREEN":1.05}},
    "Drive_Kick": {"Drive": {"PASS_KICKOUT":1.10, "PASS_EXTRA":1.15, "SHOT_RIM_LAYUP":1.05},
                   "Kickout": {"SHOT_3_CS":1.05, "PASS_EXTRA":1.08, "PASS_SKIP":1.05},
                   "ExtraPass": {"SHOT_3_CS":1.04, "PASS_SKIP":1.08}},
    "FiveOut": {"Drive": {"PASS_KICKOUT":1.10, "PASS_EXTRA":1.10, "SHOT_RIM_LAYUP":0.95},
                "Kickout": {"SHOT_3_CS":1.08, "PASS_SKIP":1.10},
                "ExtraPass": {"SHOT_3_CS":1.08, "PASS_SKIP":1.12},
                "Cut": {"SHOT_RIM_LAYUP":1.08, "RESET_HUB":0.95},
                "PostUp": {"SHOT_POST":0.80}},
    "Motion_SplitCut": {"Cut": {"SHOT_RIM_LAYUP":1.18, "PASS_KICKOUT":1.05, "RESET_HUB":0.95},
                        "ExtraPass": {"PASS_EXTRA":1.10, "SHOT_3_CS":1.05},
                        "DHO": {"RESET_REDO_DHO":0.95, "PASS_KICKOUT":1.05},
                        "PnR": {"SHOT_3_OD":0.90, "SHOT_MID_PU":0.95}},
    "DHO_Chicago": {"DHO": {"SHOT_3_OD":1.10, "SHOT_MID_PU":1.05, "RESET_REDO_DHO":0.95},
                    "Drive": {"SHOT_RIM_LAYUP":1.05}},
    "Post_InsideOut": {"PostUp": {"SHOT_POST":1.20, "PASS_KICKOUT":1.05, "FOUL_DRAW_POST":1.10, "RESET_POST_OUT":0.95},
                       "ExtraPass": {"SHOT_3_CS":1.05}},
    "Horns_Elbow": {"HornsSet": {"RESET_HUB":0.95, "PASS_EXTRA":1.05, "SHOT_MID_CS":1.10, "PASS_KICKOUT":1.05},
                    "PnR": {"PASS_SHORTROLL":1.05}},
    "Transition_Early": {"TransitionEarly": {"SHOT_RIM_DUNK":1.05, "SHOT_3_CS":0.98, "PASS_KICKOUT":0.95, "RESET_HUB":1.00, "FOUL_DRAW_RIM":1.05}},
}

DEFENSE_SCHEME_MULT: Dict[str, Dict[str, Dict[str, float]]] = {
    "Drop": {"PnR": {"SHOT_MID_PU":1.35, "SHOT_3_OD":1.15, "PASS_SHORTROLL":0.75, "SHOT_RIM_LAYUP":0.85, "SHOT_RIM_DUNK":0.85, "RESET_RESREEN":1.05},
             "Drive": {"SHOT_RIM_LAYUP":0.90}},
    "Switch_Everything": {"PnR": {"RESET_RESREEN":1.25, "TO_SHOT_CLOCK":1.15, "PASS_SHORTROLL":0.85, "SHOT_3_OD":1.10},
                          "DHO": {"RESET_REDO_DHO":1.15, "TO_HANDLE_LOSS":1.10},
                          "PostUp": {"SHOT_POST":1.35, "FOUL_DRAW_POST":1.20},
                          "Drive": {"TO_CHARGE":1.10}},
    "Hedge_ShowRecover": {"PnR": {"PASS_SHORTROLL":1.25, "PASS_KICKOUT":1.10, "RESET_RESREEN":1.10},
                          "Drive": {"SHOT_TOUCH_FLOATER":1.10}},
    "Blitz_TrapPnR": {"PnR": {"PASS_SHORTROLL":1.55, "PASS_KICKOUT":1.20, "SHOT_3_OD":0.75, "SHOT_MID_PU":0.75, "TO_BAD_PASS":1.35, "TO_HANDLE_LOSS":1.20, "FOUL_REACH_TRAP":1.20, "RESET_HUB":1.15},
                      "DHO": {"TO_BAD_PASS":1.20, "RESET_REDO_DHO":1.10},
                      "Drive": {"TO_HANDLE_LOSS":1.10}},
    "ICE_SidePnR": {"PnR": {"RESET_RESREEN":1.10, "PASS_KICKOUT":1.10, "SHOT_MID_PU":0.85, "SHOT_TOUCH_FLOATER":1.15}},
    "Zone": {"Drive": {"SHOT_RIM_LAYUP":0.75, "PASS_EXTRA":1.25, "PASS_SKIP":1.30, "SHOT_3_CS":1.15, "TO_BAD_PASS":1.10},
             "Kickout": {"PASS_EXTRA":1.15, "TO_BAD_PASS":1.08},
             "PostUp": {"SHOT_POST":0.85, "PASS_SKIP":1.15},
             "HornsSet": {"SHOT_MID_CS":1.15}},
    "PackLine_GapHelp": {"Drive": {"SHOT_RIM_LAYUP":0.65, "SHOT_RIM_DUNK":0.70, "PASS_KICKOUT":1.25, "PASS_EXTRA":1.20, "SHOT_3_CS":1.20, "TO_CHARGE":1.15},
                         "PnR": {"PASS_KICKOUT":1.15, "SHOT_MID_PU":1.05},
                         "ExtraPass": {"TO_BAD_PASS":1.05}},
}
