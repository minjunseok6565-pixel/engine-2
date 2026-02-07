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
