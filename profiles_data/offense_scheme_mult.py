from __future__ import annotations

from typing import Dict



# -------------------------
# Distortion multipliers (schemes) - same as MVP v0
# -------------------------

OFFENSE_SCHEME_MULT: Dict[str, Dict[str, Dict[str, float]]] = {
    "Spread_HeavyPnR": {"PnR": {"PASS_SHORTROLL":1.10, "PASS_KICKOUT":1.05, "SHOT_3_OD":1.10, "SHOT_MID_PU":1.05, "RESET_RESREEN":1.05},
                      "PnP": {"SHOT_3_CS":1.12, "SHOT_MID_CS":1.06, "PASS_KICKOUT":1.05, "RESET_RESREEN":1.05, "SHOT_RIM_LAYUP":0.95},
                      "ISO": {"SHOT_3_OD":1.08, "SHOT_MID_PU":1.05, "RESET_HUB":1.05, "PASS_KICKOUT":0.97}},
    "Drive_Kick": {"Drive": {"PASS_KICKOUT":1.10, "PASS_EXTRA":1.15, "SHOT_RIM_LAYUP":1.05},
                   "ISO": {"PASS_KICKOUT":1.10, "PASS_EXTRA":1.06, "SHOT_RIM_LAYUP":1.05, "SHOT_MID_PU":0.90, "SHOT_3_OD":0.95},
                   "Kickout": {"SHOT_3_CS":1.05, "PASS_EXTRA":1.08, "PASS_SKIP":1.05},
                   "ExtraPass": {"SHOT_3_CS":1.04, "PASS_SKIP":1.08}},
    "FiveOut": {"Drive": {"PASS_KICKOUT":1.10, "PASS_EXTRA":1.10, "SHOT_RIM_LAYUP":0.95},
                "PnP": {"SHOT_3_CS":1.18, "PASS_KICKOUT":1.08, "SHOT_MID_CS":1.05, "SHOT_RIM_LAYUP":0.92},
                "ISO": {"SHOT_3_OD":1.10, "PASS_KICKOUT":1.12, "SHOT_RIM_LAYUP":1.03, "SHOT_MID_PU":0.92, "SHOT_RIM_CONTACT":0.95},
                "Kickout": {"SHOT_3_CS":1.08, "PASS_SKIP":1.10},
                "ExtraPass": {"SHOT_3_CS":1.08, "PASS_SKIP":1.12},
                "Cut": {"SHOT_RIM_LAYUP":1.08, "RESET_HUB":0.95},
                "PostUp": {"SHOT_POST":0.80}},
    "Motion_SplitCut": {"Cut": {"SHOT_RIM_LAYUP":1.18, "PASS_KICKOUT":1.05, "RESET_HUB":0.95},
                        "ISO": {"PASS_EXTRA":1.10, "RESET_HUB":1.10, "PASS_KICKOUT":1.05, "SHOT_RIM_LAYUP":0.95, "SHOT_MID_PU":0.92, "SHOT_3_OD":0.95},
                        "ExtraPass": {"PASS_EXTRA":1.10, "SHOT_3_CS":1.05},
                        "DHO": {"RESET_REDO_DHO":0.95, "PASS_KICKOUT":1.05},
                        "PnR": {"SHOT_3_OD":0.90, "SHOT_MID_PU":0.95}},
    "DHO_Chicago": {"DHO": {"SHOT_3_OD":1.10, "SHOT_MID_PU":1.05, "RESET_REDO_DHO":0.95},
                    "ISO": {"SHOT_3_OD":1.05, "SHOT_RIM_LAYUP":1.03, "PASS_KICKOUT":1.06, "SHOT_MID_PU":0.95},
                    "Drive": {"SHOT_RIM_LAYUP":1.05}},
    "Post_InsideOut": {"PostUp": {"SHOT_POST":1.20, "PASS_KICKOUT":1.05, "FOUL_DRAW_POST":1.10, "RESET_POST_OUT":0.95},
                       "ISO": {"SHOT_RIM_CONTACT":1.10, "SHOT_MID_PU":1.08, "SHOT_3_OD":0.85, "PASS_KICKOUT":1.05},
                       "ExtraPass": {"SHOT_3_CS":1.05}},
    "Horns_Elbow": {"HornsSet": {"RESET_HUB":0.95, "PASS_EXTRA":1.05, "SHOT_MID_CS":1.10, "PASS_KICKOUT":1.05},
                    "PnP": {"SHOT_MID_CS":1.10, "SHOT_3_CS":1.05, "PASS_EXTRA":1.05, "RESET_HUB":1.05},
                    "ISO": {"SHOT_MID_PU":1.05, "PASS_KICKOUT":1.05, "PASS_EXTRA":1.03, "SHOT_3_OD":0.97},
                    "PnR": {"PASS_SHORTROLL":1.05}},
    "Transition_Early": {"TransitionEarly": {"SHOT_RIM_DUNK":1.05, "SHOT_3_CS":0.98, "PASS_KICKOUT":0.95, "RESET_HUB":1.00, "FOUL_DRAW_RIM":1.05},
                        "ISO": {"SHOT_RIM_LAYUP":1.07, "PASS_KICKOUT":1.05, "SHOT_MID_PU":0.85, "SHOT_3_OD":0.95}},
}
