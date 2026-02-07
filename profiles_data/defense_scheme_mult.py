from __future__ import annotations

from typing import Dict


DEFENSE_SCHEME_MULT: Dict[str, Dict[str, Dict[str, float]]] = {
    "Drop": {"PnR": {"SHOT_MID_PU":1.35, "SHOT_3_OD":1.15, "PASS_SHORTROLL":0.75, "SHOT_RIM_LAYUP":0.85, "SHOT_RIM_DUNK":0.85, "RESET_RESREEN":1.05},
             "PnP": {"SHOT_3_CS": 1.12, "SHOT_MID_CS": 1.08, "SHOT_RIM_LAYUP": 0.90},
             "ISO": {"SHOT_RIM_LAYUP":0.90, "SHOT_RIM_DUNK":0.92, "SHOT_MID_PU":1.10, "SHOT_TOUCH_FLOATER":1.08},
             "Drive": {"SHOT_RIM_LAYUP":0.90}},
    "Switch_Everything": {"PnR": {"RESET_RESREEN":1.25, "TO_SHOT_CLOCK":1.15, "PASS_SHORTROLL":0.85, "SHOT_3_OD":1.10},
                          "PnP": {"SHOT_3_CS": 0.95, "TO_HANDLE_LOSS": 1.06, "RESET_RESREEN": 1.10},
                          "ISO": {"TO_HANDLE_LOSS":1.08, "SHOT_MID_PU":1.05, "SHOT_3_OD":1.03, "SHOT_RIM_LAYUP":0.95},
                          "DHO": {"RESET_REDO_DHO":1.15, "TO_HANDLE_LOSS":1.10},
                          "PostUp": {"SHOT_POST":1.35, "FOUL_DRAW_POST":1.20},
                          "Drive": {"TO_CHARGE":1.10}},
    "Hedge_ShowRecover": {"PnR": {"PASS_SHORTROLL":1.25, "PASS_KICKOUT":1.10, "RESET_RESREEN":1.10},
                          "PnP": {"SHOT_3_CS": 0.92, "PASS_KICKOUT": 1.05, "SHOT_RIM_LAYUP": 1.03},
                          "ISO": {"TO_HANDLE_LOSS":1.06, "PASS_KICKOUT":1.05, "SHOT_TOUCH_FLOATER":1.05},
                          "Drive": {"SHOT_TOUCH_FLOATER":1.10}},
    "Blitz_TrapPnR": {"PnR": {"PASS_SHORTROLL":1.55, "PASS_KICKOUT":1.20, "SHOT_3_OD":0.75, "SHOT_MID_PU":0.75, "TO_BAD_PASS":1.35, "TO_HANDLE_LOSS":1.20, "FOUL_REACH_TRAP":1.20, "RESET_HUB":1.15},
                      "PnP": {"PASS_KICKOUT": 1.15, "PASS_EXTRA": 1.08, "TO_HANDLE_LOSS": 1.15, "FOUL_REACH_TRAP": 1.15, "SHOT_3_CS": 0.92},
                      "ISO": {"PASS_KICKOUT":1.08, "PASS_EXTRA":1.05, "TO_HANDLE_LOSS":1.12, "SHOT_3_OD":0.92, "SHOT_MID_PU":0.92, "FOUL_REACH_TRAP":1.15, "RESET_HUB":1.05},
                      "DHO": {"TO_BAD_PASS":1.20, "RESET_REDO_DHO":1.10},
                      "Drive": {"TO_HANDLE_LOSS":1.10}},
    "ICE_SidePnR": {"PnR": {"RESET_RESREEN":1.10, "PASS_KICKOUT":1.10, "SHOT_MID_PU":0.85, "SHOT_TOUCH_FLOATER":1.15},
                    "PnP": {"SHOT_3_OD": 0.92, "SHOT_MID_CS": 1.03},
                    "ISO": {"SHOT_MID_PU":1.02, "SHOT_TOUCH_FLOATER":1.04}},
    "Zone": {"Drive": {"SHOT_RIM_LAYUP":0.75, "PASS_EXTRA":1.25, "PASS_SKIP":1.30, "SHOT_3_CS":1.15, "TO_BAD_PASS":1.10},
             "PnP": {"SHOT_3_CS": 1.08, "PASS_EXTRA": 1.12, "SHOT_RIM_LAYUP": 0.92},
             "ISO": {"PASS_KICKOUT":1.15, "PASS_EXTRA":1.10, "SHOT_3_OD":1.05, "SHOT_RIM_LAYUP":0.92, "SHOT_RIM_CONTACT":0.95},
             "Kickout": {"PASS_EXTRA":1.15, "TO_BAD_PASS":1.08},
             "PostUp": {"SHOT_POST":0.85, "PASS_SKIP":1.15},
             "HornsSet": {"SHOT_MID_CS":1.15}},
    "PackLine_GapHelp": {"Drive": {"SHOT_RIM_LAYUP":0.65, "SHOT_RIM_DUNK":0.70, "PASS_KICKOUT":1.25, "PASS_EXTRA":1.20, "SHOT_3_CS":1.20, "TO_CHARGE":1.15},
                         "ISO": {"SHOT_RIM_LAYUP":0.88, "SHOT_RIM_DUNK":0.90, "PASS_KICKOUT":1.12, "SHOT_MID_PU":1.06, "TO_CHARGE":1.08},
                         "PnR": {"PASS_KICKOUT":1.15, "SHOT_MID_PU":1.05},
                         "ExtraPass": {"TO_BAD_PASS":1.05}},
}
