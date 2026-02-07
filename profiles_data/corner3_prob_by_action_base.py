from __future__ import annotations

from typing import Dict


# Chance that a 3PA is a corner 3 instead of ATB 3 (keyed by *base* action)
CORNER3_PROB_BY_ACTION_BASE = {
    "default": 0.145,
    "Kickout": 0.23,
    "ExtraPass": 0.205,
    "SpotUp": 0.165,
    "TransitionEarly": 0.14,
    "PnR": 0.09,
    "PnP": 0.07,
    "DHO": 0.09,
    "ISO": 0.06,
}
