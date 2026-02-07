from __future__ import annotations

from typing import Dict




# -------------------------
# Scheme action weights
# -------------------------

OFF_SCHEME_ACTION_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Spread_HeavyPnR": {
        "PnR": 25,
        "PnP": 9,
        "TransitionEarly": 6,
        "Drive": 7,
        "ISO": 6,
        "Kickout": 8,
        "ExtraPass": 6,
        "SpotUp": 7,
        "Cut": 4,
    },
    "Drive_Kick": {
        "Drive": 30,
        "ISO": 7,
        "Kickout": 18,
        "ExtraPass": 12,
        "SpotUp": 12,
        "Cut": 6,
        "PnR": 3,
        "DHO": 2,
        "PnP": 3,
    },
    "FiveOut": {
        "Drive": 18,
        "ISO": 9,
        "SpotUp": 16,
        "Kickout": 14,
        "ExtraPass": 10,
        "Cut": 10,
        "DHO": 8,
        "PnR": 5,
        "PnP": 10,
    },
    "Motion_SplitCut": {
        "Cut": 18,
        "ISO": 4,
        "DHO": 8,
        "Drive": 10,
        "Kickout": 6,
        "ExtraPass": 6,
        "SpotUp": 6,
        "PnR": 4,
        "PnP": 3,
    },
    "DHO_Chicago": {
        "DHO": 16,
        "ISO": 5,
        "Drive": 12,
        "Kickout": 10,
        "ExtraPass": 6,
        "SpotUp": 10,
        "PnR": 6,
        "PnP": 4,
    },
    "Post_InsideOut": {
        "PostUp": 22,
        "ISO": 5,
        "Kickout": 14,
        "ExtraPass": 8,
        "SpotUp": 12,
        "Cut": 8,
        "Drive": 4,
        "DHO": 4,
        "PnP": 3,
    },
    "Horns_Elbow": {
        "HornsSet": 18,
        "ISO": 5,
        "PnR": 12,
        "PnP": 6,
        "DHO": 8,
        "Drive": 10,
        "Kickout": 8,
        "ExtraPass": 6,
        "SpotUp": 8,
        "Cut": 6,
    },
    "Transition_Early": {
        "TransitionEarly": 40,
        "ISO": 2,
        "Drive": 8,
        "Kickout": 8,
        "SpotUp": 8,
        "PnP": 1,
    },
}
