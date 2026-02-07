from __future__ import annotations

from typing import Dict



ACTION_ALIASES = {
    "DragScreen": "PnR",
    "DoubleDrag": "PnR",
    "Rescreen": "PnR",
    "SideAnglePnR": "PnR",
    "SlipScreen": "PnR",
    "SpainPnR": "PnR",
    "ShortRollPlay": "PnR",
    "PickAndPop": "PnP",
    "PopPnR": "PnP",
    "GhostPop": "PnP",
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
    "QuickShot": "SpotUp",
}
