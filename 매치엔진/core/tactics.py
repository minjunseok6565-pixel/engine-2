from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict


# -------------------------
# Tactics config
# -------------------------

@dataclass
class TacticsConfig:
    offense_scheme: str = "Spread_HeavyPnR"
    defense_scheme: str = "Drop"
    scheme_weight_sharpness: float = 1.00
    scheme_outcome_strength: float = 1.00
    def_scheme_weight_sharpness: float = 1.00
    def_scheme_outcome_strength: float = 1.00

    action_weight_mult: Dict[str, float] = field(default_factory=dict)
    outcome_global_mult: Dict[str, float] = field(default_factory=dict)
    outcome_by_action_mult: Dict[str, Dict[str, float]] = field(default_factory=dict)

    opp_action_weight_mult: Dict[str, float] = field(default_factory=dict)

    opp_outcome_global_mult: Dict[str, float] = field(default_factory=dict)
    opp_outcome_by_action_mult: Dict[str, Dict[str, float]] = field(default_factory=dict)

    context: Dict[str, Any] = field(default_factory=dict)


# -------------------------
# Defense scheme canonicalization
# -------------------------

# Internally, we want to use a *single* canonical scheme key everywhere.
# UI / save files may provide aliases (case differences, short names, Korean names, etc.).

_CANON_DEFENSE_SCHEMES = {
    "Drop",
    "Switch_Everything",
    "Hedge_ShowRecover",
    "Blitz_TrapPnR",
    "ICE_SidePnR",
    "Zone",
    "PackLine_GapHelp",
}

# Map a normalized alias string -> canonical defense scheme key.
# Normalization is done by: strip + lower + remove spaces/underscores/hyphens.
_DEFENSE_SCHEME_ALIAS_NORM = {
    # Drop
    "drop": "Drop",

    # Switch
    "switch": "Switch_Everything",
    "switcheverything": "Switch_Everything",
    "allswitch": "Switch_Everything",
    # common UI / config spellings normalize to 'allswitch'

    # Hedge / show-recover
    "hedge": "Hedge_ShowRecover",
    "hedgeshowrecover": "Hedge_ShowRecover",
    "showrecover": "Hedge_ShowRecover",
    "hedgerecover": "Hedge_ShowRecover",

    # Blitz / trap
    "blitz": "Blitz_TrapPnR",
    "trap": "Blitz_TrapPnR",
    "blitztrappnr": "Blitz_TrapPnR",
    "blitztrap": "Blitz_TrapPnR",

    # ICE side PnR
    "ice": "ICE_SidePnR",
    "icesidepnr": "ICE_SidePnR",

    # Zone
    "zone": "Zone",
    "matchupzone": "Zone",
    "23": "Zone",
    "23zone": "Zone",

    # Pack line
    "packline": "PackLine_GapHelp",
    "packlinegaphelp": "PackLine_GapHelp",
    "gaphelp": "PackLine_GapHelp",

    # Korean / localized labels used by UI / quality tables
    "올스위치": "Switch_Everything",         # '올-스위치'
    "아이스": "ICE_SidePnR",                 # '아이스 픽앤롤'
    "아이스픽앤롤": "ICE_SidePnR",
    "23존디펜스": "Zone",                   # '2-3 존디펜스'
    "23존": "Zone",
    "헷지쇼앤리커버": "Hedge_ShowRecover",   # '헷지-쇼앤리커버'
    "블리츠트랩": "Blitz_TrapPnR",           # '블리츠-트랩'
}


def canonical_defense_scheme(value: Any) -> str:
    """Return a canonical defense scheme key.

    - Returns one of _CANON_DEFENSE_SCHEMES when recognized.
    - Otherwise returns the original string (caller should validate/fallback).
    """
    if value is None:
        s = ""
    else:
        try:
            s = str(value)
        except Exception:
            s = ""

    s = s.strip()
    if s in _CANON_DEFENSE_SCHEMES:
        return s

    # Normalize: lowercase + remove spaces/underscores/hyphens
    key = re.sub(r"[\s_\-]+", "", s.lower())
    mapped = _DEFENSE_SCHEME_ALIAS_NORM.get(key)
    if mapped in _CANON_DEFENSE_SCHEMES:
        return mapped

    return s



