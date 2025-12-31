from __future__ import annotations

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

    def_action_weight_mult: Dict[str, float] = field(default_factory=dict)
    opp_action_weight_mult: Dict[str, float] = field(default_factory=dict)

    opp_outcome_global_mult: Dict[str, float] = field(default_factory=dict)
    opp_outcome_by_action_mult: Dict[str, Dict[str, float]] = field(default_factory=dict)

    context: Dict[str, Any] = field(default_factory=dict)


