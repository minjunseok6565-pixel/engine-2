"""Split data tables from profiles_data.py (auto-packaged)."""
from __future__ import annotations

from .outcome_profiles import OUTCOME_PROFILES
from .shot_base import SHOT_BASE
from .corner3_prob_by_action_base import CORNER3_PROB_BY_ACTION_BASE
from .pass_base_success import PASS_BASE_SUCCESS
from .off_scheme_action_weights import OFF_SCHEME_ACTION_WEIGHTS
from .action_outcome_priors import ACTION_OUTCOME_PRIORS
from .action_aliases import ACTION_ALIASES
from .offense_scheme_mult import OFFENSE_SCHEME_MULT
from .defense_scheme_mult import DEFENSE_SCHEME_MULT

__all__ = [
    'OUTCOME_PROFILES',
    'SHOT_BASE',
    'CORNER3_PROB_BY_ACTION_BASE',
    'PASS_BASE_SUCCESS',
    'OFF_SCHEME_ACTION_WEIGHTS',
    'ACTION_OUTCOME_PRIORS',
    'ACTION_ALIASES',
    'OFFENSE_SCHEME_MULT',
    'DEFENSE_SCHEME_MULT',
]
