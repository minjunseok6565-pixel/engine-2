"""
Data tables for quality.py.

This package is auto-split from the original monolithic quality_data.py.
It re-exports the same public tables, so importing code can stay simple.
"""

from __future__ import annotations

from .scheme_base_outcome_labels import SCHEME_BASE_OUTCOME_LABELS
from .role_stat_profiles import ROLE_STAT_PROFILES
from .group_scheme_role_weights import GROUP_SCHEME_ROLE_WEIGHTS
from .outcome_to_group import OUTCOME_TO_GROUP
from .group_fallback import GROUP_FALLBACK
from .scheme_aliases import SCHEME_ALIASES

__all__ = [
    "SCHEME_BASE_OUTCOME_LABELS",
    "ROLE_STAT_PROFILES",
    "GROUP_SCHEME_ROLE_WEIGHTS",
    "OUTCOME_TO_GROUP",
    "GROUP_FALLBACK",
    "SCHEME_ALIASES",
]
