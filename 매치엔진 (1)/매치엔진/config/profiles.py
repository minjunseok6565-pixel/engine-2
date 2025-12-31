"""
Lightweight wrapper around profiles_data.py.

This module intentionally contains **no big tables**.
All tuning tables live in `profiles_data.py` and are managed separately.

Data contract (in profiles_data.py):
  - OUTCOME_PROFILES: {outcome: {"offense": {stat_key: w}, "defense": {stat_key: w}}}
  - SHOT_BASE: {shot_outcome: base_success_prob}
  - CORNER3_PROB_BY_ACTION_BASE: {base_action: prob_corner3_given_3pt_attempt}
  - PASS_BASE_SUCCESS: {pass_outcome: base_success_prob}
  - OFF_SCHEME_ACTION_WEIGHTS: {off_scheme: {base_action: weight}}
  - DEF_SCHEME_ACTION_WEIGHTS: {def_scheme: {base_action: weight}}
  - ACTION_OUTCOME_PRIORS: {base_action: {outcome: prior_prob}}
  - ACTION_ALIASES: {concrete_action: base_action}
  - OFFENSE_SCHEME_MULT: {off_scheme: {base_action: {outcome: multiplier}}}
  - DEFENSE_SCHEME_MULT: {def_scheme: {base_action: {outcome: multiplier}}}

LLM workflow tip:
  - Provide `profiles.py` by default.
  - Only include `profiles_data.py` when you are actively tuning tables.
"""
from __future__ import annotations

try:
    # Package execution
    from .profiles_data import (  # noqa: F401
        OUTCOME_PROFILES,
        SHOT_BASE,
        CORNER3_PROB_BY_ACTION_BASE,
        PASS_BASE_SUCCESS,
        OFF_SCHEME_ACTION_WEIGHTS,
        DEF_SCHEME_ACTION_WEIGHTS,
        ACTION_OUTCOME_PRIORS,
        ACTION_ALIASES,
        OFFENSE_SCHEME_MULT,
        DEFENSE_SCHEME_MULT,
    )
except ImportError:  # pragma: no cover
    # Script / flat-module execution
    from profiles_data import (  # type: ignore  # noqa: F401
        OUTCOME_PROFILES,
        SHOT_BASE,
        CORNER3_PROB_BY_ACTION_BASE,
        PASS_BASE_SUCCESS,
        OFF_SCHEME_ACTION_WEIGHTS,
        DEF_SCHEME_ACTION_WEIGHTS,
        ACTION_OUTCOME_PRIORS,
        ACTION_ALIASES,
        OFFENSE_SCHEME_MULT,
        DEFENSE_SCHEME_MULT,
    )

__all__ = [
    "OUTCOME_PROFILES",
    "SHOT_BASE",
    "CORNER3_PROB_BY_ACTION_BASE",
    "PASS_BASE_SUCCESS",
    "OFF_SCHEME_ACTION_WEIGHTS",
    "DEF_SCHEME_ACTION_WEIGHTS",
    "ACTION_OUTCOME_PRIORS",
    "ACTION_ALIASES",
    "OFFENSE_SCHEME_MULT",
    "DEFENSE_SCHEME_MULT",
]
