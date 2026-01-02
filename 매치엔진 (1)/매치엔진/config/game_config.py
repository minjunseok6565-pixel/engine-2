from __future__ import annotations

import copy
from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Mapping
from typing import Any


def _freeze_mapping(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze_mapping(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_mapping(v) for v in value)
    return value


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return MappingProxyType({})


@dataclass(frozen=True)
class GameConfig:
    era: Mapping[str, Any]
    knobs: Mapping[str, Any]
    prob_model: Mapping[str, Any]
    logistic_params: Mapping[str, Any]
    variance_params: Mapping[str, Any]
    role_fit: Mapping[str, Any]
    shot_base: Mapping[str, Any]
    pass_base_success: Mapping[str, Any]
    action_outcome_priors: Mapping[str, Any]
    action_aliases: Mapping[str, Any]
    off_scheme_action_weights: Mapping[str, Any]
    def_scheme_action_weights: Mapping[str, Any]
    offense_scheme_mult: Mapping[str, Any]
    defense_scheme_mult: Mapping[str, Any]


def build_game_config(era_cfg: Mapping[str, Any]) -> GameConfig:
    if not isinstance(era_cfg, Mapping):
        raise TypeError(f"build_game_config expected Mapping, got {type(era_cfg).__name__}")
    cfg_copy = copy.deepcopy(era_cfg)
    frozen = _freeze_mapping(cfg_copy)
    return GameConfig(
        era=frozen,
        knobs=_as_mapping(frozen.get("knobs", {})),
        prob_model=_as_mapping(frozen.get("prob_model", {})),
        logistic_params=_as_mapping(frozen.get("logistic_params", {})),
        variance_params=_as_mapping(frozen.get("variance_params", {})),
        role_fit=_as_mapping(frozen.get("role_fit", {})),
        shot_base=_as_mapping(frozen.get("shot_base", {})),
        pass_base_success=_as_mapping(frozen.get("pass_base_success", {})),
        action_outcome_priors=_as_mapping(frozen.get("action_outcome_priors", {})),
        action_aliases=_as_mapping(frozen.get("action_aliases", {})),
        off_scheme_action_weights=_as_mapping(frozen.get("off_scheme_action_weights", {})),
        def_scheme_action_weights=_as_mapping(frozen.get("def_scheme_action_weights", {})),
        offense_scheme_mult=_as_mapping(frozen.get("offense_scheme_mult", {})),
        defense_scheme_mult=_as_mapping(frozen.get("defense_scheme_mult", {})),
    )
