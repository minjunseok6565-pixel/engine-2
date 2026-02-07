from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any, Dict, TYPE_CHECKING

from ..core import clamp
from ..era import DEFAULT_PROB_MODEL
from ..models import Player, TeamState
from ..participants import (
    choose_orb_rebounder as _choose_orb_rebounder,
    choose_drb_rebounder as _choose_drb_rebounder,
)
from ..prob import prob_from_scores
from .resolve_context import _knob_mult

if TYPE_CHECKING:
    from ..game_config import GameConfig

def resolve_free_throws(
    rng: random.Random,
    shooter: Player,
    n: int,
    team: TeamState,
    game_cfg: "GameConfig",
) -> Dict[str, Any]:
    pm = game_cfg.prob_model if isinstance(game_cfg.prob_model, Mapping) else DEFAULT_PROB_MODEL
    ft = shooter.get("SHOT_FT")
    p = clamp(
        float(pm.get("ft_base", 0.45)) + (ft / 100.0) * float(pm.get("ft_range", 0.47)),
        float(pm.get("ft_min", 0.40)),
        float(pm.get("ft_max", 0.95)),
    )
    fta = 0
    ftm = 0
    last_made = False
    for _ in range(int(n)):
        team.fta += 1
        team.add_player_stat(shooter.pid, "FTA", 1)
        fta += 1
        made = rng.random() < p
        last_made = bool(made)
        if made:
            team.ftm += 1
            team.pts += 1
            team.add_player_stat(shooter.pid, "FTM", 1)
            team.add_player_stat(shooter.pid, "PTS", 1)
            ftm += 1
    return {"fta": fta, "ftm": ftm, "last_made": last_made, "p_ft": float(p)}

def rebound_orb_probability(
    offense: TeamState,
    defense: TeamState,
    orb_mult: float,
    drb_mult: float,
    game_cfg: "GameConfig",
) -> float:
    off_players = offense.on_court_players()
    def_players = defense.on_court_players()
    off_orb = sum(p.get("REB_OR") for p in off_players) / max(len(off_players), 1)
    def_drb = sum(p.get("REB_DR") for p in def_players) / max(len(def_players), 1)
    off_orb *= orb_mult
    def_drb *= drb_mult
    pm = game_cfg.prob_model if isinstance(game_cfg.prob_model, Mapping) else DEFAULT_PROB_MODEL
    base = float(pm.get("orb_base", 0.26)) * _knob_mult(game_cfg, "orb_base_mult", 1.0)
    return prob_from_scores(
        None,
        base,
        off_orb,
        def_drb,
        kind="rebound",
        variance_mult=1.0,
        game_cfg=game_cfg,
    )

def choose_orb_rebounder(rng: random.Random, offense: TeamState) -> Player:
    """Compatibility wrapper: rebounder selection lives in participants."""
    return _choose_orb_rebounder(rng, offense)


def choose_drb_rebounder(rng: random.Random, defense: TeamState) -> Player:
    """Compatibility wrapper: rebounder selection lives in participants."""
    return _choose_drb_rebounder(rng, defense)

