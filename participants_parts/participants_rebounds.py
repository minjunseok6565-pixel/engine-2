from __future__ import annotations

import math
import random
from typing import Dict, List, Optional

from ..core import weighted_choice
from ..models import Player, TeamState

from .participants_common import _active, _clamp

def _softmax_pick_player(
    rng: random.Random,
    players: List[Player],
    scores: List[float],
    beta: float,
    mix: float = 1.0,
) -> Optional[Player]:
    """Pick one player using softmax(exp(beta * score)).

    We use a numerically stable form by subtracting the max score before exponentiation.
    """
    if not players:
        return None
    if len(players) != len(scores) or not scores:
        return players[0]

    beta = max(float(beta), 0.0)
    mix = _clamp(float(mix), 0.0, 1.0)
    m = max(scores)
    base = 1.0 - mix
    weights = {p.pid: (base + mix * math.exp(beta * (float(s) - m))) for p, s in zip(players, scores)}
    pid = weighted_choice(rng, weights)
    for p in players:
        if p.pid == pid:
            return p
    return players[0]


# Rebound softmax expects a stable score scale. We normalize the raw rebound score
# (REB_OR/REB_DR + 0.20*PHYSICAL) into 0..1 before applying softmax.
_REB_NORM_LO: float = 75.0
_REB_NORM_HI: float = 120.0


def _reb_norm(v: float) -> float:
    if _REB_NORM_HI <= _REB_NORM_LO:
        return 0.0
    return _clamp((float(v) - _REB_NORM_LO) / (_REB_NORM_HI - _REB_NORM_LO), 0.0, 1.0)


def choose_orb_rebounder(rng: random.Random, offense: TeamState) -> Player:
    """Choose an offensive rebounder.

    Improvement 7-A: include all 5 on-court players and sample with softmax weights.
    This keeps bigs favored but allows guards/wings to occasionally grab long rebounds.

    Tuning: tactics.context['ORB_SOFTMAX_BETA'] (default 3.2)
    """
    cand = list(_active(offense))
    beta = float(getattr(getattr(offense, 'tactics', None), 'context', {}).get('ORB_SOFTMAX_BETA', 3.2))
    raw_scores = [p.get('REB_OR') + 0.20 * p.get('PHYSICAL') for p in cand]
    scores = [_reb_norm(s) for s in raw_scores]
    return _softmax_pick_player(rng, cand, scores, beta) or cand[0]


def choose_drb_rebounder(rng: random.Random, defense: TeamState) -> Player:
    """Choose a defensive rebounder.

    Improvement 7-A: include all 5 on-court players and sample with softmax weights.

    Tuning: tactics.context['DRB_SOFTMAX_BETA'] (default 2.2), tactics.context['DRB_SOFTMAX_MIX'] (default 0.92)

    DRB_SOFTMAX_MIX blends uniform weights with softmax to reduce extreme top-1 dominance.
    mix=1.0 is pure softmax; mix=0.0 is uniform.
    """
    cand = list(_active(defense))
    ctx = getattr(getattr(defense, 'tactics', None), 'context', {}) or {}
    beta = float(ctx.get('DRB_SOFTMAX_BETA', 2.2))
    mix = float(ctx.get('DRB_SOFTMAX_MIX', 0.92))
    raw_scores = [p.get('REB_DR') + 0.20 * p.get('PHYSICAL') for p in cand]
    scores = [_reb_norm(s) for s in raw_scores]
    return _softmax_pick_player(rng, cand, scores, beta) or cand[0]
