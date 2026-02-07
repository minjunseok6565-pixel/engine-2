from __future__ import annotations

import random
from typing import Dict, List, Optional

from ..core import weighted_choice
from ..models import Player, TeamState

from .participants_common import _active, _clamp

# -------------------------
# Steal / block credit assignment
# -------------------------

# These choosers decide *who* gets credited with a steal/block once the resolve
# layer has determined that the event occurred.
#
# Design goals:
# - Robust: always returns a reasonable on-court defender.
# - Role-agnostic: uses stats (and current fatigue scaling) rather than 12-role keys.
# - Tunable: tactics.context can override minimum weight and exponent.

_EVENT_ASSIGN_NORM_LO: float = 60.0
_EVENT_ASSIGN_NORM_HI: float = 100.0


def _event_assign_norm01(v: float) -> float:
    if _EVENT_ASSIGN_NORM_HI <= _EVENT_ASSIGN_NORM_LO:
        return 0.0
    return _clamp((float(v) - _EVENT_ASSIGN_NORM_LO) / (_EVENT_ASSIGN_NORM_HI - _EVENT_ASSIGN_NORM_LO), 0.0, 1.0)


def _estat(p: Player, key: str) -> float:
    # Use fatigue-sensitive values: steals/blocks are in-play defensive actions.
    return _event_assign_norm01(p.get(key, fatigue_sensitive=True))


def choose_stealer_pid(rng: random.Random, defense: TeamState) -> Optional[str]:
    """Choose a defender pid to be credited with a steal.

    Intended to be used after the resolve layer decides a turnover is a steal.

    Tuning via defense.tactics.context:
      - STEAL_ASSIGN_POWER (default 1.6)
      - STEAL_ASSIGN_W_MIN (default 0.05)
    """
    cand = list(_active(defense))
    if not cand:
        return None

    ctx = getattr(getattr(defense, 'tactics', None), 'context', {}) or {}
    power = float(ctx.get('STEAL_ASSIGN_POWER', 1.6))
    w_min = float(ctx.get('STEAL_ASSIGN_W_MIN', 0.05))
    power = max(power, 0.0)
    w_min = max(w_min, 0.0)

    weights: Dict[str, float] = {}
    for p in cand:
        steal = _estat(p, 'DEF_STEAL')
        poa = _estat(p, 'DEF_POA')
        help_ = _estat(p, 'DEF_HELP')
        w = 0.60 * steal + 0.25 * poa + 0.15 * help_
        weights[p.pid] = max(float(w), float(w_min)) ** float(power)

    if sum(max(w, 0.0) for w in weights.values()) <= 1e-12:
        return rng.choice([p.pid for p in cand])
    return weighted_choice(rng, weights)


def choose_blocker_pid(rng: random.Random, defense: TeamState, shot_kind: str) -> Optional[str]:
    """Choose a defender pid to be credited with a block.

    shot_kind should match resolve's shot-kind labels (e.g., 'shot_rim', 'shot_post',
    'shot_mid', 'shot_3'). Unknown kinds fall back to rim/post weighting.

    Tuning via defense.tactics.context:
      - BLOCK_ASSIGN_POWER (default 1.7)
      - BLOCK_ASSIGN_W_MIN (default 0.05)
    """
    cand = list(_active(defense))
    if not cand:
        return None

    k = str(shot_kind or '').lower()
    is_rim_like = (k in ('shot_rim', 'shot_post')) or ('rim' in k) or ('post' in k)

    ctx = getattr(getattr(defense, 'tactics', None), 'context', {}) or {}
    power = float(ctx.get('BLOCK_ASSIGN_POWER', 1.7))
    w_min = float(ctx.get('BLOCK_ASSIGN_W_MIN', 0.05))
    power = max(power, 0.0)
    w_min = max(w_min, 0.0)

    weights: Dict[str, float] = {}
    for p in cand:
        phys = _estat(p, 'PHYSICAL')
        help_ = _estat(p, 'DEF_HELP')
        if is_rim_like:
            rim = _estat(p, 'DEF_RIM')
            w = 0.70 * rim + 0.20 * phys + 0.10 * help_
        else:
            poa = _estat(p, 'DEF_POA')
            w = 0.55 * poa + 0.25 * phys + 0.20 * help_
        weights[p.pid] = max(float(w), float(w_min)) ** float(power)

    if sum(max(w, 0.0) for w in weights.values()) <= 1e-12:
        return rng.choice([p.pid for p in cand])
    return weighted_choice(rng, weights)
