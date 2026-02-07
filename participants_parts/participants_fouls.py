from __future__ import annotations

import math
import random
from typing import Dict, Optional, Sequence

from ..core import weighted_choice
from ..models import Player, TeamState

from .participants_common import _clamp

# -------------------------
# Foul assignment (Improvement 6-A, 6-B)
# -------------------------

_FOUL_NORM_LO = 60.0
_FOUL_NORM_HI = 100.0


def _norm01(v: float, lo: float = _FOUL_NORM_LO, hi: float = _FOUL_NORM_HI) -> float:
    if hi <= lo:
        return 0.0
    return _clamp((float(v) - float(lo)) / (float(hi) - float(lo)), 0.0, 1.0)


def _nstat(p: Player, key: str) -> float:
    # Use fatigue-insensitive stats for foul tendency; fatigue is modeled separately.
    return _norm01(p.get(key, fatigue_sensitive=False))


def _foul_tendency_score(p: Player, outcome: Optional[str]) -> float:
    """Return a 0..~1.45 score expressing 'how likely this defender commits this foul type'.

    We model: involvement * (0.55 + 0.9 * mistake)
      - involvement: who is usually involved for this foul type (role proxy)
      - mistake: undisciplined / tired / late to contest -> more likely to foul

    Outcome types seen in engine:
      - FOUL_REACH_TRAP
      - FOUL_DRAW_JUMPER
      - FOUL_DRAW_POST
      - FOUL_DRAW_RIM
    """
    phys = _nstat(p, 'PHYSICAL')
    poa = _nstat(p, 'DEF_POA')
    steal = _nstat(p, 'DEF_STEAL')
    rim = _nstat(p, 'DEF_RIM')
    post = _nstat(p, 'DEF_POST')
    help_ = _nstat(p, 'DEF_HELP')

    # DISCIPLINE isn't in the roster schema; use safety/IQ proxies.
    disc = 0.5 * _nstat(p, 'PASS_SAFE') + 0.5 * _nstat(p, 'HANDLE_SAFE')
    undisc = 1.0 - disc

    # Fatigue increases foul likelihood (separate from fatigue-sensitive stats).
    fat = _clamp(1.0 - float(getattr(p, 'energy', 1.0)), 0.0, 1.0)

    if outcome == 'FOUL_REACH_TRAP':
        inv = 0.55 * steal + 0.30 * poa + 0.15 * help_
        mist = 0.45 * undisc + 0.35 * fat + 0.20 * (1.0 - poa)
    elif outcome == 'FOUL_DRAW_JUMPER':
        inv = 0.70 * poa + 0.30 * help_
        mist = 0.45 * undisc + 0.35 * fat + 0.20 * (1.0 - poa)
    elif outcome == 'FOUL_DRAW_RIM':
        inv = 0.65 * rim + 0.20 * help_ + 0.15 * phys
        mist = 0.40 * undisc + 0.35 * fat + 0.25 * (1.0 - rim)
    elif outcome == 'FOUL_DRAW_POST':
        inv = 0.60 * post + 0.25 * phys + 0.15 * help_
        mist = 0.40 * undisc + 0.35 * fat + 0.25 * (1.0 - post)
    else:
        # Generic fallback: balanced involvement + standard mistake model.
        inv = 0.35 * poa + 0.25 * help_ + 0.20 * rim + 0.20 * post
        mist = 0.45 * undisc + 0.35 * fat + 0.20 * (1.0 - poa)

    return float(inv) * (0.55 + 0.90 * float(mist))


def choose_fouler_pid(
    rng: random.Random,
    defense: TeamState,
    def_on_court: Sequence[str],
    player_fouls: Dict[str, int],
    foul_out_limit: int,
    outcome: Optional[str] = None,
) -> Optional[str]:
    """Choose a defender pid to be credited with a foul.

    Improvements:
      - 6-A: weighted selection based on foul tendency proxies (stats + fatigue)
      - 6-B: dynamic foul-trouble penalty so high-foul players are less likely to be assigned

    Notes:
      - Excludes players already at/over foul-out limit when possible.
      - Does NOT mutate player_fouls; resolve layer remains responsible for bookkeeping.
      - Tuning can be overridden via defense.tactics.context:
          FOUL_WEIGHT_ALPHA (default 2.0)
          FOUL_WEIGHT_MIN (default 0.05)
          FOUL_TROUBLE_FREE (default 2)
          FOUL_TROUBLE_K (default 0.60)
          FOUL_TROUBLE_MIN_MULT (default 0.12)
    """
    cands = [pid for pid in (def_on_court or []) if isinstance(pid, str) and pid]
    if not cands:
        return None

    eligible = [pid for pid in cands if int(player_fouls.get(pid, 0)) < int(foul_out_limit)]
    if not eligible:
        eligible = cands

    ctx = getattr(getattr(defense, 'tactics', None), 'context', {}) or {}
    alpha = float(ctx.get('FOUL_WEIGHT_ALPHA', 2.0))
    w_min = float(ctx.get('FOUL_WEIGHT_MIN', 0.05))

    free_fouls = int(ctx.get('FOUL_TROUBLE_FREE', 2))
    k = float(ctx.get('FOUL_TROUBLE_K', 0.60))
    min_mult = float(ctx.get('FOUL_TROUBLE_MIN_MULT', 0.12))

    weights: Dict[str, float] = {}
    for pid in eligible:
        p = defense.find_player(pid)
        if p is None:
            weights[pid] = 1.0
            continue

        score = _foul_tendency_score(p, outcome)
        w_base = math.exp(alpha * (float(score) - 0.5))
        w_base = max(float(w_base), float(w_min))

        f = int(player_fouls.get(pid, 0))
        f_adj = max(0, f - int(free_fouls))
        trouble_mult = max(float(min_mult), math.exp(-float(k) * float(f_adj)))

        weights[pid] = w_base * float(trouble_mult)

    # Fallback: if something degenerated, keep legacy uniform behavior.
    if sum(max(w, 0.0) for w in weights.values()) <= 1e-12:
        return rng.choice(list(eligible))

    return weighted_choice(rng, weights)
