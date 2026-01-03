
"""
match_engine_mvp.py

Basketball tactics-driven match engine MVP
-----------------------------------------
End-to-end possession simulator using:
- offense scheme action weights + UI multipliers
- defense scheme action weights + UI multipliers (for intensity/logging)
- action -> outcome priors, distorted by offense + defense schemes and UI knobs
- outcome resolution using derived abilities (0~100)

MVP traits:
- no full shot-clock model (simple reset cap)
- no bonus/free throw rules (basic shooting foul + side-out trap foul)
- no lineup rotations (single lineup)
- lightweight fatigue (affects abilities slightly)

You can integrate by:
- feeding real Player derived stats + user-selected roles + tactics knobs
- running simulate_game() and consuming the returned dict
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math, random, json, hashlib, pickle, os, copy, warnings


ENGINE_VERSION: str = "mvp_plus_0.2"

def make_replay_token(rng: random.Random, home: 'TeamState', away: 'TeamState', era: str = "default") -> str:
    """Create a short stable token to reproduce/debug a game.

    Token is derived from: engine version, era, RNG state hash, rosters, roles, and tactics.
    """
    try:
        state_bytes = pickle.dumps(rng.getstate())
        rng_hash = hashlib.sha256(state_bytes).hexdigest()
    except Exception as exc:
        warnings.warn(f"make_replay_token: failed to hash RNG state ({type(exc).__name__}: {exc})")
        rng_hash = "no_state"

    def _player_payload(p: 'Player') -> Dict[str, Any]:
        # Keep it deterministic; derived is already 0~100 numbers
        return {
            'pid': p.pid,
            'pos': p.pos,
            'derived': p.derived,
        }

    def _tactics_payload(t: 'TacticsConfig') -> Dict[str, Any]:
        return {
            'offense_scheme': t.offense_scheme,
            'defense_scheme': t.defense_scheme,
            'scheme_weight_sharpness': t.scheme_weight_sharpness,
            'scheme_outcome_strength': t.scheme_outcome_strength,
            'def_scheme_weight_sharpness': t.def_scheme_weight_sharpness,
            'def_scheme_outcome_strength': t.def_scheme_outcome_strength,
            'action_weight_mult': t.action_weight_mult,
            'outcome_global_mult': t.outcome_global_mult,
            'outcome_by_action_mult': t.outcome_by_action_mult,
            'def_action_weight_mult': t.def_action_weight_mult,
            'opp_action_weight_mult': getattr(t, 'opp_action_weight_mult', {}),
            'opp_outcome_global_mult': t.opp_outcome_global_mult,
            'opp_outcome_by_action_mult': t.opp_outcome_by_action_mult,
            'context': t.context,
        }

    payload = {
        'engine_version': ENGINE_VERSION,
        'era': era,
        'rng_state_hash': rng_hash,
        'home': {
            'name': home.name,
            'roles': home.roles,
            'lineup': [_player_payload(p) for p in home.lineup],
            'tactics': _tactics_payload(home.tactics),
        },
        'away': {
            'name': away.name,
            'roles': away.roles,
            'lineup': [_player_payload(p) for p in away.lineup],
            'tactics': _tactics_payload(away.tactics),
        },
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()[:12]

# -------------------------
# Helpers
# -------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

def normalize_weights(d: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(v, 0.0) for v in d.values())
    if s <= 1e-12:
        n = len(d) if d else 1
        return {k: 1.0 / n for k in d} if d else {}
    return {k: max(v, 0.0) / s for k, v in d.items()}


def apply_temperature(weights: Dict[str, float], T: float) -> Dict[str, float]:
    if not weights:
        return {}
    exp = 1.0 / float(T) if float(T) != 0 else 1.0
    adj = {k: max(v, 0.0) ** exp for k, v in weights.items()}
    return normalize_weights(adj)


def apply_min_floor(probs: Dict[str, float], floor: float) -> Dict[str, float]:
    if not probs:
        return {}
    floored = {k: max(v, float(floor)) for k, v in probs.items()}
    return normalize_weights(floored)

def weighted_choice(rng: random.Random, weights: Dict[str, float]) -> str:
    total = sum(max(w, 0.0) for w in weights.values())
    if total <= 1e-12:
        return next(iter(weights.keys()))
    r = rng.random() * total
    upto = 0.0
    for k, w in weights.items():
        w = max(w, 0.0)
        upto += w
        if upto >= r:
            return k
    return next(iter(weights.keys()))

def dot_profile(vals: Dict[str, float], profile: Dict[str, float], missing_default: float = 50.0) -> float:
    s = 0.0
    for k, w in profile.items():
        s += float(vals.get(k, missing_default)) * float(w)
    return s

def apply_multipliers(base: Dict[str, float], mults: Dict[str, float]) -> Dict[str, float]:
    out = dict(base)
    for k, m in mults.items():
        if k in out:
            out[k] *= float(m)
    return out
