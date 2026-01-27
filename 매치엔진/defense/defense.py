from __future__ import annotations

import math
from typing import Any, Dict, List

from .models import TeamState

# -------------------------
# Defense snapshot (no matchups MVP)
# -------------------------
#
# Historically this used max() for a few "anchor" defensive axes (POA/RIM/STEAL),
# which can let a single elite defender represent the whole lineup.
#
# This version aggregates those axes using a configurable, low-cost method:
#   - "topk": mean of the top K values (default K=2)
#   - "softmax": softmax-weighted mean (beta controls how "max-like" it is)
#
# Call sites do not change: resolve.py consumes the returned dict keys as before.

DEF_SNAPSHOT_METHOD = "topk"        # "topk" or "softmax"
DEF_SNAPSHOT_TOPK = 2              # used when method == "topk"
DEF_SNAPSHOT_SOFTMAX_BETA = 0.18   # used when method == "softmax"


def _safe_stat(p: Any, key: str, default: float = 50.0, fatigue_sensitive: bool = True) -> float:
    """
    Safe stat getter that supports BOTH:
      - engine Player.get(key, fatigue_sensitive=...)
      - dict-like get(key, default)

    This avoids the signature mismatch bug where p.get(key, default) would pass `default`
    into `fatigue_sensitive` on the engine Player.
    """
    try:
        # 엔진 Player.get(key, fatigue_sensitive=...)
        v = p.get(key, fatigue_sensitive=fatigue_sensitive)
    except TypeError:
        # dict-like get(key, default) fallback
        try:
            v = p.get(key, default)
        except Exception:
            v = default
    except Exception:
        v = default

    return float(default if v is None else v)


def _topk_mean(vals: List[float], k: int) -> float:
    if not vals:
        return 50.0
    kk = max(1, min(int(k), len(vals)))
    top = sorted(vals, reverse=True)[:kk]
    return sum(top) / float(len(top))


def _softmax_mean(vals: List[float], beta: float) -> float:
    if not vals:
        return 50.0
    b = float(beta)
    m = max(vals)
    exps = [math.exp(b * (v - m)) for v in vals]
    s = sum(exps)
    if s <= 0.0:
        return sum(vals) / float(len(vals))
    return sum(v * w for v, w in zip(vals, exps)) / s


def _agg_anchor(lineup, key: str) -> float:
    vals = [_safe_stat(p, key) for p in (lineup or [])]
    if DEF_SNAPSHOT_METHOD == "softmax":
        return _softmax_mean(vals, DEF_SNAPSHOT_SOFTMAX_BETA)
    # default: top-k mean
    return _topk_mean(vals, DEF_SNAPSHOT_TOPK)


def team_def_snapshot(team: TeamState) -> Dict[str, float]:
    lineup = getattr(team, "lineup", None) or []
    if not lineup:
        # Defensive-neutral fallback (should be rare, but avoids crashes).
        return {
            "DEF_POA": 50.0,
            "DEF_RIM": 50.0,
            "DEF_STEAL": 50.0,
            "DEF_HELP": 50.0,
            "DEF_POST": 50.0,
            "PHYSICAL": 50.0,
            "ENDURANCE": 50.0,
        }

    # Anchor axes: use top-2 mean (or softmax mean) instead of max()
    def_poa = _agg_anchor(lineup, "DEF_POA")
    def_rim = _agg_anchor(lineup, "DEF_RIM")
    def_steal = _agg_anchor(lineup, "DEF_STEAL")

    # Remaining axes: simple lineup mean (kept as-is)
    avg_keys = ["DEF_HELP", "PHYSICAL", "ENDURANCE", "DEF_POST"]
    avg = {k: sum(_safe_stat(p, k) for p in lineup) / float(len(lineup)) for k in avg_keys}

    return {
        "DEF_POA": def_poa,
        "DEF_RIM": def_rim,
        "DEF_STEAL": def_steal,
        "DEF_HELP": avg["DEF_HELP"],
        "DEF_POST": avg["DEF_POST"],
        "PHYSICAL": avg["PHYSICAL"],
        "ENDURANCE": avg["ENDURANCE"],
    }
