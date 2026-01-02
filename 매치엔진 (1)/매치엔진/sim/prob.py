from __future__ import annotations

import math
import random
from collections.abc import Mapping
from typing import Dict, Optional, TYPE_CHECKING

from .core import clamp, sigmoid
from .era import (
    DEFAULT_LOGISTIC_PARAMS,
    DEFAULT_PROB_MODEL,
    DEFAULT_VARIANCE_PARAMS,
)

if TYPE_CHECKING:
    from config.game_config import GameConfig

# -------------------------
# Probability model
# -------------------------

def prob_from_scores(
    rng: Optional[random.Random],
    base_p: float,
    off_score: float,
    def_score: float,
    *,
    game_cfg: "GameConfig",
    kind: str = "default",
    variance_mult: float = 1.0,
    logit_delta: float = 0.0,
    fatigue_logit_delta: float = 0.0,
) -> float:
    """Convert an OffScore/DefScore matchup into a probability using a logistic model.

    Model:
      p = sigmoid( logit(base_p) + (off_score - def_score) * sensitivity + noise )

    - base_p: the baseline probability for this outcome type (e.g., 3PT base%, pass base%).
    - sensitivity: per-kind slope, externalized in the era file (logistic_params).
    - noise: optional variance knob (2-3). Uses logit-space Gaussian noise so the mean stays stable.
    """
    if game_cfg is None:
        raise ValueError("prob_from_scores requires game_cfg")
    pm = game_cfg.prob_model if isinstance(game_cfg.prob_model, Mapping) else DEFAULT_PROB_MODEL
    base_p = clamp(float(base_p), float(pm.get("base_p_min", 0.02)), float(pm.get("base_p_max", 0.98)))
    base_logit = math.log(base_p / (1.0 - base_p))

    # ---- sensitivity (2-1, 2-2) ----
    lp = game_cfg.logistic_params if isinstance(game_cfg.logistic_params, Mapping) else DEFAULT_LOGISTIC_PARAMS
    spec = lp.get(kind) or lp.get("default") or {}
    sens = spec.get("sensitivity")
    scale = spec.get("scale")

    # Back-compat fallback (older era json without logistic_params)
    if sens is None:
        if scale is not None and float(scale) > 1e-9:
            sens = 1.0 / float(scale)
        else:
            # old single-scale knobs
            if kind.startswith("pass"):
                sens = 1.0 / float(pm.get("pass_scale", 20.0))
            elif kind.startswith("rebound"):
                sens = 1.0 / float(pm.get("rebound_scale", 22.0))
            else:
                sens = 1.0 / float(pm.get("shot_scale", 18.0))

    gap = (float(off_score) - float(def_score)) * float(sens)

    # ---- variance knob (2-3) ----
    noise = 0.0
    if rng is not None:
        vp = game_cfg.variance_params if isinstance(game_cfg.variance_params, Mapping) else DEFAULT_VARIANCE_PARAMS
        std = float(vp.get("logit_noise_std", 0.0))
        kind_mult = float((vp.get("kind_mult") or {}).get(kind, 1.0)) if isinstance(vp.get("kind_mult"), Mapping) else 1.0
        # team volatility multiplier (clamped)
        tlo, thi = 0.70, 1.40
        if isinstance(vp.get("team_mult_lo"), (int, float)):
            tlo = float(vp["team_mult_lo"])
        if isinstance(vp.get("team_mult_hi"), (int, float)):
            thi = float(vp["team_mult_hi"])
        vm = clamp(float(variance_mult), tlo, thi)
        std = std * kind_mult * vm
        if std > 1e-9:
            noise = rng.gauss(0.0, std)

    p = sigmoid(base_logit + gap + noise + float(logit_delta) + float(fatigue_logit_delta))
    return clamp(p, float(pm.get("prob_min", 0.03)), float(pm.get("prob_max", 0.97)))


def _shot_kind_from_outcome(outcome: str) -> str:
    # 2-2 categories
    if outcome in ("SHOT_3_CS", "SHOT_3_OD"):
        return "shot_3"
    if outcome in ("SHOT_MID_CS", "SHOT_MID_PU", "SHOT_TOUCH_FLOATER"):
        return "shot_mid"
    if outcome == "SHOT_POST":
        return "shot_post"
    return "shot_rim"

def _team_variance_mult(team: "TeamState", game_cfg: "GameConfig") -> float:
    vp = game_cfg.variance_params if isinstance(game_cfg.variance_params, Mapping) else DEFAULT_VARIANCE_PARAMS
    try:
        vm = float((team.tactics.context or {}).get("VARIANCE_MULT", 1.0))
    except Exception:
        vm = 1.0
    lo = float(vp.get("team_mult_lo", 0.70))
    hi = float(vp.get("team_mult_hi", 1.40))
    return clamp(vm, lo, hi)
