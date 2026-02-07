from __future__ import annotations

"""Quality-driven possession bias helpers."""

import math
from typing import Any, Dict

from .. import quality
from ..core import clamp
from ..def_role_players import get_or_build_def_role_players, engine_get_stat
from ..models import TeamState


def apply_quality_to_turnover_priors(
    pri: Dict[str, float],
    base_action: str,
    offense: TeamState,
    defense: TeamState,
    tags: Dict[str, Any],
    ctx: Dict[str, Any],
) -> Dict[str, float]:
    """Adjust TO_HANDLE_LOSS prior weight using quality-driven 'pressure'.

    quality.compute_quality_score returns an offense-perspective quality score:
      + higher => more open / better for offense
      - lower  => tougher / worse for offense (better defense)

    For turnovers, we want better defense => higher TO probability, so we invert:
        pressure = -quality_score

    We apply an exponential multiplier to pri['TO_HANDLE_LOSS']:

        pri['TO_HANDLE_LOSS'] *= exp(clamp(pressure * K_TO_QUALITY, -CLAMP, +CLAMP))

    Tuning knobs (defense.tactics.context):
      - K_TO_QUALITY (default 0.25)
      - TO_QUALITY_LOG_CLAMP (default 1.0)
    """
    if "TO_HANDLE_LOSS" not in pri:
        return pri

    scheme = getattr(defense.tactics, "defense_scheme", "")
    role_players = get_or_build_def_role_players(ctx, defense, scheme=scheme)

    debug_q = bool(ctx.get("debug_quality", False))
    q_res = quality.compute_quality_score(
        scheme=str(scheme),
        base_action=str(base_action),
        outcome="TO_HANDLE_LOSS",
        role_players=role_players,
        get_stat=engine_get_stat,
        return_detail=debug_q,
    )
    q_score = float(q_res.score) if (debug_q and hasattr(q_res, "score")) else float(q_res)

    pressure = -q_score

    tctx = getattr(defense.tactics, "context", {}) or {}
    k_to = float(tctx.get("K_TO_QUALITY", 0.25))
    log_clamp = float(tctx.get("TO_QUALITY_LOG_CLAMP", 1.0))
    log_mult = clamp(pressure * k_to, -log_clamp, log_clamp)

    pri["TO_HANDLE_LOSS"] = float(pri.get("TO_HANDLE_LOSS", 0.0)) * math.exp(log_mult)

    if debug_q:
        tags["to_quality_score"] = q_score
        tags["to_pressure"] = pressure
        tags["to_log_mult"] = log_mult
        tags["to_weight_after"] = float(pri["TO_HANDLE_LOSS"])

    return pri

