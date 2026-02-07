from __future__ import annotations

"""Outcome priors bias helpers."""

from typing import Any, Dict

from ..core import clamp


def apply_help_to_priors(priors: Dict[str, float], ctx: Dict[str, Any]) -> Dict[str, float]:
    """Apply a small help-defense tradeoff to outcome priors (possession scoped).

    Uses ctx['team_help_level'] in [-1, +1]:
      +1 => strong help: more kickouts/skip/TO bad pass, fewer rim/post
      -1 => weak help: fewer kickouts/skip/TO bad pass, more rim/post
    """
    if not priors:
        return priors
    try:
        h = float(ctx.get("team_help_level", 0.0))
    except Exception:
        h = 0.0
    h = clamp(h, -1.0, 1.0)
    if abs(h) < 1e-9:
        return priors

    rim_mult = clamp(1.0 - 0.10 * h, 0.75, 1.25)
    post_mult = clamp(1.0 - 0.08 * h, 0.75, 1.25)
    c3_mult = clamp(1.0 + 0.10 * h, 0.75, 1.25)
    kick_mult = clamp(1.0 + 0.12 * h, 0.75, 1.25)
    badpass_mult = clamp(1.0 + 0.06 * h, 0.75, 1.25)

    out = dict(priors)
    for k, v in list(out.items()):
        vv = float(v)
        if k.startswith("SHOT_RIM_") or k == "SHOT_TOUCH_FLOATER":
            vv *= rim_mult
        elif k == "SHOT_POST":
            vv *= post_mult
        elif k == "SHOT_3_CS":
            vv *= c3_mult
        elif k in ("PASS_KICKOUT", "PASS_SKIP"):
            vv *= kick_mult
        elif k == "TO_BAD_PASS":
            vv *= badpass_mult
        out[k] = vv

    s = sum(float(x) for x in out.values())
    if s <= 0:
        return priors
    for k in out:
        out[k] = float(out[k]) / s
    return out
