from __future__ import annotations

"""Outcome priors bias helpers."""

from typing import Any, Dict

from ..core import clamp


def apply_help_to_priors(priors: Dict[str, float], ctx: Dict[str, Any]) -> Dict[str, float]:
    """Apply a small help-defense tradeoff to outcome priors (possession scoped).

    Uses (preferred) ctx['def_pressure']['help']['eff_priors'] in [-1, +1],
    falling back to ctx['team_help_level'] (legacy).

      +1 => strong help: more kickouts/skip/TO bad pass, fewer rim/post
      -1 => weak help: fewer kickouts/skip/TO bad pass, more rim/post
    """
    if not priors:
        return priors

    # Preferred SSOT: ctx['def_pressure']['help']['eff_priors']
    h = 0.0
    leave_cost_norm = 0.0
    try:
        dp = ctx.get("def_pressure") if isinstance(ctx.get("def_pressure"), dict) else {}
        hp = dp.get("help") if isinstance(dp.get("help"), dict) else {}
        if "eff_priors" in hp:
            h = float(hp.get("eff_priors", 0.0) or 0.0)
            leave_cost_norm = float(hp.get("leave_cost_norm", 0.0) or 0.0)
        else:
            h = float(ctx.get("team_help_level", 0.0) or 0.0)
    except Exception:
        h = 0.0
        leave_cost_norm = 0.0

    h = clamp(float(h), -1.0, 1.0)
    leave_cost_norm = clamp(float(leave_cost_norm), -1.0, 1.0)
    if abs(h) < 1e-9:
        return priors

    rim_mult = clamp(1.0 - 0.10 * h, 0.75, 1.25)
    post_mult = clamp(1.0 - 0.08 * h, 0.75, 1.25)
    # Leaving a strong shooter while helping should amplify catch&shoot 3 opportunity.
    c3_mult = clamp(1.0 + 0.10 * h + 0.06 * h * leave_cost_norm, 0.75, 1.35)
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


def apply_double_to_priors(priors: Dict[str, float], ctx: Dict[str, Any]) -> Dict[str, float]:
    """Apply an on-ball double-team pressure bias to outcome priors.

    Preferred SSOT: ctx['def_pressure']['double'].
    When active, doubles:
      - reduce on-ball shot attempts
      - increase escape passes
      - increase bad-pass / handle-loss turnover risk
      - in PnR/PnP/DHO: increase short-roll outcomes
    """
    if not priors:
        return priors

    s = 0.0
    base_action = ""
    try:
        dp = ctx.get("def_pressure") if isinstance(ctx.get("def_pressure"), dict) else {}
        base_action = str(dp.get("step_base_action", "") or "")
        dbl = dp.get("double") if isinstance(dp.get("double"), dict) else {}
        if bool(dbl.get("active", False)):
            s = float(dbl.get("strength", 0.0) or 0.0)
    except Exception:
        s = 0.0
        base_action = ""

    s = clamp(float(s), 0.0, 1.0)
    if s < 1e-9:
        return priors

    shot_mult = clamp(1.0 - 0.25 * s, 0.60, 1.10)
    pass_mult = clamp(1.0 + 0.30 * s, 0.80, 1.45)
    to_mult = clamp(1.0 + 0.18 * s, 0.75, 1.35)
    shortroll_mult = clamp(1.0 + 0.35 * s, 0.80, 1.60)
    is_pnr_like = base_action in ("PnR", "PnP", "DHO")

    out = dict(priors)
    for k, v in list(out.items()):
        vv = float(v)

        # On-ball shots are discouraged.
        if (
            k.startswith("SHOT_RIM_")
            or k == "SHOT_POST"
            or k.startswith("SHOT_MID_")
            or k == "SHOT_TOUCH_FLOATER"
            or k == "SHOT_3_OD"
        ):
            vv *= shot_mult

        # Escape passing is encouraged.
        if k.startswith("PASS_"):
            vv *= pass_mult
            if is_pnr_like and k == "PASS_SHORTROLL":
                vv *= shortroll_mult

        # Turnover risk rises under pressure.
        if k in ("TO_BAD_PASS", "TO_HANDLE_LOSS"):
            vv *= to_mult

        out[k] = vv

    ssum = sum(float(x) for x in out.values())
    if ssum <= 0:
        return priors
    for k in out:
        out[k] = float(out[k]) / ssum
    return out


def apply_rotation_advantage_to_priors(priors: Dict[str, float], ctx: Dict[str, Any]) -> Dict[str, float]:
    """Apply a 4v3 / rotation advantage bias to priors for exactly one step (ttl=1).

    Reads ctx['rotation_adv'] = {"ttl": int, "adv": 0..1, "source": ...}
    Decrements ttl and removes when expired.
    """
    if not priors:
        return priors

    rot = ctx.get("rotation_adv")
    if not isinstance(rot, dict):
        return priors

    try:
        ttl = int(rot.get("ttl", 0) or 0)
    except Exception:
        ttl = 0
    if ttl <= 0:
        ctx.pop("rotation_adv", None)
        return priors

    try:
        adv = float(rot.get("adv", 0.0) or 0.0)
    except Exception:
        adv = 0.0
    adv = clamp(float(adv), 0.0, 1.0)

    c3_mult = clamp(1.0 + 0.18 * adv, 0.85, 1.40)
    dunk_mult = clamp(1.0 + 0.10 * adv, 0.85, 1.30)
    lay_mult = clamp(1.0 + 0.06 * adv, 0.85, 1.25)
    extra_mult = clamp(1.0 + 0.10 * adv, 0.85, 1.30)
    badpass_mult = clamp(1.0 - 0.10 * adv, 0.70, 1.05)

    out = dict(priors)
    for k, v in list(out.items()):
        vv = float(v)
        if k == "SHOT_3_CS":
            vv *= c3_mult
        elif k == "SHOT_RIM_DUNK":
            vv *= dunk_mult
        elif k in ("SHOT_RIM_LAYUP", "SHOT_RIM_CONTACT"):
            vv *= lay_mult
        elif k == "PASS_EXTRA":
            vv *= extra_mult
        elif k == "TO_BAD_PASS":
            vv *= badpass_mult
        out[k] = vv

    ssum = sum(float(x) for x in out.values())
    if ssum <= 0:
        return priors
    for k in out:
        out[k] = float(out[k]) / ssum

    # Consume one step of advantage.
    ttl -= 1
    if ttl <= 0:
        ctx.pop("rotation_adv", None)
    else:
        rot["ttl"] = ttl
        ctx["rotation_adv"] = rot

    return out
