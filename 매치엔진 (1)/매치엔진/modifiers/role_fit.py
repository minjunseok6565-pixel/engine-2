# role_fit.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING


# If your project has concrete Player / TeamState classes, you can type-import them here.
# This file keeps runtime-safe fallbacks so it won't crash if imported standalone.
if TYPE_CHECKING:
    from typing import Protocol
    from config.game_config import GameConfig

    class Player(Protocol):
        def get(self, key: str) -> Any: ...

    class TeamState(Protocol):
        roles: Dict[str, Any]
        tactics: Any
        role_fit_pos_log: List[Dict[str, Any]]
        role_fit_grade_counts: Dict[str, int]
        role_fit_role_counts: Dict[str, int]

        def find_player(self, pid: Any) -> Optional[Player]: ...

else:
    Player = Any
    TeamState = Any


# -----------------------------
# Helpers
# -----------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    try:
        xf = float(x)
    except Exception:
        xf = lo
    if xf < lo:
        return lo
    if xf > hi:
        return hi
    return xf


def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    """Normalize dict values to sum to 1.0 (if possible)."""
    s = 0.0
    for v in w.values():
        try:
            s += float(v)
        except Exception:
            pass
    if s <= 1e-12:
        return w
    out: Dict[str, float] = {}
    for k, v in w.items():
        try:
            out[k] = float(v) / s
        except Exception:
            out[k] = 0.0
    return out


# -----------------------------
# NOTE ON DATA/LOGIC SPLIT
# -----------------------------
# This module is logic-focused. Large tuning tables are kept in `role_fit_data.py` and managed separately.
#
# Data contract (role_fit_data.py):
#   - ROLE_FIT_WEIGHTS: {role_name: {stat_key: weight}}
#       * fit score = sum(player.get(stat_key) * weight), then clamped to [0, 100]
#   - ROLE_FIT_CUTS: {role_name: (S_min, A_min, B_min, C_min)}
#       * if missing, a small default threshold set is used (see role_fit_grade)
#   - ROLE_PRIOR_MULT_RAW: {grade: {"GOOD": mult, "BAD": mult}}
#       * applied to priors for outcomes categorized as GOOD/BAD for the possession step
#   - ROLE_LOGIT_DELTA_RAW: {grade: delta}
#       * optional additive logit delta (scaled by strength) exposed via tags["role_logit_delta"]
#
# LLM workflow tip:
#   - Provide `role_fit.py` by default.
#   - Only include `role_fit_data.py` when tuning weights/cuts/multipliers.
# -----------------------------

# -----------------------------
# Data imports (tables moved to role_fit_data.py to keep this module logic-focused)
# -----------------------------
try:
    # Package execution
    from .role_fit_data import (
        ROLE_PRIOR_MULT_RAW,
        ROLE_LOGIT_DELTA_RAW,
        ROLE_FIT_WEIGHTS,
        ROLE_FIT_CUTS,
    )
except ImportError:  # pragma: no cover
    # Script / flat-module execution
    from role_fit_data import (  # type: ignore
        ROLE_PRIOR_MULT_RAW,
        ROLE_LOGIT_DELTA_RAW,
        ROLE_FIT_WEIGHTS,
        ROLE_FIT_CUTS,
    )

# -----------------------------
# Fit score / grade
# -----------------------------
def role_fit_score(player: Player, role: str) -> float:
    w = ROLE_FIT_WEIGHTS.get(role)
    if not w:
        return 50.0
    s = 0.0
    for k, a in w.items():
        # defensive: player.get(k) might be None depending on your data model
        try:
            v = player.get(k)
        except Exception:
            v = 0.0
        s += float(v or 0.0) * float(a)
    return clamp(s, 0.0, 100.0)


def role_fit_grade(role: str, fit: float) -> str:
    cuts = ROLE_FIT_CUTS.get(role)
    if not cuts:
        return "B" if fit >= 60 else "C" if fit >= 52 else "D"
    s_min, a_min, b_min, c_min = cuts
    if fit >= s_min:
        return "S"
    if fit >= a_min:
        return "A"
    if fit >= b_min:
        return "B"
    if fit >= c_min:
        return "C"
    return "D"


def _get_role_fit_strength(offense: TeamState, role_fit_cfg: Optional[Dict[str, Any]] = None) -> float:
    try:
        v = (offense.tactics.context or {}).get("ROLE_FIT_STRENGTH", None)
    except Exception:
        v = None
    if v is None:
        try:
            v = float((role_fit_cfg or {}).get("default_strength", 0.65))
        except Exception:
            v = 0.65
    try:
        return clamp(float(v), 0.0, 1.0)
    except Exception:
        return 0.65


def _choose_best_role(offense: TeamState, roles: List[str]) -> Optional[Tuple[str, Player, float]]:
    best: Optional[Tuple[str, Player, float]] = None
    for r in roles:
        pid = getattr(offense, "roles", {}).get(r)
        if not pid:
            continue
        p = offense.find_player(pid)
        if not p:
            continue
        fit = role_fit_score(p, r)
        if best is None or fit > best[2]:
            best = (r, p, fit)
    return best


def _collect_roles_for_action_family(action_family: str, offense: TeamState) -> List[Tuple[str, Player, float]]:
    """
    Collect role participants for a possession 'action_family'.
    This is the only place that should reference specific role keys.
    """
    parts: List[Tuple[str, Player, float]] = []
    fam = action_family

    if fam == "PnR":
        pick = _choose_best_role(offense, ["Initiator_Primary"])
        if pick:
            parts.append(pick)
        pick = _choose_best_role(offense, ["Initiator_Secondary"])
        if pick:
            parts.append(pick)

        # Roller / Short roll: evaluate both if assigned
        for r in ["Roller_Finisher", "ShortRoll_Playmaker"]:
            pid = offense.roles.get(r)
            if pid:
                p = offense.find_player(pid)
                if p:
                    parts.append((r, p, role_fit_score(p, r)))

        # Optional Pop big
        pid = offense.roles.get("Pop_Spacer_Big")
        if pid:
            p = offense.find_player(pid)
            if p:
                parts.append(("Pop_Spacer_Big", p, role_fit_score(p, "Pop_Spacer_Big")))

    elif fam == "DHO":
        for group in [
            ["Initiator_Secondary", "Connector_Playmaker"],
            ["Spacer_Movement"],
            ["Post_Hub", "Pop_Spacer_Big"],
        ]:
            pick = _choose_best_role(offense, group)
            if pick:
                parts.append(pick)

    elif fam == "Drive":
        pick = _choose_best_role(offense, ["Rim_Attacker", "Shot_Creator", "Initiator_Primary"])
        if pick:
            parts.append(pick)

    elif fam == "Kickout":
        for group in [
            ["Rim_Attacker", "Shot_Creator", "Initiator_Primary"],
            ["Spacer_CatchShoot", "Spacer_Movement"],
        ]:
            pick = _choose_best_role(offense, group)
            if pick:
                parts.append(pick)

    elif fam == "ExtraPass":
        for group in [
            ["Connector_Playmaker"],
            ["Initiator_Secondary", "Post_Hub"],
        ]:
            pick = _choose_best_role(offense, group)
            if pick:
                parts.append(pick)

    elif fam == "PostUp":
        pick = _choose_best_role(offense, ["Post_Hub"])
        if pick:
            parts.append(pick)
        pick2 = _choose_best_role(offense, ["Spacer_CatchShoot", "Spacer_Movement"])
        if pick2:
            parts.append(pick2)

    elif fam == "HornsSet":
        for group in [
            ["Initiator_Secondary", "Initiator_Primary"],
            ["Post_Hub"],
            ["Pop_Spacer_Big", "ShortRoll_Playmaker", "Roller_Finisher"],
        ]:
            pick = _choose_best_role(offense, group)
            if pick:
                parts.append(pick)

    elif fam == "SpotUp":
        pick = _choose_best_role(offense, ["Spacer_CatchShoot", "Spacer_Movement"])
        if pick:
            parts.append(pick)

    elif fam == "Cut":
        pick = _choose_best_role(offense, ["Rim_Attacker", "Roller_Finisher"])
        if pick:
            parts.append(pick)
        pick2 = _choose_best_role(offense, ["Connector_Playmaker", "Post_Hub", "Initiator_Secondary"])
        if pick2:
            parts.append(pick2)

    elif fam == "TransitionEarly":
        for group in [
            ["Transition_Handler"],
            ["Roller_Finisher", "Rim_Attacker"],
            ["Spacer_CatchShoot"],
        ]:
            pick = _choose_best_role(offense, group)
            if pick:
                parts.append(pick)

    return parts


def _role_fit_effective_score(fits: List[float]) -> float:
    """
    Effective fit score used for tags/debug and to summarize multi-role participation.
    Weighted towards the minimum fit (weakest link).
    """
    if not fits:
        return 50.0
    if len(fits) == 1:
        return float(fits[0])
    mn = min(fits)
    av = sum(fits) / len(fits)
    return clamp(0.70 * mn + 0.30 * av, 0.0, 100.0)


def _effective_grade_from_participants(participants: List[Tuple[str, Player, float]]) -> str:
    """
    Grade is taken as the worst (most severe) grade among participants,
    computed from EACH participant's own role-specific fit score.
    """
    if not participants:
        return "B"
    sev = {"S": 0, "A": 1, "B": 2, "C": 3, "D": 4}
    grades = [role_fit_grade(r, f) for (r, _, f) in participants]
    return max(grades, key=lambda g: sev.get(g, 2))


def apply_role_fit_to_priors_and_tags(
    priors: Dict[str, float],
    action_family: str,
    offense: TeamState,
    tags: Dict[str, Any],
    game_cfg: Optional["GameConfig"] = None,
) -> Dict[str, float]:
    role_fit_cfg = game_cfg.role_fit if game_cfg is not None else None
    strength = _get_role_fit_strength(offense, role_fit_cfg=role_fit_cfg)
    participants = _collect_roles_for_action_family(action_family, offense)
    applied = bool(participants)

    fits = [f for (_, _, f) in participants]
    fit_eff = _role_fit_effective_score(fits) if applied else 50.0
    grade = _effective_grade_from_participants(participants) if applied else "B"

    mults_applied: List[float] = []

    if applied and strength > 1e-9:
        for o in list(priors.keys()):
            # IMPORTANT: keep FOUL_DRAW as GOOD, and do not overwrite it later.
            if o.startswith("FOUL_DRAW_"):
                cat = "GOOD"
            elif o.startswith("FOUL_"):
                continue
            else:
                if o.startswith("SHOT_") or o.startswith("PASS_"):
                    cat = "GOOD"
                elif o.startswith("TO_") or o.startswith("RESET_"):
                    cat = "BAD"
                else:
                    cat = None

            if not cat:
                continue

            mult_raw = ROLE_PRIOR_MULT_RAW.get(grade, ROLE_PRIOR_MULT_RAW["B"])[cat]
            mult_final = 1.0 + (0.60 * strength) * (float(mult_raw) - 1.0)
            priors[o] *= mult_final
            mults_applied.append(mult_final)

        priors = normalize_weights(priors)

    avg_mult_final = (sum(mults_applied) / len(mults_applied)) if mults_applied else 1.0
    delta_raw = float(ROLE_LOGIT_DELTA_RAW.get(grade, 0.0))
    delta_final = (0.40 * strength) * delta_raw if applied else 0.0

    tags["role_fit_applied"] = bool(applied)
    tags["role_logit_delta"] = float(delta_final)
    tags["role_fit_eff"] = float(fit_eff)
    tags["role_fit_grade"] = str(grade)

    # internal debug (possession-step)
    if hasattr(offense, "role_fit_pos_log"):
        offense.role_fit_pos_log.append(
            {
                "action_family": str(action_family),
                "applied": bool(applied),
                "n_roles": int(len(participants)),
                "fit_eff": float(fit_eff),
                "grade": str(grade),
                "role_fit_strength": float(strength),
                "avg_mult_final": float(avg_mult_final),
                "delta_final": float(delta_final),
            }
        )

    # game-level aggregates (only when applied)
    if applied and hasattr(offense, "role_fit_grade_counts"):
        offense.role_fit_grade_counts[grade] = offense.role_fit_grade_counts.get(grade, 0) + 1
    if applied and hasattr(offense, "role_fit_role_counts"):
        for r, _, _ in participants:
            offense.role_fit_role_counts[r] = offense.role_fit_role_counts.get(r, 0) + 1

    return priors
