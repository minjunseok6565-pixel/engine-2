from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

from ..models import Player, TeamState

from .participants_roles import (
    ROLE_INITIATOR_PRIMARY,
    ROLE_INITIATOR_SECONDARY,
    ROLE_TRANSITION_HANDLER,
    ROLE_SHOT_CREATOR,
    ROLE_RIM_ATTACKER,
    ROLE_SPACER_CS,
    ROLE_SPACER_MOVE,
    ROLE_CONNECTOR,
    ROLE_ROLLER,
    ROLE_SHORTROLL,
    ROLE_POP_BIG,
    ROLE_POST_HUB,
)

from .participants_common import (
    _active,
    _append_unique,
    _clamp,
    _players_from_roles,
    _shot_diet_info,
    _top_k_by_stat,
    _unique_players,
    choose_weighted_player,
)

def _role_of_pid(team: TeamState, pid: str) -> str:
    pid = str(pid or "")
    if not pid:
        return ""
    roles = getattr(team, "roles", {}) or {}
    for role_name, rpid in roles.items():
        if str(rpid) == pid:
            return str(role_name)
    return ""


def _pass_family(base_action: str, outcome: str) -> str:
    ba = str(base_action or "")
    oc = str(outcome or "")

    if oc == "PASS_SHORTROLL":
        return "shortroll"

    if ba in ("Drive",):
        return "drive"

    if ba in ("TransitionEarly",):
        return "transition"

    if ba in ("PnR", "DHO", "PnP"):
        return "pnr"

    if ba in ("PostUp", "ElbowHub", "HornsSet"):
        return "posthub"

    if ba in ("Kickout", "ExtraPass"):
        return "swing"

    return "default"


_PASSER_ROLE_PRIORITY: Dict[str, List[str]] = {
    "default": [ROLE_INITIATOR_PRIMARY, ROLE_INITIATOR_SECONDARY, ROLE_CONNECTOR, ROLE_TRANSITION_HANDLER, ROLE_SHOT_CREATOR, ROLE_SHORTROLL, ROLE_POST_HUB],
    "drive": [ROLE_RIM_ATTACKER, ROLE_SHOT_CREATOR, ROLE_INITIATOR_PRIMARY, ROLE_INITIATOR_SECONDARY, ROLE_CONNECTOR],
    "swing": [ROLE_CONNECTOR, ROLE_INITIATOR_PRIMARY, ROLE_INITIATOR_SECONDARY, ROLE_SHOT_CREATOR, ROLE_TRANSITION_HANDLER],
    "pnr": [ROLE_INITIATOR_PRIMARY, ROLE_INITIATOR_SECONDARY, ROLE_SHORTROLL, ROLE_CONNECTOR, ROLE_ROLLER],
    "transition": [ROLE_TRANSITION_HANDLER, ROLE_INITIATOR_PRIMARY, ROLE_INITIATOR_SECONDARY, ROLE_CONNECTOR, ROLE_RIM_ATTACKER],
    "posthub": [ROLE_POST_HUB, ROLE_CONNECTOR, ROLE_INITIATOR_PRIMARY, ROLE_INITIATOR_SECONDARY, ROLE_SHORTROLL],
    "shortroll": [ROLE_SHORTROLL, ROLE_ROLLER, ROLE_POP_BIG, ROLE_POST_HUB, ROLE_CONNECTOR],
}

_PASSER_CAND_CAP: Dict[str, int] = {
    "default": 5,
    "drive": 4,
    "swing": 5,
    "pnr": 5,
    "transition": 4,
    "posthub": 4,
    "shortroll": 4,
}

_PASSER_KEY_POWER: Dict[str, Tuple[str, float]] = {
    "default": ("PASS_CREATE", 1.10),
    "swing": ("PASS_CREATE", 1.10),
    "pnr": ("PASS_CREATE", 1.12),
    "transition": ("PASS_CREATE", 1.08),
    "drive": ("DRIVE_CREATE", 1.10),
    "shortroll": ("SHORTROLL_PLAY", 1.12),
    "posthub": ("POST_CONTROL", 1.12),
}

_PASSER_ROLE_MULT: Dict[str, Dict[str, float]] = {
    "default": {
        ROLE_INITIATOR_PRIMARY: 1.25,
        ROLE_INITIATOR_SECONDARY: 1.15,
        ROLE_CONNECTOR: 1.12,
        ROLE_TRANSITION_HANDLER: 1.08,
        ROLE_SHOT_CREATOR: 1.05,
        ROLE_SHORTROLL: 1.03,
        ROLE_POST_HUB: 1.03,
        ROLE_RIM_ATTACKER: 0.98,
        ROLE_ROLLER: 0.96,
        ROLE_POP_BIG: 0.96,
        ROLE_SPACER_CS: 0.92,
        ROLE_SPACER_MOVE: 0.92,
        "_DEFAULT_": 0.95,
    },
    "drive": {
        ROLE_RIM_ATTACKER: 1.30,
        ROLE_SHOT_CREATOR: 1.12,
        ROLE_INITIATOR_PRIMARY: 1.10,
        ROLE_INITIATOR_SECONDARY: 1.05,
        ROLE_CONNECTOR: 1.05,
        ROLE_TRANSITION_HANDLER: 1.02,
        "_DEFAULT_": 0.95,
    },
    "swing": {
        ROLE_CONNECTOR: 1.25,
        ROLE_INITIATOR_PRIMARY: 1.18,
        ROLE_INITIATOR_SECONDARY: 1.10,
        ROLE_SHOT_CREATOR: 1.06,
        ROLE_TRANSITION_HANDLER: 1.05,
        "_DEFAULT_": 0.95,
    },
    "pnr": {
        ROLE_INITIATOR_PRIMARY: 1.28,
        ROLE_INITIATOR_SECONDARY: 1.16,
        ROLE_CONNECTOR: 1.08,
        ROLE_SHORTROLL: 1.05,
        ROLE_ROLLER: 1.02,
        ROLE_SHOT_CREATOR: 1.05,
        "_DEFAULT_": 0.95,
    },
    "transition": {
        ROLE_TRANSITION_HANDLER: 1.35,
        ROLE_INITIATOR_PRIMARY: 1.20,
        ROLE_INITIATOR_SECONDARY: 1.10,
        ROLE_CONNECTOR: 1.05,
        ROLE_RIM_ATTACKER: 1.05,
        "_DEFAULT_": 0.95,
    },
    "posthub": {
        ROLE_POST_HUB: 1.50,
        ROLE_CONNECTOR: 1.12,
        ROLE_INITIATOR_PRIMARY: 1.00,
        ROLE_INITIATOR_SECONDARY: 0.95,
        ROLE_SHORTROLL: 1.05,
        "_DEFAULT_": 0.95,
    },
    "shortroll": {
        ROLE_SHORTROLL: 1.55,
        ROLE_ROLLER: 1.18,
        ROLE_POP_BIG: 1.10,
        ROLE_POST_HUB: 1.08,
        ROLE_CONNECTOR: 1.05,
        "_DEFAULT_": 0.95,
    },
}

_PASS_OUTCOME_ROLE_BONUS: Dict[str, Dict[str, float]] = {
    "PASS_KICKOUT": {ROLE_RIM_ATTACKER: 1.08, ROLE_SHOT_CREATOR: 1.05},
    "PASS_SKIP": {ROLE_CONNECTOR: 1.05, ROLE_INITIATOR_PRIMARY: 1.05},
    "PASS_EXTRA": {ROLE_CONNECTOR: 1.12, ROLE_INITIATOR_PRIMARY: 1.05, ROLE_INITIATOR_SECONDARY: 1.03},
    "PASS_SHORTROLL": {ROLE_SHORTROLL: 1.15, ROLE_ROLLER: 1.08, ROLE_POP_BIG: 1.05},
}


def _role_mult_with_default(team: TeamState, pid: str, role_mult: Dict[str, float]) -> float:
    roles = getattr(team, "roles", {}) or {}
    base = float(role_mult.get("_DEFAULT_", 1.0))
    out = base
    for role, rpid in roles.items():
        if str(rpid) == str(pid):
            out = max(out, float(role_mult.get(role, base)))
    return out


def _role_bonus_for_outcome(team: TeamState, pid: str, outcome: str) -> float:
    bonus_map = _PASS_OUTCOME_ROLE_BONUS.get(str(outcome or ""))
    if not bonus_map:
        return 1.0
    roles = getattr(team, "roles", {}) or {}
    out = 1.0
    for role, rpid in roles.items():
        if str(rpid) == str(pid):
            out = max(out, float(bonus_map.get(role, 1.0)))
    return out


def _pass_safe_mult(p: Player) -> float:
    # 0.90 ~ 1.10
    v = max(p.get("PASS_SAFE"), 0.0)
    return _clamp(0.90 + 0.20 * (v / 100.0), 0.90, 1.10)


def _handle_safe_mult(p: Player) -> float:
    # 0.92 ~ 1.08
    v = max(p.get("HANDLE_SAFE"), 0.0)
    return _clamp(0.92 + 0.16 * (v / 100.0), 0.92, 1.08)


def _style_initiator_mult(p: Player, style: Optional[object]) -> float:
    if style is None:
        return 1.0
    info = _shot_diet_info(style)
    pid = p.pid
    if pid == info.get("primary_pid"):
        w = float(info.get("w_primary", 1.0))
        return 1.0 + (w - 1.0) * 0.70
    if pid == info.get("secondary_pid"):
        w = float(info.get("w_secondary", 1.0))
        return 1.0 + (w - 1.0) * 0.70
    return 1.0


def choose_passer(rng: random.Random, offense: TeamState, base_action: str, outcome: str, style: Optional[object] = None) -> Player:
    """Pick a passer using role pools + weighted sampling.

    This removes deterministic passer lock-in while preserving:
    - handler/connector roles pass more often
    - higher PASS_CREATE (and situation-relevant creation) increases selection odds
    """
    fam = _pass_family(base_action, outcome)
    role_priority = _PASSER_ROLE_PRIORITY.get(fam, _PASSER_ROLE_PRIORITY["default"])
    cap = int(_PASSER_CAND_CAP.get(fam, 5))

    # 1) Start from role-based candidates
    cand: List[Player] = _players_from_roles(offense, role_priority)

    # 2) Fill with top-k by relevant stats (family-specific), keeping uniqueness
    if fam == "drive":
        cand = _append_unique(cand, _top_k_by_stat(offense, "DRIVE_CREATE", 2, exclude_pids={p.pid for p in cand}))
        cand = _append_unique(cand, _top_k_by_stat(offense, "PASS_CREATE", 2, exclude_pids={p.pid for p in cand}))
    elif fam == "shortroll":
        cand = _append_unique(cand, _top_k_by_stat(offense, "SHORTROLL_PLAY", 2, exclude_pids={p.pid for p in cand}))
        cand = _append_unique(cand, _top_k_by_stat(offense, "PASS_CREATE", 2, exclude_pids={p.pid for p in cand}))
    elif fam == "posthub":
        cand = _append_unique(cand, _top_k_by_stat(offense, "POST_CONTROL", 2, exclude_pids={p.pid for p in cand}))
        cand = _append_unique(cand, _top_k_by_stat(offense, "PASS_CREATE", 2, exclude_pids={p.pid for p in cand}))
    else:
        cand = _append_unique(cand, _top_k_by_stat(offense, "PASS_CREATE", 3, exclude_pids={p.pid for p in cand}))

    cand = _unique_players(cand)[:cap]
    if not cand:
        # Safety fallback
        return max(_active(offense), key=lambda p: p.get("PASS_CREATE"))

    key, power = _PASSER_KEY_POWER.get(fam, ("PASS_CREATE", 1.10))
    role_mult_map = _PASSER_ROLE_MULT.get(fam, _PASSER_ROLE_MULT["default"])

    # 3) Extra multipliers (role, safety, initiator bias, outcome bonus)
    extra: Dict[str, float] = {}
    for p in cand:
        mult = 1.0

        mult *= _role_mult_with_default(offense, p.pid, role_mult_map)
        mult *= _role_bonus_for_outcome(offense, p.pid, outcome)

        mult *= _pass_safe_mult(p)
        mult *= _handle_safe_mult(p)
        mult *= _style_initiator_mult(p, style)

        # If we're selecting by a non-pass key (drive/post/shortroll), we still want passing skill to matter.
        if key != "PASS_CREATE":
            pass_create = max(p.get("PASS_CREATE"), 1.0)
            mult *= (pass_create / 50.0) ** 0.60

        extra[p.pid] = mult

    return choose_weighted_player(rng, cand, key, power=float(power), extra_mult_by_pid=extra)


# ---- Assister selection (deterministic) ----

_ASSIST_ROLE_PRIORITY: Tuple[str, ...] = (
    ROLE_CONNECTOR,
    ROLE_INITIATOR_PRIMARY,
    ROLE_INITIATOR_SECONDARY,
    ROLE_SHORTROLL,
    ROLE_POST_HUB,
    ROLE_TRANSITION_HANDLER,
)

def choose_assister_deterministic(team: TeamState, shooter_pid: str) -> Optional[Player]:
    # Prefer primary playmakers, but never return the shooter.
    for role in _ASSIST_ROLE_PRIORITY:
        pid = team.roles.get(role)
        if pid and pid != shooter_pid:
            p = team.find_player(pid)
            if p and team.is_on_court(p.pid):
                return p

    others = [p for p in _active(team) if p.pid != shooter_pid]
    if not others:
        return None
    return max(others, key=lambda x: x.get("PASS_CREATE"))


# ---- Assister selection (weighted; for implied assists) ----

def _assist_group(shot_outcome: str) -> str:
    o = str(shot_outcome or "")
    if "_CS" in o:
        return "cs"
    if o in ("SHOT_RIM_LAYUP", "SHOT_RIM_DUNK", "SHOT_RIM_CONTACT", "SHOT_TOUCH_FLOATER"):
        return "rim"
    if o == "SHOT_POST":
        return "post"
    if ("_PU" in o) or (o == "SHOT_3_OD"):
        return "pullup"
    return "default"


_ASSISTER_ROLE_MULT: Dict[str, Dict[str, float]] = {
    "cs": {
        ROLE_CONNECTOR: 1.28,
        ROLE_INITIATOR_PRIMARY: 1.22,
        ROLE_INITIATOR_SECONDARY: 1.12,
        ROLE_POST_HUB: 1.10,
        ROLE_SHORTROLL: 1.08,
        ROLE_TRANSITION_HANDLER: 1.06,
        ROLE_SHOT_CREATOR: 1.02,
        "_DEFAULT_": 0.95,
    },
    "rim": {
        ROLE_INITIATOR_PRIMARY: 1.28,
        ROLE_INITIATOR_SECONDARY: 1.14,
        ROLE_SHORTROLL: 1.16,
        ROLE_CONNECTOR: 1.08,
        ROLE_POST_HUB: 1.06,
        ROLE_TRANSITION_HANDLER: 1.08,
        ROLE_SHOT_CREATOR: 1.04,
        "_DEFAULT_": 0.95,
    },
    "post": {
        ROLE_POST_HUB: 1.35,
        ROLE_CONNECTOR: 1.12,
        ROLE_INITIATOR_PRIMARY: 1.10,
        ROLE_INITIATOR_SECONDARY: 1.02,
        ROLE_SHORTROLL: 1.06,
        "_DEFAULT_": 0.95,
    },
    "pullup": {
        ROLE_INITIATOR_PRIMARY: 1.22,
        ROLE_SHOT_CREATOR: 1.18,
        ROLE_INITIATOR_SECONDARY: 1.10,
        ROLE_CONNECTOR: 1.02,
        "_DEFAULT_": 0.95,
    },
    "default": {
        ROLE_INITIATOR_PRIMARY: 1.20,
        ROLE_INITIATOR_SECONDARY: 1.10,
        ROLE_CONNECTOR: 1.10,
        ROLE_SHORTROLL: 1.06,
        ROLE_POST_HUB: 1.06,
        ROLE_TRANSITION_HANDLER: 1.04,
        ROLE_SHOT_CREATOR: 1.04,
        "_DEFAULT_": 0.95,
    },
}

_ASSISTER_ROLE_PRIORITY_WEIGHTED: Tuple[str, ...] = (
    ROLE_CONNECTOR,
    ROLE_INITIATOR_PRIMARY,
    ROLE_INITIATOR_SECONDARY,
    ROLE_POST_HUB,
    ROLE_SHORTROLL,
    ROLE_TRANSITION_HANDLER,
    ROLE_SHOT_CREATOR,
)

_ASSISTER_CAND_CAP: int = 5
_ASSISTER_POWER: float = 1.08


def choose_assister_weighted(
    rng: random.Random,
    offense: TeamState,
    shooter_pid: str,
    base_action: str,
    shot_outcome: str,
    style: Optional[object] = None,
) -> Optional[Player]:
    """Choose an assister stochastically when no concrete last-pass event exists.

    This is intentionally *not* deterministic to avoid structural AST monopolization.
    """
    shooter_pid = str(shooter_pid or "")
    group = _assist_group(shot_outcome)
    role_mult_map = _ASSISTER_ROLE_MULT.get(group, _ASSISTER_ROLE_MULT["default"])

    # Role-first candidates
    cand: List[Player] = []
    roles = getattr(offense, "roles", {}) or {}
    for role in _ASSISTER_ROLE_PRIORITY_WEIGHTED:
        pid = roles.get(role)
        if not pid:
            continue
        if str(pid) == shooter_pid:
            continue
        p = offense.find_player(pid)
        if p is not None and offense.is_on_court(p.pid):
            cand.append(p)

    cand = _unique_players(cand)

    # Fill with top PASS_CREATE (up to 4) to ensure we always have a reasonable pool
    exclude = {p.pid for p in cand}
    if shooter_pid:
        exclude.add(shooter_pid)
    cand = _append_unique(cand, _top_k_by_stat(offense, "PASS_CREATE", 4, exclude_pids=exclude))

    cand = _unique_players([p for p in cand if p.pid != shooter_pid])[:_ASSISTER_CAND_CAP]
    if not cand:
        return None

    extra: Dict[str, float] = {}
    for p in cand:
        mult = 1.0
        mult *= _role_mult_with_default(offense, p.pid, role_mult_map)
        mult *= _pass_safe_mult(p)
        mult *= _handle_safe_mult(p)
        mult *= _style_initiator_mult(p, style)
        extra[p.pid] = mult

    return choose_weighted_player(rng, cand, "PASS_CREATE", power=_ASSISTER_POWER, extra_mult_by_pid=extra)


# -------------------------
# Additional choosers moved from resolve_12role
# -------------------------

# Default actor selection for outcomes that don't have a specific chooser.
_DEFAULT_ACTOR_ROLE_PRIORITY: Tuple[str, ...] = (
    ROLE_INITIATOR_PRIMARY,
    ROLE_INITIATOR_SECONDARY,
    ROLE_TRANSITION_HANDLER,
    ROLE_CONNECTOR,
    ROLE_SHOT_CREATOR,
)


def choose_default_actor(offense: TeamState) -> Player:
    """Pick the most reasonable on-ball actor (12-role first, then best passer).

    Used for generic outcomes (e.g., shot clock, generic turnover/reset) where
    a specific participant chooser is not defined.
    """
    roles = getattr(offense, "roles", {}) or {}
    for role in _DEFAULT_ACTOR_ROLE_PRIORITY:
        pid = roles.get(role)
        if isinstance(pid, str) and pid:
            p = offense.find_player(pid)
            if p is not None and offense.is_on_court(p.pid):
                return p
    # Final fallback: best creator/passer on the floor
    return max(_active(offense), key=lambda p: p.get("PASS_CREATE"))
