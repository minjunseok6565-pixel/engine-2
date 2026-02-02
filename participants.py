# -------------------------
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

from .core import weighted_choice
from .models import Player, TeamState

# Participant selection (12-role only)
# -------------------------
#
# This module intentionally does NOT use legacy role keys (e.g., "ball_handler", "screener", "post").
# TeamState.roles is expected to be a mapping: 12-role name -> pid.

# 12 roles (canonical)
ROLE_INITIATOR_PRIMARY = "Initiator_Primary"
ROLE_INITIATOR_SECONDARY = "Initiator_Secondary"
ROLE_TRANSITION_HANDLER = "Transition_Handler"
ROLE_SHOT_CREATOR = "Shot_Creator"
ROLE_RIM_ATTACKER = "Rim_Attacker"
ROLE_SPACER_CS = "Spacer_CatchShoot"
ROLE_SPACER_MOVE = "Spacer_Movement"
ROLE_CONNECTOR = "Connector_Playmaker"
ROLE_ROLLER = "Roller_Finisher"
ROLE_SHORTROLL = "ShortRoll_Playmaker"
ROLE_POP_BIG = "Pop_Spacer_Big"
ROLE_POST_HUB = "Post_Hub"


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def choose_weighted_player(
    rng: random.Random,
    players: List[Player],
    key: str,
    power: float = 1.2,
    extra_mult_by_pid: Optional[Dict[str, float]] = None,
) -> Player:
    # Weighted random choice among provided candidates.
    # NOTE: callers should pass de-duplicated players.
    extra_mult_by_pid = extra_mult_by_pid or {}
    weights = {
        p.pid: (max(p.get(key), 1.0) ** power) * float(extra_mult_by_pid.get(p.pid, 1.0))
        for p in players
    }
    pid = weighted_choice(rng, weights)
    for p in players:
        if p.pid == pid:
            return p
    return players[0]


def _shot_diet_info(style: Optional[object]) -> Dict[str, object]:
    # Extract style hints (initiator and screeners) if available.
    # We clamp initiator weights to avoid extreme bias.
    try:
        initiator = getattr(style, "initiator", None)
        screeners = getattr(style, "screeners", None)
        w_primary = float(getattr(initiator, "w_primary", 1.0)) if initiator else 1.0
        w_secondary = float(getattr(initiator, "w_secondary", 1.0)) if initiator else 1.0
        return {
            "primary_pid": getattr(initiator, "primary_pid", None) if initiator else None,
            "secondary_pid": getattr(initiator, "secondary_pid", None) if initiator else None,
            "w_primary": _clamp(w_primary, 0.75, 1.35),
            "w_secondary": _clamp(w_secondary, 0.75, 1.35),
            "screener1_pid": getattr(screeners, "screener1_pid", None) if screeners else None,
            "screener2_pid": getattr(screeners, "screener2_pid", None) if screeners else None,
        }
    except Exception:
        return {
            "primary_pid": None,
            "secondary_pid": None,
            "w_primary": 1.0,
            "w_secondary": 1.0,
            "screener1_pid": None,
            "screener2_pid": None,
        }


def _unique_players(players: Sequence[Optional[Player]]) -> List[Player]:
    seen = set()
    uniq: List[Player] = []
    for p in players:
        if not p:
            continue
        if p.pid in seen:
            continue
        seen.add(p.pid)
        uniq.append(p)
    return uniq


def _active(team: TeamState) -> List[Player]:
    return team.on_court_players()


def _role_player(team: TeamState, role_name: str) -> Optional[Player]:
    pid = team.roles.get(role_name)
    if not pid:
        return None
    p = team.find_player(pid)
    if p and team.is_on_court(p.pid):
        return p
    return None


def _players_from_roles(team: TeamState, role_priority: Sequence[str]) -> List[Player]:
    return _unique_players([_role_player(team, r) for r in role_priority])


def _top_k_by_stat(team: TeamState, stat_key: str, k: int, exclude_pids: Optional[set] = None) -> List[Player]:
    exclude_pids = exclude_pids or set()
    sorted_p = sorted(_active(team), key=lambda p: p.get(stat_key), reverse=True)
    out: List[Player] = []
    for p in sorted_p:
        if p.pid in exclude_pids:
            continue
        out.append(p)
        if len(out) >= k:
            break
    return out


def _fill_candidates_with_top_k(
    team: TeamState,
    cand: List[Player],
    cap: int,
    stat_key: str,
) -> List[Player]:
    if len(cand) >= cap:
        return cand[:cap]
    exclude = {p.pid for p in cand}
    cand.extend(_top_k_by_stat(team, stat_key, cap - len(cand), exclude))
    return _unique_players(cand)[:cap]


def _pid_role_mult(team: TeamState, pid: str, role_mult: Dict[str, float]) -> float:
    # If a player has multiple assigned roles, take the maximum multiplier.
    mult = 1.0
    for role, rpid in team.roles.items():
        if rpid == pid:
            mult = max(mult, float(role_mult.get(role, 1.0)))
    return mult


# ---- Shooter selection (catch & shoot) ----

def choose_shooter_for_three(rng: random.Random, offense: TeamState, style: Optional[object] = None) -> Player:
    # Allow any on-court player to be the shooter; keep weighting by SHOT_3_CS.
    cand = _active(offense)
    info = _shot_diet_info(style)
    apply_bias = style is not None
    weights: Dict[str, float] = {}
    for p in cand:
        mult = 1.0
        if apply_bias:
            mult = 0.85 if p.pid in (info.get("primary_pid"), info.get("secondary_pid")) else 1.10
        weights[p.pid] = (max(p.get("SHOT_3_CS"), 1.0) ** 1.35) * mult
    pid = weighted_choice(rng, weights)
    for p in cand:
        if p.pid == pid:
            return p
    return cand[0]


def choose_shooter_for_mid(rng: random.Random, offense: TeamState, style: Optional[object] = None) -> Player:
    # Allow any on-court player to be the shooter; keep weighting by SHOT_MID_CS.
    cand = _active(offense)
    info = _shot_diet_info(style)
    apply_bias = style is not None
    weights: Dict[str, float] = {}
    for p in cand:
        mult = 1.0
        if apply_bias:
            mult = 0.85 if p.pid in (info.get("primary_pid"), info.get("secondary_pid")) else 1.10
        weights[p.pid] = (max(p.get("SHOT_MID_CS"), 1.0) ** 1.25) * mult
    pid = weighted_choice(rng, weights)
    for p in cand:
        if p.pid == pid:
            return p
    return cand[0]


# ---- Creator selection (pull-up / off-dribble) ----

_CREATOR_ROLE_PRIORITY: Tuple[str, ...] = (
    ROLE_SHOT_CREATOR,
    ROLE_INITIATOR_PRIMARY,
    ROLE_INITIATOR_SECONDARY,
    ROLE_TRANSITION_HANDLER,
    ROLE_CONNECTOR,
)

def choose_creator_for_pulloff(rng: random.Random, offense: TeamState, outcome: str, style: Optional[object] = None) -> Player:
    # 12-role candidates first, then fill so that ALL on-court players can be selected.
    key = "SHOT_3_OD" if outcome == "SHOT_3_OD" else "SHOT_MID_PU"
    cand = _players_from_roles(offense, _CREATOR_ROLE_PRIORITY)
    # Previously capped to 3, which hard-limited distribution.
    # Use the on-court count (normally 5) so every player becomes a candidate.
    on_court_cap = len(_active(offense))
    cand = _fill_candidates_with_top_k(offense, cand, cap=on_court_cap, stat_key=key)

    info = _shot_diet_info(style)
    extra: Dict[str, float] = {}
    primary_pid = info.get("primary_pid")
    secondary_pid = info.get("secondary_pid")
    for p in cand:
        if p.pid == primary_pid:
            extra[p.pid] = float(info.get("w_primary", 1.0))
        elif p.pid == secondary_pid:
            extra[p.pid] = float(info.get("w_secondary", 1.0))

    return choose_weighted_player(rng, cand, key, power=1.20, extra_mult_by_pid=extra)


# ---- Rim finisher selection ----

_FINISH_ROLE_BASE: Tuple[str, ...] = (
    ROLE_RIM_ATTACKER,
    ROLE_ROLLER,
    ROLE_SPACER_MOVE,
    ROLE_SHOT_CREATOR,
    ROLE_INITIATOR_PRIMARY,
    ROLE_INITIATOR_SECONDARY,
)

_FINISH_ROLE_PNR: Tuple[str, ...] = (
    ROLE_ROLLER,
    ROLE_SHORTROLL,
    ROLE_POP_BIG,
    ROLE_RIM_ATTACKER,
    ROLE_SPACER_MOVE,
    ROLE_SHOT_CREATOR,
    ROLE_INITIATOR_PRIMARY,
    ROLE_INITIATOR_SECONDARY,
)

# Conservative dunk role multipliers (optional realism tuning).
# These are only applied when dunk_bias=True, on top of the FIN_* stat.
_DUNK_ROLE_MULT = {
    ROLE_RIM_ATTACKER: 1.10,
    ROLE_ROLLER: 1.15,
    ROLE_SHORTROLL: 1.00,
    ROLE_SPACER_MOVE: 1.00,
    ROLE_POP_BIG: 0.80,
}
_MULT_MIN = 0.70
_MULT_MAX = 1.40

def choose_finisher_rim(
    rng: random.Random,
    offense: TeamState,
    dunk_bias: bool = False,
    style: Optional[object] = None,
    base_action: Optional[str] = None,
) -> Player:
    # Choose who finishes at the rim. Candidates are role-driven (12-role only),
    # then filled with best rim-finishers from the lineup to ensure robustness.
    key = "FIN_DUNK" if dunk_bias else "FIN_RIM"
    role_priority = _FINISH_ROLE_PNR if base_action == "PnR" else _FINISH_ROLE_BASE

    cand = _players_from_roles(offense, role_priority)
    cand = _fill_candidates_with_top_k(offense, cand, cap=4, stat_key=key)

    info = _shot_diet_info(style)
    extra: Dict[str, float] = {}
    for p in cand:
        mult = 1.0

        # PnR: prioritize style-selected screeners.
        if base_action == "PnR":
            if p.pid == info.get("screener1_pid"):
                mult *= 1.25
            elif p.pid == info.get("screener2_pid"):
                mult *= 1.10

        # Optional dunk realism: discourage pop-big dunk dominance.
        if dunk_bias:
            mult *= _pid_role_mult(offense, p.pid, _DUNK_ROLE_MULT)

        extra[p.pid] = _clamp(mult, _MULT_MIN, _MULT_MAX)

    return choose_weighted_player(rng, cand, key, power=1.15, extra_mult_by_pid=extra)


# ---- Post target selection ----

_POST_FALLBACK_ROLES: Tuple[str, ...] = (
    ROLE_SHORTROLL,
    ROLE_POP_BIG,
    ROLE_ROLLER,
)

def choose_post_target(offense: TeamState) -> Player:
    # Prefer the Post_Hub. If missing, fall back to the most post-capable big-ish option.
    p = _role_player(offense, ROLE_POST_HUB)
    if p:
        return p

    cand = _players_from_roles(offense, _POST_FALLBACK_ROLES)
    if cand:
        # Deterministic: choose the best by POST_CONTROL (then POST_SCORE).
        return max(cand, key=lambda x: (x.get("POST_CONTROL"), x.get("POST_SCORE")))

    # Final fallback: pick best post controller from lineup (or simply the "biggest" by proxy).
    return max(_active(offense), key=lambda x: (x.get("POST_CONTROL"), x.get("POST_SCORE"), x.get("REB")))


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


def _append_unique(dst: List[Player], src: List[Player]) -> List[Player]:
    if not src:
        return dst
    return _unique_players(list(dst) + list(src))


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


def _softmax_pick_player(
    rng: random.Random,
    players: List[Player],
    scores: List[float],
    beta: float,
    mix: float = 1.0,
) -> Optional[Player]:
    """Pick one player using softmax(exp(beta * score)).

    We use a numerically stable form by subtracting the max score before exponentiation.
    """
    if not players:
        return None
    if len(players) != len(scores) or not scores:
        return players[0]

    beta = max(float(beta), 0.0)
    mix = _clamp(float(mix), 0.0, 1.0)
    m = max(scores)
    base = 1.0 - mix
    weights = {p.pid: (base + mix * math.exp(beta * (float(s) - m))) for p, s in zip(players, scores)}
    pid = weighted_choice(rng, weights)
    for p in players:
        if p.pid == pid:
            return p
    return players[0]


# Rebound softmax expects a stable score scale. We normalize the raw rebound score
# (REB_OR/REB_DR + 0.20*PHYSICAL) into 0..1 before applying softmax.
_REB_NORM_LO: float = 75.0
_REB_NORM_HI: float = 120.0


def _reb_norm(v: float) -> float:
    if _REB_NORM_HI <= _REB_NORM_LO:
        return 0.0
    return _clamp((float(v) - _REB_NORM_LO) / (_REB_NORM_HI - _REB_NORM_LO), 0.0, 1.0)


def choose_orb_rebounder(rng: random.Random, offense: TeamState) -> Player:
    """Choose an offensive rebounder.

    Improvement 7-A: include all 5 on-court players and sample with softmax weights.
    This keeps bigs favored but allows guards/wings to occasionally grab long rebounds.

    Tuning: tactics.context['ORB_SOFTMAX_BETA'] (default 3.2)
    """
    cand = list(_active(offense))
    beta = float(getattr(getattr(offense, 'tactics', None), 'context', {}).get('ORB_SOFTMAX_BETA', 3.2))
    raw_scores = [p.get('REB_OR') + 0.20 * p.get('PHYSICAL') for p in cand]
    scores = [_reb_norm(s) for s in raw_scores]
    return _softmax_pick_player(rng, cand, scores, beta) or cand[0]


def choose_drb_rebounder(rng: random.Random, defense: TeamState) -> Player:
    """Choose a defensive rebounder.

    Improvement 7-A: include all 5 on-court players and sample with softmax weights.

    Tuning: tactics.context['DRB_SOFTMAX_BETA'] (default 2.2), tactics.context['DRB_SOFTMAX_MIX'] (default 0.92)

    DRB_SOFTMAX_MIX blends uniform weights with softmax to reduce extreme top-1 dominance.
    mix=1.0 is pure softmax; mix=0.0 is uniform.
    """
    cand = list(_active(defense))
    ctx = getattr(getattr(defense, 'tactics', None), 'context', {}) or {}
    beta = float(ctx.get('DRB_SOFTMAX_BETA', 2.2))
    mix = float(ctx.get('DRB_SOFTMAX_MIX', 0.92))
    raw_scores = [p.get('REB_DR') + 0.20 * p.get('PHYSICAL') for p in cand]
    scores = [_reb_norm(s) for s in raw_scores]
    return _softmax_pick_player(rng, cand, scores, beta) or cand[0]

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
