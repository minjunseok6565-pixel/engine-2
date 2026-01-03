# -------------------------
from __future__ import annotations

import random
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

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

def _usage_penalty(
    ctx: Optional[Mapping[str, object]],
    pid: str,
    *,
    free: float,
    scale: float,
    power: float,
    floor: float,
) -> float:
    if not ctx:
        return 1.0
    fga_by_pid = ctx.get("fga_by_pid") if isinstance(ctx, Mapping) else None
    if not isinstance(fga_by_pid, Mapping):
        return 1.0
    try:
        fga = int(fga_by_pid.get(pid, 0))
    except Exception:
        fga = 0
    over = max(0.0, float(fga) - float(free))
    penalty = 1.0 / (1.0 + ((over / float(scale)) ** float(power)))
    return max(float(floor), float(penalty))


def _usage_penalty_cs(ctx: Optional[Mapping[str, object]], pid: str) -> float:
    return _usage_penalty(ctx, pid, free=2, scale=6.0, power=1.25, floor=0.40)


def _usage_penalty_od(ctx: Optional[Mapping[str, object]], pid: str) -> float:
    return _usage_penalty(ctx, pid, free=3, scale=7.0, power=1.20, floor=0.50)


def _recent_actor_mult(ctx: Optional[Mapping[str, object]], pid: str) -> float:
    if not ctx:
        return 1.0
    try:
        last_actor = ctx.get("last_actor_pid")
    except Exception:
        last_actor = None
    return 0.92 if last_actor == pid else 1.0


def choose_shooter_for_three(
    rng: random.Random,
    offense: TeamState,
    style: Optional[object] = None,
    ctx: Optional[Mapping[str, object]] = None,
) -> Player:
    # Role candidates first, then fill with top-K by stat.
    role_priority = (
        ROLE_SPACER_CS,
        ROLE_SPACER_MOVE,
        ROLE_POP_BIG,
        ROLE_CONNECTOR,
        ROLE_INITIATOR_SECONDARY,
    )
    cand = _players_from_roles(offense, role_priority)
    cand = _fill_candidates_with_top_k(offense, cand, cap=5, stat_key="SHOT_3_CS")
    info = _shot_diet_info(style)
    role_mult = {
        ROLE_SPACER_CS: 1.25,
        ROLE_SPACER_MOVE: 1.18,
        ROLE_POP_BIG: 1.12,
        ROLE_CONNECTOR: 1.08,
        ROLE_INITIATOR_SECONDARY: 1.05,
        ROLE_SHOT_CREATOR: 1.03,
        ROLE_INITIATOR_PRIMARY: 0.92,
    }
    extra: Dict[str, float] = {}
    for p in cand:
        if style is None:
            style_mult = 1.0
        elif p.pid == info.get("primary_pid"):
            style_mult = 0.90
        elif p.pid == info.get("secondary_pid"):
            style_mult = 0.95
        else:
            style_mult = 1.08
        mult = _pid_role_mult(offense, p.pid, role_mult)
        mult *= style_mult * _usage_penalty_cs(ctx, p.pid)
        extra[p.pid] = mult
    return choose_weighted_player(rng, cand, "SHOT_3_CS", power=1.15, extra_mult_by_pid=extra)


def choose_shooter_for_mid(
    rng: random.Random,
    offense: TeamState,
    style: Optional[object] = None,
    ctx: Optional[Mapping[str, object]] = None,
) -> Player:
    role_priority = (
        ROLE_SHOT_CREATOR,
        ROLE_CONNECTOR,
        ROLE_INITIATOR_SECONDARY,
        ROLE_SPACER_MOVE,
        ROLE_POST_HUB,
    )
    cand = _players_from_roles(offense, role_priority)
    cand = _fill_candidates_with_top_k(offense, cand, cap=5, stat_key="SHOT_MID_CS")
    info = _shot_diet_info(style)
    role_mult = {
        ROLE_SHOT_CREATOR: 1.18,
        ROLE_CONNECTOR: 1.10,
        ROLE_INITIATOR_SECONDARY: 1.06,
        ROLE_SPACER_MOVE: 1.03,
        ROLE_POST_HUB: 1.05,
        ROLE_INITIATOR_PRIMARY: 0.95,
    }
    extra: Dict[str, float] = {}
    for p in cand:
        if style is None:
            style_mult = 1.0
        elif p.pid == info.get("primary_pid"):
            style_mult = 0.92
        elif p.pid == info.get("secondary_pid"):
            style_mult = 0.97
        else:
            style_mult = 1.06
        mult = _pid_role_mult(offense, p.pid, role_mult)
        mult *= style_mult * _usage_penalty_cs(ctx, p.pid)
        extra[p.pid] = mult
    return choose_weighted_player(rng, cand, "SHOT_MID_CS", power=1.12, extra_mult_by_pid=extra)


# ---- Creator selection (pull-up / off-dribble) ----

_CREATOR_ROLE_PRIORITY: Tuple[str, ...] = (
    ROLE_SHOT_CREATOR,
    ROLE_INITIATOR_PRIMARY,
    ROLE_INITIATOR_SECONDARY,
    ROLE_TRANSITION_HANDLER,
    ROLE_CONNECTOR,
)

def choose_creator_for_pulloff(
    rng: random.Random,
    offense: TeamState,
    outcome: str,
    style: Optional[object] = None,
    ctx: Optional[Mapping[str, object]] = None,
) -> Player:
    # 12-role candidates first, then fill with top-K by the relevant off-dribble stat.
    key = "SHOT_3_OD" if outcome == "SHOT_3_OD" else "SHOT_MID_PU"
    cand = _players_from_roles(offense, _CREATOR_ROLE_PRIORITY)
    cand = _fill_candidates_with_top_k(offense, cand, cap=5, stat_key=key)

    info = _shot_diet_info(style)
    role_mult = {
        ROLE_SHOT_CREATOR: 1.20,
        ROLE_INITIATOR_PRIMARY: 1.12,
        ROLE_INITIATOR_SECONDARY: 1.07,
        ROLE_TRANSITION_HANDLER: 1.05,
        ROLE_CONNECTOR: 1.03,
    }
    extra: Dict[str, float] = {}
    for p in cand:
        if style is None:
            style_mult = 1.0
        elif p.pid == info.get("primary_pid"):
            style_mult = 1.05
        elif p.pid == info.get("secondary_pid"):
            style_mult = 1.02
        else:
            style_mult = 0.98
        mult = _pid_role_mult(offense, p.pid, role_mult)
        mult *= style_mult * _usage_penalty_od(ctx, p.pid)
        extra[p.pid] = mult

    return choose_weighted_player(rng, cand, key, power=1.18, extra_mult_by_pid=extra)


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


# ---- Passer selection ----

_DEFAULT_PASSER_PRIORITY: Tuple[str, ...] = (
    ROLE_INITIATOR_PRIMARY,
    ROLE_INITIATOR_SECONDARY,
    ROLE_CONNECTOR,
    ROLE_TRANSITION_HANDLER,
    ROLE_SHOT_CREATOR,
)

_SHORTROLL_PASSER_PRIORITY: Tuple[str, ...] = (
    ROLE_SHORTROLL,
    ROLE_ROLLER,
    ROLE_POP_BIG,
    ROLE_POST_HUB,
)

def choose_passer(rng: random.Random, offense: TeamState, base_action: str, outcome: str, style: Optional[object] = None) -> Player:
    # Heuristic passer selection using 12-role keys only.
    #
    # - Shortroll pass: short-roll playmaker (or roller/pop big)
    # - PostUp: post hub
    # - Kickout/extra/skip: style initiators when available
    # - Drive: choose between a rim attacker (or best driver) and an initiator/connector
    # - Default: primary initiator (or secondary/connector)

    if outcome == "PASS_SHORTROLL":
        cand = _players_from_roles(offense, _SHORTROLL_PASSER_PRIORITY)
        if cand:
            # Deterministic: prefer shortroll skill; fall back to passing.
            return max(cand, key=lambda x: (x.get("SHORTROLL_PLAY"), x.get("PASS_CREATE")))
        # Fallback: best shortroll playmaker if stat exists, else best passer.
        best = max(_active(offense), key=lambda x: (x.get("SHORTROLL_PLAY"), x.get("PASS_CREATE")))
        return best

    if base_action == "PostUp":
        p = _role_player(offense, ROLE_POST_HUB)
        if p:
            return p
        return max(_active(offense), key=lambda x: (x.get("POST_CONTROL"), x.get("PASS_CREATE")))

    if style is not None and outcome in ("PASS_KICKOUT", "PASS_EXTRA", "PASS_SKIP"):
        info = _shot_diet_info(style)
        cands: List[Player] = []
        for pid in (info.get("primary_pid"), info.get("secondary_pid")):
            if pid:
                p = offense.find_player(pid)
                if p and offense.is_on_court(p.pid):
                    cands.append(p)
        cands = _unique_players(cands)
        if cands:
            extra: Dict[str, float] = {}
            for p in cands:
                mult = info.get("w_primary", 1.0) if p.pid == info.get("primary_pid") else info.get("w_secondary", 1.0)
                extra[p.pid] = float(mult)
            return choose_weighted_player(rng, cands, "PASS_CREATE", power=1.10, extra_mult_by_pid=extra)
        # If no initiators on-court, fall through to default behavior.

    if base_action == "Drive":
        # Candidate A: rim attacker (if assigned), otherwise the best driver
        cand_a = _role_player(offense, ROLE_RIM_ATTACKER) or max(_active(offense), key=lambda p: p.get("DRIVE_CREATE"))
        # Candidate B: primary initiator; else secondary; else connector; else best passer
        cand_b = (
            _role_player(offense, ROLE_INITIATOR_PRIMARY)
            or _role_player(offense, ROLE_INITIATOR_SECONDARY)
            or _role_player(offense, ROLE_CONNECTOR)
            or max(_active(offense), key=lambda p: p.get("PASS_CREATE"))
        )
        cand = _unique_players([cand_a, cand_b])
        return choose_weighted_player(rng, cand, "PASS_CREATE", power=1.10)

    # Default: use the best available initiator/connector; fall back to best passer.
    for r in _DEFAULT_PASSER_PRIORITY:
        p = _role_player(offense, r)
        if p:
            return p
    return max(_active(offense), key=lambda x: x.get("PASS_CREATE"))


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


def choose_assister_from_history(
    team: TeamState,
    shooter_pid: str,
    pass_history: Optional[Sequence[str]] = None,
    last_passer_pid: Optional[str] = None,
) -> Optional[Player]:
    """Choose an assister based on recent pass history; fall back to deterministic."""
    if last_passer_pid and last_passer_pid != shooter_pid:
        p = team.find_player(last_passer_pid)
        if p and team.is_on_court(p.pid):
            return p

    if pass_history:
        for pid in reversed(list(pass_history)):
            if pid == shooter_pid:
                continue
            p = team.find_player(pid)
            if p and team.is_on_court(p.pid):
                return p

    return choose_assister_deterministic(team, shooter_pid)


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


def _big_rebounder_pids(team: TeamState) -> set[str]:
    """Classify bigs as top-2 by (REB_DR + 0.20*PHYSICAL) within the lineup."""
    lineup = _active(team)
    ranked = sorted(
        lineup,
        key=lambda p: p.get("REB_DR") + 0.20 * p.get("PHYSICAL"),
        reverse=True,
    )
    return {p.pid for p in ranked[:2]}


def _physical_mult(p: Player) -> float:
    """Physical multiplier for rebound weights (clamped)."""
    return _clamp(1.0 + 0.014 * (p.get("PHYSICAL") - 50.0), 0.70, 1.70)


def _zone_mult(zone_detail: Optional[str], *, is_big: bool, is_orb: bool) -> float:
    """Zone-aware big/perimeter multipliers for rebound selection."""
    if zone_detail in ("Corner_3", "ATB_3"):
        if is_big:
            return 0.92 if is_orb else 0.95
        return 1.15 if is_orb else 1.12
    if zone_detail in ("Restricted_Area", "Paint_Non_RA"):
        if is_big:
            return 1.18 if is_orb else 1.15
        return 0.92 if is_orb else 0.94
    if zone_detail in ("Mid_Range",):
        if is_big:
            return 0.99
        return 1.06 if is_orb else 1.04
    return 1.0


def choose_orb_rebounder(
    rng: random.Random,
    offense: TeamState,
    shot_zone_detail: Optional[str] = None,
) -> Player:
    """Choose an offensive rebounder with full-lineup weighting."""
    cand = _active(offense)
    if not cand:
        return offense.on_court_players()[0]
    big_pids = _big_rebounder_pids(offense)
    weights: Dict[str, float] = {}
    for p in cand:
        is_big = p.pid in big_pids
        mult = _physical_mult(p) * _zone_mult(shot_zone_detail, is_big=is_big, is_orb=True)
        weights[p.pid] = (max(p.get("REB_OR"), 1.0) ** 1.08) * mult
    pid = weighted_choice(rng, weights)
    for p in cand:
        if p.pid == pid:
            return p
    return cand[0]


def choose_drb_rebounder(
    rng: random.Random,
    defense: TeamState,
    shot_zone_detail: Optional[str] = None,
) -> Player:
    """Choose a defensive rebounder with full-lineup weighting."""
    cand = _active(defense)
    if not cand:
        return defense.on_court_players()[0]
    big_pids = _big_rebounder_pids(defense)
    weights: Dict[str, float] = {}
    for p in cand:
        is_big = p.pid in big_pids
        mult = _physical_mult(p) * _zone_mult(shot_zone_detail, is_big=is_big, is_orb=False)
        weights[p.pid] = (max(p.get("REB_DR"), 1.0) ** 1.06) * mult
    pid = weighted_choice(rng, weights)
    for p in cand:
        if p.pid == pid:
            return p
    return cand[0]


# Foul meta buckets and fixed foul-type mixes (non-offense-role based).
FOUL_META_BUCKET_POA = "POA"
FOUL_META_BUCKET_HELP = "HELP_CLOSEOUT"
FOUL_META_BUCKET_RIM = "RIM_PROTECT"
FOUL_META_BUCKET_POST = "POST"
FOUL_META_BUCKET_STEAL = "STEAL_PRESS"
FOUL_META_BUCKET_CHASER = "CHASER"

FOUL_META_BUCKETS = (
    FOUL_META_BUCKET_POA,
    FOUL_META_BUCKET_HELP,
    FOUL_META_BUCKET_RIM,
    FOUL_META_BUCKET_POST,
    FOUL_META_BUCKET_STEAL,
    FOUL_META_BUCKET_CHASER,
)

# Optional scheme-specific role-key -> bucket mapping overrides.
DEF_FOUL_META_BUCKET_MAP_BY_SCHEME: Dict[str, Dict[str, Sequence[str]]] = {}

_FOUL_BUCKET_MIX = {
    "FOUL_DRAW_POST": {
        FOUL_META_BUCKET_POST: 0.55,
        FOUL_META_BUCKET_RIM: 0.20,
        FOUL_META_BUCKET_HELP: 0.15,
        FOUL_META_BUCKET_POA: 0.10,
    },
    "FOUL_DRAW_RIM": {
        FOUL_META_BUCKET_RIM: 0.45,
        FOUL_META_BUCKET_POA: 0.25,
        FOUL_META_BUCKET_HELP: 0.20,
        FOUL_META_BUCKET_POST: 0.10,
    },
    "FOUL_DRAW_JUMPER": {
        FOUL_META_BUCKET_HELP: 0.45,
        FOUL_META_BUCKET_POA: 0.35,
        FOUL_META_BUCKET_CHASER: 0.15,
        FOUL_META_BUCKET_RIM: 0.05,
    },
    "FOUL_REACH_TRAP": {
        FOUL_META_BUCKET_STEAL: 0.55,
        FOUL_META_BUCKET_POA: 0.25,
        FOUL_META_BUCKET_HELP: 0.15,
        "BASELINE": 0.05,
    },
}

_FOUL_BUCKET_MIX_DEFAULT = {
    FOUL_META_BUCKET_POA: 0.30,
    FOUL_META_BUCKET_HELP: 0.35,
    FOUL_META_BUCKET_RIM: 0.20,
    FOUL_META_BUCKET_POST: 0.10,
    FOUL_META_BUCKET_STEAL: 0.05,
}


def _normalize_scheme(scheme: str) -> str:
    return str(scheme or "").strip().lower()


def _token_bucket(role_key: str) -> str:
    """Fallback token-based role->bucket mapping when a scheme map is not defined."""
    name = role_key.upper()
    if "POA" in name:
        return FOUL_META_BUCKET_POA
    if "HELP" in name or "CLOSE" in name:
        return FOUL_META_BUCKET_HELP
    if "RIM" in name or "LOWMAN" in name:
        return FOUL_META_BUCKET_RIM
    if "POST" in name:
        return FOUL_META_BUCKET_POST
    if "STEAL" in name or "PRESS" in name or "TRAP" in name:
        return FOUL_META_BUCKET_STEAL
    if "CHASE" in name:
        return FOUL_META_BUCKET_CHASER
    return FOUL_META_BUCKET_HELP


def _build_def_meta_buckets(
    role_players: Mapping[str, Player],
    scheme: str,
) -> Dict[str, List[Player]]:
    buckets: Dict[str, List[Player]] = {k: [] for k in FOUL_META_BUCKETS}
    scheme_map = DEF_FOUL_META_BUCKET_MAP_BY_SCHEME.get(_normalize_scheme(scheme), {})
    role_to_bucket: Dict[str, str] = {}
    for bucket, role_keys in scheme_map.items():
        for role_key in role_keys:
            role_to_bucket[str(role_key)] = str(bucket)

    for role_key, player in role_players.items():
        if not player:
            continue
        bucket = role_to_bucket.get(str(role_key))
        if not bucket:
            bucket = _token_bucket(str(role_key))
        buckets.setdefault(bucket, []).append(player)
    return buckets


def _physical_mult_foul(p: Player) -> float:
    return _clamp(1.0 + 0.014 * (p.get("PHYSICAL") - 50.0), 0.70, 1.70)


def _steal_mult_foul(p: Player) -> float:
    return _clamp(1.0 + 0.010 * (p.get("DEF_STEAL") - 50.0), 0.80, 1.60)


def _trouble_mult(player_fouls: Dict[str, int], pid: str, foul_out_limit: int) -> float:
    remaining = int(foul_out_limit) - int(player_fouls.get(pid, 0))
    if remaining <= 1:
        return 0.30
    if remaining == 2:
        return 0.60
    if remaining == 3:
        return 0.85
    return 1.00


def choose_fouler_pid(
    rng: random.Random,
    defense: TeamState,
    def_on_court: Sequence[str],
    player_fouls: Dict[str, int],
    foul_out_limit: int,
    *,
    outcome: str,
    base_action: str,
    attacker_pid: Optional[str] = None,
    shot_zone_detail: Optional[str] = None,
    role_players: Optional[Mapping[str, Player]] = None,
    scheme: str = "",
) -> Optional[str]:
    """Choose a defender pid to be credited with a foul.

    - Excludes players already at/over foul-out limit when possible.
    - Does NOT mutate player_fouls; resolve layer remains responsible for bookkeeping.
    """
    cands = [pid for pid in (def_on_court or []) if isinstance(pid, str) and pid]
    if not cands:
        return None

    eligible = [pid for pid in cands if int(player_fouls.get(pid, 0)) < int(foul_out_limit)]
    if not eligible:
        eligible = cands

    buckets = _build_def_meta_buckets(role_players or {}, scheme)
    mix = dict(_FOUL_BUCKET_MIX.get(outcome, _FOUL_BUCKET_MIX_DEFAULT))
    baseline_weight = float(mix.pop("BASELINE", 0.0))

    def _eligible_pids(pids: Sequence[str]) -> List[str]:
        filtered = [pid for pid in pids if pid in eligible]
        return filtered if filtered else list(pids)

    if rng.random() < 0.20:
        cand_pids = _eligible_pids(cands)
    else:
        bucket_weights = {}
        for bucket, weight in mix.items():
            players = buckets.get(bucket, [])
            pids = [p.pid for p in players if p and p.pid in cands]
            if pids:
                bucket_weights[bucket] = float(weight)

        if baseline_weight > 0 and cands:
            bucket_weights["__baseline__"] = baseline_weight

        if not bucket_weights:
            cand_pids = _eligible_pids(cands)
        else:
            bucket = weighted_choice(rng, bucket_weights)
            if bucket == "__baseline__":
                cand_pids = _eligible_pids(cands)
            else:
                pids = [p.pid for p in buckets.get(bucket, []) if p.pid in cands]
                cand_pids = _eligible_pids(pids)

    if not cand_pids:
        return None

    weights: Dict[str, float] = {}
    for pid in cand_pids:
        p = defense.find_player(pid)
        if not p:
            weights[pid] = 1.0
            continue
        mult = _physical_mult_foul(p) * _trouble_mult(player_fouls, pid, foul_out_limit)
        if outcome == "FOUL_REACH_TRAP":
            mult *= _steal_mult_foul(p)
        weights[pid] = max(mult, 0.05)

    return weighted_choice(rng, weights)


def choose_turnover_actor(
    rng: random.Random,
    offense: TeamState,
    outcome: str,
    base_action: str,
    ctx: Optional[Mapping[str, object]] = None,
) -> Player:
    ctx = ctx or {}

    def _default_actor() -> Player:
        return choose_default_actor(offense)

    def _pid_to_player(pid: Optional[str]) -> Optional[Player]:
        if not pid:
            return None
        p = offense.find_player(pid)
        if p and offense.is_on_court(p.pid):
            return p
        return None

    if outcome == "TO_SHOT_CLOCK":
        if rng.random() < 0.70:
            p = _pid_to_player(ctx.get("last_actor_pid"))
            if p:
                return p
        if rng.random() < 0.6667:
            p = _pid_to_player(ctx.get("last_passer_pid"))
            if p:
                return p
        return _default_actor()

    if outcome == "TO_TRAVEL":
        if rng.random() < 0.85:
            p = _pid_to_player(ctx.get("last_actor_pid"))
            if p:
                return p
        return _default_actor()

    if outcome == "TO_3SEC":
        cand = sorted(
            _active(offense),
            key=lambda p: p.get("PHYSICAL") + p.get("REB_OR"),
            reverse=True,
        )[:3]
        if cand:
            return choose_weighted_player(rng, cand, "PHYSICAL", power=1.05)
        return _default_actor()

    if outcome == "TO_INBOUND":
        role_priority = (
            ROLE_CONNECTOR,
            ROLE_INITIATOR_PRIMARY,
            ROLE_INITIATOR_SECONDARY,
            ROLE_SPACER_MOVE,
            ROLE_POP_BIG,
        )
        cand = _players_from_roles(offense, role_priority)
        cand = _fill_candidates_with_top_k(offense, cand, cap=4, stat_key="PASS_SAFE")
        if cand:
            weights: Dict[str, float] = {}
            for p in cand:
                stat = p.get("PASS_SAFE") if p.get("PASS_SAFE") > 0 else p.get("PASS_CREATE")
                weights[p.pid] = max(stat, 1.0) ** 1.10
            pid = weighted_choice(rng, weights)
            for p in cand:
                if p.pid == pid:
                    return p
        return _default_actor()

    if outcome == "TO_BAD_PASS":
        p = _pid_to_player(ctx.get("last_passer_pid"))
        if p:
            return p
        pass_history = ctx.get("pass_history")
        if isinstance(pass_history, Sequence):
            for pid in reversed(list(pass_history)):
                p = _pid_to_player(pid)
                if p:
                    return p

    if outcome == "TO_HANDLE_LOSS":
        last_actor = _pid_to_player(ctx.get("last_actor_pid"))
        post_actor = choose_post_target(offense) if base_action == "PostUp" else None
        drive_actor = choose_finisher_rim(rng, offense, dunk_bias=False, base_action=base_action)

        if base_action == "PostUp":
            if rng.random() < 0.65 and post_actor:
                return post_actor
            if last_actor:
                return last_actor
            return _default_actor()

        if base_action == "Drive":
            if rng.random() < 0.55 and drive_actor:
                return drive_actor
            if last_actor:
                return last_actor
            return _default_actor()

        cand = _active(offense)
        if cand:
            weights = {
                p.pid: max(100.0 - float(p.get("BALL_SECURITY")), 1.0) ** 1.10
                for p in cand
            }
            pid = weighted_choice(rng, weights)
            for p in cand:
                if p.pid == pid:
                    return p
        return _default_actor()

    if outcome == "TO_CHARGE":
        last_actor = _pid_to_player(ctx.get("last_actor_pid"))
        post_actor = choose_post_target(offense) if base_action == "PostUp" else None
        drive_actor = choose_finisher_rim(rng, offense, dunk_bias=False, base_action=base_action)

        if base_action == "PostUp":
            if rng.random() < 0.75 and post_actor:
                return post_actor
            if drive_actor:
                return drive_actor
            return _default_actor()

        if rng.random() < 0.80 and drive_actor:
            return drive_actor
        if last_actor:
            return last_actor
        return _default_actor()

    cand = _players_from_roles(offense, _DEFAULT_ACTOR_ROLE_PRIORITY)
    cand = _fill_candidates_with_top_k(offense, cand, cap=4, stat_key="PASS_CREATE")
    extra = {p.pid: _usage_penalty_od(ctx, p.pid) * _recent_actor_mult(ctx, p.pid) for p in cand}
    return choose_weighted_player(rng, cand, "PASS_CREATE", power=1.05, extra_mult_by_pid=extra)
