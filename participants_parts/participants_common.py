from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

from ..core import weighted_choice
from ..models import Player, TeamState

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


def _append_unique(dst: List[Player], src: List[Player]) -> List[Player]:
    if not src:
        return dst
    return _unique_players(list(dst) + list(src))
