"""def_role_players.py

Build a mapping of (defensive) scheme roles -> players on the court.

This module is designed to feed quality.compute_quality_score(..., role_players=...).

Key goals:
- Deterministic and debuggable role assignment.
- Small-n exact optimization (brute force) for stability.
- Optional manual overrides via TeamState.roles (role_name -> pid).
- Intended usage pattern: lazy-build in resolve.py and cache in ctx.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import quality
from .models import Player, TeamState


# --------------------------------------------------------------------------------------
# Stat access
# --------------------------------------------------------------------------------------

def engine_get_stat(p: Any, stat: str, default: float = quality.DEFAULT_NEUTRAL_STAT) -> float:
    """Engine-friendly stat getter.

    quality.default_get_stat assumes an object.get(key, default) signature.
    Our Player.get signature is (key, fatigue_sensitive=True), so passing a
    float default would be interpreted as truthy and the default value would
    not be used.

    This wrapper handles Player (and Player-like objects) safely.
    """
    if isinstance(p, Player):
        # Player.get(key, fatigue_sensitive=True)
        try:
            return float(p.get(stat))
        except Exception:
            return float(default)

    # Other objects: try a 1-arg get first, then fall back.
    if hasattr(p, "get") and callable(getattr(p, "get")):
        try:
            return float(p.get(stat))  # type: ignore[misc]
        except TypeError:
            pass
        except Exception:
            return float(default)

    return quality.default_get_stat(p, stat, default)


# --------------------------------------------------------------------------------------
# Config / detail
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RoleAssignmentConfig:
    use_manual_overrides: bool = True
    allow_player_reuse: bool = False
    get_stat: Callable[[Any, str, float], float] = engine_get_stat


@dataclass
class RoleAssignmentDetail:
    scheme: str
    roles: List[str]
    fixed_roles: Dict[str, str]            # role -> pid
    assignment: Dict[str, str]             # role -> pid
    role_fit: Dict[str, float]             # role -> role_score (0..100-ish)
    total_fit: float


# --------------------------------------------------------------------------------------
# Core builder
# --------------------------------------------------------------------------------------

def _role_score(player: Any, role_profile: Mapping[str, float], *, get_stat: Callable[[Any, str, float], float]) -> float:
    """Compute a role fit score using quality.dot_profile."""
    return float(quality.dot_profile(player, role_profile, get_stat))


def _extract_fixed_roles(defense: TeamState, roles: Sequence[str]) -> Dict[str, Player]:
    """Return fixed role assignments from defense.roles if present and valid."""
    fixed: Dict[str, Player] = {}
    for r in roles:
        pid = defense.roles.get(r)
        if not pid:
            continue
        p = defense.find_player(pid)
        if p is not None:
            fixed[r] = p
    return fixed


def build_def_role_players(
    defense: TeamState,
    scheme: Optional[str] = None,
    *,
    config: RoleAssignmentConfig = RoleAssignmentConfig(),
    return_detail: bool = False,
) -> Dict[str, Player] | RoleAssignmentDetail:
    """Build role_players mapping for a defense TeamState.

    Args:
        defense: TeamState of the defending team (must have lineup).
        scheme: Defense scheme override. If None, uses defense.tactics.defense_scheme.
        config: RoleAssignmentConfig.
        return_detail: If True, returns RoleAssignmentDetail.

    Returns:
        Dict[role_name, Player] by default, or RoleAssignmentDetail.
    """
    if scheme is None:
        scheme = getattr(defense.tactics, "defense_scheme", "")
    scheme_c = quality.canonical_scheme(str(scheme))

    role_profiles = quality.ROLE_STAT_PROFILES.get(scheme_c, {})
    roles: List[str] = list(role_profiles.keys())
    if not roles:
        # Unknown scheme or no role profiles.
        if return_detail:
            return RoleAssignmentDetail(
                scheme=scheme_c,
                roles=[],
                fixed_roles={},
                assignment={},
                role_fit={},
                total_fit=0.0,
            )
        return {}

    # 1) Fixed (manual) assignments if provided.
    fixed: Dict[str, Player] = {}
    if config.use_manual_overrides:
        fixed = _extract_fixed_roles(defense, roles)

    fixed_pids = {p.pid for p in fixed.values()}
    remaining_roles = [r for r in roles if r not in fixed]
    remaining_players = [p for p in defense.on_court_players() if p.pid not in fixed_pids]

    # If there are more roles than remaining players, we can optionally reuse.
    if remaining_roles and (len(remaining_roles) > len(remaining_players)) and not config.allow_player_reuse:
        # Fallback: allow reuse to avoid hard failure.
        # This should not happen in normal 5-man lineup schemes.
        remaining_players = list(defense.on_court_players())

    # 2) Compute score matrix for remaining roles/players.
    #    score[role_idx][player_idx]
    get_stat = config.get_stat
    score_mat: List[List[float]] = []
    for r in remaining_roles:
        prof = role_profiles.get(r, {})
        row = [_role_score(p, prof, get_stat=get_stat) for p in remaining_players]
        score_mat.append(row)

    # 3) Solve assignment (small-n brute force) for maximum total fit.
    best_total = float("-inf")
    best_perm: Optional[Tuple[int, ...]] = None

    if not remaining_roles:
        best_total = 0.0
        best_perm = tuple()
    else:
        n_roles = len(remaining_roles)
        n_players = len(remaining_players)

        if n_roles <= n_players:
            # Iterate over permutations of players indices of length n_roles.
            for perm in permutations(range(n_players), n_roles):
                total = 0.0
                for i, j in enumerate(perm):
                    total += score_mat[i][j]
                if total > best_total:
                    best_total = total
                    best_perm = perm
        else:
            # Allow reuse: assign each role its best player.
            # (Should be rare; kept for robustness.)
            perm = []
            total = 0.0
            for i in range(len(remaining_roles)):
                row = score_mat[i]
                j = max(range(len(row)), key=lambda x: row[x])
                perm.append(j)
                total += row[j]
            best_total = total
            best_perm = tuple(perm)

    # 4) Build result mapping.
    assignment: Dict[str, Player] = dict(fixed)
    role_fit: Dict[str, float] = {}
    # fixed role scores
    for r, p in fixed.items():
        role_fit[r] = _role_score(p, role_profiles.get(r, {}), get_stat=get_stat)

    if best_perm is not None:
        for i, j in enumerate(best_perm):
            r = remaining_roles[i]
            p = remaining_players[j]
            assignment[r] = p
            role_fit[r] = score_mat[i][j] if score_mat else _role_score(p, role_profiles.get(r, {}), get_stat=get_stat)

    # 5) Return.
    if return_detail:
        fixed_roles_pid = {r: p.pid for r, p in fixed.items()}
        assignment_pid = {r: p.pid for r, p in assignment.items()}
        total_fit = sum(role_fit.get(r, 0.0) for r in roles)
        return RoleAssignmentDetail(
            scheme=scheme_c,
            roles=list(roles),
            fixed_roles=fixed_roles_pid,
            assignment=assignment_pid,
            role_fit=dict(role_fit),
            total_fit=float(total_fit),
        )

    return assignment


# --------------------------------------------------------------------------------------
# Lazy + cache helper (for resolve.py)
# --------------------------------------------------------------------------------------

def get_or_build_def_role_players(
    ctx: Dict[str, Any],
    defense: TeamState,
    scheme: Optional[str] = None,
    *,
    cache_key: str = "def_role_players",
    config: RoleAssignmentConfig = RoleAssignmentConfig(),
    debug_detail_key: Optional[str] = None,
) -> Dict[str, Player]:
    """Lazy-build and cache role_players in ctx.

    Pattern:
        role_players = get_or_build_def_role_players(ctx, defense)
        score = quality.compute_quality_score(..., role_players=role_players)

    If debug_detail_key is provided, stores RoleAssignmentDetail in ctx[debug_detail_key].
    """
    cached = ctx.get(cache_key)
    if isinstance(cached, dict):
        # Assume mapping role->Player (possibly empty if scheme not supported).
        return cached  # type: ignore[return-value]

    if debug_detail_key:
        detail = build_def_role_players(defense, scheme, config=config, return_detail=True)
        # detail.assignment is role->pid; rebuild role->Player mapping.
        role_players: Dict[str, Player] = {}
        for role, pid in detail.assignment.items():
            p = defense.find_player(pid)
            if p is not None:
                role_players[role] = p
        ctx[cache_key] = role_players
        ctx[debug_detail_key] = detail
        return role_players

    role_players = build_def_role_players(defense, scheme, config=config, return_detail=False)
    ctx[cache_key] = role_players
    return role_players
