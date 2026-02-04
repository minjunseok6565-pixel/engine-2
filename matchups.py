from __future__ import annotations

"""matchups.py

Plan A (Matchup Overlay MVP)
---------------------------
This module provides a lightweight 5v5 matchup layer without simulating full
player movement. It is designed to support:

1) Base 5v5 assignments (off_pid -> def_pid) derived from on-court lineups.
2) User/AI directives via `tactics.context`:
   - Defense: MATCHUP_LOCKS, MATCHUP_HIDE_PIDS
   - Offense: HUNT_ENABLED, HUNT_RATE, HUNT_TARGET_MODE, HUNT_ACTOR_PID, HUNT_ACTOR_ROLE
3) One-play directives injected into ctx just before resolve:
   - ctx['matchup_play'] = { 'hunt_target_def_pid': ..., 'forced_primary_def_pid': ..., ... }
   - ctx['force_actor_pid'] can be set (one-shot) to bias the actor selection.
4) Selection of the *primary defender* for the current resolved play.
5) A simple per-outcome blending weight `w` that determines how strongly the
   primary defender affects the defense score versus team-aggregate defense.

The module is intentionally conservative and defensive about inputs.
"""

import itertools
import random
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple

from .builders import get_action_base
from .core import clamp
from .models import GameState, Player, TeamState
from .participants import choose_default_actor


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_stat(p: Optional[Player], key: str, default: float = 50.0) -> float:
    if p is None:
        return float(default)
    try:
        return float(p.get(key))
    except Exception:
        return float(default)


def _as_pid(x: Any) -> Optional[str]:
    if x is None:
        return None
    try:
        s = str(x).strip()
    except Exception:
        return None
    return s if s else None


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _parse_hide(def_ctx: Mapping[str, Any], def_on: List[str]) -> List[str]:
    raw = def_ctx.get("MATCHUP_HIDE_PIDS", def_ctx.get("matchup_hide_pids"))
    out: List[str] = []
    for v in _as_list(raw):
        pid = _as_pid(v)
        if pid and pid in def_on:
            out.append(pid)
    # de-dup while preserving order
    seen = set()
    uniq: List[str] = []
    for pid in out:
        if pid in seen:
            continue
        seen.add(pid)
        uniq.append(pid)
    return uniq


def _parse_locks(def_ctx: Mapping[str, Any], off_on: List[str], def_on: List[str]) -> List[Tuple[str, str]]:
    """Return list of (def_pid, off_pid) locks that are currently on-court."""
    raw = def_ctx.get("MATCHUP_LOCKS", def_ctx.get("matchup_locks"))
    locks: List[Tuple[str, str]] = []
    for item in _as_list(raw):
        dp: Optional[str] = None
        op: Optional[str] = None
        if isinstance(item, Mapping):
            dp = _as_pid(item.get("def") or item.get("def_pid") or item.get("defender"))
            op = _as_pid(item.get("off") or item.get("off_pid") or item.get("attacker"))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            dp = _as_pid(item[0])
            op = _as_pid(item[1])
        else:
            # ignore unknown format
            continue

        if not dp or not op:
            continue
        if dp not in def_on or op not in off_on:
            continue
        locks.append((dp, op))

    # de-dup (def_pid unique) while preserving order
    seen_def = set()
    uniq: List[Tuple[str, str]] = []
    for dp, op in locks:
        if dp in seen_def:
            continue
        seen_def.add(dp)
        uniq.append((dp, op))
    return uniq


def _threat_features(p: Player) -> Tuple[float, float, float, float, float]:
    """Return (guard, shoot, rim, post, total) in 0..100-ish."""
    guard = (
        0.40 * _safe_stat(p, "HANDLE_SAFE")
        + 0.35 * _safe_stat(p, "PNR_READ")
        + 0.25 * _safe_stat(p, "PASS_CREATE")
    )
    shoot = max(_safe_stat(p, "SHOT_3"), _safe_stat(p, "SHOT_3_CS"))
    rim = 0.55 * _safe_stat(p, "FIN_RIM") + 0.25 * _safe_stat(p, "FIN_CONTACT") + 0.20 * _safe_stat(p, "FIN_DUNK")
    post = 0.60 * _safe_stat(p, "POST_SCORE") + 0.40 * _safe_stat(p, "POST_CONTROL")
    total = 0.35 * guard + 0.25 * shoot + 0.25 * rim + 0.15 * post
    return guard, shoot, rim, post, total


def _axis_from_threat(guard: float, shoot: float, rim: float, post: float) -> str:
    # shoot maps to perimeter axis
    best = max((guard + 0.20 * shoot, "perim"), (rim, "rim"), (post, "post"), key=lambda x: x[0])
    return best[1]


def _def_guard_index(d: Player, axis: str) -> float:
    if axis == "rim":
        return 0.70 * _safe_stat(d, "DEF_RIM") + 0.20 * _safe_stat(d, "DEF_HELP") + 0.10 * _safe_stat(d, "PHYSICAL")
    if axis == "post":
        return 0.70 * _safe_stat(d, "DEF_POST") + 0.30 * _safe_stat(d, "PHYSICAL")
    # perim (default)
    return 0.70 * _safe_stat(d, "DEF_POA") + 0.20 * _safe_stat(d, "DEF_STEAL") + 0.10 * _safe_stat(d, "PHYSICAL")


def _pair_cost(off_p: Player, def_p: Player, hide_set: set[str]) -> float:
    guard, shoot, rim, post, total = _threat_features(off_p)
    axis = _axis_from_threat(guard, shoot, rim, post)
    dg = _def_guard_index(def_p, axis)
    cost = total * (100.0 - dg)

    # Hide directive: avoid putting this defender on high-threat attackers.
    if def_p.pid in hide_set and total > 65.0:
        cost += 12.0 * (total - 65.0)

    return float(cost)


def _violates_locks(off_players: List[Player], def_perm: Tuple[Player, ...], locks: List[Tuple[str, str]]) -> bool:
    if not locks:
        return False
    # locks are (def_pid, off_pid)
    lock_map = {off_pid: def_pid for (def_pid, off_pid) in locks}
    for i in range(len(off_players)):
        off_pid = off_players[i].pid
        if off_pid in lock_map:
            if def_perm[i].pid != lock_map[off_pid]:
                return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_base_matchups(
    offense: TeamState,
    defense: TeamState,
    def_ctx: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    """Build base 5v5 assignments (off_pid -> def_pid).

    Uses 5! exhaustive search (120 perms) to minimize a simple cost function.
    Respects defensive locks and hide directives.
    """
    off_players = offense.on_court_players()
    def_players = defense.on_court_players()
    if len(off_players) != 5 or len(def_players) != 5:
        # Ensure lineup normalized; TeamState will coerce, but still guard.
        off_players = off_players[:5]
        def_players = def_players[:5]

    ctx_map = def_ctx if isinstance(def_ctx, Mapping) else (getattr(defense.tactics, "context", {}) or {})
    off_on = [p.pid for p in off_players]
    def_on = [p.pid for p in def_players]

    locks = _parse_locks(ctx_map, off_on, def_on)
    hide = _parse_hide(ctx_map, def_on)
    hide_set = set(hide)

    best_cost = float("inf")
    best_perm: Optional[Tuple[Player, ...]] = None
    best_key: Optional[Tuple[str, ...]] = None

    for perm in itertools.permutations(def_players, 5):
        if _violates_locks(off_players, perm, locks):
            continue

        cost = 0.0
        for i in range(5):
            cost += _pair_cost(off_players[i], perm[i], hide_set)

        key = tuple(p.pid for p in perm)
        if cost < best_cost - 1e-9:
            best_cost = cost
            best_perm = perm
            best_key = key
        elif abs(cost - best_cost) <= 1e-9:
            # deterministic tie-break
            if best_key is None or key < best_key:
                best_perm = perm
                best_key = key

    if best_perm is None:
        # Fallback: identity pairing (by order)
        best_perm = tuple(def_players)

    return {off_players[i].pid: best_perm[i].pid for i in range(min(5, len(off_players), len(best_perm)))}


def ensure_matchups(
    ctx: Dict[str, Any],
    offense: TeamState,
    defense: TeamState,
    game_state: Optional[GameState],
    game_cfg: Any,
) -> None:
    """Ensure ctx contains a cached base 5v5 matchup mapping for current on-court lineups."""
    off_on = list(getattr(offense, "on_court_pids", []) or [])
    def_on = list(getattr(defense, "on_court_pids", []) or [])

    dctx = getattr(defense.tactics, "context", {})
    if not isinstance(dctx, Mapping):
        dctx = {}
    locks = _parse_locks(dctx, off_on, def_on)
    hide = _parse_hide(dctx, def_on)

    # keep sig small & stable; include only lineup + lock/hide directives
    sig = (
        tuple(off_on),
        tuple(def_on),
        tuple(sorted(locks)),
        tuple(sorted(hide)),
    )

    if ctx.get("_matchups_sig") == sig and isinstance(ctx.get("matchups"), dict):
        return

    m = build_base_matchups(offense, defense, def_ctx=dctx)
    ctx["matchups"] = m
    ctx["_matchups_rev"] = {d: o for (o, d) in m.items()}
    ctx["_matchups_sig"] = sig


def maybe_prepare_matchup_play_context(
    rng: random.Random,
    offense: TeamState,
    defense: TeamState,
    action: str,
    outcome: str,
    tags: Dict[str, Any],
    ctx: Dict[str, Any],
    game_cfg: Any,
) -> None:
    """Optionally inject one-play matchup directives into ctx.

    This is expected to be called immediately after outcome is sampled but before
    resolve_outcome() is called.
    """
    # clear previous play directive to avoid leaking
    ctx.pop("matchup_play", None)

    # If another system already forced an actor (e.g., ORB putback), do not override.
    if ctx.get("force_actor_pid"):
        return

    octx = getattr(offense.tactics, "context", {})
    if not isinstance(octx, Mapping):
        return

    if not bool(octx.get("HUNT_ENABLED", False)):
        return

    base_action = get_action_base(action, game_cfg)
    allowed = octx.get("HUNT_ALLOWED_BASE_ACTIONS")
    if not isinstance(allowed, (list, tuple)):
        allowed = ["ISO", "Drive", "PnR", "PostUp", "Post"]
    if base_action not in set(str(x) for x in allowed):
        return

    try:
        rate = clamp(float(octx.get("HUNT_RATE", 0.18)), 0.0, 1.0)
    except Exception:
        rate = 0.18
    if rng.random() >= rate:
        return

    # Pick target defender
    target_mode = str(octx.get("HUNT_TARGET_MODE", "weakest_poa"))
    target_pid: Optional[str] = None
    def_players = defense.on_court_players()
    def_on = [p.pid for p in def_players]

    if target_mode.startswith("pid:"):
        target_pid = _as_pid(target_mode.split(":", 1)[1])
        if target_pid not in def_on:
            target_pid = None

    if target_pid is None:
        if target_mode == "weakest_total":
            def score_total(d: Player) -> float:
                return 0.55 * _safe_stat(d, "DEF_POA") + 0.25 * _safe_stat(d, "DEF_HELP") + 0.20 * _safe_stat(d, "PHYSICAL")

            target_pid = min(def_players, key=score_total).pid if def_players else None
        else:
            # default: weakest_poa
            target_pid = min(def_players, key=lambda d: _safe_stat(d, "DEF_POA")).pid if def_players else None

    if target_pid is None:
        return

    # Pick hunting actor
    actor_pid: Optional[str] = _as_pid(octx.get("HUNT_ACTOR_PID"))
    if actor_pid and not offense.is_on_court(actor_pid):
        actor_pid = None

    if actor_pid is None:
        role = str(octx.get("HUNT_ACTOR_ROLE", "Shot_Creator"))
        pid = (getattr(offense, "roles", {}) or {}).get(role)
        if isinstance(pid, str) and pid and offense.is_on_court(pid):
            actor_pid = pid

    if actor_pid is None:
        actor_pid = choose_default_actor(offense).pid

    ctx["matchup_play"] = {
        "hunt_target_def_pid": target_pid,
        "hunt_actor_pid": actor_pid,
        "event_tag_hint": "HUNT",
    }
    ctx["force_actor_pid"] = actor_pid


def pick_primary_defender_for_play(
    rng: random.Random,
    actor_pid: str,
    outcome: str,
    base_action: str,
    offense: TeamState,
    defense: TeamState,
    ctx: Dict[str, Any],
    game_cfg: Any,
) -> Tuple[Optional[str], str, Dict[str, Any]]:
    """Pick the primary defender pid for the resolved play.

    Priority (Plan A):
      1) ctx['matchup_play']['forced_primary_def_pid'] (MANUAL)
      2) ctx['matchup_play']['hunt_target_def_pid'] (HUNT)
      3) ctx['matchups'][actor_pid] (BASE)
      4) fallback: best DEF_POA on the floor (FALLBACK)

    Note: consumes ctx['matchup_play'] (one-shot).
    """
    ensure_matchups(ctx, offense, defense, game_state=None, game_cfg=game_cfg)

    base_map = ctx.get("matchups") if isinstance(ctx.get("matchups"), dict) else {}
    base_def = _as_pid(base_map.get(actor_pid))

    play = ctx.pop("matchup_play", None)
    play_map = play if isinstance(play, Mapping) else {}

    forced = _as_pid(play_map.get("forced_primary_def_pid"))
    if forced and defense.is_on_court(forced):
        return forced, "MANUAL", {"base_def_pid": base_def, "play": dict(play_map)}

    hunt = _as_pid(play_map.get("hunt_target_def_pid"))
    if hunt and defense.is_on_court(hunt):
        return hunt, "HUNT", {"base_def_pid": base_def, "play": dict(play_map)}

    if base_def and defense.is_on_court(base_def):
        return base_def, "BASE", {"base_def_pid": base_def, "play": dict(play_map)}

    # fallback: best POA defender
    def_players = defense.on_court_players()
    if def_players:
        best = max(def_players, key=lambda d: _safe_stat(d, "DEF_POA"))
        return best.pid, "FALLBACK", {"base_def_pid": base_def, "play": dict(play_map)}
    return None, "NONE", {"base_def_pid": base_def, "play": dict(play_map)}


def matchup_blend_w(game_cfg: Any, outcome: str, base_action: str, matchup_event: str) -> float:
    """Return blending weight w in [0, 0.85] for defense score mixing."""
    knobs = getattr(game_cfg, "knobs", {})
    if not isinstance(knobs, Mapping):
        knobs = {}

    if isinstance(outcome, str) and outcome.startswith("SHOT_"):
        w = float(knobs.get("matchup_w_shot", 0.48))
    elif isinstance(outcome, str) and outcome.startswith("PASS_"):
        w = float(knobs.get("matchup_w_pass", 0.22))
    elif isinstance(outcome, str) and outcome.startswith("TO_"):
        w = float(knobs.get("matchup_w_to", 0.28))
    elif isinstance(outcome, str) and outcome.startswith("FOUL_"):
        w = float(knobs.get("matchup_w_foul", 0.30))
    else:
        w = float(knobs.get("matchup_w_misc", 0.20))

    if matchup_event == "HUNT":
        w += float(knobs.get("matchup_w_bonus_hunt", 0.10))
    elif matchup_event == "MANUAL":
        w += float(knobs.get("matchup_w_bonus_manual", 0.06))

    return clamp(float(w), 0.0, 0.85)
