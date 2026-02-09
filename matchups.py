"""
matchups.py

Plan-1 Matchup MVP
------------------

This module provides a lightweight 5v5 matchup map for the possession simulator.

Goals (v1):
  - Build a stable OFF_PID -> DEF_PID mapping for current on-court units.
  - Honor basic instructions from defense.tactics.context:
      * MATCHUP_LOCKS: force (def_pid -> off_pid) assignments.
      * MATCHUP_HIDE_PIDS: discourage assigning those defenders to high-threat offensive players.
      * MATCHUP_ASSIGNMENTS: defender-first preferences (def_pid -> {primary_off_pid, secondary_off_role}).
      * MATCHUP_LOCKDOWN: single hard lock (def_pid -> off_pid/off_role/tag).
  - Honor temporary locks passed through ctx (ctx["matchups_temp_locks"]).
  - Provide a helper to fetch the primary defender for a given offensive pid.

Important contracts:
  - All return values must be JSON-serializable (pids are strings; meta is dict/list/float).
  - Must be safe with the engine Player.get signature (fatigue_sensitive kwarg).
  - 5-man matching is tiny; we use brute-force permutations for stability.
"""

from __future__ import annotations

from itertools import permutations
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .core import clamp
from .def_role_players import engine_get_stat
from .models import Player, TeamState


# -------------------------
# Public API
# -------------------------


def build_matchups(
    offense: TeamState,
    defense: TeamState,
    ctx: Dict[str, Any],
    rng: Any = None,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Any]]:
    """Build a 5v5 matchup mapping for the current on-court units.

    Returns:
      - matchups_map: {off_pid: def_pid}
      - matchups_rev: {def_pid: off_pid}
      - meta: JSON-friendly debug metadata (algo/score/locks/hides)

    Notes:
      - The algorithm is deterministic given the same on-court players + instructions.
        (rng is accepted for future expansion but not used in v1.)
    """

    off_players = offense.on_court_players()
    def_players = defense.on_court_players()
    off_pids = [str(p.pid) for p in off_players]
    def_pids = [str(p.pid) for p in def_players]

    off_set = set(off_pids)
    def_set = set(def_pids)

    # Extract defensive instructions (LOCK/HIDE/ASSIGN/LOCKDOWN) – keep JSON-friendly.
    hides_raw = _extract_hides(defense)

    # Temporary locks (ctx-driven) are used for one-off, possession-scoped directives.
    # Example: a "DENY" response can pre-switch to an alternate defender.
    temp_locks_raw = _extract_temp_locks(ctx)

    # Optional single lockdown instruction.
    lockdown_spec = _extract_lockdown(defense)

    # Defender-first matchup preferences.
    assignments_raw = _extract_assignments(defense)

    # Normalize / filter hides to on-court only.
    hides = [pid for pid in hides_raw if pid in def_set]

    # Precompute threat + buckets.
    threat = {str(p.pid): _threat_score(p) for p in off_players}
    off_bucket = {str(p.pid): _bucket_offense_player(p) for p in off_players}
    def_bucket = {str(p.pid): _bucket_defense_player(p) for p in def_players}

    # Resolve defender-first assignment targets to on-court OFF pids.
    assignment_target: Dict[str, str] = {}
    if isinstance(assignments_raw, Mapping):
        for dpid, spec in assignments_raw.items():
            d = str(dpid or "").strip()
            if not d or d not in def_set:
                continue
            if not isinstance(spec, Mapping):
                continue
            target = _resolve_assignment_target(offense, spec, off_set)
            if target:
                assignment_target[d] = str(target)

    # Merge locks with precedence:
    #   1) ctx.matchups_temp_locks (highest)
    #   2) defense MATCHUP_LOCKDOWN
    #   3) defense MATCHUP_LOCKS / MATCHUP_LOCK
    locks_explicit = _extract_locks(defense)
    lockdown_lock = _resolve_lockdown_lock(lockdown_spec, offense, off_players, off_pids, def_pids, threat)
    locks_merged: List[Mapping[str, Any]] = []
    locks_merged.extend(temp_locks_raw)
    if isinstance(lockdown_lock, Mapping) and lockdown_lock.get("def_pid") and lockdown_lock.get("off_pid"):
        locks_merged.append(lockdown_lock)
    locks_merged.extend(locks_explicit)

    # Normalize / filter locks to on-court only.
    locks = _normalize_locks(locks_merged, off_pids, def_pids)

    # Fixed assignments from locks (after normalization).
    fixed_off: set[str] = set()
    fixed_def: set[str] = set()
    fixed_pairs: List[Tuple[str, str]] = []  # (off_pid, def_pid)
    for item in locks:
        dpid = str(item.get("def_pid") or "")
        opid = str(item.get("off_pid") or "")
        if not dpid or not opid:
            continue
        if opid in fixed_off or dpid in fixed_def:
            continue
        fixed_pairs.append((opid, dpid))
        fixed_off.add(opid)
        fixed_def.add(dpid)

    remaining_off = [pid for pid in off_pids if pid not in fixed_off]
    remaining_def = [pid for pid in def_pids if pid not in fixed_def]

    # Safety fallback if locks created a mismatch (should be rare due to normalization).
    if len(remaining_off) != len(remaining_def):
        fixed_pairs = []
        remaining_off = list(off_pids)
        remaining_def = list(def_pids)

    # Build pid->Player maps for fast access.
    off_by_pid = {str(p.pid): p for p in off_players}
    def_by_pid = {str(p.pid): p for p in def_players}

    def _pair_score(off_pid: str, def_pid: str) -> float:
        op = off_by_pid.get(off_pid)
        dp = def_by_pid.get(def_pid)
        if op is None or dp is None:
            return 0.0
        t = float(threat.get(off_pid, 50.0))
        t_norm = clamp((t - 40.0) / 60.0, 0.0, 1.0)

        ob = off_bucket.get(off_pid, "wing")
        db = def_bucket.get(def_pid, "wing")
        cap = _def_capability(dp, ob)

        # Base: reward defenders more when guarding higher-threat players.
        score = cap * (0.40 + 0.60 * t_norm)

        # Bucket mismatch penalties (simple, stable heuristics).
        score -= _mismatch_penalty(ob, db)

        # HIDE penalty: strong discouragement for high-threat assignments.
        if def_pid in hides:
            score -= 25.0 * t_norm

        # Assignment bonus: defender-first preference (on-court only).
        # This is a *preference*, not a hard constraint; it should not dominate LOCKs.
        # Tuned to be meaningful while still allowing the optimizer to solve conflicts.
        if assignment_target.get(def_pid) == off_pid:
            score += 18.0 + 22.0 * t_norm

        return float(score)

    fixed_score = sum(_pair_score(opid, dpid) for opid, dpid in fixed_pairs)

    best_score = float("-inf")
    best_key: Optional[Tuple[str, ...]] = None
    best_assign: Optional[Dict[str, str]] = None

    # Iterate all remaining defender permutations (<= 120).
    for perm in permutations(remaining_def):
        total = fixed_score
        mapping: Dict[str, str] = {opid: dpid for opid, dpid in fixed_pairs}
        for opid, dpid in zip(remaining_off, perm):
            mapping[opid] = dpid
            total += _pair_score(opid, dpid)

        # Deterministic tie-break: lexicographic defender tuple in off_pids order.
        key = tuple(mapping.get(opid, "") for opid in off_pids)
        if (total > best_score + 1e-9) or (abs(total - best_score) <= 1e-9 and (best_key is None or key < best_key)):
            best_score = float(total)
            best_key = key
            best_assign = mapping

    # Finalize; ensure full coverage.
    matchups_map: Dict[str, str] = {}
    if isinstance(best_assign, dict) and len(best_assign) == len(off_pids):
        matchups_map = {str(opid): str(best_assign.get(opid, "")) for opid in off_pids}
    else:
        # Absolute fallback: identity mapping by index.
        matchups_map = {str(opid): str(def_pids[i]) for i, opid in enumerate(off_pids) if i < len(def_pids)}

    # Build reverse mapping.
    matchups_rev: Dict[str, str] = {str(dpid): str(opid) for opid, dpid in matchups_map.items() if dpid}

    meta: Dict[str, Any] = {
        "algo": "permute_v1",
        "score": float(best_score if best_score != float("-inf") else fixed_score),
        "locks": [{"def_pid": str(x.get("def_pid")), "off_pid": str(x.get("off_pid"))} for x in locks],
        "hides": [str(pid) for pid in hides],
        "temp_locks": [
            {"def_pid": str(x.get("def_pid") or ""), "off_pid": str(x.get("off_pid") or "")}
            for x in temp_locks_raw
            if isinstance(x, Mapping)
        ],
        "lockdown": {
            "def_pid": str(lockdown_lock.get("def_pid") or ""),
            "off_pid": str(lockdown_lock.get("off_pid") or ""),
        }
        if isinstance(lockdown_lock, Mapping) and lockdown_lock.get("def_pid") and lockdown_lock.get("off_pid")
        else None,
        "assignments": [
            {"def_pid": str(dpid), "off_pid": str(opid)}
            for dpid, opid in sorted(assignment_target.items(), key=lambda kv: (str(kv[0]), str(kv[1])))
        ],
    }
    return matchups_map, matchups_rev, meta


def get_primary_defender_pid(
    off_pid: str,
    defense: TeamState,
    ctx: Dict[str, Any],
    off_player: Optional[Player] = None,
) -> Tuple[Optional[str], str, Optional[str]]:
    """Return (def_pid, source, event) for the given offensive pid.

    Source priority (v1):
      1) ctx["matchup_force"] if it targets this off_pid
      2) ctx["matchups_map"]
      3) fallback best on-court defender by simple heuristics

    Note: consumption (ttl decrement/pop) is handled by resolve.py per the contract.
    """

    opid = str(off_pid or "")
    if not opid:
        return None, "fallback", None

    # 1) one-shot force
    force = ctx.get("matchup_force")
    if isinstance(force, dict):
        f_opid = str(force.get("off_pid") or "")
        f_dpid = str(force.get("def_pid") or "")
        if f_opid and f_dpid and f_opid == opid and defense.is_on_court(f_dpid):
            ev = str(force.get("event") or "") or None
            return f_dpid, "force", ev

    # 2) possession map
    m = ctx.get("matchups_map")
    if isinstance(m, dict):
        dpid = str(m.get(opid) or "")
        if dpid and defense.is_on_court(dpid):
            return dpid, "map", None

    # 3) fallback
    dpid = _fallback_defender_pid(defense, off_player)
    return dpid, "fallback", None


# -------------------------
# Instruction extraction
# -------------------------


def _tactics_context(team: TeamState) -> Mapping[str, Any]:
    try:
        t = getattr(team, "tactics", None)
        ctx = getattr(t, "context", None)
        return ctx if isinstance(ctx, Mapping) else {}
    except Exception:
        return {}


def _extract_locks(defense: TeamState) -> List[Dict[str, str]]:
    """Extract MATCHUP_LOCKS from defense.tactics.context.

    Accepted shapes:
      - list of {"def_pid": "...", "off_pid": "..."}
      - single dict with keys def_pid/off_pid
      - dict mapping {def_pid: off_pid}
    """
    c = _tactics_context(defense)
    raw = c.get("MATCHUP_LOCKS")
    if raw is None:
        raw = c.get("MATCHUP_LOCK")

    if raw is None:
        return []

    out: List[Dict[str, str]] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                dpid = str(item.get("def_pid") or "")
                opid = str(item.get("off_pid") or "")
                if dpid and opid:
                    out.append({"def_pid": dpid, "off_pid": opid})
        return out

    if isinstance(raw, dict):
        # Case 1: explicit pair dict
        if "def_pid" in raw and "off_pid" in raw:
            dpid = str(raw.get("def_pid") or "")
            opid = str(raw.get("off_pid") or "")
            if dpid and opid:
                return [{"def_pid": dpid, "off_pid": opid}]
            return []

        # Case 2: mapping def_pid -> off_pid
        for dpid, opid in raw.items():
            d = str(dpid or "")
            o = str(opid or "")
            if d and o:
                out.append({"def_pid": d, "off_pid": o})
        return out

    return []


def _extract_hides(defense: TeamState) -> List[str]:
    """Extract MATCHUP_HIDE_PIDS from defense.tactics.context."""
    c = _tactics_context(defense)
    raw = c.get("MATCHUP_HIDE_PIDS")
    if raw is None:
        raw = c.get("MATCHUP_HIDE_PID")
    if raw is None:
        return []

    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x or "").strip()]

    s = str(raw or "").strip()
    return [s] if s else []

def _extract_assignments(defense: TeamState) -> Mapping[str, Mapping[str, Any]]:
    """Extract MATCHUP_ASSIGNMENTS from defense.tactics.context.

    Expected shape:
      MATCHUP_ASSIGNMENTS = {
        def_pid: {"primary_off_pid": "...", "secondary_off_role": "..."},
        ...
      }

    Notes:
      - We allow simple aliases (off_pid/off_role) to reduce friction.
      - Resolution to on-court pids is handled in build_matchups.
    """
    c = _tactics_context(defense)
    raw = c.get("MATCHUP_ASSIGNMENTS")
    if not isinstance(raw, Mapping):
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for dpid, spec in raw.items():
        d = str(dpid or "").strip()
        if not d or not isinstance(spec, Mapping):
            continue

        primary = str(spec.get("primary_off_pid") or spec.get("off_pid") or "").strip()
        secondary = str(spec.get("secondary_off_role") or spec.get("off_role") or "").strip()
        if not primary and not secondary:
            continue

        out[d] = {"primary_off_pid": primary, "secondary_off_role": secondary}

    return out


def _resolve_assignment_target(offense: TeamState, spec: Mapping[str, Any], off_set: set[str]) -> Optional[str]:
    """Resolve a single defender assignment spec to an on-court offensive pid."""
    primary = str(spec.get("primary_off_pid") or spec.get("off_pid") or "").strip()
    if primary and primary in off_set:
        return primary

    role = str(spec.get("secondary_off_role") or spec.get("off_role") or "").strip()
    if role:
        try:
            roles = getattr(offense, "roles", None)
            if isinstance(roles, Mapping):
                pid = str(roles.get(role) or "").strip()
                if pid and pid in off_set:
                    return pid
        except Exception:
            return None

    return None


def _extract_lockdown(defense: TeamState) -> Mapping[str, Any]:
    """Extract MATCHUP_LOCKDOWN from defense.tactics.context.

    Accepted shapes:
      - {"def_pid": "dp..", "target": {"off_pid": "op.."}}
      - {"def_pid": "dp..", "target": {"off_role": "..."}}
      - {"def_pid": "dp..", "target": {"tag": "BEST_THREAT"}}
      - or the same keys at the top-level (without nested "target").
    """
    c = _tactics_context(defense)
    raw = c.get("MATCHUP_LOCKDOWN")
    if not isinstance(raw, Mapping):
        return {}

    dpid = str(raw.get("def_pid") or "").strip()
    if not dpid:
        return {}

    target: Mapping[str, Any] = raw
    t = raw.get("target")
    if isinstance(t, Mapping):
        target = t

    off_pid = str(target.get("off_pid") or "").strip()
    off_role = str(target.get("off_role") or "").strip()
    tag = str(target.get("tag") or "").strip().upper()

    out: Dict[str, Any] = {"def_pid": dpid}
    if off_pid:
        out["off_pid"] = off_pid
    if off_role:
        out["off_role"] = off_role
    if tag:
        out["tag"] = tag
    return out


def _resolve_lockdown_lock(
    lockdown_spec: Mapping[str, Any],
    offense: TeamState,
    off_players: Sequence[Player],
    off_pids: Sequence[str],
    def_pids: Sequence[str],
    threat: Mapping[str, float],
) -> Optional[Dict[str, str]]:
    """Resolve a MATCHUP_LOCKDOWN spec to a concrete {def_pid, off_pid} lock (on-court only)."""
    if not isinstance(lockdown_spec, Mapping) or not lockdown_spec:
        return None

    dpid = str(lockdown_spec.get("def_pid") or "").strip()
    if not dpid or dpid not in set(str(x) for x in def_pids):
        return None

    off_set = set(str(x) for x in off_pids)

    # 1) explicit off_pid
    opid = str(lockdown_spec.get("off_pid") or "").strip()
    if opid and opid in off_set:
        return {"def_pid": dpid, "off_pid": opid}

    # 2) role -> pid (if on-court)
    off_role = str(lockdown_spec.get("off_role") or "").strip()
    if off_role:
        try:
            roles = getattr(offense, "roles", None)
            if isinstance(roles, Mapping):
                pid = str(roles.get(off_role) or "").strip()
                if pid and pid in off_set:
                    return {"def_pid": dpid, "off_pid": pid}
        except Exception:
            pass

    # 3) tag-based (currently only BEST_THREAT)
    tag = str(lockdown_spec.get("tag") or "").strip().upper()
    if tag == "BEST_THREAT":
        best_pid = None
        best_val = float("-inf")
        for p in off_players:
            pid = str(getattr(p, "pid", "") or "")
            if pid not in off_set:
                continue
            val = float(threat.get(pid, 50.0))
            if val > best_val + 1e-9:
                best_val = val
                best_pid = pid
        if best_pid:
            return {"def_pid": dpid, "off_pid": str(best_pid)}

    return None


def _extract_temp_locks(ctx: Mapping[str, Any]) -> List[Dict[str, str]]:
    """Extract temporary locks from ctx.

    Expected shape (ctx):
      matchups_temp_locks = [{"def_pid": "...", "off_pid": "...", ...}, ...]

    Notes:
      - This is a low-level hook intended for possession-scoped directives.
      - Only def_pid/off_pid are used; other keys are ignored.
    """
    raw = None
    try:
        raw = ctx.get("matchups_temp_locks")
    except Exception:
        raw = None

    if raw is None:
        return []

    out: List[Dict[str, str]] = []

    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            dpid = str(item.get("def_pid") or "").strip()
            opid = str(item.get("off_pid") or "").strip()
            if dpid and opid:
                out.append({"def_pid": dpid, "off_pid": opid})
        return out

    if isinstance(raw, Mapping):
        # Case 1: explicit pair dict
        if "def_pid" in raw and "off_pid" in raw:
            dpid = str(raw.get("def_pid") or "").strip()
            opid = str(raw.get("off_pid") or "").strip()
            if dpid and opid:
                return [{"def_pid": dpid, "off_pid": opid}]
            return []

        # Case 2: mapping def_pid -> off_pid
        for dpid, opid in raw.items():
            d = str(dpid or "").strip()
            o = str(opid or "").strip()
            if d and o:
                out.append({"def_pid": d, "off_pid": o})
        return out

    return []


def _normalize_locks(locks: Sequence[Mapping[str, Any]], off_pids: Sequence[str], def_pids: Sequence[str]) -> List[Dict[str, str]]:
    """Filter/normalize locks to valid on-court pids, de-dup deterministically."""
    off_set = set(str(x) for x in off_pids)
    def_set = set(str(x) for x in def_pids)

    out: List[Dict[str, str]] = []
    seen_off: set[str] = set()
    seen_def: set[str] = set()
    for item in locks or []:
        if not isinstance(item, Mapping):
            continue
        dpid = str(item.get("def_pid") or "").strip()
        opid = str(item.get("off_pid") or "").strip()
        if not dpid or not opid:
            continue
        if dpid not in def_set or opid not in off_set:
            continue
        if dpid in seen_def or opid in seen_off:
            continue
        out.append({"def_pid": dpid, "off_pid": opid})
        seen_def.add(dpid)
        seen_off.add(opid)
    return out


# -------------------------
# Scoring heuristics
# -------------------------


def _bucket_offense_player(p: Player) -> str:
    """Coarse bucket: guard/wing/big.

    This is intentionally heuristic and robust to varied position strings.
    """
    pos = str(getattr(p, "pos", "") or "").upper()
    # Post-heavy players behave like "big" for matchup purposes.
    post = 0.5 * engine_get_stat(p, "POST_SCORE", 50.0) + 0.5 * engine_get_stat(p, "POST_CONTROL", 50.0)
    if post >= 62.0 or ("C" in pos):
        return "big"
    if ("G" in pos) and ("F" not in pos) and ("C" not in pos):
        return "guard"
    if "F" in pos:
        return "wing"
    return "wing"


def _bucket_defense_player(p: Player) -> str:
    pos = str(getattr(p, "pos", "") or "").upper()
    rim = engine_get_stat(p, "DEF_RIM", 50.0)
    post = engine_get_stat(p, "DEF_POST", 50.0)
    if ("C" in pos) or rim >= 60.0 or post >= 60.0:
        return "big"
    if ("G" in pos) and ("F" not in pos) and ("C" not in pos):
        return "guard"
    if "F" in pos:
        return "wing"
    return "wing"


def _threat_score(p: Player) -> float:
    """Compute a compact offensive threat proxy (0..100-ish).

    We intentionally use only common derived keys and default to 50.
    """
    # Shooting / creation
    t3 = 0.55 * engine_get_stat(p, "SHOT_3_OD", 50.0) + 0.45 * engine_get_stat(p, "SHOT_3_CS", 50.0)
    mid = 0.55 * engine_get_stat(p, "SHOT_MID_PU", 50.0) + 0.45 * engine_get_stat(p, "SHOT_MID_CS", 50.0)
    rim = 0.50 * engine_get_stat(p, "FIRST_STEP", 50.0) + 0.25 * engine_get_stat(p, "FIN_LAYUP", 50.0) + 0.25 * engine_get_stat(p, "FIN_DUNK", 50.0)
    post = 0.55 * engine_get_stat(p, "POST_SCORE", 50.0) + 0.45 * engine_get_stat(p, "POST_CONTROL", 50.0)
    play = 0.55 * engine_get_stat(p, "PASS_CREATE", 50.0) + 0.45 * engine_get_stat(p, "PNR_READ", 50.0)

    # Small blend (handlers vs bigs will naturally differ by where they are strong).
    raw = 0.23 * t3 + 0.16 * mid + 0.22 * rim + 0.15 * post + 0.24 * play
    return float(clamp(raw, 0.0, 100.0))


def _def_capability(defender: Player, off_bucket: str) -> float:
    """Compute matchup-relevant defensive capability (0..100-ish)."""
    if off_bucket == "big":
        d_post = engine_get_stat(defender, "DEF_POST", 50.0)
        d_rim = engine_get_stat(defender, "DEF_RIM", 50.0)
        phys = engine_get_stat(defender, "PHYSICAL", 50.0)
        return float(clamp(0.55 * d_post + 0.20 * d_rim + 0.25 * phys, 0.0, 100.0))

    d_poa = engine_get_stat(defender, "DEF_POA", 50.0)
    d_stl = engine_get_stat(defender, "DEF_STEAL", 50.0)
    phys = engine_get_stat(defender, "PHYSICAL", 50.0)
    return float(clamp(0.55 * d_poa + 0.25 * d_stl + 0.20 * phys, 0.0, 100.0))


def _mismatch_penalty(off_bucket: str, def_bucket: str) -> float:
    if off_bucket == def_bucket:
        return 0.0
    if off_bucket == "big" and def_bucket == "guard":
        return 18.0
    if off_bucket == "guard" and def_bucket == "big":
        return 8.0
    # guard<->wing or wing<->big
    return 3.0


# -------------------------
# Fallback defender selection
# -------------------------


def _fallback_defender_pid(defense: TeamState, off_player: Optional[Player]) -> Optional[str]:
    """Pick a reasonable on-court defender when no mapping exists."""
    defenders = defense.on_court_players()
    if not defenders:
        return None

    if off_player is None:
        # Safe fallback: best POA defender.
        best = max(defenders, key=lambda p: engine_get_stat(p, "DEF_POA", 50.0))
        return str(best.pid) if best else None

    ob = _bucket_offense_player(off_player)
    best = max(defenders, key=lambda p: _def_capability(p, ob))
    return str(best.pid) if best else None
