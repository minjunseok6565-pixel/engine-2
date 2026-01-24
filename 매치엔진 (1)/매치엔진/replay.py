from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from .models import GameState, TeamState


# ---------------------------------------------------------------------------
# Replay / Play-by-play event emission
# - Single source of truth: GameState.replay_events
# - One incident => exactly one event append (call emit_event once)
# ---------------------------------------------------------------------------

# Fields that are owned by the emitter (callers must not override them via **payload).
_RESERVED_KEYS: Set[str] = {
    # core context
    "seq",
    "event_type",
    "quarter",
    "clock_sec",
    "shot_clock_sec",
    "game_elapsed_sec",
    "possession_index",
    "score_home",
    "score_away",
    "home_team_id",
    "away_team_id",
    # lineup context (owned by emitter)
    "lineup_version",
    "lineup_version_team",
    "lineup_version_by_team_id",
    "on_court_home",
    "on_court_away",
    "on_court_by_team_id",
    # team mapping
    "team_side",
    "team_id",
    "opp_side",
    "opp_team_id",
}


def _side_other(side: str) -> str:
    s = str(side)
    if s == "home":
        return "away"
    if s == "away":
        return "home"
    raise ValueError(f"invalid team_side: {side!r}")


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _fmt_clock_mmss(clock_sec: Any) -> str:
    """
    Format remaining period clock seconds into 'MM:SS'.
    - We floor/truncate toward 0 for display (9:32.9 -> 9:32).
    - Clamp negative to 0.
    """
    sec = _safe_float(clock_sec, 0.0)
    if sec < 0:
        sec = 0.0
    s = int(sec)  # truncate
    m = s // 60
    r = s % 60
    return f"{m:02d}:{r:02d}"

def _round_1dp(x: Any) -> Any:
    """
    Round numeric values to 1 decimal place (for readability in replay logs).
    Leaves None and non-numerics untouched.
    """
    if x is None:
        return None
    try:
        # bool is int subclass; treat it as non-numeric for rounding purposes
        if isinstance(x, bool):
            return x
        v = float(x)
        if v < 0:
            v = 0.0
        return round(v, 1)
    except Exception:
        return x


def _compute_game_elapsed_sec(game_state: GameState, rules: Optional[Mapping[str, Any]]) -> int:
    """
    Prefer the project's canonical elapsed-time helper from sim_rotation.py.
    Fallback to a simple best-effort computation if import fails.
    """
    # 1) Canonical: sim_rotation._game_elapsed_sec(game_state, rules)
    try:
        from .sim_rotation import _game_elapsed_sec  # type: ignore

        return _safe_int(_game_elapsed_sec(game_state, rules or {}), 0)
    except Exception:
        pass

    # 2) Fallback (best-effort): regulation quarters + OT
    q = _safe_int(getattr(game_state, "quarter", 1), 1)
    # Remaining clock is "time left in current period" in this engine.
    clock_left = _safe_float(getattr(game_state, "clock_sec", 0.0), 0.0)
    quarter_len = _safe_float((rules or {}).get("quarter_length", 720), 720.0)
    ot_len = _safe_float((rules or {}).get("overtime_length", 300), 300.0)
    reg_q = _safe_int((rules or {}).get("quarters", 4), 4)

    # elapsed in current period
    period_len = quarter_len if q <= reg_q else ot_len
    elapsed_in_period = max(0.0, float(period_len) - float(clock_left))

    # elapsed of completed prior periods
    if q <= reg_q:
        prior = max(0, q - 1) * quarter_len
    else:
        prior_reg = reg_q * quarter_len
        prior_ot = max(0, q - reg_q - 1) * ot_len
        prior = prior_reg + prior_ot
    return _safe_int(prior + elapsed_in_period, 0)


def _get_team_id(team: TeamState) -> str:
    # In current project, TeamState.name is already effectively team_id.
    # If team.team_id exists, prefer it.
    tid = getattr(team, "team_id", None)
    if tid:
        return str(tid)
    return str(getattr(team, "name", ""))


def _ensure_team_mapping(game_state: GameState, home: TeamState, away: TeamState) -> Tuple[str, str]:
    """
    Ensure game_state has stable team-id mapping for this game.
    Returns (home_team_id, away_team_id).
    """
    home_id = str(getattr(game_state, "home_team_id", None) or _get_team_id(home))
    away_id = str(getattr(game_state, "away_team_id", None) or _get_team_id(away))

    # Write-through so downstream callers can rely on this.
    game_state.home_team_id = home_id
    game_state.away_team_id = away_id

    # Build mapping dicts if missing/incomplete.
    st = getattr(game_state, "side_to_team_id", None)
    if not isinstance(st, dict):
        game_state.side_to_team_id = {}
    tt = getattr(game_state, "team_id_to_side", None)
    if not isinstance(tt, dict):
        game_state.team_id_to_side = {}

    game_state.side_to_team_id.setdefault("home", home_id)
    game_state.side_to_team_id.setdefault("away", away_id)
    game_state.team_id_to_side.setdefault(home_id, "home")
    game_state.team_id_to_side.setdefault(away_id, "away")

    return home_id, away_id


def emit_event(
    game_state: GameState,
    *,
    event_type: str,
    home: TeamState,
    away: TeamState,
    rules: Optional[Mapping[str, Any]] = None,
    # team mapping (either or both can be provided; emitter will fill/validate)
    team_side: Optional[str] = None,
    team_id: Optional[str] = None,
    opp_side: Optional[str] = None,
    opp_team_id: Optional[str] = None,
    # flow keys (optional but standardized when present)
    pos_start: Optional[str] = None,
    pos_start_next: Optional[str] = None,
    pos_start_next_override: Optional[str] = None,
    # some callers (e.g. timeout) may want to override possession index snapshot
    possession_index: Optional[int] = None,
    # include on-court snapshots (for replay seeking / exact on-court reconstruction)
    # when True: emits on_court_* and lineup_version_by_team_id in a standard format
    include_lineups: bool = False,
    # strict validation: mismatched team_side/team_id is a hard error (prevents bad logs)
    strict: bool = True,
    **payload: Any,
) -> Dict[str, Any]:
    """
    Append a replay_event dict to game_state.replay_events following the project's final spec.

    Source of truth: replay_events only.
    - seq auto-increments
    - common context auto-filled from game_state + home/away
    - team_side<->team_id derived and validated
    - payload keys are copied as-is (but cannot override reserved context keys)
    - lineup_version is always included (global monotonic counter, if present on GameState)
    - if include_lineups=True, emitter attaches:
        - on_court_home / on_court_away (lists of pids)
        - on_court_by_team_id ({home_team_id: [...], away_team_id: [...]})
        - lineup_version_by_team_id (dict copy, if present)
    """
    # Disallow accidental context overrides (prevents subtle duplicate/incorrect logs).
    bad = [k for k in payload.keys() if k in _RESERVED_KEYS]
    if bad:
        raise ValueError(f"emit_event() payload attempted to override reserved keys: {bad}")

    # Ensure mapping exists (and write-through home/away ids into game_state).
    home_team_id, away_team_id = _ensure_team_mapping(game_state, home, away)

    # Normalize event_type
    et = str(event_type).strip()
    if not et:
        raise ValueError("emit_event(): event_type must be a non-empty string")

    # Normalize context values
    q = _safe_int(getattr(game_state, "quarter", 1), 1)
    clk = _safe_float(getattr(game_state, "clock_sec", 0.0), 0.0)
    sclk = _safe_float(getattr(game_state, "shot_clock_sec", 0.0), 0.0)

    poss_idx = possession_index if possession_index is not None else getattr(game_state, "possession", 0)
    poss_idx_i = _safe_int(poss_idx, 0)

    # Score snapshot policy (UX correctness):
    # - Prefer authoritative TeamState.pts (updated immediately by resolve layer).
    # - Fallback to GameState mirrors only if TeamState.pts is missing/None.
    home_pts = getattr(home, "pts", None)
    away_pts = getattr(away, "pts", None)
    if home_pts is None:
        home_pts = getattr(game_state, "score_home", 0)
    if away_pts is None:
        away_pts = getattr(game_state, "score_away", 0)
    score_home = _safe_int(home_pts, 0)
    score_away = _safe_int(away_pts, 0)

    # seq (1..N)
    seq = _safe_int(getattr(game_state, "replay_seq", 0), 0) + 1
    game_state.replay_seq = seq

    # Derive team_side/team_id/opp fields with strict validation.
    def _norm_opt_str(x: Optional[str]) -> Optional[str]:
        if x is None:
            return None
        s = str(x).strip()
        return s if s else None

    ts = _norm_opt_str(team_side)
    tid = _norm_opt_str(team_id)
    os = _norm_opt_str(opp_side)
    oid = _norm_opt_str(opp_team_id)

    if ts is None and tid is None:
        # Neutral event (period start/end etc.)
        os = None
        oid = None
    else:
        # Fill missing side/id from mapping tables
        if ts is None and tid is not None:
            ts = game_state.team_id_to_side.get(tid)
            if ts is None:
                # last-resort: compare to known home/away ids
                if tid == home_team_id:
                    ts = "home"
                elif tid == away_team_id:
                    ts = "away"
        if tid is None and ts is not None:
            if ts == "home":
                tid = home_team_id
            elif ts == "away":
                tid = away_team_id
            else:
                raise ValueError(f"emit_event(): invalid team_side {ts!r}")

        # Validate consistency (team_side must map to the given team_id)
        if ts is not None:
            expected = home_team_id if ts == "home" else away_team_id if ts == "away" else None
            if expected is None:
                raise ValueError(f"emit_event(): invalid team_side {ts!r}")
            if tid is not None and tid != expected:
                msg = (
                    f"emit_event(): team_side/team_id mismatch: side={ts!r} expects {expected!r}, "
                    f"got {tid!r}"
                )
                if strict:
                    raise ValueError(msg)
                # Non-strict mode: force-correct to mapping.
                tid = expected

        # Opponent defaults
        if ts is not None and os is None:
            os = _side_other(ts)
        if os is not None and oid is None:
            if os == "home":
                oid = home_team_id
            elif os == "away":
                oid = away_team_id
            else:
                raise ValueError(f"emit_event(): invalid opp_side {os!r}")

        # Validate opp consistency if both provided
        if os is not None:
            expected_opp = home_team_id if os == "home" else away_team_id if os == "away" else None
            if expected_opp is None:
                raise ValueError(f"emit_event(): invalid opp_side {os!r}")
            if oid is not None and oid != expected_opp:
                msg = (
                    f"emit_event(): opp_side/opp_team_id mismatch: side={os!r} expects {expected_opp!r}, "
                    f"got {oid!r}"
                )
                if strict:
                    raise ValueError(msg)
                oid = expected_opp

        # Validate that team and opp are opposite when both present
        if ts is not None and os is not None:
            if os == ts:
                msg = f"emit_event(): opp_side must differ from team_side (both {ts!r})"
                if strict:
                    raise ValueError(msg)
                os = _side_other(ts)
                oid = home_team_id if os == "home" else away_team_id

    # Build event dict (final spec keys + payload passthrough)
    ge_sec = int(_compute_game_elapsed_sec(game_state, rules))

    # Lineup version snapshot (always included for deterministic seeking)
    lv_global = _safe_int(getattr(game_state, "lineup_version", 0), 0)
    lv_team: Optional[int] = None
    lvt_map = getattr(game_state, "lineup_version_by_team_id", None)
    if tid is not None and isinstance(lvt_map, dict):
        try:
            lv_team = _safe_int(lvt_map.get(tid, 0), 0)
        except Exception:
            lv_team = None

    evt: Dict[str, Any] = {
        "seq": seq,
        "event_type": et,
        "quarter": q,
        "clock_sec": _fmt_clock_mmss(clk),
        "shot_clock_sec": _round_1dp(sclk),
        "game_elapsed_sec": ge_sec,
        "possession_index": poss_idx_i,
        "score_home": int(score_home),
        "score_away": int(score_away),
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "lineup_version": int(lv_global),
        "team_side": ts,
        "team_id": tid,
        "opp_side": os,
        "opp_team_id": oid,
    }

    # Team-specific lineup version (only when this event has a subject team)
    if lv_team is not None:
        evt["lineup_version_team"] = int(lv_team)

    # Optional lineup snapshots (standard format)
    if include_lineups:
        on_home = list(getattr(game_state, "on_court_home", []) or [])
        on_away = list(getattr(game_state, "on_court_away", []) or [])
        evt["on_court_home"] = on_home
        evt["on_court_away"] = on_away
        evt["on_court_by_team_id"] = {home_team_id: on_home, away_team_id: on_away}
        if isinstance(lvt_map, dict):
            evt["lineup_version_by_team_id"] = dict(lvt_map)

    # Flow keys (standardized names)
    if pos_start is not None:
        evt["pos_start"] = str(pos_start)
    if pos_start_next is not None:
        evt["pos_start_next"] = str(pos_start_next)
    if pos_start_next_override is not None:
        evt["pos_start_next_override"] = str(pos_start_next_override)

    # Copy payload fields as-is (spec keeps existing names from resolve/sim_possession)
    if payload:
        for k, v in payload.items():
            evt[k] = v
            
    # Readability polish: round any "*_shotclock_sec" fields (payload-derived)
    # e.g. first_fga_shotclock_sec, etc. Keep as numeric with 1 decimal.
    for k in list(evt.keys()):
        if k == "shot_clock_sec" or k.endswith("_shotclock_sec"):
            evt[k] = _round_1dp(evt.get(k))

    # Append (single source of truth)
    game_state.replay_events.append(evt)
    return evt


def rebuild_events_of_type(replay_events: List[Dict[str, Any]], event_type: str) -> List[Dict[str, Any]]:
    """
    Convenience helper for tests/tools: derive a typed log from replay_events (e.g., TIMEOUT list).
    Not used by the engine runtime; engine must never maintain duplicate logs.
    """
    et = str(event_type)
    return [e for e in (replay_events or []) if isinstance(e, dict) and str(e.get("event_type")) == et]
