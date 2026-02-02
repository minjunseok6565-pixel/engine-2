from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Set

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
    # team mapping (derived only)
    "team_side",
    "team_id",
    "opp_side",
    "opp_team_id",
}


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
    """Format remaining period clock seconds into 'MM:SS'."""
    sec = _safe_float(clock_sec, 0.0)
    if sec < 0:
        sec = 0.0
    s = int(sec)  # truncate
    m = s // 60
    r = s % 60
    return f"{m:02d}:{r:02d}"

def _round_1dp(x: Any) -> Any:
    """Round numeric values to 1 decimal place (for readability in replay logs)."""
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


def _derive_side(*, home_team_id: str, away_team_id: str, team_id: str) -> str:
    if team_id == home_team_id:
        return "home"
    if team_id == away_team_id:
        return "away"
    raise ValueError(
        f"emit_event(): team_id {team_id!r} is not in this game: home={home_team_id!r}, away={away_team_id!r}"
    )


def _validate_game_teams(game_state: GameState, home: TeamState, away: TeamState) -> tuple[str, str]:
    home_team_id = str(getattr(game_state, "home_team_id", "") or "").strip()
    away_team_id = str(getattr(game_state, "away_team_id", "") or "").strip()
    if not home_team_id or not away_team_id:
        raise ValueError(
            f"emit_event(): GameState.home_team_id/away_team_id must be set before replay emission (home={home_team_id!r}, away={away_team_id!r})"
        )
    if home_team_id == away_team_id:
        raise ValueError(
            f"emit_event(): invalid GameState team ids (home_team_id == away_team_id == {home_team_id!r})"
        )

    # Defensive contract check: callers must pass the real home/away TeamState objects.
    if str(getattr(home, "team_id", "") or "").strip() != home_team_id:
        raise ValueError(
            f"emit_event(): home TeamState.team_id mismatch: game_state.home_team_id={home_team_id!r}, home.team_id={getattr(home, 'team_id', None)!r}"
        )
    if str(getattr(away, "team_id", "") or "").strip() != away_team_id:
        raise ValueError(
            f"emit_event(): away TeamState.team_id mismatch: game_state.away_team_id={away_team_id!r}, away.team_id={getattr(away, 'team_id', None)!r}"
        )

    return home_team_id, away_team_id


def emit_event(
    game_state: GameState,
    *,
    event_type: str,
    home: TeamState,
    away: TeamState,
    rules: Optional[Mapping[str, Any]] = None,
    # team mapping (SSOT keys: team_id / opp_team_id only)
    team_id: Optional[str] = None,
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
    # Calibration / fast-sim mode: allow callers to disable replay emission.
    # IMPORTANT: return a *truthy* stub dict so callers that test "did an event occur?"
    # (e.g. timeout/sub windows) keep identical control flow even when replay is disabled.
    if getattr(game_state, "replay_disabled", False):
        et = str(event_type).strip()
        if not et:
            raise ValueError("emit_event(): event_type must be a non-empty string")
        stub: Dict[str, Any] = {
            "event_type": et,
            "replay_disabled": True,
        }

        # Minimal context (cheap snapshots; no heavy derived computations).
        stub["quarter"] = _safe_int(getattr(game_state, "quarter", 1), 1)
        stub["clock_sec"] = _fmt_clock_mmss(getattr(game_state, "clock_sec", 0.0))
        stub["shot_clock_sec"] = _round_1dp(getattr(game_state, "shot_clock_sec", 0.0))

        poss_idx = possession_index if possession_index is not None else getattr(game_state, "possession", 0)
        stub["possession_index"] = _safe_int(poss_idx, 0)

        # Team ids (best-effort; do not enforce SSOT contracts in disabled mode).
        ht = str(getattr(game_state, "home_team_id", "") or "").strip()
        at = str(getattr(game_state, "away_team_id", "") or "").strip()
        if ht:
            stub["home_team_id"] = ht
        if at:
            stub["away_team_id"] = at

        # Score snapshot (cheap; best-effort).
        try:
            stub["score_home"] = _safe_int(getattr(home, "pts", 0) or 0, 0)
            stub["score_away"] = _safe_int(getattr(away, "pts", 0) or 0, 0)
        except Exception:
            pass

        # Flow keys (standardized names)
        if pos_start is not None:
            stub["pos_start"] = str(pos_start)
        if pos_start_next is not None:
            stub["pos_start_next"] = str(pos_start_next)
        if pos_start_next_override is not None:
            stub["pos_start_next_override"] = str(pos_start_next_override)

        # Team mapping (echo inputs; no validation)
        if team_id is not None:
            s = str(team_id).strip()
            if s:
                stub["team_id"] = s
        if opp_team_id is not None:
            s = str(opp_team_id).strip()
            if s:
                stub["opp_team_id"] = s

        return stub

    # Disallow accidental context overrides (prevents subtle duplicate/incorrect logs).
    bad = [k for k in payload.keys() if k in _RESERVED_KEYS]
    if bad:
        raise ValueError(f"emit_event() payload attempted to override reserved keys: {bad}")

    home_team_id, away_team_id = _validate_game_teams(game_state, home, away)

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

    # Score snapshot policy (strict): TeamState.pts only (no GameState fallback).
    score_home = _safe_int(getattr(home, "pts", 0) or 0, 0)
    score_away = _safe_int(getattr(away, "pts", 0) or 0, 0)

    # seq (1..N)
    seq = _safe_int(getattr(game_state, "replay_seq", 0), 0) + 1
    game_state.replay_seq = seq

    def _norm_opt_str(x: Optional[str]) -> Optional[str]:
        if x is None:
            return None
        s = str(x).strip()
        return s if s else None

    tid = _norm_opt_str(team_id)
    oid = _norm_opt_str(opp_team_id)

    ts: Optional[str] = None
    os: Optional[str] = None

    if tid is None:
        if oid is not None:
            raise ValueError(
                f"emit_event(): opp_team_id provided without team_id (opp_team_id={oid!r}); home={home_team_id!r}, away={away_team_id!r}"
            )
    else:
        # Subject-team event: validate ids belong to this game.
        ts = _derive_side(home_team_id=home_team_id, away_team_id=away_team_id, team_id=tid)

        if oid is None:
            oid = away_team_id if tid == home_team_id else home_team_id
        else:
            if oid not in (home_team_id, away_team_id):
                raise ValueError(
                    f"emit_event(): opp_team_id {oid!r} is not in this game: home={home_team_id!r}, away={away_team_id!r}"
                )
            if oid == tid:
                raise ValueError(
                    f"emit_event(): team_id and opp_team_id must differ (both {tid!r}); home={home_team_id!r}, away={away_team_id!r}"
                )

        os = _derive_side(home_team_id=home_team_id, away_team_id=away_team_id, team_id=oid)

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
        on_home = list(getattr(home, "on_court_pids", []) or [])
        on_away = list(getattr(away, "on_court_pids", []) or [])
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
    """Convenience helper for tests/tools; engine must never maintain duplicate logs."""
    et = str(event_type)
    return [e for e in (replay_events or []) if isinstance(e, dict) and str(e.get("event_type")) == et]
