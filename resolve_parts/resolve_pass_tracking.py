from __future__ import annotations

from typing import Any, Dict, Optional

from ..models import GameState, TeamState

_ASSIST_WINDOW_SEC = {
    "SHOT_3_CS": 2.00,
    "SHOT_MID_CS": 2.00,

    "SHOT_RIM_LAYUP": 3.10,
    "SHOT_RIM_DUNK": 3.10,
    "SHOT_RIM_CONTACT": 3.10,

    "SHOT_TOUCH_FLOATER": 2.40,
    "SHOT_POST": 2.70,

    "SHOT_3_OD": 0.95,
    "SHOT_MID_PU": 0.80,
}
_DEFAULT_ASSIST_WINDOW_SEC = 2.40
_PASS_HISTORY_MAXLEN = 3


def clear_pass_tracking(ctx: Dict[str, Any]) -> None:
    """Clear pass-history + staged pass event for the current possession sequence."""
    if not isinstance(ctx, dict):
        return
    ctx.pop("_pending_pass_event", None)
    ctx.pop("pass_history", None)
    ctx.pop("_pass_seq", None)


def commit_pending_pass_event(ctx: Dict[str, Any], game_state: Optional[GameState]) -> None:
    """Commit staged pass event into pass_history using *post-time-cost* clocks.

    Called by sim_possession.py immediately after apply_time_cost for a pass.
    """
    if not isinstance(ctx, dict):
        return
    ev = ctx.pop("_pending_pass_event", None)
    if not isinstance(ev, dict):
        return

    pid = str(ev.get("pid") or "")
    if not pid:
        return

    seq = int(ctx.get("_pass_seq", 0)) + 1
    ctx["_pass_seq"] = seq

    hist = ctx.get("pass_history")
    if not isinstance(hist, list):
        hist = []
        ctx["pass_history"] = hist

    sc = float(getattr(game_state, "shot_clock_sec", 0.0)) if game_state is not None else 0.0
    gc = float(getattr(game_state, "clock_sec", 0.0)) if game_state is not None else 0.0

    hist.append({
        "seq": seq,
        "pid": pid,
        "outcome": str(ev.get("outcome") or ""),
        "base_action": str(ev.get("base_action") or ""),
        "shot_clock_sec": sc,
        "game_clock_sec": gc,
    })

    if len(hist) > _PASS_HISTORY_MAXLEN:
        del hist[:-_PASS_HISTORY_MAXLEN]


def _assist_window_sec(shot_outcome: str) -> float:
    return float(_ASSIST_WINDOW_SEC.get(str(shot_outcome), _DEFAULT_ASSIST_WINDOW_SEC))


def pick_assister_from_history(
    ctx: Dict[str, Any],
    offense: TeamState,
    shooter_pid: str,
    game_state: Optional[GameState],
    shot_outcome: str,
) -> Optional[str]:
    """Pick assister from pass_history if the last pass is within the assist window."""
    if game_state is None:
        return None
    hist = ctx.get("pass_history")
    if not isinstance(hist, list) or not hist:
        return None

    win = _assist_window_sec(shot_outcome)
    shot_sc = float(getattr(game_state, "shot_clock_sec", 0.0))

    for ev in reversed(hist):
        if not isinstance(ev, dict):
            continue
        pid = str(ev.get("pid") or "")
        if not pid or pid == str(shooter_pid or ""):
            continue
        if not offense.is_on_court(pid):
            continue

        pass_sc = float(ev.get("shot_clock_sec", 0.0))
        dt = pass_sc - shot_sc
        if 0.0 <= dt <= win:
            return pid

    return None
