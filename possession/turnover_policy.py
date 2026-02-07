from __future__ import annotations

"""Turnover deadball/liveball classification policy.

Moved from engine.sim_possession to keep the possession engine modular.
"""

import warnings
from typing import Any

#
# Turnover deadball/liveball classification
# ---------------------------------------
# These are the ONLY turnover outcome strings emitted by resolve_outcome()/sim_clock in this engine:
#   TO_HANDLE_LOSS, TO_BAD_PASS, TO_CHARGE, TO_INBOUND, TO_SHOT_CLOCK
#
# Policy:
#   - Deadball turnovers: charge / inbound / shot clock
#   - Liveball turnovers: handle loss / bad pass
#
# NOTE:
#   resolve_outcome() may attach payload flags that override this classification:
#     - payload['deadball_override'] -> force deadball (e.g., bad-pass lineout)
#     - payload['pos_start_next_override'] / payload['steal'] -> force after_steal start
#
_DEADBALL_TURNOVER_OUTCOMES = {
    "TO_CHARGE",
    "TO_INBOUND",
    "TO_SHOT_CLOCK",
}

_LIVEBALL_TURNOVER_OUTCOMES = {
    "TO_HANDLE_LOSS",
    "TO_BAD_PASS",
}


def _normalize_turnover_outcome(o: Any) -> str:
    """Normalize turnover outcome keys (compat for older logs/validation)."""
    try:
        s = str(o or "")
    except Exception:
        return ""
    # legacy key sometimes appears in validation / old logs
    if s == "TO_SHOTCLOCK":
        return "TO_SHOT_CLOCK"
    return s


def _turnover_is_deadball(outcome: Any) -> bool:
    o = _normalize_turnover_outcome(outcome)
    if o in _DEADBALL_TURNOVER_OUTCOMES:
        return True
    if o in _LIVEBALL_TURNOVER_OUTCOMES:
        return False
    # Unknown TO_*: default to LIVE to preserve fastbreak/flow (and avoid surprising deadball windows)
    # but keep a warning to surface schema drift early.
    if o.startswith("TO_"):
        warnings.warn(f"[sim_possession] Unknown turnover outcome '{o}'; defaulting to liveball after_tov")
        return False
    # If payload is missing/invalid, be conservative and keep existing behavior.
    return False
