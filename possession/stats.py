from __future__ import annotations

"""Small stat helper utilities."""

from typing import Any


def _player_stat(player: Any, key: str, default: float = 50.0) -> float:
    """Best-effort stat getter for Player-like objects.

    Prefer fatigue-insensitive values when available. Falls back to derived dict,
    then default.
    """
    try:
        # Most Player objects in this engine support fatigue_sensitive.
        return float(player.get(key, fatigue_sensitive=False))
    except TypeError:
        # Back-compat: older signature without fatigue_sensitive.
        try:
            return float(player.get(key))
        except Exception:
            pass
    except Exception:
        pass
    try:
        return float(getattr(player, "derived", {}).get(key, default))
    except Exception:
        return float(default)


