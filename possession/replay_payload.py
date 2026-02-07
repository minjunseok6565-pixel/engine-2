from __future__ import annotations

"""Replay payload sanitizers."""

from typing import Any, Dict, Optional


def _clean_replay_payload(payload: Any, *, drop: Optional[set] = None) -> Dict[str, Any]:
    """
    Replay payload sanitizer.

    resolve.py returns simulation payloads that may include internal control keys.
    This helper ensures we never forward keys that would collide with emit_event()'s explicit params.
    """
    if not isinstance(payload, dict):
        return {}
    out = dict(payload)
    if drop:
        for k in drop:
            out.pop(k, None)
    return out


