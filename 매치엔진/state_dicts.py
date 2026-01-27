from __future__ import annotations

from typing import Any, Dict, Optional


class SideKeyedDict(dict):
    """dict that stores values by team_id but allows HOME/AWAY side-key access.

    - Keys "home"/"away" are translated to stable team_id via side_to_team_id mapping.
    - If key is not a known side, it is stored as-is (for non-strict use cases).
    """

    def __init__(self, side_to_team_id: Dict[str, str], initial: Optional[Dict[Any, Any]] = None, **kwargs: Any):
        super().__init__()
        self._side_to_team_id = dict(side_to_team_id or {})
        if initial:
            self.update(initial)
        if kwargs:
            self.update(kwargs)

    def _k(self, key: Any) -> Any:
        try:
            s = str(key)
            # translate only if it is a side key
            return self._side_to_team_id.get(s, key)
        except Exception:
            return key

    def __getitem__(self, key: Any) -> Any:
        return super().__getitem__(self._k(key))

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(self._k(key), value)

    def __delitem__(self, key: Any) -> None:
        super().__delitem__(self._k(key))

    def __contains__(self, key: object) -> bool:
        try:
            return super().__contains__(self._k(key))  # type: ignore[arg-type]
        except Exception:
            return super().__contains__(key)  # type: ignore[arg-type]

    def get(self, key: Any, default: Any = None) -> Any:  # noqa: A003
        return super().get(self._k(key), default)

    def setdefault(self, key: Any, default: Any = None) -> Any:
        return super().setdefault(self._k(key), default)

    def pop(self, key: Any, default: Any = None) -> Any:  # noqa: A003
        return super().pop(self._k(key), default)

    def update(self, other: Optional[Dict[Any, Any]] = None, /, **kwargs: Any) -> None:  # type: ignore[override]
        if other:
            if hasattr(other, "items"):
                for k, v in other.items():  # type: ignore[union-attr]
                    super().__setitem__(self._k(k), v)
            else:
                for k, v in other:  # type: ignore[assignment]
                    super().__setitem__(self._k(k), v)
        for k, v in kwargs.items():
            super().__setitem__(self._k(k), v)


class StrictSideKeyedDict(SideKeyedDict):
    """Strict SideKeyedDict:

    - Only accepts keys that are either:
      - a side key present in mapping (e.g. "home"/"away"), OR
      - one of the mapped team_id values.
    - Prevents accidental pollution with unknown keys (critical for state dict correctness).
    """

    def _allowed(self, raw_key: Any) -> bool:
        try:
            s = str(raw_key)
        except Exception:
            return False
        # side key allowed only if present in mapping
        if s in self._side_to_team_id:
            return True
        # team_id allowed if it is one of mapping values
        return s in set(self._side_to_team_id.values())

    def _k(self, key: Any) -> Any:
        if not self._allowed(key):
            raise KeyError(f"StrictSideKeyedDict: disallowed key {key!r}")
        return super()._k(key)
