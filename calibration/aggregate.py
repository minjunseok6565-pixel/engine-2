from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Union, Optional

Number = Union[int, float]

def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)

@dataclass
class MeanAccumulator:
    n: int = 0
    # nested sums
    sums: Dict[str, Any] = field(default_factory=dict)

    def add(self, x: Mapping[str, Any]) -> None:
        self.n += 1
        _acc_dict(self.sums, x)

    def mean(self) -> Dict[str, Any]:
        if self.n <= 0:
            return {}
        return _mean_dict(self.sums, self.n)

def _acc_dict(acc: Dict[str, Any], x: Mapping[str, Any]) -> None:
    for k, v in x.items():
        if v is None:
            continue
        if _is_num(v):
            acc[k] = float(acc.get(k, 0.0)) + float(v)
        elif isinstance(v, Mapping):
            sub = acc.get(k)
            if not isinstance(sub, dict):
                sub = {}
                acc[k] = sub
            _acc_dict(sub, v)
        # ignore lists/strings/etc. (calibration averages focus on numeric aggregates)

def _mean_dict(sums: Dict[str, Any], n: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in sums.items():
        if _is_num(v):
            out[k] = float(v) / float(n)
        elif isinstance(v, dict):
            out[k] = _mean_dict(v, n)
    return out

def safe_div(a: float, b: float) -> float:
    return (float(a) / float(b)) if b else 0.0

def pct(made: float, att: float) -> float:
    return safe_div(made, att) * 100.0
