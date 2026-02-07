from __future__ import annotations

"""Late-clock guardrails and action-selection helpers.

Mostly-mechanical extraction from engine.sim_possession.
"""

import random
from types import SimpleNamespace
from typing import Any, Dict

from ..builders import get_action_base
from ..core import clamp, weighted_choice


def build_late_clock_guardrails(game_state: Any, rules: Dict[str, Any], tempo_mult: float, game_cfg: Any):
    """Build a late-clock helper namespace (closures capture game_state / tempo_mult / rules)."""

    # Late-clock action selection guardrails
    # -------------------------------------------------------------------------
    # Problem 1/2 fix: prevent "no attempt" period ends and excessive shotclock
    # violations by selecting only feasible actions given the remaining time.

    time_costs = rules.get("time_costs", {}) or {}
    timing = rules.get("timing", {}) or {}

    def _timing_f(key: str, default: float) -> float:
        try:
            return float(timing.get(key, default))
        except Exception:
            return float(default)

    min_release_window = _timing_f("min_release_window", 0.7)
    urgent_budget_sec = _timing_f("urgent_budget_sec", 8.0)
    quickshot_cost_sec = _timing_f("quickshot_cost_sec", float(time_costs.get("QuickShot", 1.2)))
    soft_slack_span = _timing_f("soft_slack_span", 4.0)
    soft_slack_floor = _timing_f("soft_slack_floor", 0.20)
    quickshot_inject_base = _timing_f("quickshot_inject_base", 0.05)
    quickshot_inject_urgency_mult = _timing_f("quickshot_inject_urgency_mult", 0.35)
    pass_reset_suppress_urgency = _timing_f("pass_reset_suppress_urgency", 0.85)

    def _budget_sec() -> float:
        # remaining real seconds (already in game clock units)
        try:
            return float(min(float(game_state.clock_sec), float(game_state.shot_clock_sec)))
        except Exception:
            return float(game_state.clock_sec)

    def _estimate_action_cost_sec(action_name: str) -> float:
        # base seconds BEFORE tempo_mult is applied
        base = get_action_base(action_name, game_cfg)
        return float(time_costs.get(action_name, time_costs.get(base, 0.0)))

    def _is_nonterminal_base(base_action: str) -> bool:
        return base_action in ("Kickout", "ExtraPass", "Reset")

    def _normalize_prob_map(weights: Dict[str, float]) -> Dict[str, float]:
        if not weights:
            return {}
        s = sum(float(v) for v in weights.values())
        if s <= 0:
            return {}
        return {k: float(v) / s for k, v in weights.items()}

    def choose_action_with_budget(
        rng_local: random.Random,
        base_probs: Dict[str, float],
        *,
        prefer_terminal: bool = True,
        allow_quickshot: bool = True,
    ) -> str:
        # Returns a feasible action while preserving the tactical distribution.
        if not base_probs:
            return "SpotUp"

        b = _budget_sec()
        tm = float(tempo_mult) if float(tempo_mult) > 0 else 1.0

        # urgency: 0 when plenty of time, 1 when very tight
        u = clamp(1.0 - (b / urgent_budget_sec), 0.0, 1.0) if urgent_budget_sec > 0 else 0.0

        probs = dict(base_probs)
        # Add QuickShot as an emergency option. Keep it tiny unless time is tight.
        if allow_quickshot and "QuickShot" not in probs:
            probs["QuickShot"] = max((quickshot_inject_base + quickshot_inject_urgency_mult * u) * 0.25, 0.0)

        # 1) Hard feasibility filter (guardrail)
        feasible: Dict[str, float] = {}
        for act, w in probs.items():
            if float(w) <= 0:
                continue
            base = get_action_base(act, game_cfg)
            cost = _estimate_action_cost_sec(act) * tm
            margin = min_release_window if (prefer_terminal and _is_nonterminal_base(base)) else 0.0
            if cost <= max(0.0, b - margin):
                feasible[act] = float(w)

        if not feasible:
            # 2) Fallback: pick the fastest action available (or QuickShot)
            best_act = None
            best_cost = None
            for act, w in probs.items():
                if float(w) <= 0:
                    continue
                c = _estimate_action_cost_sec(act) * tm
                if best_act is None or c < float(best_cost):
                    best_act = act
                    best_cost = c
            return best_act or "QuickShot"

        # 3) Soft penalty: smoothly discourage actions that leave little slack
        penalized: Dict[str, float] = {}
        for act, w in feasible.items():
            c = _estimate_action_cost_sec(act) * tm
            slack = max(0.0, b - c)
            pen = clamp(slack / max(soft_slack_span, 0.10), soft_slack_floor, 1.0)
            penalized[act] = float(w) * pen

        # 4) Small urgency boost to faster actions as u increases (continuous transition)
        mixed: Dict[str, float] = {}
        for act, w in penalized.items():
            c = _estimate_action_cost_sec(act) * tm
            fastness = clamp(1.0 - (c / 8.0), 0.0, 1.0)
            mixed[act] = float(w) * (1.0 + u * 1.6 * fastness)

        final_probs = _normalize_prob_map(mixed)
        if not final_probs:
            return next(iter(feasible.keys()))
        return weighted_choice(rng_local, final_probs)

    def _apply_urgent_outcome_constraints(priors: Dict[str, float]) -> Dict[str, float]:
        # Reduce PASS/RESET chaining when time is tight.
        if not priors:
            return priors
        b = _budget_sec()
        u = clamp(1.0 - (b / urgent_budget_sec), 0.0, 1.0) if urgent_budget_sec > 0 else 0.0
        # When urgent, heavily suppress PASS_/RESET_ outcomes to avoid "no attempt" endings.
        suppress = clamp(1.0 - u * pass_reset_suppress_urgency, 0.02, 1.0)
        out: Dict[str, float] = {}
        for k, v in priors.items():
            w = float(v)
            if k.startswith("PASS_") or k.startswith("RESET_"):
                w *= suppress
            out[k] = w
        # Normalize
        s = sum(out.values())
        if s <= 0:
            return priors
        return {k: (v / s) for k, v in out.items()}

    return SimpleNamespace(
        time_costs=time_costs,
        timing=timing,
        min_release_window=min_release_window,
        urgent_budget_sec=urgent_budget_sec,
        quickshot_cost_sec=quickshot_cost_sec,
        soft_slack_span=soft_slack_span,
        soft_slack_floor=soft_slack_floor,
        quickshot_inject_base=quickshot_inject_base,
        quickshot_inject_urgency_mult=quickshot_inject_urgency_mult,
        pass_reset_suppress_urgency=pass_reset_suppress_urgency,
        budget_sec=_budget_sec,
        estimate_action_cost_sec=_estimate_action_cost_sec,
        is_nonterminal_base=_is_nonterminal_base,
        normalize_prob_map=_normalize_prob_map,
        choose_action_with_budget=choose_action_with_budget,
        apply_urgent_outcome_constraints=_apply_urgent_outcome_constraints,
    )
