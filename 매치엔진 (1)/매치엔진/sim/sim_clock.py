from __future__ import annotations

"""Clock / shot-clock utilities (time costs, inbound, shot-clock turnovers).

NOTE: Split from sim.py on 2025-12-27.
"""

import random
from typing import Any, Dict

from .core import clamp
from .models import GameState, TeamState

def apply_time_cost(game_state: GameState, cost: float, tempo_mult: float) -> None:
    adj = float(cost) * float(tempo_mult)
    game_state.shot_clock_sec -= adj
    game_state.clock_sec = max(game_state.clock_sec - adj, 0.0)

def apply_dead_ball_cost(game_state: GameState, cost: float, tempo_mult: float) -> None:
    """Dead-ball time: game clock runs, shot clock does not."""
    adj = float(cost) * float(tempo_mult)
    game_state.clock_sec = max(game_state.clock_sec - adj, 0.0)

def simulate_inbound(
    rng: random.Random,
    offense: TeamState,
    defense: TeamState,
    rules: Dict[str, Any],
) -> bool:
    """Return True if inbound results in a turnover (steal/5sec-like)."""
    inbound_rules = rules.get("inbound", {}) or {}
    tov_base = float(inbound_rules.get("tov_base", 0.010))
    tov_min = float(inbound_rules.get("tov_min", 0.003))
    tov_max = float(inbound_rules.get("tov_max", 0.060))
    def_scale = float(inbound_rules.get("def_scale", 0.00035))
    off_scale = float(inbound_rules.get("off_scale", 0.00030))

    if not offense.on_court_players() or not defense.on_court_players():
        return False

    offense_players = offense.on_court_players()
    defense_players = defense.on_court_players()
    inbounder = max(offense_players, key=lambda p: p.get("PASS_SAFE"))
    def_steal = sum(p.get("DEF_STEAL") for p in defense_players) / max(len(defense_players), 1)
    off_safe = inbounder.get("PASS_SAFE")

    # Scale around 50 as "league average"
    p_tov = tov_base + def_scale * (def_steal - 50.0) - off_scale * (off_safe - 50.0)
    p_tov = clamp(p_tov, tov_min, tov_max)

    if rng.random() < p_tov:
        offense.tov += 1
        offense.add_player_stat(inbounder.pid, "TOV", 1)
        offense.outcome_counts["TO_INBOUND"] = offense.outcome_counts.get("TO_INBOUND", 0) + 1
        return True
    return False


def _pick_shot_clock_tov_pid(offense: TeamState) -> str:
    """Pick who gets a shot-clock violation TOV using 12-role keys (no legacy)."""
    # 12-role priority (ball handlers first)
    for role in (
        "Initiator_Primary",
        "Initiator_Secondary",
        "Transition_Handler",
        "Connector_Playmaker",
        "Shot_Creator",
    ):
        pid = getattr(offense, 'roles', {}).get(role) if hasattr(offense, 'roles') else None
        if isinstance(pid, str) and pid:
            # ensure on-court
            if any(getattr(p, 'pid', None) == pid for p in offense.on_court_players()):
                return pid
    # Fallback: best passer on the floor
    offense_players = offense.on_court_players()
    if offense_players:
        return max(offense_players, key=lambda p: p.get('PASS_CREATE')).pid
    return ''

def commit_shot_clock_turnover(offense: TeamState) -> None:
    offense.tov += 1
    pid = _pick_shot_clock_tov_pid(offense)
    if pid:
        offense.add_player_stat(pid, "TOV", 1)
    offense.outcome_counts["TO_SHOT_CLOCK"] = offense.outcome_counts.get("TO_SHOT_CLOCK", 0) + 1


# -------------------------
# Team style diversity (per-team persistent multipliers)
# -------------------------
