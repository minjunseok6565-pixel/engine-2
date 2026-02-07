from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from .models import TeamState, GameState

from .resolve_parts.resolve_ft_rebound import (
    resolve_free_throws,
    rebound_orb_probability,
    choose_orb_rebounder,
    choose_drb_rebounder,
)
from .resolve_parts.resolve_pass_tracking import (
    clear_pass_tracking,
    commit_pending_pass_event,
)
from .resolve_parts.resolve_context import build_resolve_context
from .resolve_parts.resolve_handler_shot import handle_shot
from .resolve_parts.resolve_handler_pass import handle_pass
from .resolve_parts.resolve_handler_turnover import handle_turnover, handle_shot_clock_turnover
from .resolve_parts.resolve_handler_foul import handle_foul

if TYPE_CHECKING:
    from .game_config import GameConfig

# -------------------------
# Outcome groups
# -------------------------
def is_shot(o: str) -> bool: return str(o).startswith("SHOT_")
def is_pass(o: str) -> bool: return str(o).startswith("PASS_")
def is_to(o: str) -> bool: return str(o).startswith("TO_")
def is_foul(o: str) -> bool: return str(o).startswith("FOUL_")
def is_reset(o: str) -> bool: return str(o).startswith("RESET_")


# -------------------------
# Resolve sampled outcome into events
# -------------------------

def resolve_outcome(
    rng: random.Random,
    outcome: str,
    action: str,
    offense: TeamState,
    defense: TeamState,
    tags: Dict[str, Any],
    pass_chain: int,
    ctx: Optional[Dict[str, Any]] = None,
    game_state: Optional[GameState] = None,
    game_cfg: Optional["GameConfig"] = None,
) -> Tuple[str, Dict[str, Any]]:
    # count outcome
    offense.outcome_counts[outcome] = offense.outcome_counts.get(outcome, 0) + 1

    if not isinstance(ctx, dict):
        raise ValueError("resolve_outcome requires ctx dict")
    if game_state is None:
        raise ValueError("resolve_outcome requires game_state")
    if game_cfg is None:
        raise ValueError("resolve_outcome requires game_cfg")

    # SSOT / team_id-only contract (no legacy keys; no inference/repair here).
    game_id = ctx.get("game_id")

    off_team_id = str(ctx.get("off_team_id", "") or "").strip()
    def_team_id = str(ctx.get("def_team_id", "") or "").strip()
    if not off_team_id or not def_team_id:
        raise ValueError(
            "resolve_outcome(): ctx must include off_team_id/def_team_id "
            f"(game_id={game_id!r}, off_team_id={off_team_id!r}, def_team_id={def_team_id!r})"
        )
    if off_team_id == def_team_id:
        raise ValueError(
            "resolve_outcome(): off_team_id == def_team_id "
            f"(game_id={game_id!r}, team_id={off_team_id!r})"
        )

    off_tid_obj = str(getattr(offense, "team_id", "") or "").strip()
    def_tid_obj = str(getattr(defense, "team_id", "") or "").strip()
    if off_tid_obj != off_team_id or def_tid_obj != def_team_id:
        raise ValueError(
            "resolve_outcome(): ctx team ids do not match TeamState.team_id "
            f"(game_id={game_id!r}, ctx.off_team_id={off_team_id!r}, offense.team_id={off_tid_obj!r}, "
            f"ctx.def_team_id={def_team_id!r}, defense.team_id={def_tid_obj!r})"
        )

    home_team_id = str(getattr(game_state, "home_team_id", "") or "").strip()
    away_team_id = str(getattr(game_state, "away_team_id", "") or "").strip()
    if {off_team_id, def_team_id} != {home_team_id, away_team_id}:
        raise ValueError(
            "resolve_outcome(): off/def team_id not in {home,away} SSOT "
            f"(game_id={game_id!r}, home_team_id={home_team_id!r}, away_team_id={away_team_id!r}, "
            f"off_team_id={off_team_id!r}, def_team_id={def_team_id!r})"
        )

    # Special-case: shot clock turnover (kept exactly as before; just moved).
    if outcome == "TO_SHOT_CLOCK":
        return handle_shot_clock_turnover(
            rng, outcome, action, offense, defense, tags, pass_chain, ctx, game_state, game_cfg
        )

    def _record_exception(where: str, exc: BaseException) -> None:
        """Record exceptions into ctx for debugging without breaking sim flow."""
        try:
            errs = ctx.setdefault("errors", [])
            errs.append(
                {
                    "where": where,
                    "outcome": outcome,
                    "action": action,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        except Exception:
            # Never allow debug recording to crash the sim.
            return

    # role-fit bad outcome logging (internal; only when role-fit was applied on this step)
    try:
        if bool(tags.get("role_fit_applied", False)):
            g = str(tags.get("role_fit_grade", "B"))
            if is_to(outcome):
                offense.role_fit_bad_totals["TO"] = offense.role_fit_bad_totals.get("TO", 0) + 1
                offense.role_fit_bad_by_grade.setdefault(g, {}).setdefault("TO", 0)
                offense.role_fit_bad_by_grade[g]["TO"] += 1
            elif is_reset(outcome):
                offense.role_fit_bad_totals["RESET"] = offense.role_fit_bad_totals.get("RESET", 0) + 1
                offense.role_fit_bad_by_grade.setdefault(g, {}).setdefault("RESET", 0)
                offense.role_fit_bad_by_grade[g]["RESET"] += 1
    except Exception as e:
        _record_exception("role_fit_bad_logging", e)
        pass

    rc, early = build_resolve_context(
        rng,
        outcome,
        action,
        offense,
        defense,
        tags,
        pass_chain,
        ctx,
        game_state,
        game_cfg,
        game_id=game_id,
        off_team_id=off_team_id,
        def_team_id=def_team_id,
        _record_exception=_record_exception,
    )
    if early is not None:
        return early
    assert rc is not None

    defender_pid = rc.defender_pid
    matchup_source = rc.matchup_source
    matchup_event = rc.matchup_event

    help_level = rc.help_level
    double_strength = rc.double_strength
    double_doubler_pid = rc.double_doubler_pid
    double_source = rc.double_source
    double_label = rc.double_label

    def _with_matchup(payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        payload["defender_pid"] = defender_pid
        payload["matchup_source"] = matchup_source
        payload["matchup_event"] = matchup_event
        payload["matchups_version"] = int(ctx.get("matchups_version", 0) or 0)
        # Tactical knobs (for replay/text): always include, defaulting to neutral.
        payload["help_level"] = float(help_level)
        payload["double_strength"] = float(double_strength)
        payload["double_doubler_pid"] = double_doubler_pid
        payload["double_source"] = double_source
        payload["double_label"] = double_label
        return payload

    # resolve by type
    if is_shot(outcome):
        return handle_shot(rc, _with_matchup, _record_exception)
    if is_pass(outcome):
        return handle_pass(rc, _with_matchup, _record_exception)
    if is_to(outcome):
        return handle_turnover(rc, _with_matchup, _record_exception)
    if is_foul(outcome):
        return handle_foul(rc, _with_matchup, _record_exception)

    if is_reset(outcome):
        clear_pass_tracking(ctx)
        return "RESET", _with_matchup({"outcome": outcome})

    clear_pass_tracking(ctx)
    return "RESET", _with_matchup({"outcome": outcome})
