from __future__ import annotations

"""Possession validation helpers."""

from typing import Any, Dict, Tuple

from ..models import GameState, TeamState


def _validate_possession_team_ids(
    offense: TeamState,
    defense: TeamState,
    game_state: GameState,
    ctx: Dict[str, Any],
) -> Tuple[TeamState, TeamState, str, str]:
    """Validate SSOT for a possession and derive (home_team, away_team, off_team_id, def_team_id).

    Contract:
    - ctx must provide off_team_id / def_team_id (team_id only; never "home"/"away").
    - offense.team_id / defense.team_id must match those ctx ids.
    - GameState.home_team_id / away_team_id must be set and must match the two participating team_ids.
    - No inference / correction / fallback. Any mismatch is a hard ValueError.
    """
    game_id = str(ctx.get("game_id", "") or "").strip()

    home_team_id = str(getattr(game_state, "home_team_id", "") or "").strip()
    away_team_id = str(getattr(game_state, "away_team_id", "") or "").strip()
    if not home_team_id or not away_team_id:
        raise ValueError(
            f"simulate_possession(): GameState.home_team_id/away_team_id must be set "
            f"(game_id={game_id!r}, home={home_team_id!r}, away={away_team_id!r})"
        )
    if home_team_id == away_team_id:
        raise ValueError(
            f"simulate_possession(): invalid game team ids (home_team_id == away_team_id == {home_team_id!r}, game_id={game_id!r})"
        )

    off_team_id = str(ctx.get("off_team_id", "") or "").strip()
    def_team_id = str(ctx.get("def_team_id", "") or "").strip()
    if not off_team_id or not def_team_id:
        raise ValueError(
            f"simulate_possession(): ctx must include off_team_id/def_team_id "
            f"(game_id={game_id!r}, off_team_id={off_team_id!r}, def_team_id={def_team_id!r})"
        )
    if off_team_id == def_team_id:
        raise ValueError(
            f"simulate_possession(): off_team_id == def_team_id == {off_team_id!r} (game_id={game_id!r})"
        )

    off_tid_obj = str(getattr(offense, "team_id", "") or "").strip()
    def_tid_obj = str(getattr(defense, "team_id", "") or "").strip()
    if off_tid_obj != off_team_id:
        raise ValueError(
            f"simulate_possession(): offense.team_id mismatch "
            f"(game_id={game_id!r}, ctx.off_team_id={off_team_id!r}, offense.team_id={off_tid_obj!r})"
        )
    if def_tid_obj != def_team_id:
        raise ValueError(
            f"simulate_possession(): defense.team_id mismatch "
            f"(game_id={game_id!r}, ctx.def_team_id={def_team_id!r}, defense.team_id={def_tid_obj!r})"
        )

    if {off_team_id, def_team_id} != {home_team_id, away_team_id}:
        raise ValueError(
            f"simulate_possession(): ctx team ids do not match game teams "
            f"(game_id={game_id!r}, home={home_team_id!r}, away={away_team_id!r}, "
            f"off={off_team_id!r}, def={def_team_id!r})"
        )

    # Derive actual home/away TeamState objects from GameState SSOT.
    if off_team_id == home_team_id:
        home_team = offense
        away_team = defense
    elif def_team_id == home_team_id:
        home_team = defense
        away_team = offense
    else:
        raise ValueError(
            f"simulate_possession(): could not derive home_team from ids "
            f"(game_id={game_id!r}, home={home_team_id!r}, off={off_team_id!r}, def={def_team_id!r})"
        )

    return home_team, away_team, off_team_id, def_team_id


