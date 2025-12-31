from __future__ import annotations

from typing import Final

from .models import TeamState

HOME: Final[str] = "home"
AWAY: Final[str] = "away"


def team_key(team: TeamState, home_team: TeamState) -> str:
    return HOME if team is home_team else AWAY
