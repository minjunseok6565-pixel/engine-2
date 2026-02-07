# -------------------------
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

from .core import weighted_choice
from .models import Player, TeamState

# Participant selection (12-role only)
# -------------------------
#
# This module intentionally does NOT use legacy role keys (e.g., "ball_handler", "screener", "post").
# TeamState.roles is expected to be a mapping: 12-role name -> pid.

# NOTE: Implementation split across participants_*.py modules; this file re-exports the original API.

from .participants_parts import participants_roles as _participants_roles
from .participants_parts import participants_common as _participants_common
from .participants_parts import participants_offense as _participants_offense
from .participants_parts import participants_passing as _participants_passing
from .participants_parts import participants_rebounds as _participants_rebounds
from .participants_parts import participants_defense_events as _participants_defense_events
from .participants_parts import participants_fouls as _participants_fouls

for _m in (
    _participants_roles,
    _participants_common,
    _participants_offense,
    _participants_passing,
    _participants_rebounds,
    _participants_defense_events,
    _participants_fouls,
):
    for _k, _v in _m.__dict__.items():
        if _k.startswith("__"):
            continue
        globals()[_k] = _v

# Cleanup helper module refs to match prior surface as closely as possible.
del _m, _k, _v
del _participants_roles, _participants_common, _participants_offense, _participants_passing, _participants_rebounds, _participants_defense_events, _participants_fouls
