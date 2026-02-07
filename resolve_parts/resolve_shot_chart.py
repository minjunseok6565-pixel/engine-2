from __future__ import annotations

import random
from typing import Optional, TYPE_CHECKING

from ..builders import get_action_base
from ..profiles import CORNER3_PROB_BY_ACTION_BASE

if TYPE_CHECKING:
    from ..game_config import GameConfig

def shot_zone_from_outcome(outcome: str) -> Optional[str]:
    if outcome in ("SHOT_RIM_LAYUP", "SHOT_RIM_DUNK", "SHOT_RIM_CONTACT", "SHOT_TOUCH_FLOATER"):
        return "rim"
    if outcome in ("SHOT_MID_CS", "SHOT_MID_PU"):
        return "mid"
    if outcome in ("SHOT_3_CS", "SHOT_3_OD"):
        return "3"
    return None


def shot_zone_detail_from_outcome(
    outcome: str,
    action: str,
    game_cfg: "GameConfig",
    rng: Optional[random.Random] = None,
) -> Optional[str]:
    """Map outcome -> NBA shot-chart zone (detail).

    For 3PA, we sample corner vs ATB using a *base-action* probability table so
    we don't deterministically over-produce corner 3s.
    """
    base_action = get_action_base(action, game_cfg)

    if outcome in ("SHOT_RIM_LAYUP", "SHOT_RIM_DUNK", "SHOT_RIM_CONTACT"):
        return "Restricted_Area"
    if outcome in ("SHOT_TOUCH_FLOATER", "SHOT_POST"):
        return "Paint_Non_RA"
    if outcome in ("SHOT_MID_CS", "SHOT_MID_PU"):
        return "Mid_Range"

    if outcome in ("SHOT_3_CS", "SHOT_3_OD"):
        p = float(CORNER3_PROB_BY_ACTION_BASE.get(base_action, CORNER3_PROB_BY_ACTION_BASE.get("default", 0.12)))
        r = (rng.random() if rng is not None else random.random())
        return "Corner_3" if r < p else "ATB_3"

    return None

def outcome_points(o: str) -> int:
    return 3 if o in ("SHOT_3_CS","SHOT_3_OD") else 2 if o.startswith("SHOT_") else 0


def _should_award_fastbreak_fg(ctx: dict, first_fga_sc) -> bool:
    """Fastbreak points should be credited on the *scoring FG event*, not at possession end.

    Rules (v1):
    - Only possessions that *originated* from a live-ball transition (after DRB / after TOV).
    - Never credit during possession-continuation (dead-ball stop -> inbound -> set offense).
    - Only credit within the early clock window, using the first FGA shot-clock snapshot.
    - Free throws are intentionally excluded (handled elsewhere).
    """
    try:
        origin = str(ctx.get("_pos_origin_start") or ctx.get("pos_start") or "")
    except Exception:
        origin = ""
    if origin not in ("after_tov", "after_drb", "after_steal", "after_block"):
        return False
    # Any dead-ball continuation segment implies defense is set -> not a fastbreak score.
    if bool(ctx.get("_pos_continuation", False)):
        return False
    # Additional guardrail: explicit dead-ball inbound segments should never count as fastbreak.
    if bool(ctx.get("dead_ball_inbound", False)):
        return False
    if first_fga_sc is None:
        return False
    try:
        return float(first_fga_sc) >= 15.0
    except Exception:
        return False

