"""
quality.py

Defense scheme + role-stats driven "outcome quality" utilities.

Data sources (tables maintained separately in quality_data.py):
- real - 수비 스킴 퀼리티.txt
- 수비 역할이 반영되는 방식.txt

Conventions:
- Quality labels are from offense perspective:
    wide_open (+) ... tough (-)
- Player defense strength reduces the quality score (more negative => tougher).
- The produced score can be converted to a logit delta and added to your probability model.

Generated on: 2025-12-25
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

# --------------------------------------------------------------------------------------
# Tunables (start small; adjust after calibration)
# --------------------------------------------------------------------------------------

LABEL_SCORE: Dict[str, int] = {
    "wide_open": 2,
    "weak": 1,
    "neutral": 0,
    "tight": -1,
    "tough": -2,
}

DEFAULT_NEUTRAL_STAT = 50.0

@dataclass(frozen=True)
class QualityConfig:
    # How much player defense (vs 50) moves the quality score.
    # +10 def_index => quality_score -0.20 (tougher) by default.
    k_stat: float = 0.20

    # Clamp on the continuous quality score.
    clamp_min: float = -2.5
    clamp_max: float = 2.5

    # Score -> logit delta multipliers
    k_logit_shot: float = 0.25
    k_logit_pass_carry: float = 0.15

    # Reduce existing def_score impact on SHOTS by mixing toward 50.
    def_score_shot_mix: float = 0.60

    # Carry delta clamp (apply to next action only; prevents runaway chaining).
    carry_clamp_abs: float = 0.35

# --------------------------------------------------------------------------------------
# Data imports (tables moved to quality_data.py to keep this module logic-focused)
# --------------------------------------------------------------------------------------

try:
    # Package execution
    from .quality_data import (
        SCHEME_BASE_OUTCOME_LABELS,
        ROLE_STAT_PROFILES,
        GROUP_SCHEME_ROLE_WEIGHTS,
        OUTCOME_TO_GROUP,
        GROUP_FALLBACK,
        SCHEME_ALIASES,
    )
except ImportError:  # pragma: no cover
    # Script / flat-module execution
    from quality_data import (  # type: ignore
        SCHEME_BASE_OUTCOME_LABELS,
        ROLE_STAT_PROFILES,
        GROUP_SCHEME_ROLE_WEIGHTS,
        OUTCOME_TO_GROUP,
        GROUP_FALLBACK,
        SCHEME_ALIASES,
    )


# --------------------------------------------------------------------------------------
# Data contract (managed separately in quality_data.py)
#   - SCHEME_BASE_OUTCOME_LABELS: {scheme: {base_action: {outcome: quality_label}}}
#       * quality_label in {"wide_open","weak","neutral","tight","tough"} (case-insensitive; normalized here)
#   - ROLE_STAT_PROFILES: {scheme: {role: {stat_key: weight}}}
#       * used to compute role defense index via weighted average of player stats (0..100-ish)
#   - GROUP_SCHEME_ROLE_WEIGHTS: {group_id: {scheme: {role: weight}}}
#       * used to mix role indices into a single def_index for an outcome group
#   - OUTCOME_TO_GROUP: {outcome: group_id}
#   - GROUP_FALLBACK: {group_id: fallback_group_id}  (optional canonicalization)
#   - SCHEME_ALIASES: {alias: canonical_scheme}
# Notes:
#   - Edit/tune tables in quality_data.py; keep logic changes in this module.
#   - In LLM workflows, prefer providing quality.py only; include quality_data.py only when tuning tables.
# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def canonical_scheme(scheme: str) -> str:
    """Map input scheme string into the canonical key used by the dictionaries."""
    if scheme in SCHEME_BASE_OUTCOME_LABELS:
        return scheme
    if scheme in SCHEME_ALIASES:
        return SCHEME_ALIASES[scheme]
    # common case: user passes "Drop", "drop", etc.
    lower = scheme.strip().lower()
    return SCHEME_ALIASES.get(lower, scheme)

def normalize_label(label: str) -> str:
    if not label:
        return "neutral"
    s = label.strip()
    s = s.replace(" ", "_").replace("-", "_")
    s = s.lower()
    # handle "Neutral" or other casing
    if s == "neutral":
        return "neutral"
    if s in LABEL_SCORE:
        return s
    # some source values may be "Neutral" already handled, but just in case:
    if s.startswith("neutral"):
        return "neutral"
    return "neutral"

def clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

def mix_toward_neutral(score: float, mix: float, neutral: float = 50.0) -> float:
    """neutral + (score-neutral)*mix. mix=1 keeps as-is, mix=0 forces neutral."""
    return neutral + (score - neutral) * mix

def outcome_kind(outcome: str) -> str:
    if outcome.startswith("PASS_"):
        return "pass"
    if outcome.startswith("SHOT_"):
        return "shot"
    if outcome.startswith("FOUL_DRAW_"):
        # treat as "shot-like" for now; you can special-case in the engine if needed.
        return "foul"
    if outcome.startswith("TO_"):
        return "to"
    if outcome.startswith("RESET"):
        return "reset"
    return "other"

def get_base_quality_label(scheme: str, base_action: str, outcome: str) -> str:
    scheme = canonical_scheme(scheme)
    label = (
        SCHEME_BASE_OUTCOME_LABELS
        .get(scheme, {})
        .get(base_action, {})
        .get(outcome, "neutral")
    )
    return normalize_label(label)

def get_base_quality_score(scheme: str, base_action: str, outcome: str) -> int:
    return LABEL_SCORE[get_base_quality_label(scheme, base_action, outcome)]

def get_outcome_group(outcome: str) -> str:
    gid = OUTCOME_TO_GROUP.get(outcome, "unknown")
    return GROUP_FALLBACK.get(gid, gid)

def get_scheme_role_weights(scheme: str, outcome: str) -> Dict[str, float]:
    scheme = canonical_scheme(scheme)
    gid = get_outcome_group(outcome)
    return dict(GROUP_SCHEME_ROLE_WEIGHTS.get(gid, {}).get(scheme, {}))

def get_role_stat_profile(scheme: str, role: str) -> Optional[Dict[str, float]]:
    scheme = canonical_scheme(scheme)
    return ROLE_STAT_PROFILES.get(scheme, {}).get(role)

def default_get_stat(player: Any, stat: str, default: float = DEFAULT_NEUTRAL_STAT) -> float:
    """Support dict-like or attribute-like player objects."""
    if player is None:
        return default
    # dict-like
    if isinstance(player, Mapping):
        v = player.get(stat, default)
        try:
            return float(v)
        except Exception:
            return default
    # attribute-like
    if hasattr(player, stat):
        try:
            return float(getattr(player, stat))
        except Exception:
            return default
    # fallback: try .get
    if hasattr(player, "get"):
        try:
            v = player.get(stat, default)
            return float(v)
        except Exception:
            return default
    return default

def dot_profile(player: Any, profile: Mapping[str, float], get_stat: Callable[[Any, str, float], float]) -> float:
    """Weighted average (0..100-ish)"""
    num = 0.0
    den = 0.0
    for stat, w in profile.items():
        if w <= 0:
            continue
        num += w * get_stat(player, stat, DEFAULT_NEUTRAL_STAT)
        den += w
    if den <= 0:
        return DEFAULT_NEUTRAL_STAT
    return num / den


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------

@dataclass
class QualityDetail:
    scheme: str
    base_action: str
    outcome: str
    group_id: str
    base_label: str
    base_score: float
    def_index: float
    stat_delta: float
    score: float
    role_scores: Dict[str, float] = field(default_factory=dict)
    role_weights: Dict[str, float] = field(default_factory=dict)

def compute_quality_score(
    scheme: str,
    base_action: str,
    outcome: str,
    role_players: Mapping[str, Any],
    *,
    config: QualityConfig = QualityConfig(),
    get_stat: Callable[[Any, str, float], float] = default_get_stat,
    return_detail: bool = False,
) -> float | QualityDetail:
    """
    Compute continuous quality score for (scheme, base_action, outcome).

    role_players:
        Mapping from role name -> player object (dict or stat-bearing object).
        Role names should match those used in ROLE_STAT_PROFILES for the given scheme.
        (e.g. 'PnR_POA_Defender', 'Lowman_Helper', 'Zone_Top_Left', ...)

    Returns:
        float score by default, or QualityDetail if return_detail=True.
    """
    scheme_c = canonical_scheme(scheme)

    # Reset outcomes: we don't assign quality (as per your design).
    if outcome_kind(outcome) == "reset":
        if return_detail:
            return QualityDetail(
                scheme=scheme_c,
                base_action=base_action,
                outcome=outcome,
                group_id="reset",
                base_label="neutral",
                base_score=0.0,
                def_index=DEFAULT_NEUTRAL_STAT,
                stat_delta=0.0,
                score=0.0,
            )
        return 0.0

    base_label = get_base_quality_label(scheme_c, base_action, outcome)
    base_score = float(LABEL_SCORE.get(base_label, 0))

    role_weights = get_scheme_role_weights(scheme_c, outcome)
    gid = get_outcome_group(outcome)

    # Build a scheme+outcome specific defense index (0..100-ish).
    role_scores: Dict[str, float] = {}
    def_index = DEFAULT_NEUTRAL_STAT
    if role_weights:
        num = 0.0
        den = 0.0
        for role, w in role_weights.items():
            if w <= 0:
                continue
            prof = get_role_stat_profile(scheme_c, role)
            p = role_players.get(role)
            if prof is None or p is None:
                # If the engine hasn't assigned this role/player yet, skip it.
                continue
            rs = dot_profile(p, prof, get_stat)
            role_scores[role] = rs
            num += w * rs
            den += w
        if den > 0:
            def_index = num / den

    # Player defense higher => tougher contest => lower score (offense perspective).
    stat_delta = -config.k_stat * (def_index - DEFAULT_NEUTRAL_STAT) / 10.0

    score = base_score + stat_delta
    score = clamp(score, config.clamp_min, config.clamp_max)

    if return_detail:
        return QualityDetail(
            scheme=scheme_c,
            base_action=base_action,
            outcome=outcome,
            group_id=gid,
            base_label=base_label,
            base_score=base_score,
            def_index=def_index,
            stat_delta=stat_delta,
            score=score,
            role_scores=role_scores,
            role_weights=role_weights,
        )
    return score

def score_to_logit_delta(
    outcome: str,
    score: float,
    *,
    config: QualityConfig = QualityConfig(),
    kind_override: Optional[str] = None,
) -> float:
    """Convert quality score -> logit delta (for your prob model)."""
    kind = kind_override or outcome_kind(outcome)
    if kind == "shot" or kind == "foul":
        return score * config.k_logit_shot
    if kind == "pass":
        return score * config.k_logit_pass_carry
    return 0.0

def apply_pass_carry(
    carry_logit_delta: float,
    next_outcome: str,
    *,
    config: QualityConfig = QualityConfig(),
) -> float:
    """Clamp and return carry delta (you can add this to next action's logit_delta)."""
    return clamp(carry_logit_delta, -config.carry_clamp_abs, config.carry_clamp_abs)

def mix_def_score_for_shot(def_score: float, *, config: QualityConfig = QualityConfig()) -> float:
    """Reduce existing def_score impact on shot outcomes (to avoid double counting)."""
    return mix_toward_neutral(def_score, config.def_score_shot_mix, neutral=DEFAULT_NEUTRAL_STAT)
