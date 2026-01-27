from __future__ import annotations

"""
shot_diet.py

Shot diet module: lineup-driven, tactics-preserving probability shaping.

This module provides:
- ShotDietStyle: a cached style vector computed from on-court lineups (offense+defense)
- Multipliers for (action selection) and (outcome priors), derived from style features,
  tactic-specific weights, and conservative alpha/clamp guards.

Design principles (as agreed):
- Tactics remain the "intent": action distribution changes are soft (smaller alpha/clamp).
- Lineup is "feasibility": outcome priors changes are stronger (larger alpha/clamp).
- Initiation is handler-centric (primary/secondary with usage softmax), so a single elite guard
  can drive action frequency without being washed out by lineup averages.
- Uses action alias mappings passed into get_action_base() to map concrete actions to base actions.

Integration conditions (so you can use this file "as-is" with a UI-driven role system):
- TeamState.roles must be a dict[str, str] mapping role_fit role names -> on-court player pid strings.
  Example: roles["Initiator_Primary"] = "p123" (must match Player.pid on-court).
- Primary handler selection reads ONLY roles["Initiator_Primary"].
  If no Initiator_Primary is on-court, this module falls back to max(_onball_score) and may try to write
  roles["Initiator_Primary"] = <pid> (best effort; safe to ignore if roles is read-only).
- Secondary handler selection prefers roles["Initiator_Secondary"].
  If missing on-court, falls back to max(_onball_score) excluding primary and may try to write roles["Initiator_Secondary"].
- Scheme-specific screener priorities require passing a scheme name into compute_shot_diet_style() via either:
    * ctx["tactic_name"|"tactic"|"scheme_name"|"scheme"] (string), or
    * game_state.<tactic_name|tactic|scheme_name|scheme> (string).
  If no scheme is provided (or it doesn't match the table), screeners fall back to _screen_score().
- This module intentionally ignores legacy keys roles["ball_handler"], roles["secondary_handler"], roles["screener"].
  (So you should NOT rely on those keys once you adopt the 12-role system.)
- If your UI stores roles as Player objects, indices, or other types, convert them to pid strings before calling.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict
import math

from .core import clamp
from .models import Player, TeamState

# -------------------------
# Data imports (tables moved to shot_diet_data.py to keep this module logic-focused)
# -------------------------

try:
    # Package execution
    from .shot_diet_data import (
        BASELINE,
        TAU_USAGE,
        USAGE_MIN_PRIMARY,
        USAGE_MAX_PRIMARY,
        CLAMP_ACTION_MULT,
        CLAMP_OUTCOME_MULT,
        PROB_FLOOR,
        WEIGHTS_GLOBAL_OUTCOME,
        WEIGHTS_TACTIC_ACTION,
        WEIGHTS_TACTIC_OUTCOME_DELTA,
        TACTIC_ALPHA,
        SCREENER_ROLE_PRIORITY,
        SCHEME_ALIASES,
        ALPHA_ACTION_FALLBACK,
        ALPHA_OUTCOME_FALLBACK,
    )
except ImportError:  # pragma: no cover
    # Script / flat-module execution
    from shot_diet_data import (  # type: ignore
        BASELINE,
        TAU_USAGE,
        USAGE_MIN_PRIMARY,
        USAGE_MAX_PRIMARY,
        CLAMP_ACTION_MULT,
        CLAMP_OUTCOME_MULT,
        PROB_FLOOR,
        WEIGHTS_GLOBAL_OUTCOME,
        WEIGHTS_TACTIC_ACTION,
        WEIGHTS_TACTIC_OUTCOME_DELTA,
        TACTIC_ALPHA,
        SCREENER_ROLE_PRIORITY,
        SCHEME_ALIASES,
        ALPHA_ACTION_FALLBACK,
        ALPHA_OUTCOME_FALLBACK,
    )


# -------------------------
# Data contract (managed separately in shot_diet_data.py)
#   Scalars / clamps
#     - BASELINE: float feature baseline (typically 0.5) for missing stats
#     - TAU_USAGE, USAGE_MIN_PRIMARY, USAGE_MAX_PRIMARY: primary/secondary usage softmax controls
#     - CLAMP_ACTION_MULT, CLAMP_OUTCOME_MULT: (lo, hi) bounds for final multipliers
#     - PROB_FLOOR: minimum probability floor (if/when you apply multipliers downstream)
#
#   Tactic / scheme mapping
#     - TACTIC_ALPHA: {tactic_name: (alpha_action, alpha_outcome)}
#         * action alpha is conservative; outcome alpha can be larger (feasibility vs intent)
#     - SCREENER_ROLE_PRIORITY: {scheme_name: [role_name, ...]}  (role_fit role names in priority order)
#     - SCHEME_ALIASES: {normalized_string: canonical_scheme_name}
#
#   Weight tables (feature keys are produced by ShotDietStyle.all_features())
#     - WEIGHTS_GLOBAL_OUTCOME: {base_action: {outcome: {feature_key: weight}}}
#     - WEIGHTS_TACTIC_ACTION: {tactic_name: {base_action: {feature_key: weight}}}
#     - WEIGHTS_TACTIC_OUTCOME_DELTA: {tactic_name: {base_action: {outcome: {feature_key: delta_weight}}}}
#
#   Fallbacks
#     - ALPHA_ACTION_FALLBACK, ALPHA_OUTCOME_FALLBACK: default alphas when tactic missing
#
# Notes:
#   - Edit/tune tables in shot_diet_data.py; keep algorithmic changes in this module.
#   - In LLM workflows, prefer providing shot_diet.py only; include shot_diet_data.py only when tuning weights.
# -------------------------


# -------------------------
# Public dataclasses
# -------------------------

@dataclass(frozen=True)
class InitiatorInfo:
    primary_pid: str
    secondary_pid: str
    w_primary: float
    w_secondary: float
    onball_primary: float
    onball_secondary: float


@dataclass(frozen=True)
class ScreenersInfo:
    screener1_pid: str
    screener2_pid: Optional[str] = None


@dataclass(frozen=True)
class ShotDietStyle:
    """Cached style vector for a given on-court matchup."""
    initiator: InitiatorInfo
    screeners: ScreenersInfo
    off_features: Dict[str, float]
    def_features: Dict[str, float]
    meta: Dict[str, Any] = field(default_factory=dict)

    def all_features(self) -> Dict[str, float]:
        # Merge offense + defense feature namespaces (keys are distinct by prefix).
        out = dict(self.off_features)
        out.update(self.def_features)
        return out


# -------------------------
# Internal cache
# -------------------------

_STYLE_CACHE: "OrderedDict[Tuple[Any, ...], ShotDietStyle]" = OrderedDict()

# Keep cache bounded to avoid unbounded growth across many possessions/games.
_STYLE_CACHE_MAX: int = 2048

def _energy_bucket(val: Any) -> float:
    """Bucket player energy for cache key stability (fatigue-sensitive style)."""
    try:
        if val is None:
            return 1.0
        return round(float(val), 2)
    except (TypeError, ValueError):
        return 1.0


# -------------------------
# Utilities
# -------------------------

def get_action_base(action: str, action_aliases: Optional[Dict[str, str]] = None) -> str:
    """Map concrete action to base action using provided aliases."""
    aliases = action_aliases or {}
    return aliases.get(action, action)


def _get01(p: Player, key: str) -> float:
    # Player.get already applies fatigue adjustment by default.
    return clamp(p.get(key) / 100.0, 0.0, 1.0)


def _mean(vals: List[float]) -> float:
    if not vals:
        return BASELINE
    return sum(vals) / float(len(vals))


def _topk_mean(players: List[Player], score_fn, k: int) -> float:
    if not players:
        return BASELINE
    scored = sorted((score_fn(p) for p in players), reverse=True)
    k = min(max(k, 1), len(scored))
    return sum(scored[:k]) / float(k)


def _count_ge(players: List[Player], score_fn, thr: float) -> int:
    return sum(1 for p in players if score_fn(p) >= thr)


def _pid_to_player(lineup: List[Player]) -> Dict[str, Player]:
    return {p.pid: p for p in lineup}


# -------------------------
# Role selection (Spec v1)
# -------------------------

def _onball_score(p: Player) -> float:
    return (
        0.35 * _get01(p, "PNR_READ")
        + 0.35 * _get01(p, "DRIVE_CREATE")
        + 0.20 * _get01(p, "PASS_CREATE")
        + 0.10 * _get01(p, "HANDLE_SAFE")
    )


def _screen_score(p: Player) -> float:
    # PHYSICAL proxy: if PHYSICAL absent, Player.get returns DERIVED_DEFAULT (50 -> 0.5),
    # which is acceptable; FIN_CONTACT acts as secondary proxy.
    physical = _get01(p, "PHYSICAL")
    if abs(physical - BASELINE) < 1e-9:
        physical = _get01(p, "FIN_CONTACT")
    return (
        0.30 * _get01(p, "SHORTROLL_PLAY")
        + 0.25 * _get01(p, "FIN_RIM")
        + 0.15 * _get01(p, "FIN_CONTACT")
        + 0.15 * physical
        + 0.15 * _get01(p, "PASS_CREATE")
    )


def _pick_primary_secondary(off: TeamState) -> Tuple[str, str, float, float, Dict[str, Any]]:
    """
    Map role_fit roles -> shot_diet initiators.

    Rules:
    - ball_handler == Initiator_Primary (must be on-court). If none on-court, assign best _onball_score as Initiator_Primary.
    - secondary_handler prefers Initiator_Secondary (on-court). If none on-court, pick best _onball_score excluding primary.
    """
    lineup = off.on_court_players()
    pid_map = _pid_to_player(lineup)

    # roles values are expected to be on-court player pid strings (not Player objects).
    roles = off.roles if isinstance(getattr(off, "roles", None), dict) else {}
    role_ip = roles.get("Initiator_Primary")
    role_is = roles.get("Initiator_Secondary")

    # Primary == Initiator_Primary (must be on-court)
    if role_ip and role_ip in pid_map:
        primary = role_ip
        primary_fallback = False
    else:
        primary = max(lineup, key=_onball_score).pid
        primary_fallback = True
        # If roles dict is mutable, keep assignments consistent for downstream modules.
        if roles is getattr(off, "roles", None):
            try:
                off.roles["Initiator_Primary"] = primary
            except Exception:
                pass

    # Secondary: prefer Initiator_Secondary (on-court), else best onball excluding primary
    if role_is and role_is in pid_map and role_is != primary:
        secondary = role_is
        secondary_fallback = False
    else:
        candidates = [p for p in lineup if p.pid != primary]
        if candidates:
            secondary = max(candidates, key=_onball_score).pid
            secondary_fallback = True
            if roles is getattr(off, "roles", None) and secondary != primary:
                try:
                    off.roles["Initiator_Secondary"] = secondary
                except Exception:
                    pass
        else:
            secondary = primary
            secondary_fallback = True

    p_primary = pid_map[primary]
    s1 = _onball_score(p_primary)
    if secondary == primary:
        w1, w2 = 1.0, 0.0
        s2 = s1
    else:
        p_secondary = pid_map[secondary]
        s2 = _onball_score(p_secondary)
        # softmax with tau, then cap
        z1 = math.exp(s1 / TAU_USAGE)
        z2 = math.exp(s2 / TAU_USAGE)
        w1 = z1 / (z1 + z2)
        w1 = clamp(w1, USAGE_MIN_PRIMARY, USAGE_MAX_PRIMARY)
        w2 = 1.0 - w1

    meta = {
        "role_Initiator_Primary": role_ip,
        "role_Initiator_Secondary": role_is,
        "primary_fallback": primary_fallback,
        "secondary_fallback": secondary_fallback,
        "onball_primary": s1,
        "onball_secondary": s2,
    }
    return primary, secondary, w1, w2, meta


def _pick_screeners(
    off: TeamState,
    primary: str,
    secondary: str,
    scheme_name: Optional[str],
) -> Tuple[str, Optional[str], Dict[str, Any]]:
    """
    Pick screeners using scheme-specific role priorities (role_fit role names).

    Rules:
    - If any on-court player's assigned role matches the scheme priority list, pick the first match.
    - If none match, fallback to _screen_score selection.
    """
    lineup = off.on_court_players()
    pid_map = _pid_to_player(lineup)
    roles = off.roles if isinstance(getattr(off, "roles", None), dict) else {}

    scheme_norm = _normalize_scheme_name(scheme_name)
    prio_roles = SCREENER_ROLE_PRIORITY.get(scheme_norm, [])

    def _first_pid_by_role_priority(exclude: Tuple[str, ...]) -> Optional[str]:
        for r in prio_roles:
            pid = roles.get(r)
            if pid and pid in pid_map and pid not in exclude:
                return pid
        return None

    # Screener1: scheme priority first
    scr1 = _first_pid_by_role_priority((primary, secondary))
    scr1_fallback = False
    if not scr1:
        # Fallback by _screen_score
        candidates = [p for p in lineup if p.pid not in (primary, secondary)]
        if candidates:
            scr1 = max(candidates, key=_screen_score).pid
            scr1_fallback = True
        else:
            scr1 = max(lineup, key=_screen_score).pid
            scr1_fallback = True

    # Screener2 (optional): scheme priority among remaining, else fallback by _screen_score
    scr2 = _first_pid_by_role_priority((primary, secondary, scr1))
    if not scr2:
        candidates2 = [p for p in lineup if p.pid != scr1]
        # Prefer excluding handlers, but allow if necessary
        candidates2_pref = [p for p in candidates2 if p.pid not in (primary, secondary)]
        pool = candidates2_pref if candidates2_pref else candidates2
        if len(pool) >= 1:
            scr2 = max(pool, key=_screen_score).pid

    meta = {
        "scheme_name": scheme_name,
        "scheme_norm": scheme_norm,
        "screener_priority_roles": prio_roles,
        "screener1_fallback": scr1_fallback,
    }
    return scr1, scr2, meta


def _support_players(lineup: List[Player], exclude_pids: Tuple[str, ...]) -> Tuple[List[Player], bool]:
    support = [p for p in lineup if p.pid not in exclude_pids]
    if support:
        return support, False
    # Fallback: use whole lineup
    return list(lineup), True


# -------------------------
# Style vector computation (Spec v1)
# -------------------------

def compute_shot_diet_style(
    offense: TeamState,
    defense: TeamState,
    game_state: Any = None,
    ctx: Optional[Dict[str, Any]] = None,
) -> ShotDietStyle:
    """
    Compute (or fetch from a bounded cache) style vector for current on-court matchup.
    Cache key: on-court pids + role hints + (bucketed) on-court energy (fatigue-sensitive).
    """
    # Sort by pid to ensure stable cache keys.
    off_sorted = sorted(offense.on_court_players(), key=lambda p: p.pid)
    def_sorted = sorted(defense.on_court_players(), key=lambda p: p.pid)

    off_pids = tuple(p.pid for p in off_sorted)
    def_pids = tuple(p.pid for p in def_sorted)
    # Fatigue-sensitive: include (bucketed) on-court energy in the cache key so style can change over time.
    off_energy = tuple(_energy_bucket(getattr(p, "energy", None)) for p in off_sorted)
    def_energy = tuple(_energy_bucket(getattr(p, "energy", None)) for p in def_sorted)
    roles = offense.roles or {}
    # screener selection depends on scheme_name (ctx/game_state), and initiators depend on Initiator roles.
    # Integration note: if you want scheme-specific screener priority, pass scheme via ctx or game_state.
    scheme_name = None
    if ctx and isinstance(ctx, dict):
        for k in ("tactic_name", "tactic", "scheme_name", "scheme"):
            v = ctx.get(k)
            if isinstance(v, str) and v.strip():
                scheme_name = v.strip()
                break
    if scheme_name is None and game_state is not None:
        for k in ("tactic_name", "tactic", "scheme_name", "scheme"):
            v = getattr(game_state, k, None)
            if isinstance(v, str) and v.strip():
                scheme_name = v.strip()
                break
    scheme_norm = _normalize_scheme_name(scheme_name)

    # Include role assignments in cache key so initiator/screener selection stays coherent.
    _ROLE_KEYS = (
        "Initiator_Primary",
        "Initiator_Secondary",
        "Transition_Handler",
        "Shot_Creator",
        "Rim_Attacker",
        "Spacer_CatchShoot",
        "Spacer_Movement",
        "Connector_Playmaker",
        "Roller_Finisher",
        "ShortRoll_Playmaker",
        "Pop_Spacer_Big",
        "Post_Hub",
    )
    role_key = (scheme_norm,) + tuple(roles.get(k) for k in _ROLE_KEYS)
    cache_key = (off_pids, def_pids, off_energy, def_energy, role_key)
    if cache_key in _STYLE_CACHE:
        style = _STYLE_CACHE.pop(cache_key)
        _STYLE_CACHE[cache_key] = style  # mark as most recently used
        return style

    primary, secondary, w1, w2, meta_h = _pick_primary_secondary(offense)
    scr1, scr2, meta_s = _pick_screeners(offense, primary, secondary, scheme_norm)

    lineup = offense.on_court_players()
    pid_map = _pid_to_player(lineup)
    p_primary = pid_map[primary]
    p_secondary = pid_map[secondary]

    # Supporting cast excludes primary/secondary/screeners
    exclude = (primary, secondary, scr1) + ((scr2,) if scr2 else tuple())
    support, support_fallback = _support_players(lineup, exclude)

    # ---------- Initiator layer (usage-weighted) ----------
    def usage_weighted(fn):
        return w1 * fn(p_primary) + w2 * fn(p_secondary)

    BH_PNR = usage_weighted(lambda p: _get01(p, "PNR_READ"))
    BH_DRIVE_PRESSURE = usage_weighted(lambda p: 0.6 * _get01(p, "FIRST_STEP") + 0.4 * _get01(p, "DRIVE_CREATE"))
    BH_PULLUP_THREAT = usage_weighted(lambda p: 0.6 * _get01(p, "SHOT_3_OD") + 0.4 * _get01(p, "SHOT_MID_PU"))
    BH_PASS_CREATION = usage_weighted(lambda p: 0.6 * _get01(p, "PASS_CREATE") + 0.4 * _get01(p, "PNR_READ"))
    BH_BALL_SECURITY = usage_weighted(lambda p: 0.5 * _get01(p, "HANDLE_SAFE") + 0.5 * _get01(p, "PASS_SAFE"))

    def foul_pressure(p: Player) -> float:
        # Prefer SHOT_FT; if absent, it defaults to 0.5 which is ok.
        ft = _get01(p, "SHOT_FT")
        if abs(ft - BASELINE) < 1e-9:
            return _get01(p, "FIN_CONTACT")
        return 0.6 * ft + 0.4 * _get01(p, "FIN_CONTACT")

    BH_FOUL_PRESSURE = usage_weighted(foul_pressure)

    # ---------- Screener layer ----------
    scr1_p = pid_map[scr1]
    scr2_p = pid_map[scr2] if (scr2 and scr2 in pid_map) else None
    w_scr1, w_scr2 = (1.0, 0.0) if scr2_p is None else (0.70, 0.30)

    def screener_mix(fn):
        if scr2_p is None:
            return fn(scr1_p)
        return w_scr1 * fn(scr1_p) + w_scr2 * fn(scr2_p)

    def screen_quality(p: Player) -> float:
        physical = _get01(p, "PHYSICAL")
        if abs(physical - BASELINE) < 1e-9:
            physical = _get01(p, "FIN_CONTACT")
        return physical

    SC_SCREEN_QUALITY = screener_mix(screen_quality)
    SC_ROLL_FINISH = screener_mix(lambda p: 0.45 * _get01(p, "FIN_RIM") + 0.30 * _get01(p, "FIN_DUNK") + 0.25 * _get01(p, "FIN_CONTACT"))

    def shortroll_play(p: Player) -> float:
        sr = _get01(p, "SHORTROLL_PLAY")
        if abs(sr - BASELINE) < 1e-9:
            return 0.6 * _get01(p, "PASS_CREATE") + 0.4 * _get01(p, "PASS_SAFE")
        return sr

    SC_SHORTROLL_PLAY = screener_mix(shortroll_play)
    SC_POP_THREAT = screener_mix(lambda p: 0.7 * _get01(p, "SHOT_3_CS") + 0.3 * _get01(p, "SHOT_MID_CS"))

    # ---------- Supporting cast layer ----------
    TEAM_CATCH3_QUALITY = _mean([_get01(p, "SHOT_3_CS") for p in support])

    # TEAM_SPACING: top3 C&S + shooter count bonus
    cs_fn = lambda p: _get01(p, "SHOT_3_CS")
    base = _topk_mean(support, cs_fn, k=3)
    shooters = _count_ge(support, cs_fn, thr=0.70)
    bonus = clamp((shooters - 2) * 0.05, -0.05, 0.10)
    TEAM_SPACING = clamp(base + bonus, 0.0, 1.0)

    TEAM_CUTTING = _mean([0.6 * _get01(p, "FIRST_STEP") + 0.4 * _get01(p, "FIN_RIM") for p in support])
    TEAM_EXTRA_PASS = _mean([0.6 * _get01(p, "PASS_SAFE") + 0.4 * _get01(p, "PASS_CREATE") for p in support])

    TEAM_ORB_CRASH = _mean([_get01(p, "REB_OR") for p in lineup])

    # ---------- Team context ----------
    # TEAM_PACE: endurance + first step (proxy)
    TEAM_PACE = _mean([0.6 * _get01(p, "ENDURANCE") + 0.4 * _get01(p, "FIRST_STEP") for p in lineup])

    # TEAM_POST_GRAVITY: if POST_* keys missing, proxy with contact+rim
    def has_key_any(players: List[Player], key: str) -> bool:
        # Heuristic: if any player has value != default(50) we treat as present.
        for pl in players:
            v = pl.get(key, fatigue_sensitive=False)
            if abs(float(v) - 50.0) > 1e-6:
                return True
        return False

    has_post_score = has_key_any(lineup, "POST_SCORE")
    has_post_control = has_key_any(lineup, "POST_CONTROL")
    has_physical = has_key_any(lineup, "PHYSICAL")

    if has_post_score:
        post_score = _topk_mean(lineup, lambda p: _get01(p, "POST_SCORE"), 1)
        post_control = _topk_mean(lineup, lambda p: _get01(p, "POST_CONTROL"), 1) if has_post_control else BASELINE
        physical = _topk_mean(lineup, lambda p: _get01(p, "PHYSICAL"), 1) if has_physical else _topk_mean(lineup, lambda p: _get01(p, "FIN_CONTACT"), 1)
        TEAM_POST_GRAVITY = clamp(0.5 * post_score + 0.3 * post_control + 0.2 * physical, 0.0, 1.0)
    else:
        contact = _topk_mean(lineup, lambda p: _get01(p, "FIN_CONTACT"), 1)
        rim = _topk_mean(lineup, lambda p: _get01(p, "FIN_RIM"), 1)
        TEAM_POST_GRAVITY = clamp(0.7 * contact + 0.3 * rim, 0.0, 1.0)

    off_features = {
        # initiator
        "BH_PNR": BH_PNR,
        "BH_DRIVE_PRESSURE": BH_DRIVE_PRESSURE,
        "BH_PULLUP_THREAT": BH_PULLUP_THREAT,
        "BH_PASS_CREATION": BH_PASS_CREATION,
        "BH_BALL_SECURITY": BH_BALL_SECURITY,
        "BH_FOUL_PRESSURE": BH_FOUL_PRESSURE,
        # screener
        "SC_SCREEN_QUALITY": SC_SCREEN_QUALITY,
        "SC_ROLL_FINISH": SC_ROLL_FINISH,
        "SC_SHORTROLL_PLAY": SC_SHORTROLL_PLAY,
        "SC_POP_THREAT": SC_POP_THREAT,
        # support
        "TEAM_SPACING": TEAM_SPACING,
        "TEAM_CATCH3_QUALITY": TEAM_CATCH3_QUALITY,
        "TEAM_CUTTING": TEAM_CUTTING,
        "TEAM_EXTRA_PASS": TEAM_EXTRA_PASS,
        "TEAM_ORB_CRASH": TEAM_ORB_CRASH,
        # context
        "TEAM_PACE": TEAM_PACE,
        "TEAM_POST_GRAVITY": TEAM_POST_GRAVITY,
    }

    # ---------- Defense features (Spec v1) ----------
    dline = defense.on_court_players()

    # Top-1 mean for rim/poa/post, mean for help/steal/dreb
    # If DEF_* keys missing, Player.get returns 50 -> BASELINE; that's acceptable.
    D_RIM_PROTECT = _topk_mean(dline, lambda p: _get01(p, "DEF_RIM"), 1)
    if abs(D_RIM_PROTECT - BASELINE) < 1e-9:
        D_RIM_PROTECT = _mean([_get01(p, "DEF_HELP") for p in dline])

    D_POA = _topk_mean(dline, lambda p: _get01(p, "DEF_POA"), 1)
    if abs(D_POA - BASELINE) < 1e-9:
        D_POA = _mean([_get01(p, "DEF_HELP") for p in dline])

    D_HELP_CLOSEOUT = _mean([_get01(p, "DEF_HELP") for p in dline])
    if abs(D_HELP_CLOSEOUT - BASELINE) < 1e-9:
        D_HELP_CLOSEOUT = _mean([_get01(p, "DEF_POA") for p in dline])

    D_STEAL_PRESS = _mean([_get01(p, "DEF_STEAL") for p in dline])

    D_POST = _topk_mean(dline, lambda p: _get01(p, "DEF_POST"), 1)
    if abs(D_POST - BASELINE) < 1e-9:
        # fallback to physical proxy if DEF_POST missing
        D_POST = _topk_mean(dline, lambda p: _get01(p, "PHYSICAL"), 1)

    D_DREB = _mean([_get01(p, "REB_DR") for p in dline])

    def_features = {
        "D_RIM_PROTECT": D_RIM_PROTECT,
        "D_POA": D_POA,
        "D_HELP_CLOSEOUT": D_HELP_CLOSEOUT,
        "D_STEAL_PRESS": D_STEAL_PRESS,
        "D_POST": D_POST,
        "D_DREB": D_DREB,
    }

    initiator_info = InitiatorInfo(
        primary_pid=primary,
        secondary_pid=secondary,
        w_primary=w1,
        w_secondary=w2,
        onball_primary=meta_h["onball_primary"],
        onball_secondary=meta_h["onball_secondary"],
    )
    screeners_info = ScreenersInfo(screener1_pid=scr1, screener2_pid=scr2)

    meta = {}
    meta["scheme_name"] = scheme_name
    meta["scheme_norm"] = scheme_norm
    meta.update(meta_h)
    meta.update(meta_s)
    meta["support_fallback"] = support_fallback

    style = ShotDietStyle(
        initiator=initiator_info,
        screeners=screeners_info,
        off_features=off_features,
        def_features=def_features,
        meta=meta,
    )
    _STYLE_CACHE[cache_key] = style
    # Evict least-recently-used entries if cache grows too large.
    while len(_STYLE_CACHE) > _STYLE_CACHE_MAX:
        _STYLE_CACHE.popitem(last=False)
    return style

def _normalize_scheme_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    n = str(name).strip()
    if not n:
        return None
    # fast-path exact match
    if n in SCREENER_ROLE_PRIORITY:
        return n
    if n in TACTIC_ALPHA:
        return n
    nl = n.lower().replace(" ", "")
    return SCHEME_ALIASES.get(nl, n)

# -------------------------
# Multiplier computation
# -------------------------

def _compute_log_mult(features: Dict[str, float], weights: Dict[str, float]) -> float:
    s = 0.0
    for k, w in weights.items():
        v = features.get(k, BASELINE)
        s += float(w) * (float(v) - BASELINE)
    return s


def _exp_mult_from_log(
    log_mult: float,
    alpha: float,
    clamp_mult: Tuple[float, float],
) -> float:
    if alpha <= 0.0:
        return 1.0
    lo, hi = clamp_mult
    # Cap in log-space to guarantee exp(alpha*log_mult) is within [lo, hi]
    lo_log = math.log(lo) / alpha
    hi_log = math.log(hi) / alpha
    log_mult = clamp(log_mult, lo_log, hi_log)
    return math.exp(alpha * log_mult)


def get_tactic_alphas(tactic_name: str) -> Tuple[float, float]:
    return TACTIC_ALPHA.get(tactic_name, (ALPHA_ACTION_FALLBACK, ALPHA_OUTCOME_FALLBACK))


def get_action_multipliers(style: ShotDietStyle, tactic_name: str) -> Dict[str, float]:
    """
    Returns multipliers for BASE actions (PnR, Drive, ...).
    Builders should apply to concrete actions via get_action_base(action).
    """
    alpha_action, _ = get_tactic_alphas(tactic_name)
    features = style.all_features()
    weights_by_action = WEIGHTS_TACTIC_ACTION.get(tactic_name, {})

    out: Dict[str, float] = {}
    # We compute multipliers for all known base actions, plus any actions in the tactic map.
    base_actions = set(WEIGHTS_GLOBAL_OUTCOME.keys())
    base_actions.update(weights_by_action.keys())

    for a in base_actions:
        w = weights_by_action.get(a)
        if not w:
            out[a] = 1.0
            continue
        log_mult = _compute_log_mult(features, w)
        out[a] = _exp_mult_from_log(log_mult, alpha_action, CLAMP_ACTION_MULT)

    return out


def get_outcome_multipliers(style: ShotDietStyle, tactic_name: str, base_action: str) -> Dict[str, float]:
    """
    Returns multipliers for outcomes within a given base_action.
    Implementation uses: GLOBAL_OUTCOME_WEIGHTS[base_action] + OUTCOME_DELTA[tactic][base_action]
    """
    _, alpha_outcome = get_tactic_alphas(tactic_name)
    features = style.all_features()

    base = WEIGHTS_GLOBAL_OUTCOME.get(base_action, {})
    delta = WEIGHTS_TACTIC_OUTCOME_DELTA.get(tactic_name, {}).get(base_action, {})

    out: Dict[str, float] = {}
    outcomes = set(base.keys()) | set(delta.keys())
    for oc in outcomes:
        w_total: Dict[str, float] = {}
        # merge base weights
        if oc in base:
            w_total.update(base[oc])
        # add delta weights
        if oc in delta:
            for k, v in delta[oc].items():
                w_total[k] = w_total.get(k, 0.0) + float(v)

        log_mult = _compute_log_mult(features, w_total) if w_total else 0.0
        out[oc] = _exp_mult_from_log(log_mult, alpha_outcome, CLAMP_OUTCOME_MULT)

    return out


def get_action_multiplier_for_action(
    style: ShotDietStyle,
    tactic_name: str,
    action: str,
    action_aliases: Optional[Dict[str, str]] = None,
) -> float:
    """Convenience: multiplier for a concrete action (handles alias -> base)."""
    base = get_action_base(action, action_aliases)
    mults = get_action_multipliers(style, tactic_name)
    return mults.get(base, 1.0)


def clear_style_cache() -> None:
    _STYLE_CACHE.clear()
