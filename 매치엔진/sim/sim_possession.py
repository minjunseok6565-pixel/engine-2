from __future__ import annotations

"""Possession simulation (team style biasing, priors, resolve loop).

NOTE: Split from sim.py on 2025-12-27.
"""

import random
import math
import warnings
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from .builders import (
    build_offense_action_probs,
    build_outcome_priors,
    get_action_base,
)
from . import shot_diet
from . import quality
from .def_role_players import get_or_build_def_role_players, engine_get_stat
from .core import weighted_choice, clamp
from .models import GameState, TeamState
from .resolve import (
    choose_drb_rebounder,
    choose_orb_rebounder,
    rebound_orb_probability,
    resolve_outcome,
    commit_pending_pass_event,
)
from .sim_rotation import maybe_substitute_deadball_v1
from .role_fit import apply_role_fit_to_priors_and_tags

from .replay import emit_event
from .sim_clock import (
    apply_time_cost,
    apply_dead_ball_cost,
    simulate_inbound,
    commit_shot_clock_turnover,
)

if TYPE_CHECKING:
    from .game_config import GameConfig

#
# Turnover deadball/liveball classification
# ---------------------------------------
# These are the ONLY turnover outcome strings emitted by resolve_outcome()/sim_clock in this engine:
#   TO_HANDLE_LOSS, TO_BAD_PASS, TO_CHARGE, TO_INBOUND, TO_SHOT_CLOCK
#
# Policy:
#   - Deadball turnovers: charge / inbound / shot clock
#   - Liveball turnovers: handle loss / bad pass
#
# NOTE:
#   resolve_outcome() may attach payload flags that override this classification:
#     - payload['deadball_override'] -> force deadball (e.g., bad-pass lineout)
#     - payload['pos_start_next_override'] / payload['steal'] -> force after_steal start
#
_DEADBALL_TURNOVER_OUTCOMES = {
    "TO_CHARGE",
    "TO_INBOUND",
    "TO_SHOT_CLOCK",
}

_LIVEBALL_TURNOVER_OUTCOMES = {
    "TO_HANDLE_LOSS",
    "TO_BAD_PASS",
}

def _normalize_turnover_outcome(o: Any) -> str:
    """Normalize turnover outcome keys (compat for older logs/validation)."""
    try:
        s = str(o or "")
    except Exception:
        return ""
    # legacy key sometimes appears in validation / old logs
    if s == "TO_SHOTCLOCK":
        return "TO_SHOT_CLOCK"
    return s

def _turnover_is_deadball(outcome: Any) -> bool:
    o = _normalize_turnover_outcome(outcome)
    if o in _DEADBALL_TURNOVER_OUTCOMES:
        return True
    if o in _LIVEBALL_TURNOVER_OUTCOMES:
        return False
    # Unknown TO_*: default to LIVE to preserve fastbreak/flow (and avoid surprising deadball windows)
    # but keep a warning to surface schema drift early.
    if o.startswith("TO_"):
        warnings.warn(f"[sim_possession] Unknown turnover outcome '{o}'; defaulting to liveball after_tov")
        return False
    # If payload is missing/invalid, be conservative and keep existing behavior.
    return False


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


def _clean_replay_payload(payload: Any, *, drop: Optional[set] = None) -> Dict[str, Any]:
    """
    Replay payload sanitizer.

    resolve.py returns simulation payloads that may include internal control keys.
    This helper ensures we never forward keys that would collide with emit_event()'s explicit params.
    """
    if not isinstance(payload, dict):
        return {}
    out = dict(payload)
    if drop:
        for k in drop:
            out.pop(k, None)
    return out


def apply_quality_to_turnover_priors(
    pri: Dict[str, float],
    base_action: str,
    offense: TeamState,
    defense: TeamState,
    tags: Dict[str, Any],
    ctx: Dict[str, Any],
) -> Dict[str, float]:
    """Adjust TO_HANDLE_LOSS prior weight using quality-driven 'pressure'.

    quality.compute_quality_score returns an offense-perspective quality score:
      + higher => more open / better for offense
      - lower  => tougher / worse for offense (better defense)

    For turnovers, we want better defense => higher TO probability, so we invert:
        pressure = -quality_score

    We apply an exponential multiplier to pri['TO_HANDLE_LOSS']:

        pri['TO_HANDLE_LOSS'] *= exp(clamp(pressure * K_TO_QUALITY, -CLAMP, +CLAMP))

    Tuning knobs (defense.tactics.context):
      - K_TO_QUALITY (default 0.25)
      - TO_QUALITY_LOG_CLAMP (default 1.0)
    """
    if "TO_HANDLE_LOSS" not in pri:
        return pri

    scheme = getattr(defense.tactics, "defense_scheme", "")
    role_players = get_or_build_def_role_players(ctx, defense, scheme=scheme)

    debug_q = bool(ctx.get("debug_quality", False))
    q_res = quality.compute_quality_score(
        scheme=str(scheme),
        base_action=str(base_action),
        outcome="TO_HANDLE_LOSS",
        role_players=role_players,
        get_stat=engine_get_stat,
        return_detail=debug_q,
    )
    q_score = float(q_res.score) if (debug_q and hasattr(q_res, "score")) else float(q_res)

    pressure = -q_score

    tctx = getattr(defense.tactics, "context", {}) or {}
    k_to = float(tctx.get("K_TO_QUALITY", 0.25))
    log_clamp = float(tctx.get("TO_QUALITY_LOG_CLAMP", 1.0))
    log_mult = clamp(pressure * k_to, -log_clamp, log_clamp)

    pri["TO_HANDLE_LOSS"] = float(pri.get("TO_HANDLE_LOSS", 0.0)) * math.exp(log_mult)

    if debug_q:
        tags["to_quality_score"] = q_score
        tags["to_pressure"] = pressure
        tags["to_log_mult"] = log_mult
        tags["to_weight_after"] = float(pri["TO_HANDLE_LOSS"])

    return pri

def _draw_style_mult(
    rng: random.Random,
    std: float,
    lo: float,
    hi: float,
) -> float:
    return clamp(rng.gauss(1.0, float(std)), float(lo), float(hi))

def _stable_team_style_signature(team: TeamState) -> str:
    """Build a stable signature for when TEAM_STYLE should be recomputed.

    We want TEAM_STYLE to be *sticky* for a given team identity/roster/tactics,
    but also to automatically refresh if the roster or tactics change.
    """
    name = str(getattr(team, "name", ""))
    pids = []
    try:
        pids = [str(getattr(p, "pid", "")) for p in (getattr(team, "lineup", None) or [])]
    except Exception:
        pids = []
    pids = sorted([p for p in pids if p])

    tac = getattr(team, "tactics", None)
    off = str(getattr(tac, "offense_scheme", ""))
    de = str(getattr(tac, "defense_scheme", ""))
    return "|".join([name, ",".join(pids), off, de])


def _team_mean_stat(team: TeamState, key: str, default: float = 50.0) -> float:
    vals = []
    for p in (getattr(team, "lineup", None) or []):
        try:
            # Use fatigue-insensitive value for stable identity.
            v = float(p.get(key, fatigue_sensitive=False))
        except Exception:
            try:
                v = float(getattr(p, "derived", {}).get(key, default))
            except Exception:
                v = default
        vals.append(v)
    if not vals:
        return float(default)
    return float(sum(vals) / len(vals))


def _z_to_mult(z: float, strength: float, lo: float, hi: float) -> float:
    """Convert a coarse z-score into a small multiplicative bias."""
    return clamp(1.0 + float(z) * float(strength), float(lo), float(hi))


def _compute_team_style_deterministic(team: TeamState, rules: Dict[str, Any]) -> Dict[str, float]:
    """Deterministic TEAM_STYLE based on roster (derived stats) + offense scheme.

    Design goals:
    - Same roster+tactics => same style every game (eliminate per-game gaussian jitter)
    - Keep multipliers in a *narrow* band to avoid overpowering baseline era calibration
    - Make mapping robust to missing keys (defaults to 50)
    """
    cfg = (rules.get("team_style") or {})

    # --- Roster-driven signals (0..100 with 50 default) ---
    three_signal = 0.55 * _team_mean_stat(team, "SHOT_3_CS") + 0.45 * _team_mean_stat(team, "SHOT_3_OD")
    rim_signal = 0.55 * _team_mean_stat(team, "FIN_RIM") + 0.25 * _team_mean_stat(team, "FIN_DUNK") + 0.20 * _team_mean_stat(team, "FIN_CONTACT")

    # Turnover risk proxy: poor handle/pass safety => higher tov_bias.
    handle_safe = _team_mean_stat(team, "HANDLE_SAFE")
    pass_safe = _team_mean_stat(team, "PASS_SAFE")
    tov_signal = 100.0 - (0.55 * handle_safe + 0.45 * pass_safe)

    # FTr proxy: contact finishing + touch.
    ftr_signal = 0.55 * _team_mean_stat(team, "FIN_CONTACT") + 0.45 * _team_mean_stat(team, "SHOT_TOUCH")

    # Pace proxy: endurance + athleticism-ish. Keep small.
    tempo_signal = 0.60 * _team_mean_stat(team, "ENDURANCE") + 0.40 * _team_mean_stat(team, "FIRST_STEP")

    # Normalize around 50 into a coarse z in roughly [-2, +2].
    # Denominator 25 => 50±25 gives ±1.
    z_three = (three_signal - 50.0) / 25.0
    z_rim = (rim_signal - 50.0) / 25.0
    z_tov = (tov_signal - 50.0) / 25.0
    z_ftr = (ftr_signal - 50.0) / 25.0
    z_tempo = (tempo_signal - 50.0) / 25.0

    # --- Narrow bands (default) ---
    # You can tune these via rules['team_style'] if needed.
    tempo_lo = float(cfg.get("tempo_lo", 0.94)); tempo_hi = float(cfg.get("tempo_hi", 1.06))
    three_lo = float(cfg.get("three_lo", 0.88)); three_hi = float(cfg.get("three_hi", 1.12))
    rim_lo = float(cfg.get("rim_lo", 0.88)); rim_hi = float(cfg.get("rim_hi", 1.12))
    tov_lo = float(cfg.get("tov_lo", 0.88)); tov_hi = float(cfg.get("tov_hi", 1.12))
    ftr_lo = float(cfg.get("ftr_lo", 0.86)); ftr_hi = float(cfg.get("ftr_hi", 1.14))

    # Strengths (how aggressively roster signal moves the multiplier).
    s_tempo = float(cfg.get("tempo_strength", 0.03))
    s_three = float(cfg.get("three_strength", 0.06))
    s_rim = float(cfg.get("rim_strength", 0.06))
    s_tov = float(cfg.get("tov_strength", 0.06))
    s_ftr = float(cfg.get("ftr_strength", 0.07))

    style = {
        "tempo_mult": _z_to_mult(z_tempo, s_tempo, tempo_lo, tempo_hi),
        "three_bias": _z_to_mult(z_three, s_three, three_lo, three_hi),
        "rim_bias": _z_to_mult(z_rim, s_rim, rim_lo, rim_hi),
        "tov_bias": _z_to_mult(z_tov, s_tov, tov_lo, tov_hi),
        "ftr_bias": _z_to_mult(z_ftr, s_ftr, ftr_lo, ftr_hi),
    }

    # --- Small scheme nudges (keep small!) ---
    try:
        scheme = str(getattr(getattr(team, "tactics", None), "offense_scheme", ""))
    except Exception:
        scheme = ""

    scheme_mods = {
        # More 3s / spacing
        "Spread_HeavyPnR": {"three_bias": 1.03, "tempo_mult": 1.01},
        "Drive_Kick": {"rim_bias": 1.02, "three_bias": 1.01},
        "DHO_Chicago": {"three_bias": 1.02},
        "Transition_Early": {"tempo_mult": 1.02},
        # More rim / post
        "Post_Inside": {"rim_bias": 1.03, "three_bias": 0.98, "tempo_mult": 0.99},
    }
    mods = scheme_mods.get(scheme)
    if isinstance(mods, dict):
        for k, m in mods.items():
            if k in style:
                style[k] = clamp(float(style[k]) * float(m), 0.80, 1.25)

    return style


def ensure_team_style(rng: random.Random, team: TeamState, rules: Dict[str, Any]) -> Dict[str, float]:
    """Return a persistent TEAM_STYLE profile for a team.

    **Default behavior is deterministic** (stable across games for same roster+tactics),
    fixing the "team feels different every game" issue caused by per-game gaussian jitter.

    Modes (rules['team_style']['mode']):
      - 'deterministic' (default): roster/tactics based mapping
      - 'seeded': stable random per team signature (keeps diversity, but repeatable)
      - 'gaussian': legacy per-game gaussian jitter (NOT recommended)
    """
    cfg = (rules.get("team_style") or {})
    mode = str(cfg.get("mode", "deterministic")).lower().strip()

    sig = _stable_team_style_signature(team)

    # Prefer explicit TeamState cache fields (avoids tactics.context side-effects).
    existing = getattr(team, "team_style", None)
    existing_sig = getattr(team, "team_style_sig", None)
    if isinstance(existing, dict) and existing and existing_sig == sig:
        return existing

    # Fallback legacy cache (in case some code still expects it)
    try:
        tctx = getattr(team.tactics, "context", None)
    except Exception:
        tctx = None
    if isinstance(tctx, dict) and isinstance(tctx.get("TEAM_STYLE"), dict) and (existing_sig != sig):
        # If legacy cache exists but signature differs, ignore it and recompute.
        pass

    if mode == "gaussian":
        style = {
            "tempo_mult": _draw_style_mult(rng, std=float(cfg.get("tempo_std", 0.032)), lo=0.92, hi=1.08),
            "three_bias": _draw_style_mult(rng, std=float(cfg.get("three_std", 0.12)), lo=0.70, hi=1.35),
            "rim_bias": _draw_style_mult(rng, std=float(cfg.get("rim_std", 0.10)), lo=0.75, hi=1.30),
            "tov_bias": _draw_style_mult(rng, std=float(cfg.get("tov_std", 0.14)), lo=0.70, hi=1.40),
            "ftr_bias": _draw_style_mult(rng, std=float(cfg.get("ftr_std", 0.18)), lo=0.60, hi=1.50),
        }
    elif mode == "seeded":
        # Stable random per team signature.
        # Use a local RNG so game RNG doesn't get perturbed.
        seed = 0
        for ch in sig:
            seed = (seed * 131 + ord(ch)) & 0xFFFFFFFF
        local = random.Random(seed)
        style = {
            "tempo_mult": _draw_style_mult(local, std=float(cfg.get("tempo_std", 0.028)), lo=0.93, hi=1.07),
            "three_bias": _draw_style_mult(local, std=float(cfg.get("three_std", 0.08)), lo=0.85, hi=1.15),
            "rim_bias": _draw_style_mult(local, std=float(cfg.get("rim_std", 0.08)), lo=0.85, hi=1.15),
            "tov_bias": _draw_style_mult(local, std=float(cfg.get("tov_std", 0.09)), lo=0.85, hi=1.15),
            "ftr_bias": _draw_style_mult(local, std=float(cfg.get("ftr_std", 0.10)), lo=0.83, hi=1.17),
        }
    else:
        # Deterministic (recommended).
        style = _compute_team_style_deterministic(team, rules)

    # Store on TeamState (preferred).
    try:
        team.team_style = dict(style)
        team.team_style_sig = sig
    except Exception:
        pass

    # Also store in tactics.context for backward compatibility, but keep it in sync.
    if isinstance(tctx, dict):
        tctx["TEAM_STYLE"] = dict(style)

    return style


def _renorm(d: Dict[str, float]) -> Dict[str, float]:
    s = sum(float(v) for v in (d or {}).values())
    if s <= 0:
        return d
    return {k: float(v) / s for k, v in d.items()}

def apply_team_style_to_action_probs(
    probs: Dict[str, float],
    style: Dict[str, float],
    game_cfg: "GameConfig",
) -> Dict[str, float]:
    if not probs or not style:
        return probs
    out = dict(probs)
    three_bias = float(style.get("three_bias", 1.0))
    rim_bias = float(style.get("rim_bias", 1.0))
    tempo_mult = float(style.get("tempo_mult", 1.0))

    for k, v in list(out.items()):
        base = get_action_base(k, game_cfg)
        mult = 1.0
        if base == "TransitionEarly":
            mult *= tempo_mult ** 0.85
        if base in ("Kickout", "ExtraPass", "SpotUp"):
            mult *= three_bias
        if base in ("Drive", "Cut"):
            mult *= rim_bias
        if base in ("PnR", "DHO"):
            mult *= (0.55 * three_bias + 0.45 * rim_bias)
        out[k] = float(v) * float(mult)

    return _renorm(out)

def apply_team_style_to_outcome_priors(pri: Dict[str, float], style: Dict[str, float]) -> Dict[str, float]:
    if not pri or not style:
        return pri
    out = dict(pri)
    three_bias = float(style.get("three_bias", 1.0))
    rim_bias = float(style.get("rim_bias", 1.0))
    tov_bias = float(style.get("tov_bias", 1.0))
    ftr_bias = float(style.get("ftr_bias", 1.0))

    for k, v in list(out.items()):
        vv = float(v)
        if k.startswith("TO_"):
            vv *= tov_bias
        elif k.startswith("FOUL_DRAW_") or k == "FOUL_REACH_TRAP":
            vv *= ftr_bias
        elif k.startswith("SHOT_3_"):
            vv *= three_bias
        elif k.startswith("SHOT_RIM_"):
            vv *= rim_bias
        out[k] = vv

    return _renorm(out)


# -------------------------
# Possession simulation
# -------------------------

def simulate_possession(
    rng: random.Random,
    offense: TeamState,
    defense: TeamState,
    game_state: GameState,
    rules: Dict[str, Any],
    ctx: Dict[str, Any],
    game_cfg: Optional["GameConfig"] = None,
    max_steps: int = 7,
) -> Dict[str, Any]:
    """Simulate a single possession.

    Returns a dict describing how the possession ended so the game loop can be event-based.
    """

    if ctx is None:
        ctx = {}
    if game_cfg is None:
        raise ValueError("simulate_possession requires game_cfg")

    # Replay logging: infer which TeamState corresponds to home/away.
    # (We do this once per simulate_possession call; it's cheap and keeps logging consistent.)
    home_team, away_team, off_team_id, def_team_id = _validate_possession_team_ids(offense, defense, game_state, ctx or {})

    # Possession-continuation support:
    # Some dead-ball events (e.g. no-shot foul) can stop play and restart with the same offense.
    # In those cases, the game loop will call simulate_possession again with ctx['_pos_continuation']=True.
    # We must avoid double-counting possessions and must preserve possession-scope aggregates.
    is_continuation = bool(ctx.get("_pos_continuation", False))
    if not is_continuation:
        offense.possessions += 1
        before_pts = int(offense.pts)
    else:
        before_pts = int(ctx.get("_pos_before_pts", int(offense.pts)))

    # Current segment start vs. possession-origin start (for attribution like fastbreak/points_off_tov).
    pos_start = str(ctx.get("pos_start", ""))
    pos_origin = str(ctx.get("_pos_origin_start", pos_start))

    def _record_ctx_error(where: str, exc: BaseException) -> None:
        try:
            errs = ctx.setdefault("errors", [])
            errs.append(
                {
                    "where": where,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        except Exception:
            return

    tempo_mult = float(ctx.get("tempo_mult", 1.0))
    time_costs = rules.get("time_costs", {})
    had_orb = bool(ctx.get("_pos_had_orb", False))

    # per-team style profile (persistent; increases team diversity)
    team_style = ensure_team_style(rng, offense, rules)
    if team_style:
        tempo_mult *= float(team_style.get("tempo_mult", 1.0))
        # keep ctx immutable-ish
        ctx = dict(ctx)
        ctx["tempo_mult"] = tempo_mult
        ctx["team_style"] = team_style

    # Dead-ball start can trigger inbound (score, quarter start, dead-ball TO, no-shot foul restart, etc.)
    dead_ball_starts = {"start_q", "after_score", "after_tov_dead", "after_foul", "after_block_oob"}
    if pos_start in dead_ball_starts:
        # dead-ball inbound attempt
        if simulate_inbound(rng, offense, defense, rules):
            # IMPORTANT:
            # Inbound turnovers are dead-ball turnovers. Next possession should start as dead-ball inbound.
            # Also, inbound turnover consumes 0 action-time, so repeated inbound turnovers could freeze the
            # game clock in the outer loop unless we apply a small dead-ball admin cost.
            time_costs = rules.get("time_costs", {}) or {}
            try:
                inbound_tov_cost = float(time_costs.get("InboundTurnover", 1.0))
            except Exception:
                inbound_tov_cost = 1.0
            if inbound_tov_cost > 0:
                apply_dead_ball_cost(game_state, inbound_tov_cost, tempo_mult)
                if game_state.clock_sec <= 0:
                    game_state.clock_sec = 0
                    return {
                        "end_reason": "PERIOD_END",
                        "pos_start_next": pos_start,
                        "points_scored": int(offense.pts) - before_pts,
                        "had_orb": had_orb,
                        "pos_start": pos_origin,
                        "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
                    }
                    
            # NBA-style shot-clock foul top-up:
            # If a defensive no-shot foul results in an inbounds (offense retains), the shot clock is
            # topped up to `foul_reset` (e.g., 14) only when the remaining time is below that value.
            try:
                foul_reset = float(rules.get("foul_reset", 14))
            except Exception:
                foul_reset = 14.0
            try:
                full_sc = float(rules.get("shot_clock", 24))
            except Exception:
                full_sc = 24.0
            if foul_reset > 0:
                foul_reset = min(foul_reset, full_sc)
                try:
                    if float(game_state.shot_clock_sec) < foul_reset:
                        game_state.shot_clock_sec = foul_reset
                except Exception:
                    # If shot_clock_sec is missing or invalid, fall back to foul_reset.
                    game_state.shot_clock_sec = foul_reset
                    
            return {
                "end_reason": "TURNOVER",
                "pos_start_next": "after_tov_dead",
                "points_scored": int(offense.pts) - before_pts,
                "had_orb": had_orb,
                "pos_start": pos_origin,
                "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
                "turnover_outcome": "TO_INBOUND",
                "turnover_deadball": True,
            }

    # shot_diet wiring
    style = shot_diet.compute_shot_diet_style(offense, defense, game_state=game_state, ctx=ctx)
    tactic_name = None
    try:
        tactic_name = offense.tactics.offense_scheme
    except Exception as exc:
        _record_ctx_error("tactic_name_access", exc)
        tactic_name = None
    ctx["shot_diet_style"] = style
    ctx["tactic_name"] = tactic_name

    def _apply_contextual_action_weights(probs: Dict[str, float]) -> Dict[str, float]:
        """Soft-bias action weights by possession context (no per-team fixed style)."""
        if not probs:
            return probs
        if bool(ctx.get("dead_ball_inbound", False)):
            return probs
        pstart = str(ctx.get("pos_start", pos_start))
        if pstart not in ("after_drb", "after_tov", "after_steal", "after_block"):
            return probs
        mult_tbl = rules.get("transition_weight_mult", {}) or {}
        try:
            mult = float(mult_tbl.get(pstart, mult_tbl.get("default", 1.0)))
        except Exception:
            mult = 1.0
        if mult <= 1.0:
            return probs

        out = dict(probs)
        changed = False
        for k, v in list(out.items()):
            if get_action_base(k, game_cfg) == "TransitionEarly":
                out[k] = float(v) * mult
                changed = True
        if not changed:
            return probs
        s = sum(out.values())
        if s <= 0:
            return probs
        for k in out:
            out[k] /= s
        return out



    # -------------------------------------------------------------------------
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
    off_probs = build_offense_action_probs(offense.tactics, defense.tactics, ctx=ctx, game_cfg=game_cfg)
    off_probs = _apply_contextual_action_weights(off_probs)
    off_probs = apply_team_style_to_action_probs(off_probs, team_style, game_cfg)

    action = choose_action_with_budget(rng, off_probs)
    offense.off_action_counts[action] = offense.off_action_counts.get(action, 0) + 1

    tags = {
        "in_transition": (get_action_base(action, game_cfg) == "TransitionEarly"),
        "is_side_pnr": (action == "SideAnglePnR"),
        "avg_fatigue_off": ctx.get("avg_fatigue_off"),
        "fatigue_bad_mult_max": ctx.get("fatigue_bad_mult_max"),
        "fatigue_bad_critical": ctx.get("fatigue_bad_critical"),
        "fatigue_bad_bonus": ctx.get("fatigue_bad_bonus"),
        "fatigue_bad_cap": ctx.get("fatigue_bad_cap"),
    }

    # --- ADD: action-dependent tags refresh helper ---
    def _refresh_action_tags(_action: str, _tags: dict) -> None:
        _tags["in_transition"] = (get_action_base(_action, game_cfg) == "TransitionEarly")
        _tags["is_side_pnr"] = (_action == "SideAnglePnR")

    # ensure initial consistency (safe even if already set above)
    _refresh_action_tags(action, tags)


    # `max_steps` is used as a safety against "no-time-progress" loops (e.g. sequences of 0-cost actions/passes).
    # When we observe `max_steps` consecutive iterations with no change to either the shot clock or game clock,
    # we force a real action (a quick SpotUp) so the possession ends naturally instead of producing an
    # artificial SHOTCLOCK turnover.
    stall_steps = 0
    pass_chain = 0

    def _bump_stall(_stall: int, _sc0: float, _gc0: float) -> int:
        """Increment stall counter if no time progressed this iteration, else reset to 0."""
        try:
            if float(game_state.shot_clock_sec) == float(_sc0) and float(game_state.clock_sec) == float(_gc0):
                return _stall + 1
        except Exception:
            # If clocks are in an unexpected state, prefer forcing progress sooner.
            return _stall + 1
        return 0

    while game_state.clock_sec > 0:
        sc0 = float(game_state.shot_clock_sec)
        gc0 = float(game_state.clock_sec)

        forced_due_to_stall = False
        if stall_steps >= max_steps:
            forced_due_to_stall = True
            stall_steps = 0
            action = "QuickShot"
            tags["forced_max_steps"] = True
            _refresh_action_tags(action, tags)

        action_cost = float(_estimate_action_cost_sec(action))
        # Clamp cost so we never consume more time than remains.
        tm = float(tempo_mult) if float(tempo_mult) > 0 else 1.0
        max_base_cost = max(0.0, min(float(sc0), float(gc0)) / tm)
        if action_cost > max_base_cost:
            action_cost = max_base_cost

        clock_expired = False
        shotclock_expired = False

        if action_cost > 0:
            apply_time_cost(game_state, action_cost, tempo_mult)
        elif forced_due_to_stall:
            # When forcing a bailout due to stalling, ensure clocks advance a bit.
            forced_cost = min(0.75, max_base_cost)
            if forced_cost > 0:
                apply_time_cost(game_state, forced_cost, tempo_mult)

        # Normalize negative clocks to 0 for stability.
        if game_state.clock_sec < 0:
            game_state.clock_sec = 0
        if game_state.shot_clock_sec < 0:
            game_state.shot_clock_sec = 0

        clock_expired = (game_state.clock_sec <= 0)
        shotclock_expired = (game_state.shot_clock_sec <= 0)

        base_action_now = get_action_base(action, game_cfg)

        # If time expires during a non-terminal action (pass/reset), end immediately.
        if shotclock_expired and _is_nonterminal_base(base_action_now):
            commit_shot_clock_turnover(offense)
            # Replay log: shot-clock violation -> turnover (deadball)
            try:
                emit_event(
                    game_state,
                    event_type="TURNOVER",
                    home=home_team,
                    away=away_team,
                    rules=rules,
                    team_id=off_team_id,
                    opp_team_id=def_team_id,
                    pos_start=str(pos_origin),
                    pos_start_next="after_tov_dead",
                    outcome="TO_SHOT_CLOCK",
                    deadball_override=True,
                    tov_deadball_reason="SHOT_CLOCK",
                )
            except Exception:
                pass
            return {
                "end_reason": "SHOTCLOCK",
                "pos_start_next": "after_tov_dead",
                "points_scored": int(offense.pts) - before_pts,
                "had_orb": had_orb,
                "pos_start": pos_origin,
                "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
            }
        if clock_expired and _is_nonterminal_base(base_action_now):
            game_state.clock_sec = 0
            return {
                "end_reason": "PERIOD_END",
                "pos_start_next": pos_start,
                "points_scored": int(offense.pts) - before_pts,
                "had_orb": had_orb,
                "pos_start": pos_origin,
                "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
            }


        # shot_diet: pass ctx so outcome multipliers can apply
        pri = build_outcome_priors(action, offense.tactics, defense.tactics, tags, ctx=ctx, game_cfg=game_cfg)
        pri = apply_team_style_to_outcome_priors(pri, team_style)
        pri = apply_role_fit_to_priors_and_tags(pri, get_action_base(action, game_cfg), offense, tags, game_cfg=game_cfg)
        pri = apply_quality_to_turnover_priors(pri, get_action_base(action, game_cfg), offense, defense, tags, ctx)
        pri = _apply_urgent_outcome_constraints(pri)
        if clock_expired or shotclock_expired:
            pri_term = {k: v for k, v in pri.items() if (not k.startswith("PASS_") and not k.startswith("RESET_"))}
            if pri_term:
                pri = _normalize_prob_map(pri_term)
        outcome = weighted_choice(rng, pri)

        term, payload = resolve_outcome(
            rng,
            outcome,
            action,
            offense,
            defense,
            tags,
            pass_chain,
            ctx=ctx,
            game_state=game_state,
            game_cfg=game_cfg,
        )

        if term == "SCORE":
            # Replay log: made shot (resolve payload contains pid/points/assist/outcome etc.)
            try:
                rp = _clean_replay_payload(payload)
                emit_event(
                    game_state,
                    event_type="SCORE",
                    home=home_team,
                    away=away_team,
                    rules=rules,
                    team_id=off_team_id,
                    opp_team_id=def_team_id,
                    pos_start=str(pos_origin),
                    pos_start_next="after_score",
                    **rp,
                )
            except Exception:
                pass
            return {
                "end_reason": "SCORE",
                "pos_start_next": "after_score",
                "points_scored": int(offense.pts) - before_pts,
                "had_orb": had_orb,
                "pos_start": pos_origin,
                "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
            }

        if term == "TURNOVER":
            tov_outcome = _normalize_turnover_outcome(payload.get("outcome") if isinstance(payload, dict) else "")
            is_dead = _turnover_is_deadball(tov_outcome)
            pos_start_next = ("after_tov_dead" if is_dead else "after_tov")
            tov_deadball_reason = None
            tov_is_steal = False
            tov_stealer_pid = None
            pstart_override_for_log = None

            if isinstance(payload, dict):
                # Allow resolve layer to override live/dead classification (e.g., bad-pass lineout).
                if payload.get("deadball_override") is True:
                    is_dead = True
                elif payload.get("deadball_override") is False:
                    is_dead = False
                if payload.get("tov_deadball_reason") is not None:
                    try:
                        tov_deadball_reason = str(payload.get("tov_deadball_reason"))
                    except Exception:
                        tov_deadball_reason = None

                tov_is_steal = bool(payload.get("steal", False))
                tov_stealer_pid = payload.get("stealer_pid")

                # Allow explicit next-start override (preferred) and fallback to 'steal' flag.
                pstart_override = payload.get("pos_start_next_override")
                if pstart_override:
                    try:
                        pstart_override_for_log = str(pstart_override)
                        pos_start_next = pstart_override_for_log
                    except Exception:
                        pos_start_next = pos_start_next
                elif tov_is_steal and not is_dead:
                    pos_start_next = "after_steal"

            # Re-derive default after override, so deadball flags remain consistent.
            if pos_start_next == "after_tov_dead":
                is_dead = True
            elif pos_start_next == "after_tov":
                is_dead = False
            # Replay log: turnover (resolve payload contains pid/outcome/type/steal/...).
            try:
                rp = _clean_replay_payload(payload, drop={"pos_start_next_override"})
                emit_event(
                    game_state,
                    event_type="TURNOVER",
                    home=home_team,
                    away=away_team,
                    rules=rules,
                    team_id=off_team_id,
                    opp_team_id=def_team_id,
                    pos_start=str(pos_origin),
                    pos_start_next=str(pos_start_next),
                    pos_start_next_override=pstart_override_for_log,
                    **rp,
                )
            except Exception:
                pass
            return {
                "end_reason": "TURNOVER",
                "pos_start_next": (pos_start_next if pos_start_next else ("after_tov_dead" if is_dead else "after_tov")),
                "points_scored": int(offense.pts) - before_pts,
                "had_orb": had_orb,
                "pos_start": pos_origin,
                "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
                "turnover_outcome": tov_outcome,
                "turnover_deadball": bool(is_dead),
                "turnover_deadball_reason": tov_deadball_reason,
                "turnover_is_steal": bool(tov_is_steal),
                "turnover_stealer_pid": tov_stealer_pid,
            }

        if term == "FOUL_NO_SHOTS":
            # Dead-ball stop, offense retains ball.
            # NOTE: We intentionally do NOT run the inbound here.
            # The game loop may want to do substitutions / timeouts / UI stops between the whistle and inbound.
            # Replay log: foul (no shots). team_side is the fouling team (defense).
            try:
                rp = _clean_replay_payload(payload)
                emit_event(
                    game_state,
                    event_type="FOUL_NO_SHOTS",
                    home=home_team,
                    away=away_team,
                    rules=rules,
                    team_id=def_team_id,
                    opp_team_id=off_team_id,
                    pos_start=str(pos_origin),
                    pos_start_next="after_foul",
                    **rp,
                )
            except Exception:
                pass
            return {
                "end_reason": "DEADBALL_STOP",
                "deadball_reason": "FOUL_NO_SHOTS",
                "pos_start_next": "after_foul",
                "points_scored": int(offense.pts) - before_pts,
                "had_orb": had_orb,
                "pos_start": pos_origin,
                "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
            }



        if term == "FOUL_FT":
            # Replay log: foul + FT trip result. team_side is the fouling team (defense).
            # (Log once here, regardless of whether last FT was made; rebound (if any) is logged separately.)
            try:
                rp = _clean_replay_payload(payload)
                emit_event(
                    game_state,
                    event_type="FOUL_FT",
                    home=home_team,
                    away=away_team,
                    rules=rules,
                    team_id=def_team_id,
                    opp_team_id=off_team_id,
                    pos_start=str(pos_origin),
                    pos_start_next=("after_score" if bool(getattr(payload, "get", lambda *_: False)("last_made", False)) else None),
                    **rp,
                )
            except Exception:
                pass
            # If last FT made -> dead-ball score, possession ends.
            if bool(payload.get("last_made", False)):
                return {
                    "end_reason": "SCORE",
                    "pos_start_next": "after_score",
                    "points_scored": int(offense.pts) - before_pts,
                    "had_orb": had_orb,
                    "pos_start": pos_origin,
                    "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
                    "ended_with_ft_trip": True,
                }
                

            # last FT missed -> live rebound
            # Before we select a rebounder, force foul-out substitutions NOW so a fouled-out player
            # cannot remain in the rebounder candidate pool.
            try:
                foul_out = int(ctx.get("foul_out", rules.get("foul_out", 6)))
                pf_map = (game_state.player_fouls.get(def_team_id, {}) or {})

                forced_out = [
                    pid for pid in list(getattr(defense, "on_court_pids", []) or [])
                    if int(pf_map.get(pid, 0)) >= foul_out
                ]

                if forced_out:

                    maybe_substitute_deadball_v1(
                        rng,
                        defense,
                        home_team,
                        away_team,
                        game_state,
                        rules,
                        q_index=max(0, int(getattr(game_state, "quarter", 1)) - 1),
                        pos_start="after_foul",
                        pressure_index=float(ctx.get("pressure_index", 0.0)),
                        garbage_index=float(ctx.get("garbage_index", 0.0)),
                    )

            except ValueError:
                raise
            except Exception:
                # Forced-sub logic should not break simulation, but SSOT violations must crash.
                pass

            orb_mult = float(offense.tactics.context.get("ORB_MULT", 1.0)) * float(rules.get("ft_orb_mult", 0.75))
            drb_mult = float(defense.tactics.context.get("DRB_MULT", 1.0))
            p_orb = rebound_orb_probability(offense, defense, orb_mult, drb_mult, game_cfg=game_cfg)
            if rng.random() < p_orb:
                offense.orb += 1
                rbd = choose_orb_rebounder(rng, offense)
                offense.add_player_stat(rbd.pid, "ORB", 1)
                # Replay log: offensive rebound after missed FT
                try:
                    emit_event(
                        game_state,
                        event_type="REB",
                        home=home_team,
                        away=away_team,
                        rules=rules,
                        team_id=off_team_id,
                        opp_team_id=def_team_id,
                        pos_start=str(pos_origin),
                        pid=getattr(rbd, "pid", None),
                        outcome="ORB",
                    )
                except Exception:
                    pass
                game_state.shot_clock_sec = float(rules.get("foul_reset", rules.get("orb_reset", game_state.shot_clock_sec)))
                r2 = rng.random()
                if r2 < 0.45:
                    action = "Kickout"
                elif r2 < 0.60:
                    action = "ExtraPass"
                else:
                    action = "Drive"
                _refresh_action_tags(action, tags)
                pass_chain = 0
                had_orb = True
                stall_steps = _bump_stall(stall_steps, sc0, gc0)
                continue


            defense.drb += 1
            rbd = choose_drb_rebounder(rng, defense)
            defense.add_player_stat(rbd.pid, "DRB", 1)
            # Replay log: defensive rebound after missed FT
            try:
                emit_event(
                    game_state,
                    event_type="REB",
                    home=home_team,
                    away=away_team,
                    rules=rules,
                    team_id=def_team_id,
                    opp_team_id=off_team_id,
                    pos_start=str(pos_origin),
                    pid=getattr(rbd, "pid", None),
                    outcome="DRB",
                )
            except Exception:
                pass
            return {
                "end_reason": "DRB",
                "pos_start_next": "after_drb",
                "points_scored": int(offense.pts) - before_pts,
                "had_orb": had_orb,
                "pos_start": pos_origin,
                "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
                "ended_with_ft_trip": True,
            }

        if term == "MISS":
            blocked = bool(payload.get("blocked", False)) if isinstance(payload, dict) else False
            block_kind = str(payload.get("block_kind", "")) if (blocked and isinstance(payload, dict)) else ""
            blocker_pid = payload.get("blocker_pid") if (blocked and isinstance(payload, dict)) else None

            # If the miss was blocked, sometimes the block goes out-of-bounds -> dead-ball inbound,
            # offense retains (continuation). This is a key "game-feel" lever for rim protection.
            if blocked:
                pm = getattr(game_cfg, "prob_model", {}) or {}
                bk = (block_kind or "").lower()
                if ("rim" in bk) or (bk == "shot_rim"):
                    k = "rim"
                elif ("post" in bk) or (bk == "shot_post"):
                    k = "post"
                elif ("mid" in bk) or (bk == "shot_mid"):
                    k = "mid"
                else:
                    k = "3"

                try:
                    p_oob = float(pm.get(f"block_oob_base_{k}", 0.22))
                except Exception:
                    p_oob = 0.22

                if rng.random() < clamp(p_oob, 0.0, 0.95):
                    # BLOCK_OOB: defense last touched -> out of bounds, offense retains.
                    # NBA-style: keep the remaining (unexpired) shot clock.
                    # Replay log: miss (blocked) that ends in deadball stop / inbound retain
                    try:
                        rp = _clean_replay_payload(payload)
                        emit_event(
                            game_state,
                            event_type="MISS",
                            home=home_team,
                            away=away_team,
                            rules=rules,
                            team_id=off_team_id,
                            opp_team_id=def_team_id,
                            pos_start=str(pos_origin),
                            pos_start_next="after_block_oob",
                            deadball_reason="BLOCK_OOB",
                            **rp,
                        )
                    except Exception:
                        pass

                    return {
                        "end_reason": "DEADBALL_STOP",
                        "deadball_reason": "BLOCK_OOB",
                        "pos_start_next": "after_block_oob",
                        "points_scored": int(offense.pts) - before_pts,
                        "had_orb": had_orb,
                        "pos_start": pos_origin,
                        "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
                        "was_blocked": True,
                        "blocker_pid": blocker_pid,
                        "block_kind": block_kind,
                    }

            # Replay log: regular miss (may lead to rebound)
            try:
                rp = _clean_replay_payload(payload)
                emit_event(
                    game_state,
                    event_type="MISS",
                    home=home_team,
                    away=away_team,
                    rules=rules,
                    team_id=off_team_id,
                    opp_team_id=def_team_id,
                    pos_start=str(pos_origin),
                    **rp,
                )
            except Exception:
                pass

            orb_mult = float(offense.tactics.context.get("ORB_MULT", 1.0))
            drb_mult = float(defense.tactics.context.get("DRB_MULT", 1.0))
            p_orb = rebound_orb_probability(offense, defense, orb_mult, drb_mult, game_cfg=game_cfg)

            # Blocked misses that stay in play are harder for the offense to recover.
            if blocked:
                pm = getattr(game_cfg, "prob_model", {}) or {}
                bk = (block_kind or "").lower()
                if ("rim" in bk) or (bk == "shot_rim"):
                    k = "rim"
                elif ("post" in bk) or (bk == "shot_post"):
                    k = "post"
                elif ("mid" in bk) or (bk == "shot_mid"):
                    k = "mid"
                else:
                    k = "3"
                try:
                    mult = float(pm.get(f"blocked_orb_mult_{k}", 0.82))
                except Exception:
                    mult = 0.82
                p_orb = clamp(float(p_orb) * clamp(mult, 0.10, 1.20), 0.02, 0.60)
                
            if rng.random() < p_orb:
                offense.orb += 1
                rbd = choose_orb_rebounder(rng, offense)
                offense.add_player_stat(rbd.pid, "ORB", 1)
                # Replay log: offensive rebound after miss
                try:
                    emit_event(
                        game_state,
                        event_type="REB",
                        home=home_team,
                        away=away_team,
                        rules=rules,
                        team_id=off_team_id,
                        opp_team_id=def_team_id,
                        pos_start=str(pos_origin),
                        pid=getattr(rbd, "pid", None),
                        outcome="ORB",
                    )
                except Exception:
                    pass
                game_state.shot_clock_sec = float(rules.get("orb_reset", game_state.shot_clock_sec))
                r2 = rng.random()
                if r2 < 0.45:
                    action = "Kickout"
                elif r2 < 0.60:
                    action = "ExtraPass"
                else:
                    action = "Drive"
                _refresh_action_tags(action, tags)
                pass_chain = 0
                had_orb = True
                stall_steps = _bump_stall(stall_steps, sc0, gc0)
                continue

            defense.drb += 1
            rbd = choose_drb_rebounder(rng, defense)
            defense.add_player_stat(rbd.pid, "DRB", 1)
            # Replay log: defensive rebound after miss
            try:
                emit_event(
                    game_state,
                    event_type="REB",
                    home=home_team,
                    away=away_team,
                    rules=rules,
                    team_id=def_team_id,
                    opp_team_id=off_team_id,
                    pos_start=str(pos_origin),
                    pid=getattr(rbd, "pid", None),
                    outcome="DRB",
                )
            except Exception:
                pass
            return {
                "end_reason": "DRB",
                "pos_start_next": ("after_block" if blocked else "after_drb"),
                "points_scored": int(offense.pts) - before_pts,
                "had_orb": had_orb,
                "pos_start": pos_origin,
                "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
                "was_blocked": bool(blocked),
                "blocker_pid": blocker_pid,
                "block_kind": block_kind,
            }

        if term == "RESET":
            reset_cost = float(time_costs.get("Reset", 0.0))
            tm = float(tempo_mult) if float(tempo_mult) > 0 else 1.0
            budget_now = _budget_sec()

            # If time is tight, skip the reset and force a quick attempt.
            if budget_now <= (min_release_window + 0.05) or (reset_cost * tm) >= max(0.0, budget_now - min_release_window):
                action = "QuickShot"
                _refresh_action_tags(action, tags)
                pass_chain = 0
                stall_steps = _bump_stall(stall_steps, sc0, gc0)
                continue
                
            if reset_cost > 0:
                apply_time_cost(game_state, reset_cost, tempo_mult)
                if game_state.clock_sec < 0:
                    game_state.clock_sec = 0
                if game_state.shot_clock_sec < 0:
                    game_state.shot_clock_sec = 0
                    
                if game_state.shot_clock_sec <= 0:
                    commit_shot_clock_turnover(offense)
                    # Replay log: shot-clock violation via RESET path -> turnover (deadball)
                    try:
                        emit_event(
                            game_state,
                            event_type="TURNOVER",
                            home=home_team,
                            away=away_team,
                            rules=rules,
                            team_id=off_team_id,
                            opp_team_id=def_team_id,
                            pos_start=str(pos_origin),
                            pos_start_next="after_tov_dead",
                            outcome="TO_SHOT_CLOCK",
                            deadball_override=True,
                            tov_deadball_reason="SHOT_CLOCK",
                        )
                    except Exception:
                        pass
                    return {
                        "end_reason": "SHOTCLOCK",
                        "pos_start_next": "after_tov_dead",
                        "points_scored": int(offense.pts) - before_pts,
                        "had_orb": had_orb,
                        "pos_start": pos_origin,
                        "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
                    }
                if game_state.clock_sec <= 0:
                    game_state.clock_sec = 0
                    return {
                        "end_reason": "PERIOD_END",
                        "pos_start_next": pos_start,
                        "points_scored": int(offense.pts) - before_pts,
                        "had_orb": had_orb,
                        "pos_start": pos_origin,
                        "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
                    }
            off_probs = build_offense_action_probs(offense.tactics, defense.tactics, ctx=ctx, game_cfg=game_cfg)
            off_probs = _apply_contextual_action_weights(off_probs)
            off_probs = apply_team_style_to_action_probs(off_probs, team_style, game_cfg)
            action = choose_action_with_budget(rng, off_probs)
            offense.off_action_counts[action] = offense.off_action_counts.get(action, 0) + 1
            _refresh_action_tags(action, tags)
            pass_chain = 0
            stall_steps = _bump_stall(stall_steps, sc0, gc0)
            continue


        if term == "CONTINUE":
            pass_chain = payload.get("pass_chain", pass_chain + 1)
            pass_cost_nominal = 0.0
            if outcome in ("PASS_KICKOUT", "PASS_SKIP"):
                pass_cost_nominal = float(time_costs.get("Kickout", 0.0))
            elif outcome == "PASS_EXTRA":
                pass_cost_nominal = float(time_costs.get("ExtraPass", 0.0))
            elif outcome == "PASS_SHORTROLL":
                # Reuse ExtraPass cost for shortroll outlets if no dedicated key exists.
                pass_cost_nominal = float(time_costs.get("ExtraPass", 0.0))

            tm = float(tempo_mult) if float(tempo_mult) > 0 else 1.0
            budget_now = _budget_sec()

            # Apply as much pass time-cost as feasible, then force QuickShot if the full pass
            # would leave insufficient release window. This preserves late-clock realism while
            # still committing the pass event for assist tracking.
            pass_cost_apply = 0.0
            force_quick_after_pass = False
            if pass_cost_nominal > 0.0:
                max_cost = max(0.0, (budget_now - min_release_window) / tm) if tm > 0 else 0.0
                pass_cost_apply = min(pass_cost_nominal, max_cost)
                if pass_cost_nominal > (max_cost + 1e-9):
                    force_quick_after_pass = True

            if pass_cost_apply > 0.0:
                apply_time_cost(game_state, pass_cost_apply, tempo_mult)
                if game_state.clock_sec < 0:
                    game_state.clock_sec = 0
                if game_state.shot_clock_sec < 0:
                    game_state.shot_clock_sec = 0
                    
                if game_state.shot_clock_sec <= 0:
                    # Drop any staged pass event to avoid leaking into the next possession.
                    ctx.pop("_pending_pass_event", None)
                    commit_shot_clock_turnover(offense)
                    return {
                        "end_reason": "SHOTCLOCK",
                        "pos_start_next": "after_tov_dead",
                        "points_scored": int(offense.pts) - before_pts,
                        "had_orb": had_orb,
                        "pos_start": pos_origin,
                        "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
                    }
                if game_state.clock_sec <= 0:
                    ctx.pop("_pending_pass_event", None)
                    game_state.clock_sec = 0
                    return {
                        "end_reason": "PERIOD_END",
                        "pos_start_next": pos_start,
                        "points_scored": int(offense.pts) - before_pts,
                        "had_orb": had_orb,
                        "pos_start": pos_origin,
                        "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
                    }

            # Commit staged pass event AFTER time cost has been applied so the recorded
            # shot-clock timestamp is accurate for assist-window logic.
            commit_pending_pass_event(ctx, game_state)

            if force_quick_after_pass:
                action = "QuickShot"
            elif outcome in ("PASS_KICKOUT", "PASS_SKIP", "PASS_EXTRA"):
                # Avoid chaining extra passes late: choose a budget-feasible catch-and-shoot.
                action = choose_action_with_budget(rng, {"SpotUp": 0.72, "ExtraPass": 0.28})
            elif outcome == "PASS_SHORTROLL":
                action = choose_action_with_budget(rng, {"Drive": 0.40, "Kickout": 0.60})
            else:
                action = choose_action_with_budget(rng, off_probs)

            if (not force_quick_after_pass) and pass_chain >= 3:
                # After a long pass chain, bias toward a shot attempt.
                action = choose_action_with_budget(rng, {"SpotUp": 1.0})

            _refresh_action_tags(action, tags)
            stall_steps = _bump_stall(stall_steps, sc0, gc0)
            continue


    # If we exit the loop here, the only expected reason is the period/game clock reaching 0.
    game_state.clock_sec = 0
    return {
        "end_reason": "PERIOD_END",
        "pos_start_next": pos_start,
        "points_scored": int(offense.pts) - before_pts,
        "had_orb": had_orb,
        "pos_start": pos_origin,
        "first_fga_shotclock_sec": ctx.get("first_fga_shotclock_sec"),
    }



# -------------------------
# Game simulation

# -------------------------
