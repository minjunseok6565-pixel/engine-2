from __future__ import annotations

"""Team style (sticky per-team biases) helpers.

Moved from engine.sim_possession to keep the main possession loop slimmer.
"""

import random
from typing import Any, Dict

from ..builders import get_action_base
from ..core import clamp
from ..models import TeamState


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
        if base == "PnP":
            mult *= (0.70 * three_bias + 0.30 * rim_bias)
        if base == "ISO":
            # ISO는 on-ball 창출 성격이라 rim/three 둘 다 영향을 받되, mid-range 성격을 반영해 중립값을 섞는다.
            mult *= (0.45 * three_bias + 0.35 * rim_bias + 0.20)
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

