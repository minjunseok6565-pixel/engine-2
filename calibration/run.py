from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ---- schema shim (runner-only) ----
# sim_game.py imports `schema` as an absolute module.
# In your full project you likely already have it.
# For standalone testing, we inject a minimal shim if missing.
def _ensure_schema_module() -> None:
    try:
        import schema  # type: ignore
        _ = schema.GameContext  # type: ignore[attr-defined]
        _ = schema.normalize_team_id  # type: ignore[attr-defined]
        return
    except Exception:
        pass

    import types
    m = types.ModuleType("schema")

    def normalize_team_id(x: str) -> str:
        return str(x or "").strip()

    @dataclass
    class GameContext:
        game_id: str
        home_team_id: str
        away_team_id: str

    m.normalize_team_id = normalize_team_id  # type: ignore[attr-defined]
    m.GameContext = GameContext  # type: ignore[attr-defined]
    sys.modules["schema"] = m

_ensure_schema_module()

import schema  # type: ignore

from ..sim_game import simulate_game
from ..era import load_era_config
from ..game_config import build_game_config
from .generate import PROFILES, build_team, DEFENSE_SCHEMES, OFFENSE_SCHEMES
from .aggregate import MeanAccumulator, pct, safe_div

def _team_to_calib_metrics(team_summary: Dict[str, Any]) -> Dict[str, Any]:
    # Drop per-player heavy payloads for calibration aggregates
    keep = dict(team_summary)
    keep.pop("Players", None)
    keep.pop("PlayerBox", None)
    return keep

def _merge_scheme_counts(dst: Dict[str, int], scheme: str) -> None:
    dst[scheme] = int(dst.get(scheme, 0)) + 1

def run_calibration(
    *,
    n_games: int,
    seed: int,
    style: str = "modern",
    era: str = "default",
    replay_disabled: bool = True,
    strict_validation: bool = False,
    store_per_game: bool = False,
) -> Dict[str, Any]:
    rng = random.Random(int(seed))

    # Validate schemes from era (optional, but helps avoid drift)
    era_cfg, _, _ = load_era_config(era)
    game_cfg = build_game_config(era_cfg)
    allowed_off = set(getattr(game_cfg, "off_scheme_action_weights", {}).keys()) or set(OFFENSE_SCHEMES)
    allowed_def = set(getattr(game_cfg, "defense_scheme_mult", {}).keys()) or set(DEFENSE_SCHEMES)

    profile = PROFILES.get(style) or PROFILES["modern"]

    # Accumulators
    league_acc = MeanAccumulator()
    per_game: List[Dict[str, Any]] = []
    inputs: List[Dict[str, Any]] = []

    scheme_counts_off: Dict[str, int] = {}
    scheme_counts_def: Dict[str, int] = {}

    for i in range(int(n_games)):
        home_id = f"H{i:04d}"
        away_id = f"A{i:04d}"

        home, meta_h = build_team(rng, team_id=home_id, name=f"Home{i:04d}", profile=profile)
        away, meta_a = build_team(rng, team_id=away_id, name=f"Away{i:04d}", profile=profile)

        # Safety: clamp to allowed sets (in case profile list diverges from era tables)
        if meta_h["offense_scheme"] not in allowed_off:
            home.tactics.offense_scheme = next(iter(allowed_off))
            meta_h["offense_scheme"] = home.tactics.offense_scheme
        if meta_h["defense_scheme"] not in allowed_def:
            home.tactics.defense_scheme = next(iter(allowed_def))
            meta_h["defense_scheme"] = home.tactics.defense_scheme
        if meta_a["offense_scheme"] not in allowed_off:
            away.tactics.offense_scheme = next(iter(allowed_off))
            meta_a["offense_scheme"] = away.tactics.offense_scheme
        if meta_a["defense_scheme"] not in allowed_def:
            away.tactics.defense_scheme = next(iter(allowed_def))
            meta_a["defense_scheme"] = away.tactics.defense_scheme

        # count schemes
        scheme_counts_off[meta_h["offense_scheme"]] = scheme_counts_off.get(meta_h["offense_scheme"], 0) + 1
        scheme_counts_off[meta_a["offense_scheme"]] = scheme_counts_off.get(meta_a["offense_scheme"], 0) + 1
        scheme_counts_def[meta_h["defense_scheme"]] = scheme_counts_def.get(meta_h["defense_scheme"], 0) + 1
        scheme_counts_def[meta_a["defense_scheme"]] = scheme_counts_def.get(meta_a["defense_scheme"], 0) + 1

        ctx = schema.GameContext(
            game_id=f"CALIB_{seed}_{i}",
            home_team_id=home_id,
            away_team_id=away_id,
        )

        result = simulate_game(
            rng,
            home,
            away,
            context=ctx,
            era=era,
            strict_validation=bool(strict_validation),
            replay_disabled=bool(replay_disabled),
        )

        # Extract and accumulate team summaries as "league samples" (2 samples per game)
        teams = result.get("teams", {}) or {}
        for tid, summ in teams.items():
            league_acc.add(_team_to_calib_metrics(summ))

        if store_per_game:
            per_game.append({
                "game_index": i,
                "meta": result.get("meta", {}),
                "possessions_per_team": result.get("possessions_per_team", None),
                "teams": {k: _team_to_calib_metrics(v) for k, v in teams.items()},
            })
            inputs.append({"home": meta_h, "away": meta_a})

    avg = league_acc.mean()

    # Add a few derived rates (from averaged totals)
    pts = float(avg.get("PTS", 0.0))
    poss = float(avg.get("Possessions", 0.0))
    fgm = float(avg.get("FGM", 0.0)); fga = float(avg.get("FGA", 0.0))
    tpm = float(avg.get("3PM", 0.0)); tpa = float(avg.get("3PA", 0.0))
    ftm = float(avg.get("FTM", 0.0)); fta = float(avg.get("FTA", 0.0))
    tov = float(avg.get("TOV", 0.0))
    orb = float(avg.get("ORB", 0.0)); drb = float(avg.get("DRB", 0.0))

    derived = {
        "PTS_per_100": safe_div(pts, poss) * 100.0,
        "TOV_per_100": safe_div(tov, poss) * 100.0,
        "ORB_rate_pct": safe_div(orb, max(orb + drb, 1e-6)) * 100.0,
        "FG_pct": pct(fgm, fga),
        "TP_pct": pct(tpm, tpa),
        "FT_pct": pct(ftm, fta),
    }

    out: Dict[str, Any] = {
        "meta": {
            "n_games": int(n_games),
            "seed": int(seed),
            "style": str(profile.name),
            "era": str(era),
            "replay_disabled": bool(replay_disabled),
            "strict_validation": bool(strict_validation),
        },
        "inputs_summary": {
            "offense_scheme_counts": dict(sorted(scheme_counts_off.items(), key=lambda x: (-x[1], x[0]))),
            "defense_scheme_counts": dict(sorted(scheme_counts_def.items(), key=lambda x: (-x[1], x[0]))),
        },
        "league_avg_team_game": avg,         # mean of team-game samples (2*N)
        "league_avg_derived": derived,
    }
    if store_per_game:
        out["inputs"] = inputs
        out["per_game"] = per_game

    return out

def main() -> None:
    ap = argparse.ArgumentParser(description="MatchEngine calibration runner (fast sim).")
    ap.add_argument("--n_games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--style", type=str, default="modern", choices=sorted(PROFILES.keys()))
    ap.add_argument("--era", type=str, default="default")
    ap.add_argument("--replay", action="store_true", help="Include replay emission (slower, bigger output).")
    ap.add_argument("--strict", action="store_true", help="Strict input validation (raise on issues).")
    ap.add_argument("--store_per_game", action="store_true", help="Store per-game outputs (very large).")
    ap.add_argument("--out", type=str, default="calibration_output.json")
    args = ap.parse_args()

    res = run_calibration(
        n_games=args.n_games,
        seed=args.seed,
        style=args.style,
        era=args.era,
        replay_disabled=(not args.replay),
        strict_validation=args.strict,
        store_per_game=args.store_per_game,
    )

    out_path = str(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(out_path)

if __name__ == "__main__":
    main()
