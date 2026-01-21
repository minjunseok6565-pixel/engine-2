from __future__ import annotations

"""Game orchestration (era/validation, period loop, overtime, reporting).

NOTE: Split from sim.py on 2025-12-27.
"""

import random
import math
from typing import Any, Dict, Optional, List

from .core import ENGINE_VERSION, make_replay_token, clamp
from .models import GameState, TeamState
from .validation import (
    ValidationConfig,
    ValidationReport,
    validate_and_sanitize_team,
)
from config.game_config import build_game_config
from .era import get_mvp_rules, load_era_config

from .sim_clock import apply_dead_ball_cost
from .sim_fatigue import _apply_break_recovery, _apply_fatigue_loss
from .sim_rotation import _get_on_court, _init_targets, _perform_rotation, _set_on_court, _update_minutes
from .sim_timeout import ensure_timeout_state, maybe_timeout_deadball, update_timeout_trackers
from .team_keys import AWAY, HOME, team_key
from .sim_possession import simulate_possession

# -------------------------
# Rotation plan helpers
# -------------------------
def _get_offense_role_by_pid(team: TeamState) -> Dict[str, str]:
    """Return pid -> offensive role name map if provided by UI/config.

    Priority:
    1) TeamState.rotation_offense_role_by_pid
    2) tactics.context (ROTATION_OFFENSE_ROLE_BY_PID / OFFENSE_ROLE_BY_PID)
    """
    m = getattr(team, "rotation_offense_role_by_pid", None)
    if isinstance(m, dict) and m:
        return {str(k): str(v) for k, v in m.items()}
    ctx = getattr(getattr(team, "tactics", None), "context", None)
    if isinstance(ctx, dict):
        rm = ctx.get("ROTATION_OFFENSE_ROLE_BY_PID") or ctx.get("OFFENSE_ROLE_BY_PID")
        if isinstance(rm, dict) and rm:
            return {str(k): str(v) for k, v in rm.items()}
    return {}


def _enforce_initiator_primary_start(
    team: TeamState,
    start_pids: List[str],
    targets_sec_by_pid: Dict[str, int],
    rules: Dict[str, Any],
) -> List[str]:
    """Best-effort: ensure Initiator_Primary is on-court exactly once at tip-off.

    If the user configured at least one Initiator_Primary-eligible player, enforce:
      - start lineup contains exactly 1 Initiator_Primary
    If constraint is impossible (e.g., no eligible initiator), lineup is returned unchanged.

    This is a one-time pre-game correction; in-game rotation logic will continue to enforce the constraint.
    """
    role_by_pid = _get_offense_role_by_pid(team)
    roster_pids = [p.pid for p in team.lineup]
    eligible = [pid for pid in roster_pids if role_by_pid.get(pid) == "Initiator_Primary"]
    if not eligible:
        return list(start_pids)

    def is_init(pid: str) -> bool:
        return role_by_pid.get(pid) == "Initiator_Primary"

    start = list(start_pids)[:5]
    init_in_start = [pid for pid in start if is_init(pid)]
    n = len(init_in_start)

    # Helper for choosing replacements: prefer higher target minutes for IN, lower target minutes for OUT.
    def tgt(pid: str) -> int:
        return int(targets_sec_by_pid.get(pid, 0))

    # Case 1: 0 initiators -> bring in best eligible, push out lowest-target player.
    if n == 0:
        pid_in = max(eligible, key=tgt)
        if pid_in in start:
            return start
        # choose OUT: lowest target (ties: random order is fine)
        pid_out = min(start, key=tgt)
        start[start.index(pid_out)] = pid_in
        return start

    # Case 2: 2+ initiators -> keep highest-target one, swap out the rest for best non-initiators.
    if n > 1:
        keep = max(init_in_start, key=tgt)
        extras = [pid for pid in init_in_start if pid != keep]

        bench_non_init = [pid for pid in roster_pids if pid not in start and not is_init(pid)]
        bench_non_init.sort(key=tgt, reverse=True)

        for pid_out in extras:
            if not bench_non_init:
                break
            pid_in = bench_non_init.pop(0)
            start[start.index(pid_out)] = pid_in

        # still 2+ (no bench non-init)? best-effort: return as-is
        return start

    # Case 3: exactly 1 initiator -> ok
    return start


def _choose_ot_start_offense(
    rng: random.Random,
    rules: Dict[str, Any],
    game_state: GameState,
    home: TeamState,
    away: TeamState,
) -> TeamState:
    mode = str(rules.get("ot_start_possession_mode", "jumpball")).lower().strip()

    if mode == "random":
        return home if rng.random() < 0.5 else away

    # default: jumpball
    a_on = _get_on_court(game_state, home, home)
    b_on = _get_on_court(game_state, away, home)

    def strength(team: TeamState, pids: List[str]) -> float:
        vals: List[float] = []
        for pid in pids:
            p = team.find_player(pid)
            if p:
                # fatigue-insensitive for jumpball
                r = float(p.get("REB_DR", fatigue_sensitive=False))
                ph = float(p.get("PHYSICAL", fatigue_sensitive=False))
                vals.append(r + 0.6 * ph)
        return max(vals) if vals else 50.0

    sA = strength(home, a_on)
    sB = strength(away, b_on)

    jb = rules.get("ot_jumpball", {}) or {}
    scale = float(jb.get("scale", 12.0))
    scale = max(scale, 1e-6)

    # sigmoid on strength gap
    pA = 1.0 / (1.0 + math.exp(-(sA - sB) / scale))
    return home if rng.random() < pA else away

def init_player_boxes(team: TeamState) -> None:
    for p in team.lineup:
        team.player_stats[p.pid] = {"PTS":0,"FGM":0,"FGA":0,"3PM":0,"3PA":0,"FTM":0,"FTA":0,"TOV":0,"ORB":0,"DRB":0}

def _safe_pct(made: int, att: int) -> float:
    return round((float(made) / float(att)) * 100.0, 2) if att else 0.0

def build_player_box(
    team: TeamState,
    game_state: Optional[GameState] = None,
    home: Optional[TeamState] = None,
) -> Dict[str, Dict[str, Any]]:
    """Return per-player box score with derived percentages + minutes + fouls.

    Note: this engine currently does NOT track AST/STL/BLK; only fields that exist in
    TeamState.player_stats and GameState are included.
    """
    key = team_key(team, home) if game_state is not None and home is not None else None
    fouls = dict(getattr(game_state, "player_fouls", {}).get(key, {}) or {}) if key else {}
    mins = dict(getattr(game_state, "minutes_played_sec", {}).get(key, {}) or {}) if key else {}

    out: Dict[str, Dict[str, Any]] = {}
    for p in team.lineup:
        pid = p.pid
        s = team.player_stats.get(pid, {}) or {}
        fgm, fga = int(s.get("FGM", 0)), int(s.get("FGA", 0))
        tpm, tpa = int(s.get("3PM", 0)), int(s.get("3PA", 0))
        ftm, fta = int(s.get("FTM", 0)), int(s.get("FTA", 0))
        orb, drb = int(s.get("ORB", 0)), int(s.get("DRB", 0))
        out[pid] = {
            "Name": p.name,
            "MIN": round(float(mins.get(pid, 0)) / 60.0, 2),
            "PTS": int(s.get("PTS", 0)),
            "FGM": fgm, "FGA": fga, "FG%": _safe_pct(fgm, fga),
            "3PM": tpm, "3PA": tpa, "3P%": _safe_pct(tpm, tpa),
            "FTM": ftm, "FTA": fta, "FT%": _safe_pct(ftm, fta),
            "ORB": orb, "DRB": drb, "REB": orb + drb,
            "TOV": int(s.get("TOV", 0)),
            "AST": int(s.get("AST", 0)),
            "PF": int(fouls.get(pid, 0)),
        }
    return out

def summarize_team(
    team: TeamState,
    game_state: Optional[GameState] = None,
    home: Optional[TeamState] = None,
) -> Dict[str, Any]:
    key = team_key(team, home) if game_state is not None and home is not None else None
    fat_map = game_state.fatigue.get(key, {}) if key else {}
    return {
        "PTS": team.pts,
        "FGM": team.fgm, "FGA": team.fga,
        "3PM": team.tpm, "3PA": team.tpa,
        "FTM": team.ftm, "FTA": team.fta,
        "TOV": team.tov,
        "ORB": team.orb, "DRB": team.drb,
        "Possessions": team.possessions,
        "AST": team.ast,
        "PITP": team.pitp,
        "FastbreakPTS": team.fastbreak_pts,
        "SecondChancePTS": team.second_chance_pts,
        "PointsOffTOV": team.points_off_tov,
        "PossessionEndCounts": dict(team.possession_end_counts),
        "ShotZoneDetail": dict(team.shot_zone_detail),
        "OffActionCounts": dict(sorted(team.off_action_counts.items(), key=lambda x: -x[1])),
        "OutcomeCounts": dict(sorted(team.outcome_counts.items(), key=lambda x: -x[1])),
        "Players": team.player_stats,
        "PlayerBox": build_player_box(team, game_state, home=home),
        "AvgFatigue": (sum((fat_map.get(p.pid, 1.0) if game_state else 1.0) for p in team.lineup) / max(len(team.lineup), 1)),
        "ShotZones": dict(team.shot_zones),
    }

def simulate_game(
    rng: random.Random,
    home: TeamState,
    away: TeamState,
    era: str = "default",
    strict_validation: bool = True,
    validation: Optional[ValidationConfig] = None,
) -> Dict[str, Any]:
    """Simulate a full game with input validation/sanitization.

    0-2 (commercial safety):
    - clamps all UI multipliers to [0.70, 1.40]
    - ignores unknown tactic keys (but logs warnings)
    - validates required derived keys (error by default; can 'fill' via ValidationConfig)
    """
    report = ValidationReport()
    cfg = validation if validation is not None else ValidationConfig(strict=strict_validation)

    # 0-1: load era tuning parameters (priors/base%/scheme multipliers/prob model)
    era_cfg, era_warnings, era_errors = load_era_config(era)
    for w in era_warnings:
        report.warn(f"era[{era}]: {w}")
    for e in era_errors:
        report.error(f"era[{era}]: {e}")

    game_cfg = build_game_config(era_cfg)

    # If caller did not pass a custom ValidationConfig, adopt knob clamp bounds from era.
    if validation is None:
        k = game_cfg.knobs
        if isinstance(k.get("mult_lo"), (int, float)):
            cfg.mult_lo = float(k["mult_lo"])
        if isinstance(k.get("mult_hi"), (int, float)):
            cfg.mult_hi = float(k["mult_hi"])

    validate_and_sanitize_team(home, cfg, report, label=f"team[{home.name}]", game_cfg=game_cfg)
    validate_and_sanitize_team(away, cfg, report, label=f"team[{away.name}]", game_cfg=game_cfg)

    if cfg.strict and report.errors:
        # Raise with a compact, actionable message (full list is also in report)
        head = "\n".join(report.errors[:6])
        more = f"\n... (+{len(report.errors)-6} more)" if len(report.errors) > 6 else ""
        raise ValueError(f"MatchEngine input validation failed:\n{head}{more}")

    init_player_boxes(home)
    init_player_boxes(away)

    rules = get_mvp_rules()
    targets_home = _init_targets(home, rules)
    targets_away = _init_targets(away, rules)

    # Starting 5 defaults to lineup order, but if Initiator_Primary is configured we enforce:
    # - exactly 1 Initiator_Primary on-court at tip-off (best-effort)
    start_home = [p.pid for p in home.lineup[:5]]
    start_away = [p.pid for p in away.lineup[:5]]
    start_home = _enforce_initiator_primary_start(home, start_home, targets_home, rules)
    start_away = _enforce_initiator_primary_start(away, start_away, targets_away, rules)
    game_state = GameState(
        quarter=1,
        clock_sec=0,
        shot_clock_sec=0,
        score_home=home.pts,
        score_away=away.pts,
        possession=0,
        team_fouls={HOME: 0, AWAY: 0},
        player_fouls={HOME: {}, AWAY: {}},
        fatigue={
            HOME: {p.pid: 1.0 for p in home.lineup},
            AWAY: {p.pid: 1.0 for p in away.lineup},
        },
        minutes_played_sec={
            HOME: {p.pid: 0 for p in home.lineup},
            AWAY: {p.pid: 0 for p in away.lineup},
        },
        on_court_home=list(start_home),
        on_court_away=list(start_away),
        targets_sec_home=targets_home,
        targets_sec_away=targets_away,
    )
    # Initialize timeout state + flow trackers (safe no-ops if rules disable it)
    ensure_timeout_state(game_state, rules)
    home.set_on_court(start_home)
    away.set_on_court(start_away)

    regulation_quarters = int(rules.get("quarters", 4))
    overtime_length = float(rules.get("overtime_length", 300))
    total_possessions = 0
    overtime_periods = 0
    replay_token = ""
    debug_errors: List[Dict[str, Any]] = []

    # Dead-ball windows where substitutions are allowed.
    # NOTE: start_q is treated as a dead-ball window ONLY for Q2+ (and OT),
    # not for the opening tip (Q1 start), to avoid subbing starters before any play.
    DEADBALL_SUB_STARTS = ("start_q", "after_score", "after_tov_dead", "after_foul")

    def _maybe_open_sub_window_deadball(
        q_index: int,
        pos_start: str,
        is_garbage: bool,
        timeout_evt: Optional[Dict[str, Any]],
    ) -> None:
        """
        Substitution-eligibility window (dead-ball only).

        This function is the single choke-point where "subs are allowed" in the engine.
        For now, we call the existing _perform_rotation() so the game remains playable.
        Later, you can replace these calls with your own rotation/substitution logic.
        """
        ps = str(pos_start)
        if ps not in DEADBALL_SUB_STARTS and not timeout_evt:
            return
        # Skip Q1 opening dead-ball (before any possession has happened).
        if ps == "start_q" and q_index == 0 and total_possessions == 0:
            return
        try:
            _perform_rotation(rng, home, home, game_state, rules, is_garbage)
            _perform_rotation(rng, away, home, game_state, rules, is_garbage)
        except Exception:
            # Sub logic must never break simulation.
            pass

    def _play_period(q: int, period_length_sec: float) -> None:
        nonlocal total_possessions, replay_token
        game_state.quarter = q + 1
        game_state.clock_sec = float(period_length_sec)
        game_state.team_fouls[HOME] = 0
        game_state.team_fouls[AWAY] = 0

        # Period start possession:
        # - Regulation: alternate (A starts Q1/Q3, B starts Q2/Q4)
        # - OT: jumpball/random (configurable)
        if q < regulation_quarters:
            offense = home if (q % 2 == 0) else away
        else:
            offense = _choose_ot_start_offense(rng, rules, game_state, home, away)

        defense = away if offense is home else home
        pos_start = "start_q"

        # Possession-continuation state: some dead-ball events (e.g. no-shot foul) can stop play
        # and restart with the same offense. In those cases we re-enter the loop without counting
        # a new possession, and we must preserve possession-scope aggregates.
        pos_is_continuation = False
        pos_before_pts = 0
        pos_had_orb = False
        pos_origin_start = ""
        pos_first_fga_sc = None


        while game_state.clock_sec > 0:
            game_state.possession = total_possessions

            # For continuation segments (e.g. after a no-shot foul), preserve the current
            # shot clock value (the foul-stop logic may have applied a 14s reset already).
            if not (pos_is_continuation and pos_start == "after_foul"):
                game_state.shot_clock_sec = float(rules.get("shot_clock", 24))
                
            start_clock = game_state.clock_sec

            # Initialize possession-scope aggregates only once per possession.
            if not pos_is_continuation:
                pos_before_pts = int(offense.pts)
                pos_had_orb = False
                pos_origin_start = str(pos_start)
                pos_first_fga_sc = None

            # Game context that does NOT depend on the on-court lineup.
            score_diff = home.pts - away.pts
            is_clutch = game_state.quarter >= regulation_quarters and game_state.clock_sec <= 120 and abs(score_diff) <= 8
            is_garbage = game_state.quarter == regulation_quarters and game_state.clock_sec <= 360 and abs(score_diff) >= 20
            variance_mult = 0.80 if is_clutch else 1.25 if is_garbage else 1.0
            tempo_mult = (1.0 / 1.08) if is_garbage else 1.0

            # --- Dead-ball timeout phase (v1) ---
            # Only attempts on dead-ball windows (start_q / after_score / after_tov_dead / after_foul).
            # Does not consume game clock and does not affect shot clock (we only log the snapshot).
            # NOTE: we intentionally do NOT change offense/defense here; timeout is a side event.
            timeout_evt = None
            try:
                home_on = list(game_state.on_court_home or [])
                away_on = list(game_state.on_court_away or [])
                home_fmap = game_state.fatigue.get(HOME, {}) if isinstance(game_state.fatigue, dict) else {}
                away_fmap = game_state.fatigue.get(AWAY, {}) if isinstance(game_state.fatigue, dict) else {}
                avg_energy_home = sum(float(home_fmap.get(pid, 1.0)) for pid in home_on) / max(len(home_on), 1)
                avg_energy_away = sum(float(away_fmap.get(pid, 1.0)) for pid in away_on) / max(len(away_on), 1)
                timeout_evt = maybe_timeout_deadball(
                    rng,
                    game_state,
                    rules,
                    pos_start=str(pos_start),
                    next_offense_side=str(off_key),
                    is_clutch=bool(is_clutch),
                    avg_energy_home=float(avg_energy_home),
                    avg_energy_away=float(avg_energy_away),
                    home_team_id=home_team_id,
                    away_team_id=away_team_id,
                )
                rec = rules.get("timeout_recovery", {})
                if timeout_evt and isinstance(rec, dict) and bool(rec.get("enabled", False)):
                    break_sec = float(rec.get("equiv_break_sec", 12.0))
                    if break_sec > 0:
                        _apply_break_recovery(home, home_on, game_state, rules, break_sec, home)
                        _apply_break_recovery(away, away_on, game_state, rules, break_sec, home)
            except Exception:
                # Timeout logic must never break simulation.
                pass

            # --- Substitution window (dead-ball only, 8-A ready) ---
            # Substitutions are allowed ONLY on dead-ball windows:
            #   - after_score
            #   - after_tov_dead (dead-ball turnovers: inbound/charge/shot-clock, per prior patch)
            #   - after_foul (DEADBALL_STOP continuation)
            #   - start_q (between quarters / OT; Q1 opening tip is excluded)
            #
            # This runs AFTER timeout recovery so the rotation/sub logic can see updated energy.
            _maybe_open_sub_window_deadball(
                q_index=q,
                pos_start=str(pos_start),
                is_garbage=bool(is_garbage),
                timeout_evt=timeout_evt if isinstance(timeout_evt, dict) else None,
            )

            # Now (after potential substitutions), re-read the actual on-court lineups and compute fatigue context.
            off_on_court = _get_on_court(game_state, offense, home)
            def_on_court = _get_on_court(game_state, defense, home)

            offense.set_on_court(off_on_court)
            defense.set_on_court(def_on_court)
            off_on_court = list(offense.on_court_pids)
            def_on_court = list(defense.on_court_pids)

            off_players = offense.on_court_players()
            def_players = defense.on_court_players()
            off_key = team_key(offense, home)
            def_key = team_key(defense, home)
            off_fatigue_map = game_state.fatigue.setdefault(off_key, {})
            def_fatigue_map = game_state.fatigue.setdefault(def_key, {})

            for p in off_players:
                p.energy = clamp(off_fatigue_map.get(p.pid, 1.0), 0.0, 1.0)
            for p in def_players:
                p.energy = clamp(def_fatigue_map.get(p.pid, 1.0), 0.0, 1.0)

            avg_off_fatigue = sum(off_fatigue_map.get(pid, 1.0) for pid in off_on_court) / max(len(off_on_court), 1)
            avg_def_fatigue = sum(def_fatigue_map.get(pid, 1.0) for pid in def_on_court) / max(len(def_on_court), 1)
            def_eff_mult = float(rules.get("fatigue_effects", {}).get("def_mult_min", 0.90)) + 0.10 * avg_def_fatigue

            bonus_threshold = (
                int(rules.get("overtime_bonus_threshold", rules.get("bonus_threshold", 5)))
                if game_state.quarter > regulation_quarters
                else int(rules.get("bonus_threshold", 5))
            )

            ctx = {
                "off_team_key": off_key,
                "def_team_key": def_key,
                "score_diff": score_diff,
                "is_clutch": is_clutch,
                "is_garbage": is_garbage,
                "variance_mult": variance_mult,
                "tempo_mult": tempo_mult,
                "avg_fatigue_off": avg_off_fatigue,
                "fatigue_bad_mult_max": float(rules.get("fatigue_effects", {}).get("bad_mult_max", 1.12)),
                "fatigue_bad_critical": float(rules.get("fatigue_effects", {}).get("bad_critical", 0.25)),
                "fatigue_bad_bonus": float(rules.get("fatigue_effects", {}).get("bad_bonus", 0.08)),
                "fatigue_bad_cap": float(rules.get("fatigue_effects", {}).get("bad_cap", 1.20)),
                "fatigue_logit_max": float(rules.get("fatigue_effects", {}).get("logit_delta_max", -0.25)),
                "def_eff_mult": def_eff_mult,
                "fatigue_map": off_fatigue_map,
                "def_on_court": def_on_court,
                "off_on_court": off_on_court,
                "team_fouls": game_state.team_fouls,
                "player_fouls_by_team": game_state.player_fouls,
                "foul_out": int(rules.get("foul_out", 6)),
                "bonus_threshold": bonus_threshold,
                "pos_start": pos_start,
                "dead_ball_inbound": pos_start in ("start_q", "after_score", "after_tov_dead", "after_foul"),

                # Possession-continuation support (used by sim_possession).
                "_pos_continuation": pos_is_continuation,
                "_pos_before_pts": pos_before_pts,
                "_pos_had_orb": pos_had_orb,
                "_pos_origin_start": pos_origin_start,
                "first_fga_shotclock_sec": pos_first_fga_sc,
            }

            # Setup time: dead-ball only (game clock runs; shot clock should start at full).
            # For continuation after a no-shot foul, the foul-stop already accounted for the
            # stoppage time, so we skip additional setup here.
            if pos_is_continuation and pos_start == "after_foul":
                setup_cost = 0.0
            else:
                setup_map = {
                    "start_q": "setup_start_q",
                    "after_score": "setup_after_score",
                    "after_drb": "setup_after_drb",
                    "after_tov": "setup_after_tov",
                    "after_tov_dead": "setup_after_tov",
                    "after_foul": "setup_after_foul",
                }
                setup_key = setup_map.get(pos_start, "possession_setup")
                setup_cost = float(rules.get("time_costs", {}).get(setup_key, rules.get("time_costs", {}).get("possession_setup", 0.0)))
            # Late-clock guardrail: never allow dead-ball setup to delete the possession entirely.
            timing = rules.get("timing", {}) or {}
            try:
                min_release_window = float(timing.get("min_release_window", 0.7))
            except Exception:
                min_release_window = 0.7
            # apply_dead_ball_cost consumes (setup_cost * tempo_mult) seconds from the game clock.
            # Ensure we leave at least `min_release_window` seconds for a live attempt.
            if setup_cost > 0:
                tm = float(tempo_mult) if float(tempo_mult) > 0 else 1.0
                max_setup = max(0.0, (float(game_state.clock_sec) - min_release_window) / tm)
                setup_cost = min(setup_cost, max_setup)
            if setup_cost > 0:
                apply_dead_ball_cost(game_state, setup_cost, tempo_mult)
                if game_state.clock_sec <= 0:
                    # account minutes for the setup time
                    elapsed = max(start_clock - game_state.clock_sec, 0.0)
                    _update_minutes(game_state, off_on_court, elapsed, offense, home)
                    _update_minutes(game_state, def_on_court, elapsed, defense, home)
                    game_state.clock_sec = 0
                    break

            # Full shot clock starts after setup (unless this is a continuation segment
            # where the shot clock value must be preserved).
            if not (pos_is_continuation and pos_start == "after_foul"):
                game_state.shot_clock_sec = float(rules.get("shot_clock", 24))
            pos_res = simulate_possession(rng, offense, defense, game_state, rules, ctx, game_cfg=game_cfg)
            pos_errors = ctx.get("errors") if isinstance(ctx, dict) else None
            if isinstance(pos_errors, list) and pos_errors:
                for err in pos_errors:
                    debug_errors.append(
                        {
                            "possession": int(game_state.possession),
                            "quarter": int(game_state.quarter),
                            "offense": offense.name,
                            "defense": defense.name,
                            "error": dict(err) if isinstance(err, dict) else {"error": str(err)},
                        }
                    )
                ctx["errors"] = []

            elapsed = max(start_clock - game_state.clock_sec, 0.0)
            _update_minutes(game_state, off_on_court, elapsed, offense, home)
            _update_minutes(game_state, def_on_court, elapsed, defense, home)

            intensity_off = {
                "transition_emphasis": bool(offense.tactics.context.get("TRANSITION_EMPHASIS", False)),
                "heavy_pnr": bool(offense.tactics.context.get("HEAVY_PNR", False)) or "PnR" in offense.tactics.offense_scheme,
            }
            intensity_def = {
                "transition_emphasis": bool(defense.tactics.context.get("TRANSITION_EMPHASIS", False)),
                "heavy_pnr": bool(defense.tactics.context.get("HEAVY_PNR", False)) or "PnR" in defense.tactics.defense_scheme,
            }
            _apply_fatigue_loss(offense, off_on_court, game_state, rules, intensity_off, elapsed, home)
            _apply_fatigue_loss(defense, def_on_court, game_state, rules, intensity_def, elapsed, home)

            # Track possession-scope aggregates across dead-ball stop continuations.
            if bool(pos_res.get("had_orb", False)):
                pos_had_orb = True
            if pos_first_fga_sc is None and pos_res.get("first_fga_shotclock_sec") is not None:
                pos_first_fga_sc = pos_res.get("first_fga_shotclock_sec")

            # Dead-ball stop (e.g. no-shot foul): same offense retains the ball.
            # We do NOT count a new possession, and we do NOT swap offense/defense.
            if pos_res.get("end_reason") == "DEADBALL_STOP":
                pos_is_continuation = True
                game_state.score_home = home.pts
                game_state.score_away = away.pts
                if game_state.clock_sec <= 0:
                    game_state.clock_sec = 0
                    break
                pos_start = str(pos_res.get("pos_start_next", "after_foul"))
                continue

            pts_scored = int(pos_res.get("points_scored", 0))
            had_orb = bool(pos_res.get("had_orb", False))
            pos_start_val = str(pos_res.get("pos_start", ""))
            first_fga_sc = pos_res.get("first_fga_shotclock_sec")
            end_key = "OTHER"
            if bool(pos_res.get("ended_with_ft_trip")):
                end_key = "FT_TRIP"
            elif pos_res.get("end_reason") in ("TURNOVER", "SHOTCLOCK"):
                end_key = "TOV"
            elif pos_res.get("end_reason") in ("SCORE", "DRB"):
                end_key = "FGA"
            offense.possession_end_counts[end_key] = offense.possession_end_counts.get(end_key, 0) + 1

            if pts_scored > 0 and had_orb:
                offense.second_chance_pts += pts_scored
            if pts_scored > 0 and pos_start_val in ("after_tov", "after_tov_dead"):
                offense.points_off_tov += pts_scored
            if pts_scored > 0 and pos_start_val in ("after_tov", "after_drb") and first_fga_sc is not None:
                try:
                    if float(first_fga_sc) >= 16.0:
                        offense.fastbreak_pts += pts_scored
                except Exception as exc:
                    report.warn(
                        f"fastbreak_pts: invalid first_fga_shotclock_sec '{first_fga_sc}' "
                        f"({type(exc).__name__}: {exc})"
                    )

            # --- NOTE (8-A): substitutions are NOT allowed here anymore ---
            # Previously, we rotated after every possession end, which could cause unrealistic
            # "robotic" per-possession substitutions (including live-ball DRB/steal transitions).
            #
            # Substitution eligibility is handled ONLY in the dead-ball window at the top of the loop.

            # Possession ended.
            pos_is_continuation = False

            # Update timeout flow trackers only on true possession ends (not DEADBALL_STOP).
            # This drives run/turnover-streak triggers for future dead-ball timeouts.
            try:
                update_timeout_trackers(game_state, offense_side=str(off_key), pos_res=pos_res)
            except Exception:
                pass

            total_possessions += 1
            game_state.score_home = home.pts
            game_state.score_away = away.pts

            if game_state.clock_sec <= 0 or pos_res.get("end_reason") == "PERIOD_END":
                game_state.clock_sec = 0
                break

            # event-based possession change: after any terminal end, ball goes to defense
            offense, defense = defense, offense
            pos_start = str(pos_res.get("pos_start_next", "after_tov"))

        replay_token = make_replay_token(rng, home, away, era=era)

    def _apply_period_break(break_sec: float) -> None:
        if break_sec <= 0:
            return
        onA = _get_on_court(game_state, home, home)
        onB = _get_on_court(game_state, away, home)
        _apply_break_recovery(home, onA, game_state, rules, break_sec, home)
        _apply_break_recovery(away, onB, game_state, rules, break_sec, home)

    break_between = float(rules.get("break_sec_between_periods", 0.0))
    break_before_ot = float(rules.get("break_sec_before_ot", break_between))

    # Regulation
    for q in range(regulation_quarters):
        _play_period(q, float(rules.get("quarter_length", 720)))

        # apply break after Q1/Q2/Q3 (not after Q4)
        if q < regulation_quarters - 1:
            _apply_period_break(break_between)

    # If tie after regulation, apply break before OT1
    if home.pts == away.pts:
        _apply_period_break(break_before_ot)

    # Overtime(s)
    while home.pts == away.pts:
        overtime_periods += 1
        _play_period(regulation_quarters - 1 + overtime_periods, overtime_length)

        # if still tied, apply break before next OT
        if home.pts == away.pts:
            _apply_period_break(break_before_ot)

    return {
        "meta": {
            "engine_version": ENGINE_VERSION,
            "era": era,
            "era_version": str(game_cfg.era.get("version", "1.0")),
            "replay_token": replay_token,
            "overtime_periods": overtime_periods,
            "validation": report.to_dict(),
            "internal_debug": {
                "errors": list(debug_errors),
                "role_fit": {
                    "role_counts": {home.name: home.role_fit_role_counts, away.name: away.role_fit_role_counts},
                    "grade_counts": {home.name: home.role_fit_grade_counts, away.name: away.role_fit_grade_counts},
                    "pos_log": {home.name: home.role_fit_pos_log, away.name: away.role_fit_pos_log},
                    "bad_totals": {home.name: home.role_fit_bad_totals, away.name: away.role_fit_bad_totals},
                    "bad_by_grade": {home.name: home.role_fit_bad_by_grade, away.name: away.role_fit_bad_by_grade},
                },
                "timeouts": {
                    "remaining": _side_to_team_keyed(dict(getattr(game_state, "timeouts_remaining", {})), home_team_id, away_team_id),
                    "used": _side_to_team_keyed(dict(getattr(game_state, "timeouts_used", {})), home_team_id, away_team_id),
                    "run_pts_by_scoring_side": _side_to_team_keyed(dict(getattr(game_state, "run_pts_by_scoring_side", {})), home_team_id, away_team_id),
                    "consecutive_team_tos": _side_to_team_keyed(dict(getattr(game_state, "consecutive_team_tos", {})), home_team_id, away_team_id),
                    "log": list(getattr(game_state, "timeout_log", []) or []),
                },
            },
        },
        "possessions_per_team": max(home.possessions, away.possessions),
        "teams": {
            home.name: summarize_team(home, game_state, home=home),
            away.name: summarize_team(away, game_state, home=home),
        },
        "game_state": {
            "team_fouls": dict(game_state.team_fouls),
            "player_fouls": dict(game_state.player_fouls),
            "fatigue": dict(game_state.fatigue),
            "minutes_played_sec": dict(game_state.minutes_played_sec),
        }
    }
