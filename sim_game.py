from __future__ import annotations

"""Game orchestration (era/validation, period loop, overtime, reporting).

NOTE: Split from sim.py on 2025-12-27.
"""

import random
import math
from typing import Any, Dict, Optional, List, Tuple

import schema

from .core import ENGINE_VERSION, make_replay_token, clamp
from .models import GameState, TeamState
from .replay import emit_event
from .validation import (
    ValidationConfig,
    ValidationReport,
    validate_and_sanitize_team,
)
from .game_config import build_game_config
from .era import get_mvp_rules, load_era_config

from .sim_clock import apply_dead_ball_cost
from .sim_fatigue import _apply_break_recovery, _apply_fatigue_loss
from .sim_rotation import (
    _get_on_court,
    _init_targets,
    _update_minutes,
    ensure_rotation_v1_state,
    maybe_substitute_deadball_v1,
)
from .sim_timeout import ensure_timeout_state, maybe_timeout_deadball, update_timeout_trackers
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
    a_on = _get_on_court(home)
    b_on = _get_on_court(away)

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
        # Initialize all tracked boxscore keys to keep downstream reporting stable.
        # (Some keys may be absent if legacy callers bypass init_player_boxes.)
        team.player_stats[p.pid] = {
            "PTS": 0,
            "FGM": 0, "FGA": 0,
            "3PM": 0, "3PA": 0,
            "FTM": 0, "FTA": 0,
            "ORB": 0, "DRB": 0,
            "AST": 0,
            "STL": 0,
            "BLK": 0,
            "TOV": 0,
        }

def _safe_pct(made: int, att: int) -> float:
    return round((float(made) / float(att)) * 100.0, 2) if att else 0.0

def build_player_box(
    team: TeamState,
    game_state: Optional[GameState] = None,
) -> Dict[str, Dict[str, Any]]:
    """Return per-player box score with derived percentages + minutes + fouls.

    Only fields tracked in TeamState.player_stats and GameState are included.
    """
    if game_state is None:
        fouls: Dict[str, int] = {}
        mins: Dict[str, float] = {}
    else:
        tid = str(getattr(team, "team_id", "") or "").strip()
        fouls = dict(getattr(game_state, "player_fouls", {}).get(tid, {}) or {})
        mins = dict(getattr(game_state, "minutes_played_sec", {}).get(tid, {}) or {})

    out: Dict[str, Dict[str, Any]] = {}
    for p in team.lineup:
        pid = p.pid
        s = team.player_stats.get(pid, {}) or {}
        fgm, fga = int(s.get("FGM", 0)), int(s.get("FGA", 0))
        tpm, tpa = int(s.get("3PM", 0)), int(s.get("3PA", 0))
        ftm, fta = int(s.get("FTM", 0)), int(s.get("FTA", 0))
        orb, drb = int(s.get("ORB", 0)), int(s.get("DRB", 0))
        stl = int(s.get("STL", 0))
        blk = int(s.get("BLK", 0))
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
            "STL": stl,
            "BLK": blk,
            "PF": int(fouls.get(pid, 0)),
        }
    return out

def summarize_team(
    team: TeamState,
    game_state: Optional[GameState] = None,
) -> Dict[str, Any]:
    tid = str(getattr(team, "team_id", "") or "").strip()
    fat_map = game_state.fatigue.get(tid, {}) if game_state is not None else {}
    # Team-level STL/BLK are derived from per-player credits.
    stl_total = sum(int((team.player_stats.get(pid, {}) or {}).get("STL", 0)) for pid in team.player_stats.keys())
    blk_total = sum(int((team.player_stats.get(pid, {}) or {}).get("BLK", 0)) for pid in team.player_stats.keys())
    return {
        "PTS": team.pts,
        "FGM": team.fgm, "FGA": team.fga,
        "3PM": team.tpm, "3PA": team.tpa,
        "FTM": team.ftm, "FTA": team.fta,
        "TOV": team.tov,
        "ORB": team.orb, "DRB": team.drb,
        "Possessions": team.possessions,
        "AST": team.ast,
        "STL": stl_total,
        "BLK": blk_total,
        "PITP": team.pitp,
        "FastbreakPTS": team.fastbreak_pts,
        "SecondChancePTS": team.second_chance_pts,
        "PointsOffTOV": team.points_off_tov,
        "PossessionEndCounts": dict(team.possession_end_counts),
        "ShotZoneDetail": dict(team.shot_zone_detail),
        "OffActionCounts": dict(sorted(team.off_action_counts.items(), key=lambda x: -x[1])),
        "OutcomeCounts": dict(sorted(team.outcome_counts.items(), key=lambda x: -x[1])),
        "Players": team.player_stats,
        "PlayerBox": build_player_box(team, game_state),
        "AvgFatigue": (sum((fat_map.get(p.pid, 1.0) if game_state else 1.0) for p in team.lineup) / max(len(team.lineup), 1)),
        "ShotZones": dict(team.shot_zones),
    }

def simulate_game(
    rng: random.Random,
    home: TeamState,
    away: TeamState,
    *,
    context: schema.GameContext,
    era: str = "default",
    strict_validation: bool = True,
    validation: Optional[ValidationConfig] = None,
    replay_disabled: bool = False,
) -> Dict[str, Any]:
    """Simulate a full game with input validation/sanitization.

    0-2 (commercial safety):
    - clamps all UI multipliers to [0.70, 1.40]
    - ignores unknown tactic keys (but logs warnings)
    - validates required derived keys (error by default; can 'fill' via ValidationConfig)
    """
    report = ValidationReport()
    cfg = validation if validation is not None else ValidationConfig(strict=strict_validation)

    # ------------------------------------------------------------
    # SSOT contract (Home/Away fixed by external GameContext)
    # ------------------------------------------------------------
    game_id = str(getattr(context, "game_id", "") or "").strip()
    hid = schema.normalize_team_id(str(getattr(context, "home_team_id", "") or ""))
    aid = schema.normalize_team_id(str(getattr(context, "away_team_id", "") or ""))
    if hid == aid:
        raise ValueError(
            f"simulate_game(): invalid SSOT (home_team_id == away_team_id == {hid!r}, game_id={game_id!r})"
        )
    home_tid = schema.normalize_team_id(str(getattr(home, "team_id", "") or ""))
    away_tid = schema.normalize_team_id(str(getattr(away, "team_id", "") or ""))
    if home_tid != hid:
        raise ValueError(
            f"simulate_game(): SSOT mismatch (game_id={game_id!r}, context.home_team_id={hid!r}, home.team_id={home_tid!r})"
        )
    if away_tid != aid:
        raise ValueError(
            f"simulate_game(): SSOT mismatch (game_id={game_id!r}, context.away_team_id={aid!r}, away.team_id={away_tid!r})"
        )

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

    # SSOT team identifiers for this game (fixed by external context).
    home_team_id = hid
    away_team_id = aid

    # Lineup SSOT is TeamState (GameState does not store on-court lists).
    home.set_on_court(list(start_home))
    away.set_on_court(list(start_away))

    # Timeout defaults (ensure_timeout_state is still called for safety, but we seed non-empty
    # team_id-keyed dicts so we never need to infer/repair keys later).
    to_rules = rules.get("timeouts", {}) if isinstance(rules, dict) else {}
    per_team_timeouts = int(to_rules.get("per_team", 7))

    game_state = GameState(
        quarter=1,
        clock_sec=0.0,
        shot_clock_sec=0.0,
        possession=0,
        # Team ids are fixed for the whole game (SSOT).
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        # Team-scoped state is stored by team_id only.
        team_fouls={home_team_id: 0, away_team_id: 0},
        player_fouls={home_team_id: {}, away_team_id: {}},
        fatigue={
            home_team_id: {p.pid: 1.0 for p in home.lineup},
            away_team_id: {p.pid: 1.0 for p in away.lineup},
        },
        minutes_played_sec={
            home_team_id: {p.pid: 0.0 for p in home.lineup},
            away_team_id: {p.pid: 0.0 for p in away.lineup},
        },
        # Timeout / flow trackers (state, not logs)
        timeouts_remaining={home_team_id: per_team_timeouts, away_team_id: per_team_timeouts},
        timeouts_used={home_team_id: 0, away_team_id: 0},
        timeout_last_possession={home_team_id: -999999, away_team_id: -999999},
        run_pts={home_team_id: 0, away_team_id: 0},
        consecutive_team_tos={home_team_id: 0, away_team_id: 0},
        last_scoring_team_id=None,
        # Rotation v1.0 state (initialized below).
        rotation_last_sub_game_sec={},
        rotation_last_in_game_sec={},
        rotation_checkpoint_mask={},
        rotation_checkpoint_quarter={},
    )
    # Fast-sim / calibration runner can disable replay emission.
    # NOTE: This must be set immediately after GameState creation so all emit_event()
    # calls (including GAME_START) can observe it.
    game_state.replay_disabled = bool(replay_disabled)
    # Initialize timeout state + flow trackers (safe no-ops if rules disable it)
    ensure_timeout_state(game_state, rules)
    ensure_rotation_v1_state(game_state, home, away, rules)

    # Seed lineup versions for deterministic replay seeking (Q1 tip-off snapshot uses these).
    game_state.lineup_version = 1
    game_state.lineup_version_by_team_id = {home_team_id: 1, away_team_id: 1}

    regulation_quarters = int(rules.get("quarters", 4))
    quarter_length_sec = float(rules.get("quarter_length", 720))
    overtime_length = float(rules.get("overtime_length", 300))
    total_possessions = 0
    overtime_periods = 0
    replay_token = ""
    debug_errors: List[Dict[str, Any]] = []

    def _push_debug_error(where: str, exc: Exception, extra: Optional[Dict[str, Any]] = None) -> None:
        """Record non-fatal internal errors instead of silently swallowing them.

        We intentionally keep simulation running, but we must never lose visibility
        into missing replay events / timeout-sub windows due to exceptions.
        """
        err: Dict[str, Any] = {
            "where": str(where),
            "quarter": int(getattr(game_state, "quarter", 0) or 0),
            "clock_sec": float(getattr(game_state, "clock_sec", 0.0) or 0.0),
            "shot_clock_sec": float(getattr(game_state, "shot_clock_sec", 0.0) or 0.0),
            "possession": int(getattr(game_state, "possession", 0) or 0),
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(extra, dict):
            for k, v in extra.items():
                if k not in err:
                    err[k] = v
        debug_errors.append(err)
   
    # Dead-ball windows where substitutions are allowed.
    # NOTE: start_q is treated as a dead-ball window ONLY for Q2+ (and OT),
    # not for the opening tip (Q1 start), to avoid subbing starters before any play.
    DEADBALL_SUB_STARTS = ("start_q", "after_score", "after_tov_dead", "after_foul", "after_block_oob")

    # Replay event: game start (Q1 tip-off state / starting lineups).
    # This guarantees the consumer can reconstruct on-court players at 12:00 even if they jump into the middle.
    try:
        game_state.quarter = 1
        game_state.clock_sec = float(quarter_length_sec)
        game_state.shot_clock_sec = float(rules.get("shot_clock", 24))
        game_state.possession = 0
        emit_event(
            game_state,
            event_type="GAME_START",
            home=home,
            away=away,
            rules=rules,
            include_lineups=True,
            pos_start="start_game",
        )
    except Exception as e:
        _push_debug_error(
            "replay.emit_event.GAME_START",
            e,
            {"event_type": "GAME_START", "q_index": 0, "pos_start": "start_game"},
        )

    # Team IDs are fixed for this game (already computed above).

    def _maybe_open_sub_window_deadball(
        q_index: int,
        pos_start: str,
        pressure_index: float,
        garbage_index: float,
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
            # Run substitution logic once per team (team-specific decision making).
            # NOTE: home/away objects are passed as *real* TeamState instances (no stubs, no inference).

            maybe_substitute_deadball_v1(
                rng,
                home,   # team
                home,   # home
                away,   # away
                game_state,
                rules,
                q_index=int(q_index),
                pos_start=str(pos_start),
                pressure_index=float(pressure_index),
                garbage_index=float(garbage_index),
            )
            maybe_substitute_deadball_v1(
                rng,
                away,   # team
                home,   # home
                away,   # away
                game_state,
                rules,
                q_index=int(q_index),
                pos_start=str(pos_start),
                pressure_index=float(pressure_index),
                garbage_index=float(garbage_index),
            )
        except ValueError:
            # SSOT / team_id contract violations must crash immediately.
            raise
        except Exception as e:
            # Sub logic must never break simulation.
            _push_debug_error(
                "rotation.sub_window_deadball",
                e,
                {
                    "q_index": int(q_index),
                    "pos_start": str(pos_start),
                    "timeout_evt": bool(timeout_evt),
                },
            )

    def _play_period(q: int, period_length_sec: float) -> None:
        nonlocal total_possessions, replay_token
        game_state.quarter = q + 1
        game_state.clock_sec = float(period_length_sec)
        game_state.team_fouls[home_team_id] = 0
        game_state.team_fouls[away_team_id] = 0

        # Period start possession:
        # - Regulation: alternate (A starts Q1/Q3, B starts Q2/Q4)
        # - OT: jumpball/random (configurable)
        if q < regulation_quarters:
            offense = home if (q % 2 == 0) else away
        else:
            offense = _choose_ot_start_offense(rng, rules, game_state, home, away)

        defense = away if str(offense.team_id) == str(home_team_id) else home
        pos_start = "start_q"

        start_q_deadball_handled = False

        def _compute_phase_indices() -> Tuple[int, float, float]:
            """
            Compute + update smoothed game-phase indices based on current clock/score:
              - pressure_index (clutch pressure; EMA-smoothed)
              - garbage_index (garbage time; EMA-smoothed)
            Also updates:
              - dominant_mode, clutch_level, garbage_level
            """
            score_diff = home.pts - away.pts
            poss_diff = abs(float(score_diff)) / 3.0  # 3 points ~= 1 possession

            pressure_raw = 0.0
            if int(game_state.quarter) >= int(regulation_quarters):
                time_pressure = math.sqrt(
                    clamp((300.0 - float(game_state.clock_sec)) / 300.0, 0.0, 1.0)
                )
                score_pressure = 1.0 / (1.0 + (poss_diff / 4.0) ** 2)
                pressure_raw = float(time_pressure) * float(score_pressure)
                if int(game_state.quarter) > int(regulation_quarters):
                    pressure_raw = min(1.0, float(pressure_raw) + 0.15)

            garbage_raw = 0.0
            if int(game_state.quarter) == int(regulation_quarters):
                time_garb = math.sqrt(
                    clamp((360.0 - float(game_state.clock_sec)) / 360.0, 0.0, 1.0)
                )
                x = float(poss_diff) / 4.5
                score_garb = (x * x) / (1.0 + x * x)
                garbage_raw = float(time_garb) * float(score_garb)

            prev_p = float(getattr(game_state, "pressure_smoothed", 0.0))
            prev_g = float(getattr(game_state, "garbage_smoothed", 0.0))
            game_state.pressure_smoothed = 0.75 * prev_p + 0.25 * float(pressure_raw)
            game_state.garbage_smoothed = 0.75 * prev_g + 0.25 * float(garbage_raw)

            pressure_index = float(game_state.pressure_smoothed)
            garbage_index = float(game_state.garbage_smoothed)

            # Dominant mode hysteresis (prevents CLUTCH/GARBAGE flicker).
            prev_mode = str(getattr(game_state, "dominant_mode", "NEUTRAL") or "NEUTRAL")
            mode = prev_mode
            if float(pressure_index) >= float(garbage_index) + 0.08:
                mode = "CLUTCH"
            elif float(garbage_index) >= float(pressure_index) + 0.08:
                mode = "GARBAGE"
            game_state.dominant_mode = mode

            # Level hysteresis (STRONG/MID/OFF) for each index.
            prev_cl = str(getattr(game_state, "clutch_level", "OFF") or "OFF")
            clutch_level = "OFF"
            if float(pressure_index) >= 0.70 or (prev_cl == "STRONG" and float(pressure_index) >= 0.55):
                clutch_level = "STRONG"
            elif float(pressure_index) >= 0.40 or (
                prev_cl in ("MID", "STRONG") and float(pressure_index) >= 0.30
            ):
                clutch_level = "MID"
            game_state.clutch_level = clutch_level

            prev_gl = str(getattr(game_state, "garbage_level", "OFF") or "OFF")
            garbage_level = "OFF"
            if float(garbage_index) >= 0.70 or (prev_gl == "STRONG" and float(garbage_index) >= 0.55):
                garbage_level = "STRONG"
            elif float(garbage_index) >= 0.45 or (
                prev_gl in ("MID", "STRONG") and float(garbage_index) >= 0.35
            ):
                garbage_level = "MID"
            game_state.garbage_level = garbage_level

            return int(score_diff), float(pressure_index), float(garbage_index)

        # ------------------------------------------------------------
        # START-OF-PERIOD DEAD-BALL (RUN ONCE, BEFORE PERIOD_START LOG)
        # ------------------------------------------------------------
        # We run the start_q dead-ball window BEFORE emitting PERIOD_START so that
        # PERIOD_START (with include_lineups=True) reflects the FINAL on-court five
        # after between-quarters substitutions / timeout recovery.
        #
        # IMPORTANT: Q1 opening tip is excluded from dead-ball (timeouts + subs),
        # consistent with the engine note for starters (no pre-tip adjustments).
        game_state.possession = total_possessions
        game_state.shot_clock_sec = float(rules.get("shot_clock", 24))

        score_diff, pressure_index, garbage_index = _compute_phase_indices()
        variance_mult = 1.0
        tempo_mult = 1.0

        timeout_evt = None
        opening_tip = (q == 0 and total_possessions == 0)
        if not opening_tip:
            # --- Dead-ball timeout phase (v1) ---
            try:
                next_offense_team_id = str(offense.team_id)
                home_on = list(getattr(home, "on_court_pids", []) or [])
                away_on = list(getattr(away, "on_court_pids", []) or [])
                home_fmap = dict(game_state.fatigue.get(home_team_id, {}) or {})
                away_fmap = dict(game_state.fatigue.get(away_team_id, {}) or {})
                avg_energy_home = sum(float(home_fmap.get(pid, 1.0)) for pid in home_on) / max(len(home_on), 1)
                avg_energy_away = sum(float(away_fmap.get(pid, 1.0)) for pid in away_on) / max(len(away_on), 1)
                timeout_evt = maybe_timeout_deadball(
                    rng,
                    game_state,
                    rules,
                    pos_start=str(pos_start),
                    next_offense_team_id=next_offense_team_id,
                    pressure_index=float(pressure_index),
                    avg_energy_home=float(avg_energy_home),
                    avg_energy_away=float(avg_energy_away),
                    home=home,
                    away=away,
                )
                rec = rules.get("timeout_recovery", {})
                if timeout_evt and isinstance(rec, dict) and bool(rec.get("enabled", False)):
                    break_sec = float(rec.get("equiv_break_sec", 12.0))
                    if break_sec > 0:
                        _apply_break_recovery(home, home_on, game_state, rules, break_sec, home)
                        _apply_break_recovery(away, away_on, game_state, rules, break_sec, home)
            except ValueError:
                raise
            except Exception as e:
                _push_debug_error(
                    "timeout.deadball_phase.start_q_pre_period_start",
                    e,
                    {
                        "pos_start": str(pos_start),
                        "next_offense_team_id": str(getattr(offense, "team_id", "")),
                        "q_index": int(q),
                    },
                )

            # --- Substitution window (dead-ball only) ---
            _maybe_open_sub_window_deadball(
                q_index=q,
                pos_start=str(pos_start),
                pressure_index=float(pressure_index),
                garbage_index=float(garbage_index),
                timeout_evt=timeout_evt if isinstance(timeout_evt, dict) else None,
            )

        start_q_deadball_handled = True

        # Replay event: period start (neutral) - snapshot FINAL on-court after start_q dead-ball.
        try:
            emit_event(
                game_state,
                event_type="PERIOD_START",
                home=home,
                away=away,
                rules=rules,
                include_lineups=True,
                pos_start="start_q",
            )
        except Exception as e:
            _push_debug_error(
                "replay.emit_event.PERIOD_START",
                e,
                {"event_type": "PERIOD_START", "q_index": int(q), "pos_start": "start_q"},
            )

        # Possession-continuation state: some dead-ball events (e.g. no-shot foul) can stop play
        # and restart with the same offense. In those cases we re-enter the loop without counting
        # a new possession, and we must preserve possession-scope aggregates.
        pos_is_continuation = False
        pos_before_pts = 0
        pos_had_orb = False
        pos_origin_start = ""
        pos_first_fga_sc = None

        # Possession-scope context:
        # Preserve a single ctx dict across DEADBALL_STOP continuation segments so that
        # per-possession guards in sim_possession (e.g., _matchup_set_emitted to ensure
        # MATCHUP_SET is emitted only once per possession) remain effective.
        pos_ctx: Optional[Dict[str, Any]] = None


        while game_state.clock_sec > 0:
            game_state.possession = total_possessions

            # For continuation segments (e.g. after a no-shot foul), preserve the current
            # shot clock value (the foul-stop logic may have applied a 14s reset already).
            if not pos_is_continuation:
                game_state.shot_clock_sec = float(rules.get("shot_clock", 24))
                
            start_clock = game_state.clock_sec

            # Initialize possession-scope aggregates only once per possession.
            if not pos_is_continuation:
                pos_before_pts = int(offense.pts)
                pos_had_orb = False
                pos_origin_start = str(pos_start)
                pos_first_fga_sc = None

            # Game context that does NOT depend on the on-court lineup.
            score_diff, pressure_index, garbage_index = _compute_phase_indices()

            # Legacy knobs: keep neutral for now (new behavior should be driven by indices).
            variance_mult = 1.0
            tempo_mult = 1.0

            # --- Dead-ball timeout phase (v1) ---
            # Only attempts on dead-ball windows (start_q / after_score / after_tov_dead / after_foul).
            # Does not consume game clock and does not affect shot clock (we only log the snapshot).
            # NOTE: we intentionally do NOT change offense/defense here; timeout is a side event.
            timeout_evt = None
            skip_deadball_phase = bool(
                start_q_deadball_handled
                and (not pos_is_continuation)
                and (str(pos_start) == "start_q")
            )
            if skip_deadball_phase:
                timeout_evt = None
            else:
                try:
                    next_offense_team_id = str(offense.team_id)
                    home_on = list(getattr(home, "on_court_pids", []) or [])
                    away_on = list(getattr(away, "on_court_pids", []) or [])
                    home_fmap = dict(game_state.fatigue.get(home_team_id, {}) or {})
                    away_fmap = dict(game_state.fatigue.get(away_team_id, {}) or {})
                    avg_energy_home = sum(float(home_fmap.get(pid, 1.0)) for pid in home_on) / max(len(home_on), 1)
                    avg_energy_away = sum(float(away_fmap.get(pid, 1.0)) for pid in away_on) / max(len(away_on), 1)
                    timeout_evt = maybe_timeout_deadball(
                        rng,
                        game_state,
                        rules,
                        pos_start=str(pos_start),
                        next_offense_team_id=next_offense_team_id,
                        pressure_index=float(pressure_index),
                        avg_energy_home=float(avg_energy_home),
                        avg_energy_away=float(avg_energy_away),
                        home=home,
                        away=away,
                    )
                    rec = rules.get("timeout_recovery", {})
                    if timeout_evt and isinstance(rec, dict) and bool(rec.get("enabled", False)):
                        break_sec = float(rec.get("equiv_break_sec", 12.0))
                        if break_sec > 0:
                            _apply_break_recovery(home, home_on, game_state, rules, break_sec, home)
                            _apply_break_recovery(away, away_on, game_state, rules, break_sec, home)
                except ValueError:
                    raise
                except Exception as e:
                    # Timeout logic must never break simulation.
                    _push_debug_error(
                        "timeout.deadball_phase",
                        e,
                        {
                            "pos_start": str(pos_start),
                            "next_offense_team_id": str(getattr(offense, "team_id", "")),
                        },
                    )

            # --- Substitution window (dead-ball only, 8-A ready) ---
            # Substitutions are allowed ONLY on dead-ball windows:
            #   - after_score
            #   - after_tov_dead (dead-ball turnovers: inbound/charge/shot-clock, per prior patch)
            #   - after_foul (DEADBALL_STOP continuation)
            #   - start_q (between quarters / OT; Q1 opening tip is excluded)
            #
            # This runs AFTER timeout recovery so the rotation/sub logic can see updated energy.
            if not skip_deadball_phase:
                _maybe_open_sub_window_deadball(
                    q_index=q,
                    pos_start=str(pos_start),
                    pressure_index=float(pressure_index),
                    garbage_index=float(garbage_index),
                    timeout_evt=timeout_evt if isinstance(timeout_evt, dict) else None,
                )

            # Now (after potential substitutions), use TeamState lineup SSOT.
            off_on_court = list(getattr(offense, "on_court_pids", []) or [])
            def_on_court = list(getattr(defense, "on_court_pids", []) or [])
            if not off_on_court or not def_on_court:
                raise ValueError(
                    f"simulate_game(): on_court_pids missing after dead-ball window "
                    f"(game_id={game_id!r}, quarter={int(game_state.quarter)}, "
                    f"offense_team_id={getattr(offense,'team_id',None)!r}, defense_team_id={getattr(defense,'team_id',None)!r})"
                )

            off_players = offense.on_court_players()
            def_players = defense.on_court_players()
            off_team_id = str(getattr(offense, "team_id", "") or "").strip()
            def_team_id = str(getattr(defense, "team_id", "") or "").strip()
            if {off_team_id, def_team_id} != {home_team_id, away_team_id}:
                raise ValueError(
                    f"simulate_game(): invalid possession team ids "
                    f"(game_id={game_id!r}, home={home_team_id!r}, away={away_team_id!r}, "
                    f"off={off_team_id!r}, def={def_team_id!r})"
                )
            if off_team_id not in game_state.fatigue or def_team_id not in game_state.fatigue:
                raise ValueError(
                    f"simulate_game(): fatigue dict missing required team_id keys "
                    f"(game_id={game_id!r}, missing={[k for k in (off_team_id, def_team_id) if k not in game_state.fatigue]!r})"
                )
            off_fatigue_map = game_state.fatigue[off_team_id]
            def_fatigue_map = game_state.fatigue[def_team_id]

            # Sync energy for the ENTIRE roster from fatigue SSOT.
            # This prevents stale bench energy values from leaking into any stat reads.
            for p in (getattr(offense, "lineup", None) or []):
                p.energy = clamp(off_fatigue_map.get(p.pid, 1.0), 0.0, 1.0)
            for p in (getattr(defense, "lineup", None) or []):
                p.energy = clamp(def_fatigue_map.get(p.pid, 1.0), 0.0, 1.0)

            avg_off_fatigue = sum(off_fatigue_map.get(pid, 1.0) for pid in off_on_court) / max(len(off_on_court), 1)

            bonus_threshold = (
                int(rules.get("overtime_bonus_threshold", rules.get("bonus_threshold", 5)))
                if game_state.quarter > regulation_quarters
                else int(rules.get("bonus_threshold", 5))
            )

            # Possession-scope ctx:
            # - Create once at possession start.
            # - Reuse across DEADBALL_STOP continuation segments so per-possession flags
            #   (e.g., _matchup_set_emitted) remain effective.
            # - Overwrite segment-varying values each loop iteration.
            if (not pos_is_continuation) or (pos_ctx is None):
                ctx = {
                    "game_id": game_id,
                    "off_team_id": off_team_id,
                    "def_team_id": def_team_id,
                    "score_diff": score_diff,
                    "pressure_index": float(pressure_index),
                    "garbage_index": float(garbage_index),
                    "variance_mult": variance_mult,
                    "tempo_mult": tempo_mult,
                    "avg_fatigue_off": avg_off_fatigue,
                    "fatigue_bad_mult_max": float(rules.get("fatigue_effects", {}).get("bad_mult_max", 1.12)),
                    "fatigue_bad_critical": float(rules.get("fatigue_effects", {}).get("bad_critical", 0.25)),
                    "fatigue_bad_bonus": float(rules.get("fatigue_effects", {}).get("bad_bonus", 0.08)),
                    "fatigue_bad_cap": float(rules.get("fatigue_effects", {}).get("bad_cap", 1.20)),
                    "fatigue_logit_max": float(rules.get("fatigue_effects", {}).get("logit_delta_max", -0.25)),
                    "fatigue_logit_red_crit": float(rules.get("fatigue_effects", {}).get("logit_red_crit", 0.0)),
                    "fatigue_logit_red_max": float(rules.get("fatigue_effects", {}).get("logit_red_max", 0.0)),
                    "fatigue_logit_red_pow": float(rules.get("fatigue_effects", {}).get("logit_red_pow", 1.0)),
                    "fatigue_map": off_fatigue_map,
                    "def_on_court": def_on_court,
                    "off_on_court": off_on_court,
                    "foul_out": int(rules.get("foul_out", 6)),
                    "bonus_threshold": bonus_threshold,
                    "pos_start": pos_start,
                    "dead_ball_inbound": pos_start in ("start_q", "after_score", "after_tov_dead", "after_foul", "after_block_oob"),

                    # Possession-continuation support (used by sim_possession).
                    "_pos_continuation": pos_is_continuation,
                    "_pos_before_pts": pos_before_pts,
                    "_pos_had_orb": pos_had_orb,
                    "_pos_origin_start": pos_origin_start,
                    "first_fga_shotclock_sec": pos_first_fga_sc,

                    # --- Matchups (Plan-1 MVP) ---
                    # Version increments when sim_possession rebuilds matchups for a new segment/lineup.
                    "matchups_version": 0,
                    # Defender-vs-team blending weights for defense keys (0..1 = primary defender weight).
                    "matchup_def_blend": {
                        "DEF_POA": 0.85,
                        "DEF_STEAL": 0.75,
                        "DEF_POST": 0.80,
                        "DEF_RIM": 0.25,
                        "DEF_HELP": 0.30,
                        "PHYSICAL": 0.50,
                        "ENDURANCE": 0.50,
                    },
                    # If true, sim_possession may emit MATCHUP_SET / MATCHUP_EVENT debug replay entries.
                    "debug_matchups": bool(rules.get("debug_matchups", False)),
                }
                pos_ctx = ctx
            else:
                ctx = pos_ctx
                ctx.update(
                    {
                        "game_id": game_id,
                        "off_team_id": off_team_id,
                        "def_team_id": def_team_id,
                        "score_diff": score_diff,
                        "pressure_index": float(pressure_index),
                        "garbage_index": float(garbage_index),
                        "variance_mult": variance_mult,
                        "tempo_mult": tempo_mult,
                        "avg_fatigue_off": avg_off_fatigue,
                        "fatigue_map": off_fatigue_map,
                        "def_on_court": def_on_court,
                        "off_on_court": off_on_court,
                        "foul_out": int(rules.get("foul_out", 6)),
                        "bonus_threshold": bonus_threshold,
                        "pos_start": pos_start,
                        "dead_ball_inbound": pos_start in ("start_q", "after_score", "after_tov_dead", "after_foul", "after_block_oob"),

                        # Continuation aggregate snapshots.
                        "_pos_continuation": pos_is_continuation,
                        "_pos_before_pts": pos_before_pts,
                        "_pos_had_orb": pos_had_orb,
                        "_pos_origin_start": pos_origin_start,
                        "first_fga_shotclock_sec": pos_first_fga_sc,
                    }
                )
                pos_ctx = ctx

            # Optional override from defense tactics context (JSON-friendly dict of {DEF_KEY: weight}).
            try:
                dctx = getattr(getattr(defense, "tactics", None), "context", None)
                blend_override = dctx.get("MATCHUP_DEF_BLEND") if isinstance(dctx, dict) else None
                if isinstance(blend_override, dict):
                    for k, v in blend_override.items():
                        kk = str(k)
                        try:
                            ctx["matchup_def_blend"][kk] = clamp(float(v), 0.0, 1.0)
                        except Exception:
                            continue
            except Exception:
                pass

            # --- Possession time segmentation (supports in-possession forced subs) ---
            # NOTE: sim_possession shallow-copies ctx (ctx = dict(ctx)), so these must be MUTABLE
            # and updated in-place inside sim_possession to be visible here.
            ctx["_seg_last_clock_sec"] = {"v": float(start_clock)}
            ctx["_seg_off_on_court"] = list(off_on_court)
            ctx["_seg_def_on_court"] = list(def_on_court)
            ctx["_time_segments"] = []
            ctx["_fatigue_seg_usage"] = {"transition_sec": 0.0, "pnr_sec": 0.0}
            
            # Setup time: admin / bring-up segments.
            # - Game clock always runs.
            # - Shot clock runs only for "live" starts (DRB/steal/live-TOV recovery/etc).
            # For continuation segments (dead-ball stop restarts), the stop logic already accounted
            # for the stoppage time and any shot-clock reset, so we skip additional setup here.
            if pos_is_continuation:
                setup_cost = 0.0
            else:
                setup_map = {
                    "start_q": "setup_start_q",
                    "after_score": "setup_after_score",
                    "after_drb": "setup_after_drb",
                    "after_tov": "setup_after_tov",
                    "after_steal": "setup_after_steal",
                    "after_block": "setup_after_block",
                    "after_tov_dead": "setup_after_tov",
                    "after_foul": "setup_after_foul",
                    # Safety: if a caller ever uses after_block_oob as a non-continuation start,
                    # treat it like a dead-ball inbound start (similar to after_foul).
                    "after_block_oob": "setup_after_foul",
                }
                setup_key = setup_map.get(pos_start, "possession_setup")
                setup_cost = float(rules.get("time_costs", {}).get(setup_key, rules.get("time_costs", {}).get("possession_setup", 0.0)))

            # Live starts: treat setup time as live-clock time (shot clock elapses too).
            # Dead-ball starts: shot clock does not elapse during setup.
            live_setup_starts = {"after_drb", "after_tov", "after_steal", "after_block"}
            setup_runs_shot_clock = str(pos_start) in live_setup_starts 
            # Late-clock guardrail: never allow dead-ball setup to delete the possession entirely.
            timing = rules.get("timing", {}) or {}
            try:
                min_release_window = float(timing.get("min_release_window", 0.7))
            except Exception:
                min_release_window = 0.7
            # apply_dead_ball_cost consumes (setup_cost * tempo_mult) seconds from the game clock,
            # and optionally from the shot clock (for live starts).
            # Ensure we leave at least `min_release_window` seconds for a live attempt.
            if setup_cost > 0:
                tm = float(tempo_mult) if float(tempo_mult) > 0 else 1.0
                max_setup_gc = max(0.0, (float(game_state.clock_sec) - min_release_window) / tm)
                max_setup = max_setup_gc
                if setup_runs_shot_clock:
                    max_setup_sc = max(0.0, (float(game_state.shot_clock_sec) - min_release_window) / tm)
                    max_setup = min(max_setup, max_setup_sc)
                setup_cost = min(setup_cost, max_setup)
            if setup_cost > 0:
                apply_dead_ball_cost(game_state, setup_cost, tempo_mult, run_shot_clock=setup_runs_shot_clock)
                if game_state.clock_sec <= 0:
                    # account minutes for the setup time
                    elapsed = max(start_clock - game_state.clock_sec, 0.0)
                    _update_minutes(game_state, off_on_court, elapsed, offense, home)
                    _update_minutes(game_state, def_on_court, elapsed, defense, home)
                    game_state.clock_sec = 0
                    break

            # Full shot clock starts after setup (unless this is a continuation segment
            # where the shot clock value must be preserved).
            # Shot clock policy:
            # - Fresh segment: set full shot clock once at segment entry (see earlier `if not pos_is_continuation:`).
            # - Continuation segment (dead-ball stop where offense retains): preserve the value that
            #   sim_possession applied (ORB reset, special cases, etc).
            # Do NOT reset shot clock here based on pos_start; it causes future continuation types
            # (e.g., after_block_oob) to be accidentally overwritten.
            # Replay event: possession start (only once per possession; not for continuation segments)
            if not pos_is_continuation:
                try:
                    emit_event(
                        game_state,
                        event_type="POSSESSION_START",
                        home=home,
                        away=away,
                        rules=rules,
                        team_id=off_team_id,
                        opp_team_id=def_team_id,
                        pos_start=str(pos_start),
                    )
                except Exception as e:
                    _push_debug_error(
                        "replay.emit_event.POSSESSION_START",
                        e,
                        {"event_type": "POSSESSION_START", "team_id": off_team_id, "pos_start": str(pos_start)},
                    )

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


            # Fatigue intensity is derived from actual in-possession usage (tracked in sim_possession)
            # and applied proportionally per time-segment.
            def _intensity_from_usage(usage, seg_elapsed: float):
                try:
                    e = float(seg_elapsed)
                except Exception:
                    e = 0.0
                if e <= 0:
                    return {"transition_emphasis": 0.0, "heavy_pnr": 0.0}
                try:
                    u = usage if isinstance(usage, dict) else {}
                    t = float(u.get("transition_sec", 0.0) or 0.0)
                    p = float(u.get("pnr_sec", 0.0) or 0.0)
                except Exception:
                    t, p = 0.0, 0.0
                return {
                    "transition_emphasis": clamp(t / e, 0.0, 1.0),
                    "heavy_pnr": clamp(p / e, 0.0, 1.0),
                }

            # --- Apply minutes/fatigue by time segments (handles in-possession lineup changes) ---
            segments = ctx.get("_time_segments") if isinstance(ctx, dict) else None
            if not isinstance(segments, list):
                segments = None

            # Close final segment
            try:
                last_ref = ctx.get("_seg_last_clock_sec") if isinstance(ctx, dict) else None
                if not isinstance(last_ref, dict):
                    last_ref = {}
                last_clock = float(last_ref.get("v", float(start_clock)))
            except Exception:
                last_clock = float(start_clock)

            final_elapsed = max(last_clock - float(game_state.clock_sec), 0.0)
            if segments is not None and final_elapsed > 0:
                segments.append(
                    {
                        "elapsed": float(final_elapsed),
                        "off": list(ctx.get("_seg_off_on_court") or off_on_court),
                        "def": list(ctx.get("_seg_def_on_court") or def_on_court),
                        "fatigue_usage": dict((ctx.get("_fatigue_seg_usage") or {}) if isinstance(ctx, dict) else {}),
                    }
                )

            if segments:
                for seg in segments:
                    try:
                        seg_elapsed = float(seg.get("elapsed", 0.0))
                    except Exception:
                        seg_elapsed = 0.0
                    if seg_elapsed <= 0:
                        continue

                    off_seg = list(seg.get("off") or off_on_court)
                    def_seg = list(seg.get("def") or def_on_court)

                    _update_minutes(game_state, off_seg, seg_elapsed, offense, home)
                    _update_minutes(game_state, def_seg, seg_elapsed, defense, home)

                    seg_intensity = _intensity_from_usage(seg.get("fatigue_usage"), seg_elapsed)

                    _apply_fatigue_loss(offense, off_seg, game_state, rules, seg_intensity, seg_elapsed, home)
                    _apply_fatigue_loss(defense, def_seg, game_state, rules, seg_intensity, seg_elapsed, home)
            else:
                # Safety fallback
                elapsed = max(float(start_clock) - float(game_state.clock_sec), 0.0)
                _update_minutes(game_state, off_on_court, elapsed, offense, home)
                _update_minutes(game_state, def_on_court, elapsed, defense, home)
                fb_intensity = _intensity_from_usage((ctx.get("_fatigue_seg_usage") if isinstance(ctx, dict) else None), elapsed)
                _apply_fatigue_loss(offense, off_on_court, game_state, rules, fb_intensity, elapsed, home)
                _apply_fatigue_loss(defense, def_on_court, game_state, rules, fb_intensity, elapsed, home)
            # Track possession-scope aggregates across dead-ball stop continuations.
            if bool(pos_res.get("had_orb", False)):
                pos_had_orb = True
            if pos_first_fga_sc is None and pos_res.get("first_fga_shotclock_sec") is not None:
                pos_first_fga_sc = pos_res.get("first_fga_shotclock_sec")

            # Dead-ball stop (e.g. no-shot foul): same offense retains the ball.
            # We do NOT count a new possession, and we do NOT swap offense/defense.
            if pos_res.get("end_reason") == "DEADBALL_STOP":
                pos_is_continuation = True
                if game_state.clock_sec <= 0:
                    game_state.clock_sec = 0
                    break
                pos_start = str(pos_res.get("pos_start_next", "after_foul"))
                continue

            # Replay event: possession end (terminal only; DEADBALL_STOP is a continuation and is not an end)
            try:
                emit_event(
                    game_state,
                    event_type="POSSESSION_END",
                    home=home,
                    away=away,
                    rules=rules,
                    team_id=off_team_id,
                    opp_team_id=def_team_id,
                    end_reason=pos_res.get("end_reason"),
                    points_scored=pos_res.get("points_scored"),
                    had_orb=pos_res.get("had_orb"),
                    first_fga_shotclock_sec=pos_res.get("first_fga_shotclock_sec"),
                    deadball_reason=pos_res.get("deadball_reason"),
                    ended_with_ft_trip=pos_res.get("ended_with_ft_trip"),
                    turnover_outcome=pos_res.get("turnover_outcome"),
                    turnover_is_steal=pos_res.get("turnover_is_steal"),
                    turnover_stealer_pid=pos_res.get("turnover_stealer_pid"),
                    turnover_deadball=pos_res.get("turnover_deadball"),
                    turnover_deadball_reason=pos_res.get("turnover_deadball_reason"),
                    was_blocked=pos_res.get("was_blocked"),
                    blocker_pid=pos_res.get("blocker_pid"),
                    block_kind=pos_res.get("block_kind"),
                    pos_start=pos_res.get("pos_start"),
                    pos_start_next=pos_res.get("pos_start_next"),
                    pos_start_next_override=pos_res.get("pos_start_next_override"),
                )
            except Exception as e:
                _push_debug_error(
                    "replay.emit_event.POSSESSION_END",
                    e,
                    {"event_type": "POSSESSION_END", "team_id": off_team_id, "end_reason": pos_res.get("end_reason")},
                )

            pts_scored = int(pos_res.get("points_scored", 0))
            had_orb = bool(pos_res.get("had_orb", False))
            pos_start_val = str(pos_res.get("pos_start", ""))
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
            if pts_scored > 0 and pos_start_val in ("after_tov", "after_tov_dead", "after_steal"):
                offense.points_off_tov += pts_scored

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
                update_timeout_trackers(game_state, offense_team_id=off_team_id, pos_res=pos_res)
            except Exception as e:
                _push_debug_error(
                    "timeout.update_timeout_trackers",
                    e,
                    {"offense_team_id": off_team_id, "end_reason": pos_res.get("end_reason")},
                )

            total_possessions += 1

            if game_state.clock_sec <= 0 or pos_res.get("end_reason") == "PERIOD_END":
                game_state.clock_sec = 0
                break

            # event-based possession change: after any terminal end, ball goes to defense
            offense, defense = defense, offense
            pos_start = str(pos_res.get("pos_start_next", "after_tov"))

        # Replay event: period end (neutral)
        try:
            emit_event(
                game_state,
                event_type="PERIOD_END",
                home=home,
                away=away,
                rules=rules,
                include_lineups=True,
            )
        except Exception as e:
            _push_debug_error(
                "replay.emit_event.PERIOD_END",
                e,
                {"event_type": "PERIOD_END", "q_index": int(q)},
            )

        replay_token = make_replay_token(rng, home, away, era=era)

    def _apply_period_break(break_sec: float) -> None:
        if break_sec <= 0:
            return
        onA = list(getattr(home, "on_court_pids", []) or [])
        onB = list(getattr(away, "on_court_pids", []) or [])
        _apply_break_recovery(home, onA, game_state, rules, break_sec, home)
        _apply_break_recovery(away, onB, game_state, rules, break_sec, home)

    break_between = float(rules.get("break_sec_between_periods", 0.0))
    break_halftime = float(rules.get("break_sec_halftime", break_between))
    break_before_ot = float(rules.get("break_sec_before_ot", break_between))

    # Regulation
    for q in range(regulation_quarters):
        _play_period(q, float(quarter_length_sec))

        # apply break after Q1/Q2/Q3 (not after Q4): halftime (after Q2) uses break_halftime
        if q < regulation_quarters - 1:
            _apply_period_break(break_halftime if q == 1 else break_between)

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
                        "role_counts": {home_team_id: home.role_fit_role_counts, away_team_id: away.role_fit_role_counts},
                        "grade_counts": {home_team_id: home.role_fit_grade_counts, away_team_id: away.role_fit_grade_counts},
                        "pos_log": {home_team_id: home.role_fit_pos_log, away_team_id: away.role_fit_pos_log},
                        "bad_totals": {home_team_id: home.role_fit_bad_totals, away_team_id: away.role_fit_bad_totals},
                        "bad_by_grade": {home_team_id: home.role_fit_bad_by_grade, away_team_id: away.role_fit_bad_by_grade},
                },
                "timeouts": {
                    "remaining": dict(getattr(game_state, "timeouts_remaining", {}) or {}),
                    "used": dict(getattr(game_state, "timeouts_used", {}) or {}),
                    "run_pts": dict(getattr(game_state, "run_pts", {}) or {}),
                    "consecutive_team_tos": dict(getattr(game_state, "consecutive_team_tos", {}) or {}),
                    "last_scoring_team_id": getattr(game_state, "last_scoring_team_id", None),
                },
            },
        },
        "replay_events": list(getattr(game_state, "replay_events", []) or []),
        "possessions_per_team": max(home.possessions, away.possessions),
        "teams": {
            home_team_id: summarize_team(home, game_state),
            away_team_id: summarize_team(away, game_state),
        },
        "game_state": {
            "quarter": game_state.quarter,
            "clock_sec": game_state.clock_sec,
            "shot_clock_sec": game_state.shot_clock_sec,
            "scores": {home_team_id: int(home.pts), away_team_id: int(away.pts)},
            "possession": game_state.possession,
            "team_fouls": dict(game_state.team_fouls),
            "player_fouls": dict(game_state.player_fouls),
            "fatigue": dict(game_state.fatigue),
            "minutes_played_sec": dict(game_state.minutes_played_sec),
        }
    }
