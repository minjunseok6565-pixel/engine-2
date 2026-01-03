from __future__ import annotations

import random
import argparse
import sys

from .core import clamp
from .era import get_mvp_rules
from .models import Player, TeamState
from .sim import simulate_game
from .sim.team_keys import team_key
from .tactics import TacticsConfig


# -------------------------
# Pretty printing (player boxscore table)
# -------------------------

def _safe_div(n: float, d: float) -> float:
    return (n / d) if d else 0.0

def _fmt_pct(made: float, att: float) -> str:
    return f"{_safe_div(made, att) * 100:.1f}" if att else "0.0"

def _fmt_min(sec: float) -> str:
    return f"{sec / 60.0:.1f}"

def print_player_boxscore_table(team: TeamState, res: dict, home: TeamState) -> None:
    team_res = (res.get("teams") or {}).get(team.name, {}) or {}
    players_raw = team_res.get("Players") or {}
    # Optional precomputed box (from patched sim). If present but raw missing, fall back to it.
    pre_box = team_res.get("PlayerBox") or {}

    gs = res.get("game_state") or {}
    key = team_key(team, home)
    mins_map = (gs.get("minutes_played_sec") or {}).get(key, {}) or {}
    pf_map = (gs.get("player_fouls") or {}).get(key, {}) or {}

    rows = []

    tot_sec = 0.0
    tot_pts = 0.0
    tot_fgm = 0.0; tot_fga = 0.0
    tot_tpm = 0.0; tot_tpa = 0.0
    tot_ftm = 0.0; tot_fta = 0.0
    tot_orb = 0.0; tot_drb = 0.0
    tot_tov = 0.0
    tot_pf = 0.0

    for p in team.lineup:
        pid = getattr(p, "player_id", None) or getattr(p, "id", None) or getattr(p, "pid", None) or str(p)
        st = players_raw.get(pid) or pre_box.get(pid) or {}
        fgm = float(st.get("FGM", 0)); fga = float(st.get("FGA", 0))
        tpm = float(st.get("3PM", 0)); tpa = float(st.get("3PA", 0))
        ftm = float(st.get("FTM", 0)); fta = float(st.get("FTA", 0))
        pts = float(st.get("PTS", 0))
        orb = float(st.get("ORB", 0)); drb = float(st.get("DRB", 0))
        tov = float(st.get("TOV", 0))
        pf = float(pf_map.get(pid, st.get("PF", 0)))
        mins = float(mins_map.get(pid, st.get("MIN", 0) * 60.0))

        tot_sec += mins
        tot_pts += pts
        tot_fgm += fgm; tot_fga += fga
        tot_tpm += tpm; tot_tpa += tpa
        tot_ftm += ftm; tot_fta += fta
        tot_orb += orb; tot_drb += drb
        tot_tov += tov
        tot_pf += pf

        rows.append({
            "PLAYER": getattr(p, "name", pid),
            "MIN": _fmt_min(mins),
            "PTS": int(pts),
            "FG": f"{int(fgm)}-{int(fga)}",
            "FG%": _fmt_pct(fgm, fga),
            "3P": f"{int(tpm)}-{int(tpa)}",
            "3P%": _fmt_pct(tpm, tpa),
            "FT": f"{int(ftm)}-{int(fta)}",
            "FT%": _fmt_pct(ftm, fta),
            "ORB": int(orb),
            "DRB": int(drb),
            "REB": int(orb + drb),
            "TOV": int(tov),
            "PF": int(pf),
        })

    # TOTAL row (team totals across lineup)
    rows.append({
        "PLAYER": "TOTAL",
        "MIN": _fmt_min(tot_sec),
        "PTS": int(tot_pts),
        "FG": f"{int(tot_fgm)}-{int(tot_fga)}",
        "FG%": _fmt_pct(tot_fgm, tot_fga),
        "3P": f"{int(tot_tpm)}-{int(tot_tpa)}",
        "3P%": _fmt_pct(tot_tpm, tot_tpa),
        "FT": f"{int(tot_ftm)}-{int(tot_fta)}",
        "FT%": _fmt_pct(tot_ftm, tot_fta),
        "ORB": int(tot_orb),
        "DRB": int(tot_drb),
        "REB": int(tot_orb + drb),
        "TOV": int(tot_tov),
        "PF": int(tot_pf),
    })

    headers = ["PLAYER","MIN","PTS","FG","FG%","3P","3P%","FT","FT%","ORB","DRB","REB","TOV","PF"]

    widths = {
        "PLAYER": 12,
        "MIN": 5,
        "PTS": 4,
        "FG": 7,
        "FG%": 5,
        "3P": 7,
        "3P%": 5,
        "FT": 7,
        "FT%": 5,
        "ORB": 4,
        "DRB": 4,
        "REB": 4,
        "TOV": 4,
        "PF": 3,
    }

    def fmt_row(r: dict) -> str:
        parts = []
        for h in headers:
            parts.append(str(r.get(h, "")).ljust(widths[h])[:widths[h]])
        return " ".join(parts)

    print(f"\n[{team.name}] Player Boxscore")
    print(fmt_row({h: h for h in headers}))
    print("-" * (sum(widths.values()) + (len(headers) - 1)))
    for i, r in enumerate(rows):
        # add a separator line before TOTAL
        if i == len(rows) - 1:
            print("-" * (sum(widths.values()) + (len(headers) - 1)))
        print(fmt_row(r))


# -------------------------
# Demo (sample derived stats)
# -------------------------

def make_sample_player(rng: random.Random, pid: str, name: str, archetype: str) -> Player:
    keys = [
        "FIN_RIM","FIN_DUNK","FIN_CONTACT","SHOT_MID_CS","SHOT_3_CS","SHOT_FT","SHOT_MID_PU","SHOT_3_OD","SHOT_TOUCH",
        "POST_SCORE","POST_CONTROL","SEAL_POWER",
        "DRIVE_CREATE","HANDLE_SAFE","FIRST_STEP",
        "PASS_SAFE","PASS_CREATE","PNR_READ","SHORTROLL_PLAY",
        "DEF_POA","DEF_HELP","DEF_STEAL","DEF_RIM","DEF_POST",
        "REB_OR","REB_DR","PHYSICAL","ENDURANCE"
    ]
    base = {k: 50.0 for k in keys}

    def bump(ks, lo, hi):
        for k in ks:
            base[k] = clamp(base[k] + rng.uniform(lo, hi), 25, 95)

    if archetype == "PG_SHOOT":
        bump(["SHOT_3_CS","SHOT_3_OD","PASS_CREATE","PASS_SAFE","PNR_READ","HANDLE_SAFE","FIRST_STEP","DRIVE_CREATE"], 12, 25)
        bump(["DEF_POA","ENDURANCE"], 5, 12)
    elif archetype == "WING_3D":
        bump(["SHOT_3_CS","DEF_POA","DEF_HELP","DEF_STEAL","ENDURANCE"], 10, 20)
        bump(["DRIVE_CREATE","HANDLE_SAFE"], 2, 10)
    elif archetype == "BIG_RIM":
        bump(["DEF_RIM","DEF_POST","REB_DR","PHYSICAL","ENDURANCE"], 12, 25)
        bump(["FIN_RIM","FIN_DUNK","FIN_CONTACT","SHORTROLL_PLAY","REB_OR"], 6, 15)
    elif archetype == "BIG_SKILL":
        bump(["SHOT_MID_CS","PASS_SAFE","PASS_CREATE","SHORTROLL_PLAY","POST_SCORE","POST_CONTROL"], 8, 18)
        bump(["DEF_HELP","DEF_POST","ENDURANCE"], 6, 14)
    elif archetype == "SLASH":
        bump(["FIN_RIM","FIN_CONTACT","FIRST_STEP","DRIVE_CREATE","HANDLE_SAFE","ENDURANCE"], 12, 24)
        bump(["SHOT_3_CS"], 0, 10)
    else:
        bump(keys, -5, 10)

    return Player(pid=pid, name=name, derived=base)

def demo(seed: int = 7) -> None:
    rules = get_mvp_rules()

    def run_game(def_scheme: str, label: str) -> None:
        rng = random.Random(seed)

        tA_tac = TacticsConfig(
            offense_scheme="Spread_HeavyPnR",
            defense_scheme="Drop",
            scheme_weight_sharpness=1.10,
            scheme_outcome_strength=1.05,
            def_scheme_weight_sharpness=1.00,
            def_scheme_outcome_strength=1.00,
            action_weight_mult={"PnR":1.15},
            outcome_global_mult={"SHOT_3_CS":1.05},
            outcome_by_action_mult={"PnR":{"PASS_SHORTROLL":1.10}},
            context={"PACE_MULT":1.05}
        )

        tB_tac = TacticsConfig(
            offense_scheme="Drive_Kick",
            defense_scheme=def_scheme,
            scheme_weight_sharpness=1.05,
            scheme_outcome_strength=1.05,
            def_scheme_weight_sharpness=1.05,
            def_scheme_outcome_strength=1.05,
            outcome_global_mult={"PASS_KICKOUT":1.10},
            context={"PACE_MULT":1.02}
        )

        home = TeamState(
            name="A_SpreadPnR",
            lineup=[
                make_sample_player(rng,"A1","A1_PG","PG_SHOOT"),
                make_sample_player(rng,"A2","A2_W","WING_3D"),
                make_sample_player(rng,"A3","A3_S","SLASH"),
                make_sample_player(rng,"A4","A4_B","BIG_SKILL"),
                make_sample_player(rng,"A5","A5_C","BIG_RIM"),
            ],
            roles={"ball_handler":"A1","secondary_handler":"A2","screener":"A5","post":"A4","shooter":"A2","cutter":"A3","rim_runner":"A5"},
            tactics=tA_tac
        )

        away = TeamState(
            name="B_DriveKick",
            lineup=[
                make_sample_player(rng,"B1","B1_PG","SLASH"),
                make_sample_player(rng,"B2","B2_W","WING_3D"),
                make_sample_player(rng,"B3","B3_W","WING_3D"),
                make_sample_player(rng,"B4","B4_B","BIG_SKILL"),
                make_sample_player(rng,"B5","B5_C","BIG_RIM"),
            ],
            roles={"ball_handler":"B1","secondary_handler":"B2","screener":"B5","post":"B4","shooter":"B2","cutter":"B3","rim_runner":"B5"},
            tactics=tB_tac
        )

        res = simulate_game(rng, home, away)
        score_a = res["teams"][home.name]["PTS"]
        score_b = res["teams"][away.name]["PTS"]
        fouls = res.get("game_state", {}).get("team_fouls", {})
        fatigue = res.get("game_state", {}).get("fatigue", {})
        hist = res["teams"][home.name]["OffActionCounts"]
        total = sum(hist.values()) or 1
        freq = {k: round(v / total * 100, 2) for k, v in sorted(hist.items(), key=lambda kv: -kv[1])}

        print(f"\n=== Run: {label} (Defense scheme={def_scheme}) ===")
        print(f"Final Score: {home.name} {score_a} - {away.name} {score_b}")
        print_player_boxscore_table(home, res, home=home)
        print_player_boxscore_table(away, res, home=home)
        print("Team fouls:", fouls)
        home_key = team_key(home, home)
        away_key = team_key(away, home)
        sample_fatigue = {
            home_key: {pid: round(fatigue.get(home_key, {}).get(pid, 1.0), 3) for pid in list(fatigue.get(home_key, {}).keys())[:4]},
            away_key: {pid: round(fatigue.get(away_key, {}).get(pid, 1.0), 3) for pid in list(fatigue.get(away_key, {}).keys())[:4]},
        }
        print("Sample fatigue:", sample_fatigue)
        print("Action frequency (home offense):", freq)
        print("Possessions per team:", res["possessions_per_team"])

    print(f"Quarter length: {rules['quarter_length']}s | Shot clock: {rules['shot_clock']}s")
    run_game("Drop", "Baseline Drop")
    run_game("Switch_Everything", "Switch Everything")


# -------------------------
# CLI entrypoint
# -------------------------

def _parse_csv_list(s: str):
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def run_excel_match(args) -> None:
    try:
        from .roster_adapter import build_team_state_from_excel
    except Exception:
        print("Failed to import roster_adapter. Make sure match_engine/roster_adapter.py exists and pandas/openpyxl are installed.")
        raise

    rng = random.Random(args.seed)
    rules = get_mvp_rules()

    home = build_team_state_from_excel(
        args.xlsx, args.home,
        lineup_selectors=_parse_csv_list(args.playersA),
        offense_scheme=args.offA,
        defense_scheme=args.defA,
    )
    away = build_team_state_from_excel(
        args.xlsx, args.away,
        lineup_selectors=_parse_csv_list(args.playersB),
        offense_scheme=args.offB,
        defense_scheme=args.defB,
    )

    print(f"Quarter length: {rules['quarter_length']}s | Shot clock: {rules['shot_clock']}s")
    label = f"{home.name} vs {away.name} (xlsx)"

    res = simulate_game(rng, home, away, era=args.era, strict_validation=True)

    # ✅ sim.py는 최상위 "score" 키를 반환하지 않음. 점수는 teams 요약의 "PTS"에 있음.
    score_a = (res.get("teams") or {}).get(home.name, {}).get("PTS", 0)
    score_b = (res.get("teams") or {}).get(away.name, {}).get("PTS", 0)

    fouls = (res.get("game_state") or {}).get("team_fouls", {})

    # ✅ fatigue_map이 아니라 fatigue
    fatigue = (res.get("game_state") or {}).get("fatigue", {})

    # ✅ action_frequency는 현재 sim.py 반환값에 없음 → teams[home]["OffActionCounts"] 기반으로 계산
    freq = {}
    try:
        hist = (res.get("teams") or {}).get(home.name, {}).get("OffActionCounts", {}) or {}
        total = sum(hist.values()) or 1
        freq = {k: round(v / total * 100.0, 2) for k, v in sorted(hist.items(), key=lambda kv: -kv[1])}
    except Exception:
        freq = {}

    print(f"\n=== Run: {label} ===")
    print(f"Final Score: {home.name} {score_a} - {away.name} {score_b}")
    print_player_boxscore_table(home, res, home=home)
    print_player_boxscore_table(away, res, home=home)
    print("Team fouls:", fouls)
    home_key = team_key(home, home)
    away_key = team_key(away, home)
    sample_fatigue = {
        home_key: {pid: round(fatigue.get(home_key, {}).get(pid, 1.0), 3) for pid in list(fatigue.get(home_key, {}).keys())[:4]},
        away_key: {pid: round(fatigue.get(away_key, {}).get(pid, 1.0), 3) for pid in list(fatigue.get(away_key, {}).keys())[:4]},
    }
    print("Sample fatigue:", sample_fatigue)
    print("Action frequency (team A offense):", freq)
    print("Possessions per team:", res.get("possessions_per_team"))

def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description="match_engine demo")
    ap.add_argument("--xlsx", default=None, help="Excel roster path. If set, runs an Excel-based matchup instead of the built-in random demo.")
    ap.add_argument("--home", default=None, help="Home team name/code in Excel (column 'Team'). Required if --xlsx is set.")
    ap.add_argument("--away", default=None, help="Away team name/code in Excel (column 'Team'). Required if --xlsx is set.")
    ap.add_argument("--playersA", default="", help="Comma-separated player pids or names for Team A lineup (auto-fill to 5).")
    ap.add_argument("--playersB", default="", help="Comma-separated player pids or names for Team B lineup (auto-fill to 5).")
    ap.add_argument("--offA", default="Spread_HeavyPnR", help="Offense scheme for Team A.")
    ap.add_argument("--defA", default="Drop", help="Defense scheme for Team A.")
    ap.add_argument("--offB", default="Drive_Kick", help="Offense scheme for Team B.")  # ✅ 기본값 수정
    ap.add_argument("--defB", default="Switch_Everything", help="Defense scheme for Team B.")
    ap.add_argument("--era", default="default", help="Era config name.")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed for reproducibility.")
    args = ap.parse_args(argv)

    if args.xlsx:
        if not args.home or not args.away:
            ap.error("--home and --away are required when using --xlsx")
        run_excel_match(args)
    else:
        demo()

if __name__ == "__main__":
    main()
