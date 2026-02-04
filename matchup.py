from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Tuple

from .core import clamp, dot_profile, weighted_choice
from .era import DEFAULT_LOGISTIC_PARAMS, DEFAULT_PROB_MODEL
from .models import Player, TeamState

# -------------------------
# Context keys (tactics.context)
# -------------------------
K_LOCK_MAP = "MATCHUP_LOCK_MAP"                 # {off_pid: def_pid}
K_LOCK_STRENGTH = "MATCHUP_LOCK_STRENGTH"       # 0..1 (락 강제 확률)

K_HIDE_DEF_PID = "MATCHUP_HIDE_DEF_PID"         # 숨길 수비자 pid
K_HIDE_STRENGTH = "MATCHUP_HIDE_STRENGTH"       # 0..1 (위협 상대로 숨김 회피 강도)
K_HIDE_THREAT_TH = "MATCHUP_HIDE_THREAT_THRESHOLD"  # 0..100 (기본 60)

K_HUNT_ATTACKER_PID = "MATCHUP_HUNT_ATTACKER_PID"   # 헌팅 수행자 pid
K_HUNT_TARGET_DEF_PID = "MATCHUP_HUNT_TARGET_DEF_PID"  # 헌팅 대상(약점 수비) pid
K_HUNT_FREQ = "MATCHUP_HUNT_FREQ"                 # 0..1 (헌팅 시도 빈도)

# matchup effect scaling (optional knobs)
K_LOGIT_MULT_SHOT = "MATCHUP_LOGIT_MULT_SHOT"     # 기본 0.55
K_LOGIT_MULT_PASS = "MATCHUP_LOGIT_MULT_PASS"     # 기본 0.35
K_LOGIT_MULT_FOUL = "MATCHUP_LOGIT_MULT_FOUL"     # 기본 0.50
K_LOGIT_MULT_STEAL = "MATCHUP_LOGIT_MULT_STEAL"   # 기본 0.60

@dataclass
class MatchupPick:
    defender_pid: Optional[str]
    event: str  # "LOCK" | "HUNT" | "HIDE_SUCCESS" | "HIDE_BROKEN" | "NATURAL"
    detail: Dict[str, Any]

def _ctx_map(obj: Any) -> Dict[str, Any]:
    return obj if isinstance(obj, dict) else {}

def _getf(m: Mapping[str, Any], k: str, default: float) -> float:
    try:
        return float(m.get(k, default))
    except Exception:
        return float(default)

def _gets(m: Mapping[str, Any], k: str, default: str = "") -> str:
    try:
        v = m.get(k, default)
        return "" if v is None else str(v)
    except Exception:
        return str(default)

def _get_sensitivity(game_cfg: Any, kind: str) -> float:
    lp = getattr(game_cfg, "logistic_params", None)
    lp = lp if isinstance(lp, Mapping) else DEFAULT_LOGISTIC_PARAMS
    spec = lp.get(kind) or lp.get("default") or {}
    sens = spec.get("sensitivity")
    scale = spec.get("scale")

    if sens is None:
        if scale is not None and float(scale) > 1e-9:
            sens = 1.0 / float(scale)
        else:
            pm = getattr(game_cfg, "prob_model", None)
            pm = pm if isinstance(pm, Mapping) else DEFAULT_PROB_MODEL
            if kind.startswith("pass"):
                sens = 1.0 / float(pm.get("pass_scale", 20.0))
            elif kind.startswith("rebound"):
                sens = 1.0 / float(pm.get("rebound_scale", 22.0))
            else:
                sens = 1.0 / float(pm.get("shot_scale", 18.0))
    return float(sens)

def threat_score(p: Player) -> float:
    # 0..100 근사: “숨김” 판단용(가드/윙/빅 모두 어느 정도 반영)
    w = {
        "SHOT_3_OD": 0.18,
        "SHOT_MID_PU": 0.12,
        "DRIVE_CREATE": 0.18,
        "SHOT_3_CS": 0.12,
        "POST_SCORE": 0.14,
        "FIN_RIM": 0.14,
        "FIN_CONTACT": 0.12,
    }
    vals = {k: float(p.get(k)) for k in w.keys()}
    return float(dot_profile(vals, w, missing_default=50.0))

def _def_select_profile(kind: str) -> Dict[str, float]:
    # 온볼 수비자 “선정”에 사용할 간단 프로파일(가볍게)
    if kind in ("shot_3", "shot_mid"):
        return {"DEF_POA": 0.55, "DEF_HELP": 0.10, "PHYSICAL": 0.10, "ENDURANCE": 0.25}
    if kind == "shot_post":
        return {"DEF_POST": 0.60, "PHYSICAL": 0.25, "DEF_RIM": 0.05, "ENDURANCE": 0.10}
    if kind == "shot_rim":
        return {"DEF_RIM": 0.55, "PHYSICAL": 0.20, "DEF_POA": 0.10, "ENDURANCE": 0.15}
    if kind == "pass":
        return {"DEF_STEAL": 0.55, "DEF_POA": 0.25, "DEF_HELP": 0.10, "ENDURANCE": 0.10}
    # turnover/pressure 기본
    return {"DEF_POA": 0.45, "DEF_STEAL": 0.30, "PHYSICAL": 0.10, "ENDURANCE": 0.15}

def _onball_keys_for_kind(kind: str) -> Set[str]:
    # “def_score(team snapshot)”에 덧붙일 때 defender 스탯으로 대체할 키들
    if kind in ("shot_3", "shot_mid"):
        return {"DEF_POA", "PHYSICAL", "ENDURANCE"}
    if kind == "shot_post":
        return {"DEF_POST", "PHYSICAL"}
    if kind == "shot_rim":
        return {"DEF_RIM", "PHYSICAL", "DEF_POA"}
    if kind == "pass":
        return {"DEF_POA", "DEF_STEAL"}
    # steal/pressure
    return {"DEF_POA", "DEF_STEAL", "PHYSICAL"}

def classify_kind(outcome: str, shot_kind: Optional[str] = None) -> str:
    # resolve.py가 _shot_kind_from_outcome(outcome) 이미 갖고 있으니,
    # 여기서는 shot_kind를 넘겨받는 형태를 기본으로 추천.
    if shot_kind:
        return shot_kind
    if outcome.startswith("PASS_"):
        return "pass"
    if outcome.startswith("TO_"):
        return "turnover"
    if outcome.startswith("FOUL_"):
        return "turnover"
    return "turnover"

def pick_onball_defender(
    rng: Any,
    offense: TeamState,
    defense: TeamState,
    *,
    actor_pid: str,
    kind: str,
    ctx: Dict[str, Any],
) -> MatchupPick:
    # ---- cache ----
    cache = ctx.get("_matchup_cache")
    if not isinstance(cache, dict):
        cache = {}
        ctx["_matchup_cache"] = cache
    by_actor = cache.get("by_actor")
    if not isinstance(by_actor, dict):
        by_actor = {}
        cache["by_actor"] = by_actor

    if actor_pid in by_actor:
        prev = by_actor.get(actor_pid) or {}
        dpid = prev.get("defender_pid")
        if isinstance(dpid, str) and dpid and defense.is_on_court(dpid):
            return MatchupPick(dpid, str(prev.get("event") or "NATURAL"), dict(prev.get("detail") or {}))

    off_ctx = _ctx_map(getattr(getattr(offense, "tactics", None), "context", None))
    def_ctx = _ctx_map(getattr(getattr(defense, "tactics", None), "context", None))

    # ---- hard/soft lock ----
    lock_strength = clamp(_getf(def_ctx, K_LOCK_STRENGTH, 1.0), 0.0, 1.0)
    lock_map = def_ctx.get(K_LOCK_MAP)
    if isinstance(lock_map, Mapping):
        forced = lock_map.get(actor_pid)
        if forced is not None:
            forced_pid = str(forced)
            if forced_pid and defense.is_on_court(forced_pid):
                if rng.random() < lock_strength:
                    pick = MatchupPick(forced_pid, "LOCK", {"lock_strength": lock_strength})
                    by_actor[actor_pid] = {"defender_pid": pick.defender_pid, "event": pick.event, "detail": pick.detail}
                    return pick

    # ---- hunt ----
    hunt_freq = clamp(_getf(off_ctx, K_HUNT_FREQ, 0.0), 0.0, 1.0)
    hunt_attacker = _gets(off_ctx, K_HUNT_ATTACKER_PID, "")
    hunt_target_def = _gets(off_ctx, K_HUNT_TARGET_DEF_PID, "")
    if hunt_freq > 1e-9 and hunt_target_def and defense.is_on_court(hunt_target_def):
        if (not hunt_attacker) or (hunt_attacker == actor_pid):
            if rng.random() < hunt_freq:
                pick = MatchupPick(hunt_target_def, "HUNT", {"hunt_freq": hunt_freq})
                by_actor[actor_pid] = {"defender_pid": pick.defender_pid, "event": pick.event, "detail": pick.detail}
                return pick

    # ---- natural weighted selection (with hide) ----
    hide_def = _gets(def_ctx, K_HIDE_DEF_PID, "")
    hide_strength = clamp(_getf(def_ctx, K_HIDE_STRENGTH, 0.0), 0.0, 1.0)
    hide_th = clamp(_getf(def_ctx, K_HIDE_THREAT_TH, 60.0), 0.0, 100.0)

    actor = offense.find_player(actor_pid)
    t = threat_score(actor) if actor is not None else 50.0

    prof = _def_select_profile(kind)

    weights: Dict[str, float] = {}
    for dp in defense.on_court_players():
        dpid = dp.pid
        vals = {k: float(dp.get(k)) for k in prof.keys()}
        score = float(dot_profile(vals, prof, missing_default=50.0))
        w = clamp(score / 100.0, 0.05, 2.0)

        if hide_def and dpid == hide_def and (t >= hide_th):
            # 숨김: 위협 높은 공격자에겐 회피 확률 증가
            w *= max(0.0, 1.0 - hide_strength)

        weights[dpid] = w

    chosen = weighted_choice(rng, weights) if weights else None
    event = "NATURAL"
    detail: Dict[str, Any] = {}
    if hide_def and defense.is_on_court(hide_def) and (t >= hide_th):
        if chosen == hide_def:
            event = "HIDE_BROKEN"
        else:
            event = "HIDE_SUCCESS"
        detail = {"hide_def_pid": hide_def, "hide_strength": hide_strength, "threat": t, "th": hide_th}

    pick = MatchupPick(chosen, event, detail)
    by_actor[actor_pid] = {"defender_pid": pick.defender_pid, "event": pick.event, "detail": pick.detail}
    return pick

def def_score_delta_from_defender(
    *,
    def_profile: Mapping[str, float],
    def_snap: Mapping[str, float],
    defender: Player,
    kind: str,
) -> float:
    # team snapshot 기반 def_score와, 일부 키를 defender로 대체한 def_score의 차이를 계산
    keys = list(def_profile.keys())
    team_vals = {k: float(def_snap.get(k, 50.0)) for k in keys}
    mb_vals = dict(team_vals)
    for k in _onball_keys_for_kind(kind):
        if k in mb_vals:
            mb_vals[k] = float(defender.get(k))
    team_score = float(dot_profile(team_vals, dict(def_profile), missing_default=50.0))
    mb_score = float(dot_profile(mb_vals, dict(def_profile), missing_default=50.0))
    return float(mb_score - team_score)

def delta_to_logit(
    *,
    game_cfg: Any,
    kind: str,
    delta_score: float,
    sign: float,
    mult: float,
    clamp_abs: float = 0.55,
) -> float:
    sens = _get_sensitivity(game_cfg, kind)
    d = float(delta_score) * float(sens) * float(mult) * float(sign)
    return float(clamp(d, -clamp_abs, clamp_abs))

def kind_mult_from_ctx(kind: str, offense: TeamState, defense: TeamState) -> float:
    off_ctx = _ctx_map(getattr(getattr(offense, "tactics", None), "context", None))
    def_ctx = _ctx_map(getattr(getattr(defense, "tactics", None), "context", None))

    if kind.startswith("shot_"):
        return clamp(_getf(off_ctx, K_LOGIT_MULT_SHOT, _getf(def_ctx, K_LOGIT_MULT_SHOT, 0.55)), 0.0, 2.0)
    if kind == "pass":
        return clamp(_getf(off_ctx, K_LOGIT_MULT_PASS, _getf(def_ctx, K_LOGIT_MULT_PASS, 0.35)), 0.0, 2.0)
    if kind == "steal":
        return clamp(_getf(off_ctx, K_LOGIT_MULT_STEAL, _getf(def_ctx, K_LOGIT_MULT_STEAL, 0.60)), 0.0, 2.0)
    if kind == "foul":
        return clamp(_getf(off_ctx, K_LOGIT_MULT_FOUL, _getf(def_ctx, K_LOGIT_MULT_FOUL, 0.50)), 0.0, 2.0)
    return 0.5
