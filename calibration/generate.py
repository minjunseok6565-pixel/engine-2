from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

from ..tactics import TacticsConfig
from ..models import Player, TeamState
from ..validation import REQUIRED_DERIVED_KEYS
from ..profiles_data import OFF_SCHEME_ACTION_WEIGHTS
from ..role_fit import role_fit_score
from ..sim_rotation import ROLE_TO_GROUPS

# -----------------------------
# Calibration generator design
# -----------------------------
# Goal:
# - Create structured (but stochastic) dummy rosters
# - Sample tactics/roles with directional bias (profiles)
# - Keep everything valid under engine validation contracts
#
# Note: we deliberately avoid importing `schema` here; runner will handle it.
# -----------------------------

OFFENSE_SCHEMES: Tuple[str, ...] = tuple(OFF_SCHEME_ACTION_WEIGHTS.keys())

# Defense schemes are defined by era tables; validation allows keys from defense_scheme_mult.
# We keep a conservative canonical list here; runner can override if needed.
DEFENSE_SCHEMES: Tuple[str, ...] = (
    "Drop",
    "Switch_Everything",
    "Hedge_ShowRecover",
    "Blitz_TrapPnR",
    "ICE_SidePnR",
    "Zone",
    "PackLine_GapHelp",
)

# 12-role list (canonical)
ROLES_12: Tuple[str, ...] = tuple(ROLE_TO_GROUPS.keys())

# -----------------------------
# Directional profiles
# -----------------------------

@dataclass(frozen=True)
class DirectionProfile:
    name: str
    offense_w: Dict[str, float]
    defense_w: Dict[str, float]
    # knob priors
    off_sharp_mu: float = 1.05
    off_sharp_sd: float = 0.10
    off_str_mu: float = 1.03
    off_str_sd: float = 0.10
    def_sharp_mu: float = 1.02
    def_sharp_sd: float = 0.10
    def_str_mu: float = 1.02
    def_str_sd: float = 0.10

# A few presets you can expand later
PROFILES: Dict[str, DirectionProfile] = {
    "modern": DirectionProfile(
        name="modern",
        offense_w={
            "Spread_HeavyPnR": 0.28,
            "FiveOut": 0.20,
            "Drive_Kick": 0.20,
            "Motion_SplitCut": 0.12,
            "DHO_Chicago": 0.10,
            "Transition_Early": 0.06,
            "Horns_Elbow": 0.03,
            "Post_InsideOut": 0.01,
        },
        defense_w={
            "Drop": 0.42,
            "Switch_Everything": 0.28,
            "Hedge_ShowRecover": 0.10,
            "ICE_SidePnR": 0.08,
            "Blitz_TrapPnR": 0.06,
            "Zone": 0.04,
            "PackLine_GapHelp": 0.02,
        },
        off_sharp_mu=1.10,
        off_str_mu=1.06,
        def_sharp_mu=1.05,
        def_str_mu=1.04,
    ),
    "motion": DirectionProfile(
        name="motion",
        offense_w={
            "Motion_SplitCut": 0.30,
            "DHO_Chicago": 0.22,
            "FiveOut": 0.14,
            "Spread_HeavyPnR": 0.12,
            "Drive_Kick": 0.10,
            "Horns_Elbow": 0.06,
            "Transition_Early": 0.04,
            "Post_InsideOut": 0.02,
        },
        defense_w={
            "Switch_Everything": 0.34,
            "Drop": 0.30,
            "PackLine_GapHelp": 0.14,
            "Zone": 0.10,
            "ICE_SidePnR": 0.06,
            "Hedge_ShowRecover": 0.04,
            "Blitz_TrapPnR": 0.02,
        },
        off_sharp_mu=1.08,
        off_str_mu=1.04,
        def_sharp_mu=1.05,
        def_str_mu=1.03,
    ),
    "post": DirectionProfile(
        name="post",
        offense_w={
            "Post_InsideOut": 0.32,
            "Horns_Elbow": 0.20,
            "Spread_HeavyPnR": 0.16,
            "Drive_Kick": 0.10,
            "Motion_SplitCut": 0.10,
            "DHO_Chicago": 0.06,
            "FiveOut": 0.04,
            "Transition_Early": 0.02,
        },
        defense_w={
            "Drop": 0.44,
            "Zone": 0.18,
            "PackLine_GapHelp": 0.16,
            "Switch_Everything": 0.10,
            "ICE_SidePnR": 0.06,
            "Hedge_ShowRecover": 0.04,
            "Blitz_TrapPnR": 0.02,
        },
        off_sharp_mu=1.02,
        off_str_mu=1.05,
        def_sharp_mu=1.00,
        def_str_mu=1.03,
    ),
    "pace": DirectionProfile(
        name="pace",
        offense_w={
            "Transition_Early": 0.32,
            "Drive_Kick": 0.22,
            "Spread_HeavyPnR": 0.18,
            "FiveOut": 0.12,
            "Motion_SplitCut": 0.08,
            "DHO_Chicago": 0.04,
            "Horns_Elbow": 0.03,
            "Post_InsideOut": 0.01,
        },
        defense_w={
            "Drop": 0.38,
            "Switch_Everything": 0.22,
            "ICE_SidePnR": 0.14,
            "Hedge_ShowRecover": 0.10,
            "Blitz_TrapPnR": 0.08,
            "PackLine_GapHelp": 0.04,
            "Zone": 0.04,
        },
        off_sharp_mu=1.08,
        off_str_mu=1.02,
        def_sharp_mu=1.04,
        def_str_mu=1.02,
    ),
}

def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def _weighted_choice(rng: random.Random, w: Dict[str, float]) -> str:
    # simple weighted choice (no external deps)
    items = [(k, float(v)) for k, v in w.items() if float(v) > 0]
    if not items:
        return list(w.keys())[0] if w else ""
    total = sum(v for _, v in items)
    r = rng.random() * total
    acc = 0.0
    for k, v in items:
        acc += v
        if r <= acc:
            return k
    return items[-1][0]

def _trunc_norm(rng: random.Random, mu: float, sd: float, lo: float, hi: float) -> float:
    # cheap truncated normal (rejection; bounded to a few tries)
    for _ in range(12):
        x = rng.gauss(mu, sd)
        if lo <= x <= hi:
            return x
    return _clamp(mu, lo, hi)

# -----------------------------
# Roster archetypes
# -----------------------------

@dataclass(frozen=True)
class Archetype:
    name: str
    pos: str  # "G" | "F" | "C"
    bumps: Dict[str, Tuple[float, float]]  # key -> (mu, sd) absolute override-ish

# helpers for bumps
def _b(mu: float, sd: float = 8.0) -> Tuple[float, float]:
    return (mu, sd)

ARCHETYPES: Dict[str, Archetype] = {
    "lead_guard": Archetype("lead_guard", "G", {
        "HANDLE_SAFE": _b(82, 7),
        "PASS_CREATE": _b(80, 8),
        "PASS_SAFE": _b(76, 7),
        "PNR_READ": _b(78, 8),
        "DRIVE_CREATE": _b(76, 9),
        "FIRST_STEP": _b(74, 9),
        "SHOT_3_OD": _b(70, 10),
    }),
    "scoring_guard": Archetype("scoring_guard", "G", {
        "SHOT_3_OD": _b(78, 9),
        "SHOT_MID_PU": _b(76, 9),
        "HANDLE_SAFE": _b(72, 9),
        "DRIVE_CREATE": _b(68, 10),
        "SHOT_FT": _b(72, 8),
    }),
    "movement_shooter": Archetype("movement_shooter", "G", {
        "SHOT_3_CS": _b(82, 7),
        "SHOT_MID_CS": _b(68, 9),
        "ENDURANCE": _b(78, 8),
        "PASS_SAFE": _b(62, 9),
    }),
    "three_d_wing": Archetype("three_d_wing", "F", {
        "SHOT_3_CS": _b(74, 9),
        "DEF_POA": _b(76, 8),
        "DEF_HELP": _b(72, 9),
        "DEF_STEAL": _b(66, 10),
        "ENDURANCE": _b(74, 9),
        "PHYSICAL": _b(68, 9),
    }),
    "slashing_wing": Archetype("slashing_wing", "F", {
        "DRIVE_CREATE": _b(76, 9),
        "FIRST_STEP": _b(78, 8),
        "FIN_RIM": _b(74, 9),
        "FIN_CONTACT": _b(74, 9),
        "SHOT_FT": _b(74, 9),
    }),
    "connector_forward": Archetype("connector_forward", "F", {
        "PASS_SAFE": _b(76, 8),
        "PASS_CREATE": _b(70, 9),
        "HANDLE_SAFE": _b(66, 10),
        "DEF_HELP": _b(74, 9),
        "SHOT_3_CS": _b(66, 10),
    }),
    "stretch_big": Archetype("stretch_big", "C", {
        "SHOT_3_CS": _b(76, 9),
        "SHOT_MID_CS": _b(66, 10),
        "PASS_SAFE": _b(66, 10),
        "SHORTROLL_PLAY": _b(62, 11),
        "REB_DR": _b(64, 11),
        "DEF_RIM": _b(60, 12),
        "PHYSICAL": _b(66, 10),
    }),
    "roll_big": Archetype("roll_big", "C", {
        "FIN_RIM": _b(78, 8),
        "FIN_DUNK": _b(82, 7),
        "FIN_CONTACT": _b(76, 9),
        "REB_OR": _b(74, 10),
        "PHYSICAL": _b(80, 7),
        "DEF_RIM": _b(72, 10),
        "REB_DR": _b(70, 10),
        "SHORTROLL_PLAY": _b(56, 11),
    }),
    "rim_protector": Archetype("rim_protector", "C", {
        "DEF_RIM": _b(84, 7),
        "DEF_POST": _b(76, 8),
        "REB_DR": _b(78, 9),
        "PHYSICAL": _b(82, 7),
        "FIN_RIM": _b(70, 10),
        "FIN_DUNK": _b(72, 10),
    }),
    "post_big": Archetype("post_big", "C", {
        "POST_SCORE": _b(80, 8),
        "POST_CONTROL": _b(78, 8),
        "SHOT_TOUCH": _b(74, 9),
        "PASS_SAFE": _b(70, 9),
        "PASS_CREATE": _b(64, 10),
        "PHYSICAL": _b(82, 7),
        "DEF_POST": _b(74, 9),
        "REB_DR": _b(72, 9),
    }),
}

# scheme -> archetype mix (12 players)
_SCHEME_ROSTER_PLAN: Dict[str, List[str]] = {
    "Spread_HeavyPnR": ["lead_guard","scoring_guard","three_d_wing","three_d_wing","slashing_wing","connector_forward","movement_shooter","stretch_big","roll_big","rim_protector","three_d_wing","connector_forward"],
    "FiveOut": ["lead_guard","scoring_guard","movement_shooter","three_d_wing","three_d_wing","connector_forward","slashing_wing","stretch_big","stretch_big","rim_protector","three_d_wing","connector_forward"],
    "Drive_Kick": ["lead_guard","scoring_guard","slashing_wing","slashing_wing","three_d_wing","three_d_wing","movement_shooter","connector_forward","roll_big","rim_protector","three_d_wing","connector_forward"],
    "Motion_SplitCut": ["lead_guard","movement_shooter","movement_shooter","three_d_wing","three_d_wing","connector_forward","connector_forward","stretch_big","roll_big","rim_protector","three_d_wing","connector_forward"],
    "DHO_Chicago": ["lead_guard","movement_shooter","scoring_guard","three_d_wing","connector_forward","three_d_wing","slashing_wing","stretch_big","stretch_big","rim_protector","three_d_wing","connector_forward"],
    "Post_InsideOut": ["lead_guard","scoring_guard","three_d_wing","connector_forward","three_d_wing","movement_shooter","slashing_wing","post_big","roll_big","rim_protector","three_d_wing","connector_forward"],
    "Horns_Elbow": ["lead_guard","scoring_guard","connector_forward","three_d_wing","movement_shooter","three_d_wing","slashing_wing","stretch_big","post_big","rim_protector","three_d_wing","connector_forward"],
    "Transition_Early": ["lead_guard","scoring_guard","slashing_wing","three_d_wing","movement_shooter","three_d_wing","slashing_wing","connector_forward","roll_big","rim_protector","three_d_wing","connector_forward"],
}

_EXTRA_KEYS = set()
for a in ARCHETYPES.values():
    _EXTRA_KEYS.update(a.bumps.keys())

def _sample_stat(rng: random.Random, mu: float, sd: float, lo: float = 5.0, hi: float = 99.0) -> float:
    return _trunc_norm(rng, mu, sd, lo, hi)

def generate_player(
    rng: random.Random,
    *,
    pid: str,
    name: str,
    archetype: str,
) -> Player:
    arch = ARCHETYPES.get(archetype) or ARCHETYPES["three_d_wing"]

    # baseline distribution (league-ish)
    derived: Dict[str, float] = {}
    all_keys = set(REQUIRED_DERIVED_KEYS) | set(_EXTRA_KEYS)
    for k in all_keys:
        derived[k] = _sample_stat(rng, mu=55.0, sd=12.0, lo=15.0, hi=92.0)

    # apply bumps
    for k, (mu, sd) in arch.bumps.items():
        derived[k] = _sample_stat(rng, mu=mu, sd=sd, lo=20.0, hi=98.0)

    # a few weak correlations for sanity
    if arch.pos == "C":
        # bigs slightly worse at handle/pnr by default
        for k in ("HANDLE_SAFE","PNR_READ","SHOT_3_OD"):
            derived[k] = _sample_stat(rng, mu=min(derived[k], 55.0), sd=10.0, lo=15.0, hi=80.0)

    return Player(pid=str(pid), name=str(name), pos=str(arch.pos), derived=derived)

def generate_tactics(
    rng: random.Random,
    profile: DirectionProfile,
) -> TacticsConfig:
    off = _weighted_choice(rng, profile.offense_w)
    if off not in OFFENSE_SCHEMES:
        off = "Spread_HeavyPnR"
    de = _weighted_choice(rng, profile.defense_w)
    if de not in DEFENSE_SCHEMES:
        de = "Drop"

    tac = TacticsConfig(
        offense_scheme=off,
        defense_scheme=de,
        scheme_weight_sharpness=_trunc_norm(rng, profile.off_sharp_mu, profile.off_sharp_sd, 0.70, 1.40),
        scheme_outcome_strength=_trunc_norm(rng, profile.off_str_mu, profile.off_str_sd, 0.70, 1.40),
        def_scheme_weight_sharpness=_trunc_norm(rng, profile.def_sharp_mu, profile.def_sharp_sd, 0.70, 1.40),
        def_scheme_outcome_strength=_trunc_norm(rng, profile.def_str_mu, profile.def_str_sd, 0.70, 1.40),
    )
    return tac

def _role_priority_for_scheme(off_scheme: str) -> List[str]:
    # Order matters only for uniqueness assignment; feel free to tune.
    # A few scheme-specific boosts are applied later via scoring.
    base = [
        "Initiator_Primary",
        "Roller_Finisher",
        "Pop_Spacer_Big",
        "Post_Hub",
        "Shot_Creator",
        "Initiator_Secondary",
        "Spacer_CatchShoot",
        "Spacer_Movement",
        "Connector_Playmaker",
        "ShortRoll_Playmaker",
        "Rim_Attacker",
        "Transition_Handler",
    ]
    if off_scheme == "FiveOut":
        return ["Pop_Spacer_Big","Spacer_CatchShoot","Initiator_Primary","Connector_Playmaker","Spacer_Movement","Shot_Creator","Initiator_Secondary","ShortRoll_Playmaker","Rim_Attacker","Transition_Handler","Roller_Finisher","Post_Hub"]
    if off_scheme == "Post_InsideOut":
        return ["Post_Hub","Initiator_Primary","Connector_Playmaker","Spacer_CatchShoot","Spacer_Movement","Shot_Creator","Initiator_Secondary","Roller_Finisher","ShortRoll_Playmaker","Rim_Attacker","Transition_Handler","Pop_Spacer_Big"]
    if off_scheme == "Transition_Early":
        return ["Transition_Handler","Initiator_Primary","Rim_Attacker","Spacer_CatchShoot","Spacer_Movement","Shot_Creator","Initiator_Secondary","Roller_Finisher","Connector_Playmaker","ShortRoll_Playmaker","Pop_Spacer_Big","Post_Hub"]
    if off_scheme in ("Motion_SplitCut","DHO_Chicago"):
        return ["Spacer_Movement","Connector_Playmaker","Initiator_Primary","Initiator_Secondary","Shot_Creator","Spacer_CatchShoot","ShortRoll_Playmaker","Pop_Spacer_Big","Roller_Finisher","Rim_Attacker","Transition_Handler","Post_Hub"]
    return base

def assign_roles_12(
    rng: random.Random,
    players: List[Player],
    off_scheme: str,
    *,
    unique_first_n: int = 8,
) -> Dict[str, str]:
    # Score table
    scores: Dict[str, List[Tuple[str, float]]] = {}
    for role in ROLES_12:
        lst: List[Tuple[str, float]] = []
        for p in players:
            s = float(role_fit_score(p, role))
            # scheme hints (small, but consistent)
            if off_scheme == "FiveOut" and role == "Pop_Spacer_Big":
                s *= 1.08
            if off_scheme == "Post_InsideOut" and role == "Post_Hub":
                s *= 1.10
            if off_scheme == "Transition_Early" and role == "Transition_Handler":
                s *= 1.10
            if off_scheme in ("Motion_SplitCut","DHO_Chicago") and role in ("Spacer_Movement","Connector_Playmaker"):
                s *= 1.06
            lst.append((p.pid, s))
        lst.sort(key=lambda x: x[1], reverse=True)
        scores[role] = lst

    priority = _role_priority_for_scheme(off_scheme)
    out: Dict[str, str] = {}
    used: set[str] = set()

    # Phase 1: enforce uniqueness for key roles
    for i, role in enumerate(priority):
        if role not in scores:
            continue
        pick = None
        for pid, _ in scores[role]:
            if (i < unique_first_n) and (pid in used):
                continue
            pick = pid
            break
        if pick is None:
            pick = scores[role][0][0]
        out[role] = pick
        if i < unique_first_n:
            used.add(pick)

    # Phase 2: fill any missing roles (allow overlaps)
    for role in ROLES_12:
        if role in out:
            continue
        out[role] = scores[role][0][0]

    return out

def _overall_rating(p: Player) -> float:
    # Keep it simple: blended offense/defense/physical
    keys = [
        ("PASS_CREATE", 0.12),
        ("HANDLE_SAFE", 0.10),
        ("SHOT_3_CS", 0.10),
        ("SHOT_3_OD", 0.08),
        ("FIN_RIM", 0.08),
        ("DEF_POA", 0.10),
        ("DEF_HELP", 0.10),
        ("DEF_RIM", 0.10),
        ("REB_DR", 0.08),
        ("PHYSICAL", 0.07),
        ("ENDURANCE", 0.07),
    ]
    s = 0.0
    for k, w in keys:
        try:
            s += float(p.get(k)) * float(w)
        except Exception:
            pass
    return s

def _choose_starters(players: List[Player], roles: Dict[str, str]) -> List[str]:
    # Ensure initiator primary and one big are starters.
    init = roles.get("Initiator_Primary")
    big_candidates = [roles.get("Roller_Finisher"), roles.get("Pop_Spacer_Big"), roles.get("Post_Hub")]
    big_candidates = [pid for pid in big_candidates if pid]
    big = None
    for pid in big_candidates:
        p = next((x for x in players if x.pid == pid), None)
        if p and p.pos in ("C","F"):
            big = pid
            break
    # pick remaining by overall
    ranked = sorted(players, key=_overall_rating, reverse=True)
    starters: List[str] = []
    if init:
        starters.append(init)
    if big and big not in starters:
        starters.append(big)
    for p in ranked:
        if p.pid in starters:
            continue
        starters.append(p.pid)
        if len(starters) >= 5:
            break
    return starters[:5]

def _build_rotation_targets(players: List[Player], starters: List[str]) -> Dict[str, int]:
    # Basic NBA-ish minute targets in seconds (sum ~240)
    ranked = sorted(players, key=_overall_rating, reverse=True)
    targets: Dict[str, int] = {}
    for i, p in enumerate(ranked):
        if p.pid in starters:
            targets[p.pid] = int(32 * 60)
        elif i < 8:
            targets[p.pid] = int(20 * 60)
        elif i < 10:
            targets[p.pid] = int(12 * 60)
        else:
            targets[p.pid] = int(6 * 60)
    return targets

def build_team(
    rng: random.Random,
    *,
    team_id: str,
    name: str,
    profile: DirectionProfile,
) -> Tuple[TeamState, Dict[str, Any]]:
    tac = generate_tactics(rng, profile)
    plan = _SCHEME_ROSTER_PLAN.get(tac.offense_scheme) or _SCHEME_ROSTER_PLAN["Spread_HeavyPnR"]

    players: List[Player] = []
    for i, archetype in enumerate(plan):
        pid = f"{team_id}_{i:02d}"
        players.append(generate_player(rng, pid=pid, name=f"{name}_{archetype}_{i:02d}", archetype=archetype))

    roles = assign_roles_12(rng, players, tac.offense_scheme, unique_first_n=8)
    starters = _choose_starters(players, roles)

    # Reorder lineup: starters first (keeps sim_game default tip-off stable)
    pid_to_player = {p.pid: p for p in players}
    lineup = [pid_to_player[pid] for pid in starters if pid in pid_to_player]
    lineup += [p for p in players if p.pid not in starters]

    team = TeamState(
        team_id=str(team_id),
        name=str(name),
        lineup=lineup,
        roles=roles,
        tactics=tac,
    )

    # Provide pid->role map for rotation/fatigue (and for tip-off initiator enforcement hooks)
    pid_role: Dict[str, str] = {}
    for role, pid in roles.items():
        if not pid:
            continue
        # if multiple roles map to same pid, keep the one with "most demanding" primary group
        prev = pid_role.get(pid)
        if prev is None:
            pid_role[pid] = role
        else:
            # choose by group priority (Handler > Wing > Big)
            def pri(rn: str) -> int:
                groups = ROLE_TO_GROUPS.get(rn, tuple())
                g0 = groups[0] if groups else ""
                return 3 if g0 == "Handler" else 2 if g0 == "Wing" else 1 if g0 == "Big" else 0
            if pri(role) > pri(prev):
                pid_role[pid] = role
    team.rotation_offense_role_by_pid = dict(pid_role)

    team.rotation_target_sec_by_pid = _build_rotation_targets(players, starters)

    meta = {
        "team_id": team.team_id,
        "name": team.name,
        "offense_scheme": tac.offense_scheme,
        "defense_scheme": tac.defense_scheme,
        "knobs": {
            "scheme_weight_sharpness": tac.scheme_weight_sharpness,
            "scheme_outcome_strength": tac.scheme_outcome_strength,
            "def_scheme_weight_sharpness": tac.def_scheme_weight_sharpness,
            "def_scheme_outcome_strength": tac.def_scheme_outcome_strength,
        },
        "roles": dict(roles),
        "starters": list(starters),
        "profile": profile.name,
    }
    return team, meta
