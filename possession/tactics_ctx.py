from __future__ import annotations

"""Possession-scoped tactical context helpers (matchups, help, hunt).

This module is a mostly-mechanical extraction from engine.sim_possession to reduce
the size of the main possession loop without changing behavior.
"""

import random
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .. import matchups
from ..participants import choose_default_actor
from ..tactics import canonical_defense_scheme
from ..models import GameState, TeamState


_SCHEME_HELP_BASELINE = {
    "PackLine_GapHelp": 0.65,
    "Zone": 0.45,
    "Blitz_TrapPnR": 0.55,
    "Hedge_ShowRecover": 0.30,
    "Drop": 0.20,
    "ICE_SidePnR": 0.15,
    "Switch_Everything": -0.25,
}


def make_possession_tactics_ctx(
    *,
    offense: TeamState,
    defense: TeamState,
    game_state: GameState,
    ctx: Dict[str, Any],
    rng: random.Random,
    home_team: TeamState,
    away_team: TeamState,
    rules: Dict[str, Any],
    off_team_id: str,
    def_team_id: str,
    pos_origin: str,
    is_continuation: bool,
    emit_event: Any,
    clamp: Any,
    player_stat: Any,
    record_ctx_error: Any,
):
    """Return possession-scoped tactical helpers.

    Returns:
        (_ensure_matchups, _cache_help_levels, _maybe_apply_hunt_plan, _maybe_inject_matchup_force, _update_def_pressure_for_step)
    """
    # Bind helper names used by the extracted code (preserve original variable names).
    _player_stat = player_stat
    _record_ctx_error = record_ctx_error

    # --- Matchups (Plan-1 MVP) ---
    # Build and maintain a 5v5 OFF_PID -> DEF_PID matchup map for the current on-court units.
    # This mapping is used by resolve.py to pick a primary defender and blend defensive values.
    def _matchups_instr_signature() -> str:
        """Build a stable signature for matchup-affecting tactical directives.

        This is intentionally narrow (matchup-related keys only), so we don't
        accidentally thrash the 5v5 map cache on unrelated context changes.
        """
        def _tctx(team: TeamState) -> Dict[str, Any]:
            try:
                c = getattr(getattr(team, "tactics", None), "context", None)
            except Exception:
                c = None
            return c if isinstance(c, dict) else {}

        dctx = _tctx(defense)

        # --- LOCKS ---
        locks_pairs: List[Tuple[str, str]] = []
        raw_locks = dctx.get("MATCHUP_LOCKS")
        if raw_locks is None:
            # Match matchups._extract_locks: fallback only when the primary key is present but None.
            raw_locks = dctx.get("MATCHUP_LOCK")
        if isinstance(raw_locks, list):
            for item in raw_locks:
                if not isinstance(item, dict):
                    continue
                dp = str(item.get("def_pid") or "").strip()
                op = str(item.get("off_pid") or "").strip()
                if dp and op:
                    locks_pairs.append((dp, op))
        elif isinstance(raw_locks, dict):
            if "def_pid" in raw_locks and "off_pid" in raw_locks:
                dp = str(raw_locks.get("def_pid") or "").strip()
                op = str(raw_locks.get("off_pid") or "").strip()
                if dp and op:
                    locks_pairs.append((dp, op))
            else:
                for dp, op in raw_locks.items():
                    dps = str(dp or "").strip()
                    ops = str(op or "").strip()
                    if dps and ops:
                        locks_pairs.append((dps, ops))
        locks_pairs.sort()

        # --- HIDES ---
        hides: List[str] = []
        raw_hides = dctx.get("MATCHUP_HIDE_PIDS")
        if raw_hides is None:
            # Match matchups._extract_hides: fallback only when the primary key is present but None.
            raw_hides = dctx.get("MATCHUP_HIDE_PID")
        if isinstance(raw_hides, (list, tuple)):
            # Match matchups._extract_hides: treat None as empty so it doesn't become the string "None".
            hides = [str(x).strip() for x in raw_hides if str(x or "").strip()]
        else:
            s = str(raw_hides or "").strip()
            hides = [s] if s else []
        hides = sorted(set(hides))

        # --- ASSIGNMENTS ---
        assigns: List[Tuple[str, str, str]] = []
        raw_assign = dctx.get("MATCHUP_ASSIGNMENTS")
        if isinstance(raw_assign, dict):
            for dp, spec in raw_assign.items():
                dps = str(dp or "").strip()
                if not dps or not isinstance(spec, dict):
                    continue
                primary = str(spec.get("primary_off_pid") or spec.get("off_pid") or "").strip()
                secondary = str(spec.get("secondary_off_role") or spec.get("off_role") or "").strip()
                assigns.append((dps, primary, secondary))
        assigns.sort()

        # --- LOCKDOWN ---
        lockdown = dctx.get("MATCHUP_LOCKDOWN")
        l_def = ""; l_off = ""; l_role = ""; l_tag = ""
        if isinstance(lockdown, dict):
            l_def = str(lockdown.get("def_pid") or "").strip()
            tgt = lockdown.get("target")
            if isinstance(tgt, dict):
                l_off = str(tgt.get("off_pid") or "").strip()
                l_role = str(tgt.get("off_role") or "").strip()
                l_tag = str(tgt.get("tag") or "").strip().upper()
            else:
                l_off = str(lockdown.get("off_pid") or "").strip()
                l_role = str(lockdown.get("off_role") or "").strip()
                l_tag = str(lockdown.get("tag") or "").strip().upper()

        # --- TEMP LOCKS (ctx) ---
        temp_pairs: List[Tuple[str, str]] = []
        raw_temp = ctx.get("matchups_temp_locks")
        if isinstance(raw_temp, list):
            for item in raw_temp:
                if not isinstance(item, dict):
                    continue
                dp = str(item.get("def_pid") or "").strip()
                op = str(item.get("off_pid") or "").strip()
                if dp and op:
                    temp_pairs.append((dp, op))
        elif isinstance(raw_temp, dict):
            if "def_pid" in raw_temp and "off_pid" in raw_temp:
                dp = str(raw_temp.get("def_pid") or "").strip()
                op = str(raw_temp.get("off_pid") or "").strip()
                if dp and op:
                    temp_pairs.append((dp, op))
            else:
                for dp, op in raw_temp.items():
                    dps = str(dp or "").strip()
                    ops = str(op or "").strip()
                    if dps and ops:
                        temp_pairs.append((dps, ops))
        temp_pairs.sort()

        # Keep signature compact and stable.
        return "|".join(
            [
                "L=" + ",".join([f"{a}>{b}" for a, b in locks_pairs]),
                "H=" + ",".join(hides),
                "A=" + ",".join([f"{d}:{p}:{r}" for d, p, r in assigns]),
                "LD=" + ":".join([l_def, l_off, l_role, l_tag]),
                "T=" + ",".join([f"{a}>{b}" for a, b in temp_pairs]),
            ]
        )

    def _ensure_matchups(reason: str = "pos_start") -> None:
        try:
            new_off = list(getattr(offense, "on_court_pids", []) or [])

            new_def = list(getattr(defense, "on_court_pids", []) or [])

            # Normalize to 5-man units if needed.
            if len(new_off) != 5:
                new_off = [p.pid for p in offense.on_court_players()]
            if len(new_def) != 5:
                new_def = [p.pid for p in defense.on_court_players()]

            # PID normalization: keep a stable on-court order for emit, and a sorted snapshot for cache checks.
            off_order = [str(x) for x in new_off if str(x)]
            def_order = [str(x) for x in new_def if str(x)]
            snap_off = sorted(off_order)
            snap_def = sorted(def_order)
            instr_sig = _matchups_instr_signature()

            # No-op if lineup snapshot is unchanged.
            if isinstance(ctx.get("matchups_map"), dict):
                if (
                    ctx.get("matchups_off_on_court") == snap_off
                    and ctx.get("matchups_def_on_court") == snap_def
                    and ctx.get("matchups_instr_sig") == instr_sig
                ):
                    # Even if the 5v5 map is unchanged, we still want one MATCHUP_SET at possession start
                    # for replay / text commentary.
                    if reason == "pos_start" and not bool(ctx.get("_matchup_set_emitted", False)):
                        try:
                            m_map = ctx.get("matchups_map") or {}
                            m_meta = ctx.get("matchups_meta") or {}
                            pairs = [
                                {"off_pid": opid, "def_pid": str(m_map.get(opid, "") or "")}
                                for opid in off_order
                            ]
                            emit_event(
                                game_state,
                                event_type="MATCHUP_SET",
                                home=home_team,
                                away=away_team,
                                rules=rules,
                                team_id=off_team_id,
                                opp_team_id=def_team_id,
                                pos_start=str(pos_origin),
                                matchups_version=int(ctx.get("matchups_version", 0) or 0),
                                pairs=pairs,
                                meta=dict(m_meta) if isinstance(m_meta, dict) else {},
                            )
                            ctx["_matchup_set_emitted"] = True
                        except Exception:
                            pass
                    return

            m_map, m_rev, m_meta = matchups.build_matchups(offense, defense, ctx, rng=rng)

            # Defensive normalization (belt-and-suspenders): ensure ctx always stores string pids.
            try:
                if isinstance(m_map, dict):
                    m_map = {str(k): str(v) for k, v in m_map.items() if str(k)}
                if isinstance(m_rev, dict):
                    m_rev = {str(k): str(v) for k, v in m_rev.items() if str(k)}
            except Exception:
                pass

            ctx["matchups_version"] = int(ctx.get("matchups_version", 0) or 0) + 1
            ctx["matchups_off_on_court"] = snap_off
            ctx["matchups_def_on_court"] = snap_def
            ctx["matchups_instr_sig"] = instr_sig
            ctx["matchups_map"] = m_map
            ctx["matchups_rev"] = m_rev
            ctx["matchups_meta"] = m_meta

            # Always emit once at possession start (for replay / text commentary).
            if reason == "pos_start" and not bool(ctx.get("_matchup_set_emitted", False)):
                try:
                    pairs = [
                        {"off_pid": opid, "def_pid": str(m_map.get(opid, "") or "")}
                        for opid in off_order
                    ]
                    emit_event(
                        game_state,
                        event_type="MATCHUP_SET",
                        home=home_team,
                        away=away_team,
                        rules=rules,
                        team_id=off_team_id,
                        opp_team_id=def_team_id,
                        pos_start=str(pos_origin),
                        matchups_version=int(ctx.get("matchups_version", 0) or 0),
                        pairs=pairs,
                        meta=dict(m_meta) if isinstance(m_meta, dict) else {},
                    )
                    ctx["_matchup_set_emitted"] = True
                except Exception:
                    pass
        except Exception as exc:
            _record_ctx_error("matchups.ensure", exc)

    def _maybe_inject_matchup_force() -> None:
        """Optionally inject a one-shot forced matchup for this possession step.

        Input shape (offense.tactics.context):
            MATCHUP_FORCE: {off_pid, def_pid, event?, reason?, force_actor?}
        Output (ctx):
            matchup_force: {off_pid, def_pid, event, reason, ttl=1}
        """
        if ctx.get("_matchup_force_injected"):
            return
        try:
            tctx = getattr(getattr(offense, "tactics", None), "context", None)
            # Policy A: Always consume (pop) the one-shot command the first time we see it
            # in this possession, regardless of whether it is valid/applicable.
            # This ensures the UX matches "one-shot" and prevents repetition across
            # continuation segments or future possessions.
            raw = None
            if isinstance(tctx, dict):
                raw = tctx.pop("MATCHUP_FORCE", None)
            if not isinstance(raw, dict):
                return

            opid = str(raw.get("off_pid") or "").strip()
            dpid = str(raw.get("def_pid") or "").strip()
            if not opid or not dpid:
                return
            if not offense.is_on_court(opid) or not defense.is_on_court(dpid):
                return

            ev = str(raw.get("event") or "USER_FORCE").strip() or "USER_FORCE"
            reason = raw.get("reason")
            force_actor = bool(raw.get("force_actor", False))

            ctx["matchup_force"] = {
                "off_pid": opid,
                "def_pid": dpid,
                "event": ev,
                "reason": (str(reason) if reason is not None else None),
                "ttl": 1,
            }
            if force_actor:
                ctx["force_actor_pid"] = opid

            ctx["_matchup_force_injected"] = True

            if bool(ctx.get("debug_matchups", False)):
                try:
                    emit_event(
                        game_state,
                        event_type="MATCHUP_EVENT",
                        home=home_team,
                        away=away_team,
                        rules=rules,
                        team_id=off_team_id,
                        opp_team_id=def_team_id,
                        pos_start=str(pos_origin),
                        matchups_version=int(ctx.get("matchups_version", 0) or 0),
                        event=ev,
                        off_pid=opid,
                        def_pid=dpid,
                        reason=(str(reason) if reason is not None else None),
                        ttl=1,
                    )
                except Exception:
                    pass
        except Exception as exc:
            _record_ctx_error("matchups.force_inject", exc)

    # --- Help defense (possession-scoped) ---
    def _cache_help_levels() -> None:
        """Cache per-defender help levels + a small team help scalar in ctx.

        Input (defense.tactics.context):
          HELP_LEVEL_BY_PID: {pid: "WEAK"/"NORMAL"/"STRONG"} or {pid: -1/0/+1}
        Output (ctx):
          help_level_by_pid: {pid: -1/0/+1}
          team_help_delta: float in [-1, +1]
          team_help_level: float in [-1, +1]  (backward-compat alias for team_help_delta)
        """
        try:
            dctx = getattr(getattr(defense, "tactics", None), "context", None)
        except Exception:
            dctx = None
        raw = dctx.get("HELP_LEVEL_BY_PID") if isinstance(dctx, dict) else None

        lvl: Dict[str, float] = {}
        if isinstance(raw, dict):
            for k, v in raw.items():
                pid = str(k or "").strip()
                if not pid:
                    continue
                if isinstance(v, (int, float)):
                    vv = float(v)
                else:
                    s = str(v or "").strip().upper()
                    if s in ("STRONG", "HIGH", "AGGRESSIVE", "+1"):
                        vv = 1.0
                    elif s in ("WEAK", "LOW", "CONSERVATIVE", "-1"):
                        vv = -1.0
                    else:
                        vv = 0.0
                lvl[pid] = clamp(vv, -1.0, 1.0)

        # Weighted team scalar: defenders with higher DEF_HELP matter a bit more.
        on_def = defense.on_court_players()
        if not on_def:
            ctx["help_level_by_pid"] = lvl
            ctx["team_help_delta"] = 0.0
            ctx["team_help_level"] = 0.0
            return

        num = 0.0
        den = 0.0
        for p in on_def:
            pid = str(getattr(p, "pid", "") or "").strip()
            if not pid:
                continue
            w = 0.5 + (_player_stat(p, "DEF_HELP", 50.0) / 100.0)
            num += w * float(lvl.get(pid, 0.0))
            den += w
        team_h = (num / den) if den > 0 else 0.0
        ctx["help_level_by_pid"] = lvl
        ctx["team_help_delta"] = clamp(team_h, -1.0, 1.0)
        # Backward-compat alias (older code reads team_help_level).
        ctx["team_help_level"] = ctx["team_help_delta"]

    # --- Help / Double pressure helpers (possession-scoped) ---
    def _scheme_help_baseline() -> float:
        try:
            ds = canonical_defense_scheme(getattr(defense.tactics, "defense_scheme", ""))
        except Exception:
            ds = ""
        return float(_SCHEME_HELP_BASELINE.get(ds, 0.0))

    def _leave_cost(off_pid: str) -> float:
        """Return a compact 'leave cost' (0..100) for helping off an offensive player."""
        if not off_pid:
            return 50.0
        op = offense.find_player(off_pid)
        if op is None:
            return 50.0
        c3 = float(_player_stat(op, "SHOT_3_CS", 50.0))
        o3 = float(_player_stat(op, "SHOT_3_OD", 50.0))
        return float(clamp(0.70 * c3 + 0.30 * o3, 0.0, 100.0))

    def _leave_cost_norm(cost: float) -> float:
        # Normalize around 50 to [-1, +1].
        return float(clamp((float(cost) - 50.0) / 50.0, -1.0, 1.0))

    def _predicted_actor_pid() -> Optional[str]:
        """Best-effort actor guess for pre-prior tactical evaluation."""
        fp = str(ctx.get("force_actor_pid") or "").strip()
        if fp and offense.is_on_court(fp):
            return fp
        try:
            a = choose_default_actor(offense)
            pid = str(getattr(a, "pid", "") or "").strip()
            return pid or None
        except Exception:
            return None

    def _choose_helper_pid(target_off_pid: Optional[str]) -> Optional[str]:
        """Choose a weak-side helper using DEF_HELP, stance, and leave-cost."""
        lvl = ctx.get("help_level_by_pid") if isinstance(ctx.get("help_level_by_pid"), dict) else {}
        m_map = ctx.get("matchups_map") if isinstance(ctx.get("matchups_map"), dict) else {}
        m_rev = ctx.get("matchups_rev") if isinstance(ctx.get("matchups_rev"), dict) else {}

        primary = str(m_map.get(str(target_off_pid or ""), "") or "").strip() if target_off_pid else ""
        best_pid = None
        best_score = float("-inf")

        for dp in defense.on_court_players():
            dpid = str(getattr(dp, "pid", "") or "").strip()
            if not dpid or (primary and dpid == primary):
                continue

            leave_off = str(m_rev.get(dpid, "") or "").strip()
            cost = _leave_cost(leave_off)
            stance = float(clamp(float(lvl.get(dpid, 0.0)), -1.0, 1.0)) * 12.0
            d_help = float(_player_stat(dp, "DEF_HELP", 50.0))

            score = 0.90 * d_help + stance - 1.00 * cost
            if score > best_score + 1e-9:
                best_score = score
                best_pid = dpid

        return best_pid

    def _choose_doubler_pid(
        target_off_pid: Optional[str],
        prefer_tag: Optional[str] = None,
        *,
        exclude: Optional[set] = None,
    ) -> Optional[str]:
        """Choose a doubler using steal/help/role skill and leave-cost."""
        m_map = ctx.get("matchups_map") if isinstance(ctx.get("matchups_map"), dict) else {}
        m_rev = ctx.get("matchups_rev") if isinstance(ctx.get("matchups_rev"), dict) else {}

        primary = str(m_map.get(str(target_off_pid or ""), "") or "").strip() if target_off_pid else ""
        tag = str(prefer_tag or "").strip().upper()
        ex = set(exclude or set())
        if primary:
            ex.add(primary)

        if tag == "BEST_HELP":
            w_steal, w_help, w_poa, w_post, w_leave = 0.3, 1.0, 0.0, 0.0, 1.10
        elif tag == "BEST_POA":
            w_steal, w_help, w_poa, w_post, w_leave = 0.3, 0.4, 0.8, 0.0, 1.10
        elif tag == "BEST_POST":
            w_steal, w_help, w_poa, w_post, w_leave = 0.3, 0.4, 0.0, 0.8, 1.10
        else:  # BEST_STEAL(default)
            w_steal, w_help, w_poa, w_post, w_leave = 0.8, 0.5, 0.0, 0.0, 1.10

        best_pid = None
        best_score = float("-inf")

        for dp in defense.on_court_players():
            dpid = str(getattr(dp, "pid", "") or "").strip()
            if not dpid or dpid in ex:
                continue

            leave_off = str(m_rev.get(dpid, "") or "").strip()
            cost = _leave_cost(leave_off)

            stl = float(_player_stat(dp, "DEF_STEAL", 50.0))
            hlp = float(_player_stat(dp, "DEF_HELP", 50.0))
            poa = float(_player_stat(dp, "DEF_POA", 50.0))
            pst = float(_player_stat(dp, "DEF_POST", 50.0))
            rim = float(_player_stat(dp, "DEF_RIM", 50.0))

            anchor_pen = 10.0 if rim >= 65.0 else 0.0

            score = (w_steal * stl) + (w_help * hlp) + (w_poa * poa) + (w_post * pst) - (w_leave * cost) - anchor_pen
            if score > best_score + 1e-9:
                best_score = score
                best_pid = dpid

        return best_pid

    # --- Hunting plans (possession start) ---
    def _choose_def_by_tag(tag: str, *, exclude: Optional[set] = None) -> Optional[str]:
        exclude = exclude or set()
        t = str(tag or "").strip().upper()
        best_pid = None
        best_val = None
        for p in defense.on_court_players():
            pid = str(getattr(p, "pid", "") or "").strip()
            if not pid or pid in exclude:
                continue
            if t == "WEAKEST_POA":
                val = -_player_stat(p, "DEF_POA", 50.0)
            elif t == "WEAKEST_OVERALL":
                # Lower is weaker.
                val = -(0.70 * _player_stat(p, "DEF_POA", 50.0) + 0.30 * _player_stat(p, "PHYSICAL", 50.0))
            elif t == "BEST_POA":
                val = _player_stat(p, "DEF_POA", 50.0)
            elif t == "BEST_POST":
                val = _player_stat(p, "DEF_POST", 50.0)
            elif t == "BEST_STEAL":
                val = _player_stat(p, "DEF_STEAL", 50.0)
            elif t == "BEST_HELP":
                val = _player_stat(p, "DEF_HELP", 50.0)
            else:
                # Default: best POA.
                val = _player_stat(p, "DEF_POA", 50.0)
            if best_pid is None or float(val) > float(best_val):
                best_pid = pid
                best_val = val
        return best_pid

    def _resolve_actor_pid(plan: Mapping[str, Any]) -> Optional[str]:
        opid = str(plan.get("actor_pid") or "").strip()
        if opid and offense.is_on_court(opid):
            return opid
        role = str(plan.get("actor_role") or "").strip()
        if role:
            try:
                roles = getattr(offense, "roles", None)
                if isinstance(roles, dict):
                    pid = str(roles.get(role) or "").strip()
                    if pid and offense.is_on_court(pid):
                        return pid
            except Exception:
                return None
        return None

    def _resolve_target_def_pid(plan: Mapping[str, Any]) -> Optional[str]:
        dpid = str(plan.get("target_def_pid") or "").strip()
        if dpid and defense.is_on_court(dpid):
            return dpid

        tag = str(plan.get("target_def_tag") or "").strip().upper()
        if tag == "HIDE":
            try:
                dctx = getattr(getattr(defense, "tactics", None), "context", None)
            except Exception:
                dctx = None
            hides = []
            if isinstance(dctx, dict):
                raw = dctx.get("MATCHUP_HIDE_PIDS", dctx.get("MATCHUP_HIDE_PID"))
                if isinstance(raw, list):
                    hides = [str(x).strip() for x in raw if str(x).strip()]
                else:
                    s = str(raw or "").strip()
                    hides = [s] if s else []
            hides = [h for h in hides if h and defense.is_on_court(h)]
            if hides:
                # Pick the weakest POA among hidden candidates (stable).
                cand = None
                cand_val = None
                for hp in hides:
                    p = defense.find_player(hp)
                    v = _player_stat(p, "DEF_POA", 50.0) if p is not None else 50.0
                    if cand is None or float(v) < float(cand_val):
                        cand = hp
                        cand_val = v
                if cand:
                    return cand
            return None

        if tag:
            return _choose_def_by_tag(tag)
        return None

    def _hunt_response_mode(target_def_pid: str) -> Tuple[str, Dict[str, Any]]:
        """Return (mode, response_ctx) where mode is ALLOW/DENY/TRAP."""
        try:
            dctx = getattr(getattr(defense, "tactics", None), "context", None)
        except Exception:
            dctx = None
        raw = dctx.get("HUNT_RESPONSE") if isinstance(dctx, dict) else None
        if not isinstance(raw, dict):
            return "ALLOW", {}

        # Determine if target is a hidden defender or the lockdown defender.
        hides = set()
        try:
            raw_h = dctx.get("MATCHUP_HIDE_PIDS", dctx.get("MATCHUP_HIDE_PID"))
        except Exception:
            raw_h = None
        if isinstance(raw_h, list):
            hides = set(str(x).strip() for x in raw_h if str(x).strip())
        else:
            s = str(raw_h or "").strip()
            hides = {s} if s else set()

        lockdown_def = None
        try:
            ld = dctx.get("MATCHUP_LOCKDOWN")
        except Exception:
            ld = None
        if isinstance(ld, dict):
            lockdown_def = str(ld.get("def_pid") or "").strip() or None

        if target_def_pid and target_def_pid in hides:
            mode = str(raw.get("vs_hide", raw.get("default", "ALLOW")) or "ALLOW").strip().upper()
        elif lockdown_def and target_def_pid == lockdown_def:
            mode = str(raw.get("vs_lockdown", raw.get("default", "ALLOW")) or "ALLOW").strip().upper()
        else:
            mode = str(raw.get("default", "ALLOW") or "ALLOW").strip().upper()

        # Normalize mode.
        if mode not in ("ALLOW", "DENY", "TRAP"):
            mode = "ALLOW"
        return mode, dict(raw)

    def _maybe_apply_hunt_plan() -> None:
        """Apply an offensive hunt plan at possession start (one plan per possession)."""
        if is_continuation:
            return
        if bool(ctx.get("_hunt_plan_applied", False)):
            return

        try:
            octx = getattr(getattr(offense, "tactics", None), "context", None)
        except Exception:
            octx = None
        plans = octx.get("HUNT_PLANS") if isinstance(octx, dict) else None
        if not isinstance(plans, list) or not plans:
            return

        # Choose first plan that passes its frequency roll and has valid actor/target.
        for plan in plans:
            if not isinstance(plan, dict):
                continue
            try:
                freq = float(plan.get("frequency", 1.0))
            except Exception:
                freq = 1.0
            freq = clamp(freq, 0.0, 1.0)
            if freq <= 0.0:
                continue
            if freq < 1.0 and rng.random() > freq:
                continue

            actor_pid = _resolve_actor_pid(plan)
            if not actor_pid:
                continue
            target_def_pid = _resolve_target_def_pid(plan)
            if not target_def_pid:
                continue

            label = str(plan.get("label") or "").strip() or None

            # Action multipliers (base-action keyed) for this possession.
            mult_by_base: Dict[str, float] = {}
            raw_mult = plan.get("action_mult_by_base")
            if isinstance(raw_mult, dict):
                for k, v in raw_mult.items():
                    kk = str(k or "").strip()
                    if not kk:
                        continue
                    try:
                        vv = float(v)
                    except Exception:
                        continue
                    if vv <= 0:
                        continue
                    mult_by_base[kk] = clamp(vv, 0.25, 2.50)

            force_actor = bool(plan.get("force_actor", True))
            force_matchup = bool(plan.get("force_matchup", True))

            # Emit intent first (text-replay can narrate "A targets B").
            try:
                emit_event(
                    game_state,
                    event_type="HUNT_CALL",
                    home=home_team,
                    away=away_team,
                    rules=rules,
                    team_id=off_team_id,
                    opp_team_id=def_team_id,
                    pos_start=str(pos_origin),
                    label=label,
                    off_pid=actor_pid,
                    target_def_pid=target_def_pid,
                    action_mult_by_base=dict(mult_by_base),
                )
            except Exception:
                pass

            mode, resp_cfg = _hunt_response_mode(target_def_pid)

            alt_def_pid = None
            doubler_pid = None
            trap_strength = None

            if mode == "DENY":
                # Choose an alternate defender to pre-switch onto the hunt actor.
                deny_tag = str(resp_cfg.get("deny_alt_def_tag", "BEST_POA") or "BEST_POA").strip().upper()
                # If actor is physical, prefer BEST_POST even when deny_tag is BEST_POA.
                actor_obj = offense.find_player(actor_pid)
                actor_phys = _player_stat(actor_obj, "PHYSICAL", 50.0) if actor_obj is not None else 50.0
                if deny_tag == "BEST_POA" and actor_phys >= 60.0:
                    deny_tag_eff = "BEST_POST"
                else:
                    deny_tag_eff = deny_tag
                alt_def_pid = _choose_def_by_tag(deny_tag_eff, exclude={target_def_pid})
                if alt_def_pid:
                    ctx["matchups_temp_locks"] = [{"off_pid": actor_pid, "def_pid": alt_def_pid, "event": "HUNT_DENY"}]
                    # Rebuild matchups immediately so the denial is reflected in ctx.matchups_map.
                    _ensure_matchups(reason="hunt_deny")
                else:
                    # If we cannot deny (no alternate), fall back to allowing the hunt.
                    mode = "ALLOW"

            if mode == "TRAP":
                try:
                    trap_strength = float(resp_cfg.get("trap_strength", 0.65))
                except Exception:
                    trap_strength = 0.65
                trap_strength = clamp(trap_strength, 0.0, 1.0)
                trap_dpid = str(resp_cfg.get("trap_doubler_pid") or "").strip()
                if trap_dpid and defense.is_on_court(trap_dpid):
                    doubler_pid = trap_dpid
                else:
                    tag = str(resp_cfg.get("trap_doubler_tag", "BEST_STEAL") or "BEST_STEAL").strip().upper()
                    doubler_pid = _choose_doubler_pid(actor_pid, prefer_tag=tag, exclude={target_def_pid})
                if doubler_pid:
                    ctx["double_active"] = {
                        "off_pid": actor_pid,
                        "primary_def_pid": target_def_pid,
                        "doubler_pid": doubler_pid,
                        "strength": float(trap_strength),
                        "ttl": 2,
                        "source": "HUNT_TRAP",
                    }

            # Apply offensive intent (still meaningful even if DENY pre-switches).
            if mult_by_base:
                ctx["hunt_action_mult_by_base"] = dict(mult_by_base)
            if force_actor:
                ctx["force_actor_pid"] = actor_pid

            if force_matchup and mode in ("ALLOW", "TRAP"):
                ctx["matchup_force"] = {
                    "off_pid": actor_pid,
                    "def_pid": target_def_pid,
                    "event": "HUNT",
                    "reason": label,
                    "ttl": 1,
                }

            # Emit defensive response (text can narrate deny/trap).
            try:
                emit_event(
                    game_state,
                    event_type="HUNT_RESPONSE",
                    home=home_team,
                    away=away_team,
                    rules=rules,
                    team_id=def_team_id,
                    opp_team_id=off_team_id,
                    pos_start=str(pos_origin),
                    response=mode,
                    label=label,
                    off_pid=actor_pid,
                    target_def_pid=target_def_pid,
                    alt_def_pid=alt_def_pid,
                    doubler_pid=doubler_pid,
                    strength=(float(trap_strength) if trap_strength is not None else None),
                )
            except Exception:
                pass

            ctx["_hunt_plan_applied"] = True
            return


    # --- Per-step defensive pressure context (help/double for priors + resolve) ---
    def _update_def_pressure_for_step(action: str, base_action: str, tags: Dict[str, Any]) -> None:
        """Update ctx['def_pressure'] for the current possession step.

        - Computes help pressure as: scheme baseline + team delta (scaled by scheme).
        - Chooses a helper and tracks who is being left (rotation risk).
        - Evaluates DOUBLE_RULES ahead of priors (so doubles affect outcome choice).
        """
        # Ensure matchup maps exist.
        try:
            _ensure_matchups(reason="step")
        except Exception:
            pass

        baseline = float(_scheme_help_baseline())
        try:
            delta = float(ctx.get("team_help_delta", 0.0) or 0.0)
        except Exception:
            delta = 0.0
        delta = float(clamp(delta, -1.0, 1.0))

        # If scheme is already very "help-heavy" (or very switchy), reduce the impact of the delta.
        delta_scale = float(clamp(1.0 - 0.65 * abs(baseline), 0.35, 1.0))

        # Priors use only delta (avoid double-counting scheme multipliers), resolve uses baseline+delta.
        eff_priors = float(clamp(delta * delta_scale, -1.0, 1.0))
        eff_resolve = float(clamp(baseline + (delta * delta_scale), -1.0, 1.0))

        # Choose helper and compute leave cost.
        target_off_pid = _predicted_actor_pid()
        helper_pid = _choose_helper_pid(target_off_pid)
        m_rev = ctx.get("matchups_rev") if isinstance(ctx.get("matchups_rev"), dict) else {}
        leave_off_pid = str(m_rev.get(helper_pid, "") or "").strip() if helper_pid else ""
        h_leave_cost = _leave_cost(leave_off_pid)
        h_leave_cost_norm = _leave_cost_norm(h_leave_cost)

        # Seed base def_pressure.
        ctx["def_pressure"] = {
            "step_base_action": str(base_action or ""),
            "help": {
                "baseline": float(baseline),
                "delta": float(delta),
                "delta_scale": float(delta_scale),
                "eff_priors": float(eff_priors),
                "eff_resolve": float(eff_resolve),
                "helper_pid": (helper_pid if helper_pid else None),
                "leave_off_pid": (leave_off_pid if leave_off_pid else None),
                "leave_cost": float(h_leave_cost),
                "leave_cost_norm": float(h_leave_cost_norm),
            },
            "double_plan_evaluated": True,
        }

        # If a double is already active (e.g., HUNT_TRAP), keep it; otherwise evaluate DOUBLE_RULES.
        double_spec = ctx.get("double_active")
        if not isinstance(double_spec, dict):
            # Evaluate rules with a pre-prior actor guess.
            try:
                dctx = getattr(getattr(defense, "tactics", None), "context", None)
            except Exception:
                dctx = None
            rules_list = dctx.get("DOUBLE_RULES") if isinstance(dctx, dict) else None

            if isinstance(rules_list, list) and rules_list:
                # Helper: compute best threat pid once if needed.
                best_threat_pid = None

                def _best_threat_pid() -> Optional[str]:
                    nonlocal best_threat_pid
                    if best_threat_pid is not None:
                        return best_threat_pid
                    best_pid = None
                    best_val = float("-inf")
                    for op in offense.on_court_players():
                        try:
                            v = float(matchups._threat_score(op))
                        except Exception:
                            v = 50.0
                        pid = str(getattr(op, "pid", "") or "").strip()
                        if pid and v > best_val + 1e-9:
                            best_val = v
                            best_pid = pid
                    best_threat_pid = best_pid
                    return best_threat_pid

                actor_pid = target_off_pid

                for rule in rules_list:
                    if not isinstance(rule, dict):
                        continue

                    # Base-action gating.
                    wba = rule.get("when_base_actions")
                    if isinstance(wba, list) and wba:
                        if str(base_action or "") not in set(str(x) for x in wba if str(x)):
                            continue

                    target = rule.get("target")
                    if not isinstance(target, Mapping):
                        target = rule

                    off_pid = str(target.get("off_pid") or "").strip()
                    off_role = str(target.get("off_role") or "").strip()
                    tag = str(target.get("tag") or "").strip().upper()

                    matched = False
                    if actor_pid:
                        if off_pid and off_pid == actor_pid:
                            matched = True
                        elif off_role:
                            try:
                                roles = getattr(offense, "roles", None)
                                if isinstance(roles, Mapping):
                                    rp = str(roles.get(off_role) or "").strip()
                                    if rp and rp == actor_pid:
                                        matched = True
                            except Exception:
                                matched = False
                        elif tag == "BEST_THREAT":
                            bt = _best_threat_pid()
                            if bt and bt == actor_pid:
                                matched = True

                    if not matched:
                        continue

                    try:
                        strength = float(rule.get("strength", 0.65) or 0.65)
                    except Exception:
                        strength = 0.65
                    strength = float(clamp(strength, 0.0, 1.0))
                    if strength <= 1e-9:
                        continue

                    try:
                        freq = float(rule.get("frequency", 1.0) or 1.0)
                    except Exception:
                        freq = 1.0
                    freq = float(clamp(freq, 0.0, 1.0))

                    # Roll.
                    p = float(clamp(freq * strength, 0.0, 1.0))
                    try:
                        if float(getattr(game_state, "shot_clock_sec", 24.0) or 24.0) <= 8.0:
                            p = float(clamp(p + 0.12, 0.0, 1.0))
                    except Exception:
                        pass
                    if bool(tags.get("in_transition", False)):
                        p = float(clamp(p - 0.12, 0.0, 1.0))

                    if p < 1.0 and rng.random() > p:
                        continue

                    # Choose doubler.
                    doubler_pid = str(rule.get("doubler_pid") or "").strip()
                    if doubler_pid and not defense.is_on_court(doubler_pid):
                        doubler_pid = ""

                    if not doubler_pid:
                        prefer = str(rule.get("doubler_tag", "BEST_STEAL") or "BEST_STEAL").strip().upper()
                        doubler_pid = _choose_doubler_pid(actor_pid, prefer_tag=prefer)

                    # Primary defender (best-effort, from current map).
                    m_map = ctx.get("matchups_map") if isinstance(ctx.get("matchups_map"), dict) else {}
                    primary_def_pid = str(m_map.get(actor_pid or "", "") or "").strip() if actor_pid else ""

                    label = str(rule.get("label") or "").strip() or None

                    ctx["double_active"] = {
                        "off_pid": actor_pid,
                        "primary_def_pid": (primary_def_pid if primary_def_pid else None),
                        "doubler_pid": (doubler_pid if doubler_pid else None),
                        "strength": float(strength),
                        "ttl": 2,
                        "source": "RULE",
                        "label": label,
                    }
                    break  # One rule per step.

        # Populate def_pressure.double from ctx.double_active (or inactive).
        spec = ctx.get("double_active")
        active = False
        off_pid = None
        primary_def_pid = None
        doubler_pid = None
        strength = 0.0
        source = None
        label = None

        if isinstance(spec, dict):
            try:
                ttl = int(spec.get("ttl", 0) or 0)
            except Exception:
                ttl = 0
            off = str(spec.get("off_pid") or "").strip()
            if ttl > 0 and off and offense.is_on_court(off):
                active = True
                off_pid = off
                primary_def_pid = str(spec.get("primary_def_pid") or "").strip() or None
                doubler_pid = str(spec.get("doubler_pid") or "").strip() or None
                try:
                    strength = float(spec.get("strength", 0.0) or 0.0)
                except Exception:
                    strength = 0.0
                strength = float(clamp(strength, 0.0, 1.0))
                source = str(spec.get("source") or "").strip() or None
                label = str(spec.get("label") or "").strip() or None

        d_leave_off = ""
        d_leave_cost = 50.0
        d_leave_cost_norm = 0.0
        if active and doubler_pid:
            m_rev = ctx.get("matchups_rev") if isinstance(ctx.get("matchups_rev"), dict) else {}
            d_leave_off = str(m_rev.get(doubler_pid, "") or "").strip()
            d_leave_cost = _leave_cost(d_leave_off)
            d_leave_cost_norm = _leave_cost_norm(d_leave_cost)

        ctx["def_pressure"]["double"] = {
            "active": bool(active),
            "off_pid": off_pid,
            "primary_def_pid": primary_def_pid,
            "doubler_pid": doubler_pid,
            "strength": float(strength),
            "leave_off_pid": (d_leave_off if d_leave_off else None),
            "leave_cost": float(d_leave_cost),
            "leave_cost_norm": float(d_leave_cost_norm),
            "source": source,
            "label": label,
        }

    return _ensure_matchups, _cache_help_levels, _maybe_apply_hunt_plan, _maybe_inject_matchup_force, _update_def_pressure_for_step
