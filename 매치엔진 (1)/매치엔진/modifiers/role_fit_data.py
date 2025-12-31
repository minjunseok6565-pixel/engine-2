    # =============================================================================
    # [DATA FILE ONLY]  (자동 분리됨)
    # 이 파일은 로직이 아니라 '튜닝 테이블/상수'만 담는 **데이터 모듈**입니다.
    # LLM 컨텍스트에는 기본적으로 포함하지 말고, 테이블을 수정/튜닝할 때만 열어보세요.
    #
    # 포함 데이터(요약):
    #   - ROLE_FIT_WEIGHTS: {role: {stat_key: weight}}
#   - ROLE_FIT_CUTS: {role: (s_min, a_min, b_min, c_min)}
#   - ROLE_PRIOR_MULT_RAW: {grade: {'GOOD': mult, 'BAD': mult}}
#   - ROLE_LOGIT_DELTA_RAW: {grade: delta}
#   - 로직 파일: role_fit.py
    # =============================================================================

"""
Data tables for role_fit.py.

This module is intentionally data-heavy.
Avoid including it in LLM context unless you are tuning role weights / cuts.
"""
from __future__ import annotations

from typing import Dict, Tuple

# -----------------------------
# Role prior / logit tuning
# -----------------------------
ROLE_PRIOR_MULT_RAW = {
    "S": {"GOOD": 1.06, "BAD": 0.94},
    "A": {"GOOD": 1.03, "BAD": 0.97},
    "B": {"GOOD": 1.00, "BAD": 1.00},
    "C": {"GOOD": 0.93, "BAD": 1.10},
    "D": {"GOOD": 0.85, "BAD": 1.25},
}
ROLE_LOGIT_DELTA_RAW = {"S": 0.18, "A": 0.10, "B": 0.00, "C": -0.18, "D": -0.35}


# -----------------------------
# Role fit weights (12 roles)
# Sum of weights per role == 1.0
# -----------------------------
ROLE_FIT_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Initiator_Primary": {
        "PNR_READ": 0.22,
        "DRIVE_CREATE": 0.18,
        "HANDLE_SAFE": 0.16,
        "PASS_CREATE": 0.16,
        "PASS_SAFE": 0.10,
        "SHOT_3_OD": 0.08,
        "SHOT_MID_PU": 0.05,
        "FIRST_STEP": 0.05,
    },
    "Initiator_Secondary": {
        "SHOT_3_CS": 0.20,
        "PASS_SAFE": 0.16,
        "PASS_CREATE": 0.14,
        "PNR_READ": 0.14,
        "HANDLE_SAFE": 0.12,
        "DRIVE_CREATE": 0.10,
        "SHOT_3_OD": 0.06,
        "FIRST_STEP": 0.04,
        "SHOT_MID_PU": 0.04,
    },
    "Transition_Handler": {
        "FIRST_STEP": 0.18,
        "DRIVE_CREATE": 0.14,
        "HANDLE_SAFE": 0.14,
        "PASS_CREATE": 0.14,
        "PASS_SAFE": 0.12,
        "ENDURANCE": 0.12,
        "FIN_RIM": 0.10,
        "SHOT_3_CS": 0.06,
    },
    "Shot_Creator": {
        "SHOT_3_OD": 0.24,
        "SHOT_MID_PU": 0.22,
        "HANDLE_SAFE": 0.14,
        "DRIVE_CREATE": 0.14,
        "FIRST_STEP": 0.08,
        "SHOT_FT": 0.07,
        "PNR_READ": 0.06,
        "PASS_CREATE": 0.05,
    },
    "Rim_Attacker": {
        "DRIVE_CREATE": 0.20,
        "FIRST_STEP": 0.16,
        "FIN_RIM": 0.14,
        "FIN_CONTACT": 0.12,
        "SHOT_FT": 0.10,
        "HANDLE_SAFE": 0.10,
        "PASS_CREATE": 0.08,
        "PASS_SAFE": 0.06,
        "SHOT_TOUCH": 0.04,
    },
    "Spacer_CatchShoot": {
        "SHOT_3_CS": 0.45,
        "SHOT_MID_CS": 0.15,
        "PASS_SAFE": 0.10,
        "ENDURANCE": 0.10,
        "HANDLE_SAFE": 0.08,
        "FIRST_STEP": 0.05,
        "FIN_RIM": 0.04,
        "SHOT_FT": 0.03,
    },
    "Spacer_Movement": {
        "SHOT_3_CS": 0.34,   # fixed to make sum exactly 1.0
        "ENDURANCE": 0.18,
        "SHOT_MID_CS": 0.10,
        "FIRST_STEP": 0.08,
        "HANDLE_SAFE": 0.08,
        "PASS_SAFE": 0.08,
        "SHOT_3_OD": 0.05,
        "DRIVE_CREATE": 0.04,
        "SHOT_FT": 0.03,
        "SHOT_TOUCH": 0.02,
    },
    "Connector_Playmaker": {
        "PASS_SAFE": 0.28,
        "PASS_CREATE": 0.22,
        "HANDLE_SAFE": 0.14,
        "PNR_READ": 0.10,
        "SHOT_3_CS": 0.10,
        "DRIVE_CREATE": 0.06,
        "SHORTROLL_PLAY": 0.05,
        "ENDURANCE": 0.05,
    },
    "Roller_Finisher": {
        "FIN_RIM": 0.20,
        "FIN_DUNK": 0.18,
        "FIN_CONTACT": 0.14,
        "PHYSICAL": 0.14,
        "REB_OR": 0.12,
        "ENDURANCE": 0.08,
        "SEAL_POWER": 0.08,
        "SHORTROLL_PLAY": 0.06,
    },
    "ShortRoll_Playmaker": {
        "SHORTROLL_PLAY": 0.28,
        "PASS_SAFE": 0.18,
        "PASS_CREATE": 0.16,
        "HANDLE_SAFE": 0.10,
        "FIN_RIM": 0.10,
        "PNR_READ": 0.06,
        "SHOT_MID_CS": 0.06,
        "PHYSICAL": 0.06,
    },
    "Pop_Spacer_Big": {
        "SHOT_3_CS": 0.32,
        "SHOT_MID_CS": 0.14,
        "PASS_SAFE": 0.14,
        "SHORTROLL_PLAY": 0.10,
        "PHYSICAL": 0.10,
        "HANDLE_SAFE": 0.06,
        "PNR_READ": 0.06,
        "REB_DR": 0.05,
        "ENDURANCE": 0.03,
    },
    "Post_Hub": {
        "POST_CONTROL": 0.22,
        "POST_SCORE": 0.20,
        "PASS_SAFE": 0.16,
        "PASS_CREATE": 0.14,
        "SHOT_TOUCH": 0.08,
        "SEAL_POWER": 0.07,
        "FIN_CONTACT": 0.06,
        "SHOT_MID_CS": 0.04,
        "FIN_RIM": 0.03,
    },
}


# -----------------------------
# Role fit grade cuts (S/A/B/C thresholds)
# (s_min, a_min, b_min, c_min)
# -----------------------------
ROLE_FIT_CUTS: Dict[str, Tuple[int, int, int, int]] = {
    "Initiator_Primary": (80, 72, 64, 56),
    "Initiator_Secondary": (78, 70, 62, 54),
    "Transition_Handler": (75, 67, 59, 51),
    "Shot_Creator": (79, 71, 63, 55),
    "Rim_Attacker": (76, 68, 60, 52),
    "Spacer_CatchShoot": (80, 72, 64, 56),
    "Spacer_Movement": (80, 72, 64, 56),
    "Connector_Playmaker": (78, 70, 62, 54),
    "Roller_Finisher": (75, 67, 59, 51),
    "ShortRoll_Playmaker": (78, 70, 62, 54),
    "Pop_Spacer_Big": (80, 72, 64, 56),
    "Post_Hub": (78, 70, 62, 54),
}
