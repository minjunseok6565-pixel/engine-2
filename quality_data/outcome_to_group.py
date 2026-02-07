# =============================================================================
# [DATA FILE ONLY]  (quality_data 패키지로 분리됨)
# 이 파일은 로직이 아니라 '튜닝 테이블/상수'만 담는 **데이터 모듈**입니다.
# 로직 파일: quality.py
# =============================================================================

from __future__ import annotations

from typing import Dict

# outcome -> group_id (used to choose the role-weight profile)
OUTCOME_TO_GROUP: Dict[str, str] = {'FOUL_DRAW_JUMPER': 'pullup',
 'FOUL_DRAW_POST': 'post_interior',
 'FOUL_DRAW_RIM': 'rim_finish',
 'PASS_EXTRA': 'pass_extra',
 'PASS_KICKOUT': 'pass_kickout',
 'PASS_SHORTROLL': 'pass_shortroll',
 'PASS_SKIP': 'pass_skip',
 'SHOT_3_CS': 'catch_shoot',
 'SHOT_3_OD': 'pullup',
 'SHOT_MID_CS': 'catch_shoot',
 'SHOT_MID_PU': 'pullup',
 'SHOT_POST': 'post_interior',
 'SHOT_RIM_CONTACT': 'rim_finish',
 'SHOT_RIM_DUNK': 'rim_finish',
 'SHOT_RIM_LAYUP': 'rim_finish',
 'SHOT_TOUCH_FLOATER': 'paint_non_rim',
 'TO_BAD_PASS': 'pass_skip',
 'TO_CHARGE': 'to_charge',
 'TO_HANDLE_LOSS': 'to_handle_loss'}
