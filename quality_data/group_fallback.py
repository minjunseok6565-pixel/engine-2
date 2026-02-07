# =============================================================================
# [DATA FILE ONLY]  (quality_data 패키지로 분리됨)
# 이 파일은 로직이 아니라 '튜닝 테이블/상수'만 담는 **데이터 모듈**입니다.
# 로직 파일: quality.py
# =============================================================================

from __future__ import annotations

from typing import Dict

# Groups that should reuse an existing role-weight profile.
GROUP_FALLBACK: Dict[str, str] = {
    # Post/Interior outcomes didn't have their own role-weight block in the source doc.
    # They share very similar tasks (T2/T3/T7/T1) with rim_finish.
    "post_interior": "rim_finish",
}
