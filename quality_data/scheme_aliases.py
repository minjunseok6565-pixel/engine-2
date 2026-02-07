# =============================================================================
# [DATA FILE ONLY]  (quality_data 패키지로 분리됨)
# 이 파일은 로직이 아니라 '튜닝 테이블/상수'만 담는 **데이터 모듈**입니다.
# 로직 파일: quality.py
# =============================================================================

from __future__ import annotations

from typing import Dict

# Scheme aliases: accept short english ids too.
SCHEME_ALIASES: Dict[str, str] = {
    "drop": "drop",
    "Drop": "drop",
    "Switch_Everything": "올-스위치",
    "Hedge_ShowRecover": "헷지-쇼앤리커버",
    "Blitz_TrapPnR": "블리츠-트랩",
    "ICE_SidePnR": "아이스 픽앤롤",
    "Zone": "2-3 존디펜스",
    "all_switch": "올-스위치",
    "all-switch": "올-스위치",
    "switch": "올-스위치",
    "올-스위치": "올-스위치",
    "hedge": "헷지-쇼앤리커버",
    "hedge-recover": "헷지-쇼앤리커버",
    "헷지-쇼앤리커버": "헷지-쇼앤리커버",
    "blitz": "블리츠-트랩",
    "trap": "블리츠-트랩",
    "blitz-trap": "블리츠-트랩",
    "블리츠-트랩": "블리츠-트랩",
    "ice": "아이스 픽앤롤",
    "아이스": "아이스 픽앤롤",
    "아이스 픽앤롤": "아이스 픽앤롤",
    "zone": "2-3 존디펜스",
    "2-3": "2-3 존디펜스",
    "2-3 zone": "2-3 존디펜스",
    "2-3 존디펜스": "2-3 존디펜스",
}
