from __future__ import annotations


TARGET_COLUMN = "전력_평균_실수"
DATE_COLUMN = "날짜"
HOUR_COLUMN = "시간"

# Every candidate receives these same past-only inputs in this exact order.
FEATURE_COLUMNS = (
    TARGET_COLUMN,
    HOUR_COLUMN,
    "15분",
    "30분",
    "45분",
    "60분",
    "평균",
    "생산량",
    "기온",
    "풍속",
    "습도",
    "강수량",
    "전기요금(계절)",
    "요일",
    "일",
    "월",
    "공장인원",
    "인건비",
    "시간_복원여부",
    "공장_셧다운_여부",
)
