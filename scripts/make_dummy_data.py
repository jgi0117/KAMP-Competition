# -*- coding: utf-8 -*-
"""코드 동작 확인용 가짜 데이터 생성기.

실제 데이터(정현님 전처리 결과)와 같은 컬럼·기간·대략적인 패턴만 흉내 낸다.
여기서 나온 성능 수치는 아무 의미가 없다.

    python scripts/make_dummy_data.py   →  data/dummy/okm_dummy.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "dummy" / "okm_dummy.csv"


def main():
    rng = np.random.default_rng(0)
    dt = pd.date_range("2021-01-01", "2021-09-14 23:00", freq="h")
    n = len(dt)
    hour, dow = dt.hour.to_numpy(), dt.dayofweek.to_numpy()

    working = (dow < 5) & (hour >= 8) & (hour <= 17) & (hour != 12)
    production = np.where(working, rng.integers(200, 1500, n), 0)
    temp = 15 - 12 * np.cos(2 * np.pi * (dt.dayofyear.to_numpy() - 15) / 365) \
        + 4 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 1.5, n)

    base = np.where(working, 130, 25) + 0.01 * production + 1.5 * np.clip(temp - 22, 0, None)
    quarters = np.clip(base[:, None] + rng.normal(0, 8, (n, 4)), 0, None).round()

    shutdown = (dt >= "2021-08-28 18:00") & (dt <= "2021-08-29 10:00")
    quarters[shutdown] = 0

    df = pd.DataFrame({
        "날짜": dt.strftime("%Y-%m-%d"),
        "시간": hour,
        "15분": quarters[:, 0], "30분": quarters[:, 1],
        "45분": quarters[:, 2], "60분": quarters[:, 3],
        "생산량": production,
        "기온": temp.round(1),
        "풍속": rng.uniform(0, 5, n).round(1),
        "습도": rng.integers(20, 95, n),
        "강수량": np.where(rng.random(n) < 0.05, rng.uniform(0, 10, n), 0).round(1),
        "요일": dow + 1,
    })
    df["전력_평균_실수"] = df[["15분", "30분", "45분", "60분"]].mean(axis=1)
    df["공장_셧다운_여부"] = shutdown

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"저장: {OUT} ({len(df)}행)")


if __name__ == "__main__":
    main()
