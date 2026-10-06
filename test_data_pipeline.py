# -*- coding: utf-8 -*-
"""데이터 파이프라인 확인 (구현가이드 12장). fold 별 기간과 Sequence shape 를 출력한다.

    python test_data_pipeline.py                     # data/processed/okm_cleaned.csv
    python test_data_pipeline.py data/dummy/okm_dummy.csv
"""

import sys

from src import data_pipeline as dp


def describe(df, split, data):
    def period(mask):
        dts = df.loc[mask, "datetime"]
        return f"{dts.min():%Y-%m-%d %H}시 ~ {dts.max():%Y-%m-%d %H}시"

    print(f"[{split.name}]")
    print(f"  train      {period(split.train)}  X={data['X_train'].shape}")
    print(f"  early_stop {period(split.early_stop)}  X={data['X_es'].shape}")
    print(f"  eval       {period(split.eval)}  X={data['X_eval'].shape}")
    print(f"  peak 기준  {data['peak_threshold']:.1f} kW")


def main():
    df = dp.load_data(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"데이터 {df.shape}, {df['datetime'].min()} ~ {df['datetime'].max()}")
    print(f"셧다운 행 {int(df['공장_셧다운_여부'].sum())}개 (Target 에서 제외: {dp.EXCLUDE_SHUTDOWN_TARGETS})")
    print(f"Feature {len(dp.FEATURE_COLUMNS)}개: {dp.FEATURE_COLUMNS}\n")

    for lookback in (24, 168):
        print(f"----- lookback={lookback} -----")
        for split in dp.get_folds(df) + [dp.get_final_split(df)]:
            describe(df, split, dp.prepare_split(df, split, lookback))
        print()


if __name__ == "__main__":
    main()
