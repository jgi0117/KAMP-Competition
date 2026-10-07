# -*- coding: utf-8 -*-
"""전처리 → 피처 생성 → (튜닝) → 학습·추론·평가 → 오류분석 → 결과 요약을 한 번에 실행한다 (요구사항 6번).

입력 데이터는 셋 중 하나만 주면 된다. 앞 단계 파일이 있으면 그 단계부터 자동으로 이어진다.
  --raw       원본 okm_augumented_2021.csv       → 정현님 전처리 → 피처 생성 → …
  --cleaned   정제 okm_cleaned_2021.csv          → 정현님 피처 생성 → …
  --features  피처 okm_features2_2021.csv        → 바로 학습·평가 (기본: data/processed/okm_features2_2021.csv)

튜닝
  --tune none (기본)  저장소의 1차 그리드 서치 결과(results/grid_search/*.csv)에서 최종 설정을 읽어 바로 학습한다.
  --tune full         1차 그리드 서치(315회)를 처음부터 다시 실행한다. GPU 권장 (Colab T4 약 5시간).

결과 (--out, 기본 results/reproduce)
  data/okm_features2_2021.csv  생성한 피처 데이터
  final/                        최종 Test 비교표·시간별 예측·이상 확률 (final_evaluate.py)
  final/analysis_*/             영향요인·FN/FP 분석 그림과 표 (analysis/error_analysis.py)
  run_manifest.json             실행 환경·데이터 지문·단계별 시간·결과 요약
  REPORT.md                     최종 비교표와 오류분석 요약

사용 예 (저장소 최상위에서)
  python run_all.py                                        # 기본: 저장된 피처 파일로 최종 평가·분석 (CPU 약 2.5시간)
  python run_all.py --cleaned data/processed/okm_cleaned_2021.csv
  python run_all.py --raw data/raw/okm_augumented_2021.csv --tune full
  python run_all.py --quick                                # 동작 확인용 (epoch 1번, 몇 분)
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# 팀이 실제로 쓴 피처 파일의 지문. 같은 데이터로 재현되는지 확인하는 데 쓴다.
EXPECTED_FEATURES_MD5 = "65bb158c064dffdbe81ca2f102c4195c"
EXPECTED_CLEANED_MD5 = "45f66ed55cfbb12432bbec31be201daa"
DEFAULT_FEATURES = ROOT / "data" / "processed" / "okm_features2_2021.csv"
MODELS = ["lstm", "tcn", "tcn_lstm"]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def banner(text):
    print(f"\n{'=' * 70}\n {text}\n{'=' * 70}", flush=True)


def run(cmd, log):
    """하위 스크립트를 실행하고 출력을 화면과 로그에 함께 남긴다."""
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with subprocess.Popen([str(c) for c in cmd], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace", env=env) as p, \
            open(log, "a", encoding="utf-8") as f:
        for line in p.stdout:
            if any(s in line for s in ("absl::", "oneDNN custom", "I0000 ", "GPU support is not available")):
                continue
            print(line, end="", flush=True)
            f.write(line)
    if p.returncode != 0:
        raise SystemExit(f"[실패] {' '.join(str(c) for c in cmd)} (종료 코드 {p.returncode}). 로그: {log}")


def versions():
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name, mod in [("tensorflow", "tensorflow"), ("keras", "keras"), ("numpy", "numpy"), ("pandas", "pandas"),
                      ("scikit-learn", "sklearn"), ("scipy", "scipy"), ("matplotlib", "matplotlib")]:
        try:
            out[name] = __import__(mod).__version__
        except Exception:
            out[name] = None
    try:
        out["git_commit"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        out["git_commit"] = None
    return out


# =========================================================
# 1~2. 전처리 · 피처 생성 (정현님 코드: src/team_kjh)
# =========================================================

def prepare_features(args, out):
    data_dir = out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    info = {}

    if args.raw:
        banner("1단계: 원본 → 정제 (정현님 전처리, src/team_kjh/preprocessing.py)")
        from src.team_kjh.preprocessing import run_preprocessing_pipeline
        cleaned_path = data_dir / "okm_cleaned_2021.csv"
        run_preprocessing_pipeline(data_path=Path(args.raw), save_output=True, output_path=cleaned_path)
        info["raw"] = {"path": str(args.raw), "md5": md5(args.raw)}
        args.cleaned = cleaned_path

    if args.cleaned:
        banner("2단계: 정제 → 피처 (정현님 피처 생성, src/team_kjh/features.py)")
        import pandas as pd
        from src.team_kjh.features import build_feature_pipeline
        cmd5 = md5(args.cleaned)
        info["cleaned"] = {"path": str(args.cleaned), "md5": cmd5, "matches_team_file": cmd5 == EXPECTED_CLEANED_MD5}
        cleaned = pd.read_csv(args.cleaned, encoding="utf-8-sig")
        feats = build_feature_pipeline(cleaned)
        features_path = data_dir / "okm_features2_2021.csv"
        feats.to_csv(features_path, index=False, encoding="utf-8-sig")
        check_same_as_team(features_path, info)
    else:
        features_path = Path(args.features)
        if not features_path.exists():
            raise SystemExit(f"피처 파일이 없습니다: {features_path}\n"
                             "--features, --cleaned, --raw 중 하나로 데이터 위치를 지정하세요.")
        banner("1~2단계 생략: 준비된 피처 파일 사용")
        shutil.copy2(features_path, data_dir / "okm_features2_2021.csv")
        features_path = data_dir / "okm_features2_2021.csv"
        check_same_as_team(features_path, info)

    info["features"]["path"] = str(features_path)
    return features_path, info


def check_same_as_team(path, info):
    """생성·복사한 피처가 팀이 실험에 쓴 파일과 같은 값인지 확인한다 (바이트가 아니라 값 비교)."""
    import numpy as np
    import pandas as pd
    m = md5(path)
    same_bytes = m == EXPECTED_FEATURES_MD5
    same_values = None
    if not same_bytes and DEFAULT_FEATURES.exists() and Path(path).resolve() != DEFAULT_FEATURES.resolve():
        a = pd.read_csv(path, encoding="utf-8-sig")
        b = pd.read_csv(DEFAULT_FEATURES, encoding="utf-8-sig")
        same_values = list(a.columns) == list(b.columns) and all(
            np.allclose(a[c].astype(float), b[c].astype(float), equal_nan=True, atol=1e-6)
            if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])
            else (a[c].astype(str).values == b[c].astype(str).values).all()
            for c in a.columns)
    info["features"] = {"md5": m, "same_file_as_team": same_bytes, "same_values_as_team": same_values}
    if same_bytes or same_values:
        print(f"[확인] 피처 데이터가 팀 실험 데이터와 같습니다 (md5 {m}).")
    else:
        print(f"[주의] 피처 데이터 지문이 팀 실험 데이터와 다릅니다 (md5 {m}). 결과 수치가 달라질 수 있습니다.")


# =========================================================
# 메인
# =========================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--raw", help="원본 okm_augumented_2021.csv")
    src.add_argument("--cleaned", help="정제 okm_cleaned_2021.csv")
    src.add_argument("--features", default=str(DEFAULT_FEATURES), help="피처 okm_features2_2021.csv")
    ap.add_argument("--tune", choices=["none", "full"], default="none",
                    help="none: 저장된 튜닝 결과 사용 (기본) / full: 1차 그리드 서치부터 다시 (GPU 권장)")
    ap.add_argument("--out", default=str(ROOT / "results" / "reproduce"), help="결과 폴더")
    ap.add_argument("--quick", action="store_true", help="동작 확인용: epoch 1번 (수치는 의미 없음)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "run_all.log"
    log.write_text(f"run_all.py 시작 {datetime.now():%Y-%m-%d %H:%M:%S}\n", encoding="utf-8")
    epochs = ["--max-epochs", "1"] if args.quick else []
    manifest = {"started": datetime.now().isoformat(timespec="seconds"), "args": vars(args),
                "environment": versions(), "seed": 42, "peak_threshold_kw": 177.0, "steps": {}}
    t0 = time.time()

    # 1~2. 데이터
    t = time.time()
    features_path, data_info = prepare_features(args, out)
    manifest["data"] = data_info
    manifest["steps"]["data"] = round(time.time() - t, 1)

    # 3. 튜닝
    grid_dir = ROOT / "results" / "grid_search"
    if args.tune == "full":
        banner("3단계: 1차 그리드 서치 (lookback → 구조 → lr·dropout → batch, fold 3개, seed 42)")
        grid_dir = out / "grid_search"
        t = time.time()
        for m in MODELS:
            for stage in ["1", "2", "3", "4"]:
                run([sys.executable, "-u", "grid_search.py", "--model", m, "--stage", stage,
                     "--data", features_path, "--results-dir", grid_dir, *epochs], log)
        manifest["steps"]["tune"] = round(time.time() - t, 1)
    else:
        banner("3단계 생략: 저장된 1차 그리드 서치 결과 사용 (results/grid_search)")
        missing = [m for m in MODELS if not (grid_dir / f"{m}.csv").exists()]
        if missing:
            raise SystemExit(f"튜닝 결과가 없습니다: {missing}. --tune full 로 실행하세요.")
    manifest["grid_dir"] = str(grid_dir)

    # 4. 학습·추론·평가
    banner("4단계: 최종 학습·추론·평가 (검증 fold 3개 + Test, 베이스라인 포함, 이상 확률·경보)")
    final_dir = out / "final"
    t = time.time()
    run([sys.executable, "-u", "final_evaluate.py", "--data", features_path, "--grid-dir", grid_dir,
         "--out-dir", final_dir, *epochs], log)
    manifest["steps"]["evaluate"] = round(time.time() - t, 1)

    # 5. 오류분석
    banner("5단계: 영향요인·오류분석 (Ensemble FN/FP, LSTM·TCN 영향 변수)")
    t = time.time()
    for m in ["ensemble", "lstm", "tcn"]:
        run([sys.executable, "-u", "analysis/error_analysis.py", "--pred-dir", final_dir, "--data", features_path,
             "--model", m], log)
    manifest["steps"]["analysis"] = round(time.time() - t, 1)

    # 6. 결과 요약
    banner("6단계: 결과 요약 (REPORT.md, run_manifest.json)")
    import pandas as pd
    s = pd.read_csv(final_dir / "test_summary.csv", encoding="utf-8-sig")
    cols = [("mse_mean", "MSE", 1), ("rmse_mean", "RMSE", 2), ("r2_mean", "R²", 3), ("mae_mean", "MAE", 2),
            ("alert_f1_mean", "F1", 3), ("alert_recall_mean", "Recall", 3), ("alert_precision_mean", "Precision", 3),
            ("pr_auc_mean", "PR-AUC", 3), ("fn_mean", "FN", 0), ("fp_mean", "FP", 0)]
    lines = ["# 재현 실행 결과", "",
             f"- 실행: {manifest['started']} · 소요 {round((time.time() - t0) / 60, 1)}분 · "
             f"커밋 {manifest['environment']['git_commit']} · {'동작 확인(--quick)' if args.quick else '정식 실행'}",
             f"- 데이터 md5 {data_info['features']['md5']} "
             f"({'팀 실험 데이터와 동일' if data_info['features']['same_file_as_team'] or data_info['features']['same_values_as_team'] else '팀 실험 데이터와 다름'})",
             "- Test 2021-08-16 ~ 09-14 (셧다운 제외), 이상 기준 177 kW, seed 42", "",
             "| 모델 | " + " | ".join(c[1] for c in cols) + " |", "| --- |" + " --- |" * len(cols)]
    for _, r in s.iterrows():
        lines.append(f"| {r['label']} | " + " | ".join(f"{r[c]:.{d}f}" for c, _, d in cols) + " |")
    lines += ["", "## 오류분석 요약 (Weighted Ensemble)", ""]
    ens = final_dir / "analysis_ensemble" / "summary.md"
    if ens.exists():
        lines += ens.read_text(encoding="utf-8").splitlines()[2:]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    manifest["finished"] = datetime.now().isoformat(timespec="seconds")
    manifest["minutes_total"] = round((time.time() - t0) / 60, 1)
    manifest["test_results"] = s[["model", "rmse_mean", "r2_mean", "alert_f1_mean", "alert_recall_mean"]].round(4).to_dict("records")
    (out / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\n".join(lines[:len(cols) + 8]))
    print(f"\n[완료] 결과 폴더: {out}  (REPORT.md, run_manifest.json, final/, data/)")


if __name__ == "__main__":
    main()
