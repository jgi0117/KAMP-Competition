"""Build the competition source ZIP from an explicit, reviewable file set."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "OKM_소스코드_제출.zip"
PACKAGE = "OKM_소스코드_제출"
DIRECTORIES = (
    "data", "src", "model_search", "neural", "scripts", "dashboard",
    "notebooks/kjh", "results", "docs/templates", "docs/report/evidence",
    "docs/report/figures",
)
SINGLE_FILES = ("requirements.txt", "docs/model-comparison.md")
EXCLUDED_NAMES = {
    "__pycache__", ".pytest_cache", "mock_data.py", "design_system.md",
    "INTEGRATION_GUIDE.md",
}


README = """# OKM 전력 피크 예측 — 소스코드 제출물

Python 3.12와 Windows에서 확인한 실행 환경입니다. 모든 명령은 이 README가 있는 폴더에서 실행합니다.

## 포함 파일

- `requirements.txt`: 전처리·학습·평가·대시보드·보고서 공통 환경
- `data/okm_augumented_2021.csv`: 원본 6,168시간 자료
- `data/okm_cleaned_2021.csv`: 전처리 결과 6,168행 × 22열
- `src/`, `notebooks/kjh/`: 전처리 코드와 EDA 재현 노트북
- `model_search/`, `neural/`, `scripts/`: 9개 공통 입력의 모델 탐색·학습·평가 코드
- `results/`, `neural/results/`: 저장된 그리드 탐색, 모델, 검증·Test 예측과 지표
- `dashboard/`: 저장된 Test 703시간을 재생하는 Dash 대시보드
- `test_predictions.csv`: 다섯 모델의 동일 Test 703시간 실측·예측·경보 확률 통합표
- `docs/templates/`, `docs/report/evidence/`, `docs/report/figures/`: 결과보고서 재생성용 양식과 근거
- `manifest.json`: ZIP 내부 파일의 SHA-256 해시와 크기

## 빠른 재현

```powershell
python -m pip install -r requirements.txt
python scripts/verify_cleaned_data.py
python scripts/build_model_comparison.py
python scripts/reproduce_submission.py
python dashboard/app.py
```

대시보드는 `http://127.0.0.1:8050`에서 열립니다. 저장된 2021년 Test 결과를 한 시간씩 재생합니다. `Ctrl+C`로 종료합니다.

`reproduce_submission.py`는 제공한 raw 자료와 cleaned 자료의 일치 여부, 저장된 다섯 모델 결과의 시각·실측값 일치, 근거표·그림·HWPX 생성과 HWPX 구조 검증을 차례로 수행합니다. 생성 보고서는 `docs/report/OKM_경진대회_결과보고서_작성본.hwpx`에 저장됩니다.

## 학습을 처음부터 실행할 때

```powershell
foreach ($model in @('lstm', 'tcn')) {
  foreach ($stage in 1..4) {
    python neural/grid_search.py --model $model --stage $stage
  }
}
python neural/final_evaluate.py --models lstm tcn
python scripts/train_tree_models.py --models lightgbm xgboost --force-search --n-jobs 8
python scripts/build_model_comparison.py
```

신경망 탐색과 트리 전체 그리드 재학습에는 상당한 시간이 필요합니다. 제공된 검증·Test 예측과 탐색 CSV로 보고서와 대시보드는 즉시 재현할 수 있습니다. 저장된 트리 모델 파일은 `results/tree_models/models/`에 있습니다.

브라우저 동작을 검증하려면 Chrome 또는 Playwright Chromium 설치 후 `python scripts/verify_dashboard.py`를 실행합니다. 네 화면을 다시 촬영하려면 `python scripts/capture_dashboard.py`를 실행합니다.
"""


def source_files() -> list[Path]:
    paths = []
    for name in DIRECTORIES:
        directory = ROOT / name
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        paths.extend(
            path for path in directory.rglob("*")
            if path.is_file()
            and not any(part in EXCLUDED_NAMES for part in path.relative_to(ROOT).parts)
            and path.suffix not in {".pyc", ".pyo"}
        )
    for name in SINGLE_FILES:
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)
    return sorted(set(paths), key=lambda path: path.relative_to(ROOT).as_posix())


def test_predictions() -> bytes:
    base = pd.read_csv(ROOT / "neural/results/final/test_predictions.csv")
    if len(base) != 703 or base.datetime.duplicated().any():
        raise ValueError("Expected 703 unique Test timestamps")
    columns = {
        "actual": "actual_kw", "is_peak": "actual_peak",
        "pred_lstm_seed42": "lstm_pred_kw", "prob_lstm_seed42": "lstm_peak_probability",
        "pred_tcn_seed42": "tcn_pred_kw", "prob_tcn_seed42": "tcn_peak_probability",
        "pred_ensemble_seed42": "ensemble_pred_kw",
        "prob_ensemble_seed42": "ensemble_peak_probability",
    }
    result = base.rename(columns=columns)
    for model in ("xgboost", "lightgbm"):
        tree = pd.read_csv(ROOT / f"results/tree_models/predictions/{model}_test.csv")
        if not pd.to_datetime(tree.datetime).equals(pd.to_datetime(base.datetime)):
            raise ValueError(f"{model} Test timestamps differ")
        if not np.allclose(tree.actual, base.actual, atol=1e-6):
            raise ValueError(f"{model} Test actual values differ")
        result[f"{model}_pred_kw"] = tree.predicted.to_numpy()
        result[f"{model}_peak_probability"] = tree.probability.to_numpy()
    result["ensemble_alert"] = result.ensemble_peak_probability.ge(0.2)
    result["peak_threshold_kw"] = 177.0
    if result.isna().any().any():
        raise ValueError("Unified Test predictions contain missing values")
    return result.to_csv(index=False).encode("utf-8-sig")


def build() -> Path:
    payloads = {path.relative_to(ROOT).as_posix(): path.read_bytes() for path in source_files()}
    payloads["README.md"] = README.encode("utf-8")
    payloads["test_predictions.csv"] = test_predictions()
    required = {
        "requirements.txt", "data/okm_augumented_2021.csv", "data/okm_cleaned_2021.csv",
        "src/preprocessing.py", "dashboard/app.py", "results/model_comparison.csv",
        "neural/results/final/test_predictions.csv", "test_predictions.csv", "README.md",
    }
    if not required <= payloads.keys():
        raise ValueError(f"Required submission files missing: {required - payloads.keys()}")
    manifest = {
        "package": PACKAGE,
        "files": {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                  for name, data in sorted(payloads.items())},
    }
    payloads["manifest.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(payloads.items()):
            info = zipfile.ZipInfo(f"{PACKAGE}/{name}", date_time=(2021, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    with zipfile.ZipFile(OUTPUT) as archive:
        if archive.testzip() is not None:
            raise ValueError("Source ZIP contains a corrupt member")
    print(f"{OUTPUT} ({len(payloads)} files, {OUTPUT.stat().st_size:,} bytes)")
    return OUTPUT


if __name__ == "__main__":
    build()
