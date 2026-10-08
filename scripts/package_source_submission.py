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
    "data", "src", "model_search", "neural", "dashboard",
    "notebooks/kjh", "results", "docs/report/evidence",
)
SINGLE_FILES = (
    "requirements.txt", "docs/model-comparison.md",
    "scripts/verify_cleaned_data.py", "scripts/train_tree_models.py",
    "scripts/build_model_comparison.py", "scripts/build_report_evidence.py",
    "scripts/analyze_feature_importance.py", "scripts/build_eda_evidence.py",
    "scripts/verify_dashboard.py", "scripts/capture_dashboard.py",
    "scripts/package_source_submission.py",
)
EXCLUDED_NAMES = {
    "__pycache__", ".pytest_cache", "mock_data.py", "design_system.md",
    "INTEGRATION_GUIDE.md",
}


README = r"""# OKM 전력 피크 예측 — 소스코드 제출물

**확인한 환경: Windows, Python 3.12.x.** 모든 명령은 이 README가 있는 폴더에서 실행합니다.

## 포함 파일

- `requirements.txt`: 전처리·학습·평가·대시보드 공통 환경
- `data/okm_augumented_2021.csv`: 원본 6,168시간 자료
- `data/okm_cleaned_2021.csv`: 전처리 결과 6,168행 × 22열
- `src/`, `notebooks/kjh/`: 전처리 코드와 EDA 재현 노트북
- `model_search/`, `neural/`, `scripts/`: 9개 공통 입력의 모델 탐색·학습·평가 코드
- `results/`, `neural/results/`: 저장된 그리드 탐색, 모델, 검증·Test 예측과 지표
- `dashboard/`: 저장된 Test 703시간을 재생하는 Dash 대시보드
- `test_predictions.csv`: 다섯 모델의 동일 Test 703시간 실측·예측·경보 확률 통합표
- `docs/report/evidence/`: 대시보드와 오류 분석에 쓰는 검증 근거
- `manifest.json`: ZIP 내부 파일의 SHA-256 해시와 크기

## 1. Python 3.12 환경 만들기

```powershell
py -3.12 --version
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install -r requirements.txt
```

`python --version`이 `Python 3.12.x`로 나오는지 확인합니다.

## 2. 전처리부터 결과·대시보드까지 확인

```powershell
python scripts/verify_cleaned_data.py
python scripts/build_model_comparison.py
python scripts/build_report_evidence.py
python scripts/analyze_feature_importance.py
python scripts/build_eda_evidence.py
python dashboard/app.py
```

첫 명령은 raw CSV로 정제 결과를 다시 만들어 제공한 cleaned CSV의 모든 열을 비교합니다. 다음 명령은 다섯 모델의 동일 Test 시각·실측값과 탐색 결과를 검사해 `results/model_comparison.csv`를 갱신합니다. 나머지 명령은 진단 근거·EDA 그림을 만들고 Dash를 실행합니다.

대시보드는 `http://127.0.0.1:8050`에서 열립니다. 저장된 2021년 Test 703시간을 한 시간씩 재생합니다. `Ctrl+C`로 종료합니다. 제출된 Test 예측은 루트의 `test_predictions.csv`와 모델별 `neural/results/final/`, `results/tree_models/predictions/`에 있습니다.

브라우저 자동 검증을 실행할 때는 `python -m playwright install chromium`으로 브라우저를 설치한 뒤 `python scripts/verify_dashboard.py`를 실행합니다. 관제·기여도·모델 성능·오류 검토 4개 화면과 1시간 선행 예측 시각을 검사합니다.

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

신경망 탐색과 트리 전체 그리드 재학습에는 상당한 시간이 필요합니다. 제공된 검증·Test 예측과 탐색 CSV로 결과 비교와 대시보드는 즉시 재현할 수 있습니다. 저장된 트리 모델 파일은 `results/tree_models/models/`에 있습니다.

네 화면을 다시 촬영하려면 `python scripts/capture_dashboard.py`를 실행합니다. 결과 PNG는 `docs/report/figures/`에 생성됩니다.
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
    if any(name.endswith(".hwpx") or name.endswith("fill_result_report.py")
           for name in payloads):
        raise ValueError("The source submission contains a report-generation artifact")
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
