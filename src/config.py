"""설정 및 디렉터리 경로 관리 모듈.

.env 파일에서 환경 변수를 로드하여 프로젝트 전반에서 일관된 경로 객체(pathlib.Path)를 제공합니다.
"""

from pathlib import Path
import os
import sys
from dotenv import load_dotenv

# 프로젝트 루트 디렉터리 (src의 상위 디렉터리)
ROOT_DIR = Path(__file__).resolve().parent.parent

# .env 파일 로드
ENV_PATH = ROOT_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()  # 시스템 환경 변수 또는 상위 디렉터리 탐색

# feature engineering 입출력에 사용하는 프로젝트 로컬 데이터 디렉터리
LOCAL_DATA_DIR = ROOT_DIR / "data"

# 원본 데이터 디렉터리 경로 (.env의 DATA_DIR 우선)
_env_data_dir = os.getenv("DATA_DIR")
if _env_data_dir:
    DATA_DIR = Path(_env_data_dir).expanduser().resolve()
else:
    # 기본 폴백 경로: 레포지토리 외부 또는 로컬 data 폴더 예시
    DATA_DIR = ROOT_DIR.parent / "data"

# 보고서 및 시각화 저장 디렉터리
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# 디렉터리 자동 생성 (reports, reports/figures)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def check_data_dir() -> bool:
    """DATA_DIR이 실제로 존재하고 읽을 수 있는지 확인하고 상태를 출력합니다."""
    print(f"[config] 프로젝트 루트: {ROOT_DIR}")
    print(f"[config] 원본 데이터 경로: {DATA_DIR}")
    
    if not DATA_DIR.exists():
        print(
            f"[경고] DATA_DIR 디렉터리가 존재하지 않습니다: {DATA_DIR}\n"
            f"       .env 파일에서 DATA_DIR=<실제_경로>로 수정해 주세요.",
            file=sys.stderr,
        )
        return False
    print(f"[config] 원본 데이터 경로 확인 완료 (디렉터리 존재함)")
    return True


if __name__ == "__main__":
    check_data_dir()
