"""데이터 로딩 유틸리티 모듈.

다양한 인코딩(utf-8 -> cp949 -> utf-8-sig) 순차 시도, 대용량 파일 샘플링 로드,
전체 파일 탐색 기능을 제공합니다.
"""

from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import pandas as pd
from .config import DATA_DIR


def list_data_files(dir_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """지정된 디렉터리(기본: DATA_DIR) 내의 모든 데이터 파일 목록과 기본 메타데이터를 반환합니다.

    Args:
        dir_path (Path, optional): 탐색할 디렉터리. 기본값은 DATA_DIR.

    Returns:
        List[Dict[str, Any]]: 파일 경로, 상대 경로, 확장자, 파일 크기(MB) 등을 담은 딕셔너리 리스트.
    """
    target_dir = dir_path if dir_path is not None else DATA_DIR
    if not target_dir.exists():
        print(f"[load] 디렉터리를 찾을 수 없습니다: {target_dir}")
        return []

    files_info = []
    # 하위 폴더까지 재귀적으로 탐색
    for file_path in target_dir.rglob("*"):
        if file_path.is_file():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            files_info.append(
                {
                    "path": file_path,
                    "rel_path": str(file_path.relative_to(target_dir)),
                    "filename": file_path.name,
                    "extension": file_path.suffix.lower(),
                    "size_mb": round(size_mb, 2),
                }
            )
    return files_info


def read_file_safe(
    file_path: Path,
    sample_rows: Optional[int] = None,
    encodings: Tuple[str, ...] = ("utf-8", "cp949", "utf-8-sig"),
    **kwargs,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """지정된 인코딩 순서(utf-8 -> cp949 -> utf-8-sig)로 파일 읽기를 시도합니다.

    Args:
        file_path (Path): 읽을 파일 경로.
        sample_rows (int, optional): 대용량 파일의 경우 읽을 상위 행 수 (None이면 전체 로드).
        encodings (Tuple[str, ...]): 시도할 인코딩 목록.
        **kwargs: pandas read 함수에 전달할 추가 키워드 인자.

    Returns:
        Tuple[Optional[pd.DataFrame], Optional[str]]: (로드된 DataFrame, 성공한 인코딩 이름). 실패 시 (None, None).
    """
    ext = file_path.suffix.lower()

    # CSV 파일 읽기
    if ext in [".csv", ".txt"]:
        for enc in encodings:
            try:
                df = pd.read_csv(file_path, encoding=enc, nrows=sample_rows, **kwargs)
                return df, enc
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                print(f"[load] {file_path.name} 로드 중 오류 ({enc}): {e}")
                break
        return None, None

    # Excel 파일 읽기
    elif ext in [".xlsx", ".xls"]:
        try:
            df = pd.read_excel(file_path, nrows=sample_rows, **kwargs)
            return df, "excel_engine"
        except Exception as e:
            print(f"[load] Excel 로드 오류 ({file_path.name}): {e}")
            return None, None

    # Parquet 파일 읽기
    elif ext in [".parquet", ".pq"]:
        try:
            df = pd.read_parquet(file_path, **kwargs)
            if sample_rows is not None:
                df = df.head(sample_rows)
            return df, "parquet"
        except Exception as e:
            print(f"[load] Parquet 로드 오류 ({file_path.name}): {e}")
            return None, None

    else:
        print(f"[load] 지원하지 않는 파일 형식입니다: {ext}")
        return None, None


if __name__ == "__main__":
    files = list_data_files()
    print(f"[load] 발견된 파일 수: {len(files)}")
    for f in files[:5]:
        print(f" - {f['rel_path']} ({f['size_mb']} MB)")
