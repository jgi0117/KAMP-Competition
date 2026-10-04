# KAMP-Competition

2026년 제6회 K-인공지능(AI) 제조데이터 분석 경진대회 참가 프로젝트입니다.

## 대회 개요

KAMP 제조AI데이터셋으로 중소 제조기업의 문제를 해결하거나 개선하는 AI 분석 모델을 개발하는 대회입니다. 장비 이상 조기 탐지, 품질 이상 탐지·진단, 공정 운영 최적화 등이 주제 예시입니다.

| 항목 | 내용 |
| --- | --- |
| 주최 | 중소벤처기업부 |
| 주관 | 중소기업기술정보진흥원, 포항공과대학교, 이노비즈협회 |
| 신청 기간 | 2026년 8월 24일 ~ 9월 17일 |
| 참가 자격 | 만 19세 이상 대한민국 국민, 팀당 최대 3명 |
| 데이터 | KAMP 등재 제조AI데이터셋 50종 중 1종 또는 여러 종을 활용해 출제 예정 |
| 개발 환경 | Python, Anaconda·Jupyter Notebook; KAMP-NOTE 또는 KAMP AI-PaaS 활용 권장 |
| 시상 | 총 8팀, 총상금 3,800만 원 |

## 참가 트랙 및 평가

| 트랙 | 대상 | 평가 중점 |
| --- | --- | --- |
| 일반 국민·대학(원)생 | 대학(원)생, 개발자, 데이터 분석가, AI 관심자 등 | 창의성, 분석 성능, 데이터 활용도 |
| 중소·중견기업 재직자 | 팀원 모두 중소·중견기업 재직자 | 현장 적용성, 문제해결성, 실효성, 확산 가능성 |

각 트랙에서 대상·최우수상·우수상·장려상 각 1팀을 선정합니다.

## 신청 방법

1. KAMP 포털 로그인 후 **알림마당 → KAMP 경진대회**에서 신청합니다.
2. 참가신청서, 참가자 서약서, 개인정보 동의서를 포털에서 작성합니다.
3. 재학·재직증명서 등 신분 증빙서류를 팀별 PDF로 취합하여 제출합니다. 주민등록번호 뒷자리는 가립니다.

신청 문의·기술지원은 **9월 17일 18시까지**입니다. 이는 지원 종료 시각이며, 접수 마감 시각은 공식 공고에서 확인해야 합니다.

## 브랜치

| 브랜치 | 용도 |
| --- | --- |
| `main` | 공통 기준 브랜치 |
| `lhs` | 개별 작업 브랜치 |
| `kjh` | 개별 작업 브랜치 |
| `kgj` | 개별 작업 브랜치 |

## 참고 자료

- [KAMP 공식 대회 안내](https://www.kamp-ai.kr/contestDetail?CPT_SEQ=38)
- [부산외국어대학교 대회 공고](https://www.bufs.ac.kr/bbs/board.php?bo_table=startup_board4&wr_id=1078)
- [위비티 대회 안내](https://www.wevity.com/?c=find&gbn=viewok&gp=28&ix=110622&s=1)

2026년 9월 9일 확인 기준입니다. 공식 페이지의 접근 제한으로 공개된 재게시 공고를 바탕으로 요약했습니다. 세부 진행 일정, 제출물, 평가 지표는 공식 공고와 후속 안내를 확인하세요.

---

## 프로젝트 디렉터리 구조

```text
KAMP-Competition/
├── .vscode/
│   └── extensions.json         # VS Code 권장 확장 (Python, Jupyter)
├── notebooks/                  # 개인별 작업 노트북
│   └── kjh/                    # kjh 브랜치 분석 노트북
│       └── 01_data_overview.ipynb
├── src/                        # 공통 재사용 소스 코드
│   ├── __init__.py
│   ├── config.py               # .env 기반 데이터 및 프로젝트 경로 관리 (pathlib)
│   ├── load.py                 # 인코딩 자동 감지 및 안전한 데이터 로딩 유틸리티
│   └── viz.py                  # 한글 폰트(Malgun Gothic) 및 마이너스 기호 깨짐 방지 설정
├── reports/                    # 분석 산출물 (수치 요약 및 그래프)
│   ├── column_summary.csv      # 컬럼 프로파일 통계 요약표 (원본 데이터 미포함)
│   └── figures/                # 결측치 시각화 및 주요 플롯
├── .env.example                # 환경 변수 예시 파일 (DATA_DIR 경로 설정 안내)
├── .env                        # 로컬 환경 변수 설정 (Git 커밋 금지)
├── .gitignore                  # 원본 데이터, .env, 대용량 파일 등 추적 제외
├── .gitattributes              # 개행 문자 자동 처리 (* text=auto)
├── environment.yml             # 동일 분석 환경 재현용 Conda 설정 파일
└── README.md                   # 프로젝트 및 환경 세팅 가이드
```

### 각 디렉터리 용도
- **`src/`**: 노트북 및 스크립트 전반에서 재사용할 모듈(설정, 데이터 로딩, 한글 시각화 등)을 정의합니다.
- **`notebooks/`**: 팀원별 EDA 및 실험 노트북을 관리합니다. (Git에는 `nbstripout`을 통해 결과 셀이 자동으로 비워진 상태로 깔끔하게 커밋됩니다.)
- **`reports/`**: 원본 데이터는 레포에 포함되지 않으므로, 분석 결과 통계치(`column_summary.csv`)와 그림(`figures/`)만 버전 관리합니다.

---

## 팀원용 환경 재현 방법 (Quickstart)

모든 팀원은 동일한 Python 및 패키지 버전을 사용하여 환경 차이로 인한 버그를 방지합니다.

1. **저장소 복제 및 브랜치 이동**
   ```bash
   git clone https://github.com/jgi0117/KAMP-Competition.git
   cd KAMP-Competition
   git checkout kjh   # 또는 본인 작업 브랜치
   ```

2. **Conda 가상환경 생성 및 활성화**
   ```bash
   conda env create -f environment.yml
   conda activate power-peak
   ```

3. **환경 변수(.env) 설정**
   - `.env.example` 파일을 복사하여 `.env`를 만듭니다.
   - 본인의 로컬 PC 원본 데이터가 위치한 경로를 입력합니다.
   ```bash
   # .env 내용 예시
   DATA_DIR=D:/data/resource_opt
   ```

4. **Jupyter 커널 등록 및 Git 노트북 출력 제거 필터 설치**
   ```bash
   python -m ipykernel install --user --name power-peak --display-name "Python (power-peak)"
   nbstripout --install
   ```

5. **VS Code 실행 및 커널 선택**
   - VS Code에서 `KAMP-Competition` 폴더를 엽니다.
   - `Ctrl + Shift + P` → `Python: Select Interpreter` → `power-peak` 선택
   - 노트북(`.ipynb`)을 열고 우측 상단 커널에서 `Python (power-peak)` 선택

