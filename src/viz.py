"""시각화 공통 설정 모듈.

Windows 환경에 최적화된 한글 폰트(Malgun Gothic) 및 마이너스 기호 깨짐 방지 설정을 제공합니다.
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns


def set_korean_font(font_family: str = "Malgun Gothic") -> None:
    """matplotlib 및 seaborn에 한글 폰트와 마이너스 부호 깨짐 방지를 적용합니다.

    Args:
        font_family (str): 적용할 폰트 이름 (기본값: 'Malgun Gothic' - Windows 맑은 고딕)
    """
    # 폰트 패밀리 지정
    plt.rcParams["font.family"] = font_family
    
    # 음수/마이너스 기호 깨짐 방지 (True일 경우 마이너스가 네모/깨짐 현상 발생)
    plt.rcParams["axes.unicode_minus"] = False
    
    # 해상도 및 기본 레이아웃 설정
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["figure.autolayout"] = True
    
    # seaborn 기본 스타일 및 폰트 연동
    sns.set_theme(style="whitegrid", font=font_family, rc={"axes.unicode_minus": False})


# 모듈 import 시 자동으로 한글 폰트 적용
set_korean_font()


if __name__ == "__main__":
    print("[viz] 맑은 고딕 한글 폰트 및 마이너스 기호 깨짐 방지 설정 완료.")
