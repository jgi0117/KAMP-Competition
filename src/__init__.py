"""KAMP Competition - 전력사용량 예측 및 피크 분석 패키지"""

from .config import DATA_DIR, REPORTS_DIR, FIGURES_DIR


def set_korean_font(*args, **kwargs):
    """Load plotting dependencies only when visualization is requested."""

    from .viz import set_korean_font as _set_korean_font

    return _set_korean_font(*args, **kwargs)

__all__ = ["DATA_DIR", "REPORTS_DIR", "FIGURES_DIR", "set_korean_font"]
