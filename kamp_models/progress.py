from __future__ import annotations

from datetime import datetime


def log_progress(message: str) -> None:
    """Write a timestamped progress message immediately."""

    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def torch_device_summary(device: str) -> str:
    try:
        import torch
    except ImportError:
        return f"selected={device} | PyTorch unavailable"

    cuda_available = torch.cuda.is_available()
    parts = [
        f"selected={device}",
        f"CUDA connected={cuda_available}",
        f"torch={torch.__version__}",
        f"CUDA runtime={torch.version.cuda or 'none'}",
    ]
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        parts.extend(
            [
                f"GPU={properties.name}",
                f"VRAM={properties.total_memory / 1024**3:.1f} GB",
            ]
        )
    return " | ".join(parts)
