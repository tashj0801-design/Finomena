"""Runtime hardware and software inspection."""

from __future__ import annotations

import os
import platform
import sys
from typing import Any

import torch


def collect_environment() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "pytorch": str(torch.__version__),
        "cpu": platform.processor() or platform.machine(),
        "logical_cpus": os.cpu_count(),
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "gpu_memory_bytes": (
            torch.cuda.get_device_properties(0).total_memory if cuda_available else None
        ),
    }


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch reports CUDA unavailable")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    return torch.device(requested)
