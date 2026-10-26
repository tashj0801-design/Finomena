# Environment report

Observed on 2026-10-27 in the implementation session:

- OS: Windows (PowerShell).
- CPU: AMD Ryzen 7 7730U with Radeon Graphics; 8 cores / 16 logical processors.
- Python: 3.14.5.
- PyTorch: 2.12.1+cpu.
- CUDA available: no. GPU count: 0. CUDA build: none.
- Streamlit: 1.59.1.
- pytest: 9.1.1.
- Git: 2.56.0.windows.2.
- Free space on C:: 273,122,467,840 bytes at inspection.
- Available RAM: Not available (the optional `psutil` probe was not installed;
  the Windows memory query did not return values).

The project was implemented and trained with CPU-only PyTorch. Training runs
are intentionally short; the measurements are specific to this host and
configuration, not universal performance claims.
