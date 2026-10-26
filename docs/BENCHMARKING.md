# Benchmarking

`scripts/benchmark.py` loads one local checkpoint and reports model-load time
separately from steady-state inference. Forward timing uses a fixed
`[batch=1, sequence=32]` input by default. Generation timing uses the same
checkpoint, a fixed-length prompt, and 16 autoregressive characters. Both
paths warm up twice, enter inference mode, and synchronize CUDA before and
after each timed operation when running on CUDA. The measured device,
software/hardware, shape, warm-up count and repeat count are stored with each
result.

Mean and median are reported for any valid repeat count. P95 is reported only
at 20 or more observations; smaller runs show it unavailable. Forward
throughput is input tokens divided by forward latency; generation throughput
is generated characters divided by end-to-end generation duration. These
definitions are not interchangeable. Peak allocated GPU memory is recorded
only on CUDA; a CPU run explicitly marks it unavailable.

Warm-up reduces one-time initialization effects but does not remove operating
system scheduling noise, thermal variation, concurrent load or CPU/GPU
differences. Compare models on the same host with the same settings. Do not
claim a universal speedup from one short experiment. Loading time includes
restricted checkpoint deserialization and model construction.

The measured CPU starter results for both models and their exact settings are
recorded in [MEASURED_RESULTS.md](MEASURED_RESULTS.md). The real downloaded
corpus contained only three documents, so those timings are software
measurements rather than evidence of general model quality.
