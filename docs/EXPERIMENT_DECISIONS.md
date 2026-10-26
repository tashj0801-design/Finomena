# Experiment and implementation decisions

| Decision | Reason/evidence | Trade-off or limitation |
|---|---|---|
| Use the ECB official English press-release feed | The RSS endpoint and the ECB's reuse terms were reachable and inspected on 2026-10-27. Terms permit accurate reuse with source attribution. | Official institutional commentary is not independent journalism; feed history is limited and rolling. |
| Filter the shared ECB feed to `/press/pr/` HTML pages | The feed also includes named speeches/interviews and PDFs. Selecting only press releases avoids treating author-attributed documents as general reusable article text. | Excludes potentially useful financial commentary; simple HTML extraction is not semantic article parsing. |
| Keep raw/processed records and checkpoints local | Reuse terms require attribution; no reason exists to redistribute article text or weights in this educational source repository. | Users need to fetch data locally before real training. |
| Include a synthetic fixture for offline use | Tests and smoke training must not depend on network access. | Synthetic results are not financial-language evidence. |
| Implement attention explicitly with PyTorch projections and a triangular mask | This makes Q/K/V shapes and the causal rule inspectable and directly testable. | Less optimized than fused scaled-dot-product attention, especially on GPU. |
| Use a sorted Unicode character vocabulary with `<unk>` | Deterministic, serializable, and handles held-out symbols without leakage. | Character sequences are long and unfamiliar grapheme sequences may split. |
| Use chronological splits when all dates parse; otherwise seeded whole-document splits | Matches temporal evaluation when possible while retaining reproducibility for local files. | Small feed samples produce coarse split sizes; random fallback is not temporal. |
| Save best and latest weights, optimizer state, and safe state dictionaries | Preserves selected model and last training state while using restricted weights-only loading. | Exact resumable training and restoration of every RNG stream are not yet wired into a resume CLI. |
| Use synchronous Streamlit training | Keeps the interface small and avoids background-worker lifecycle complexity. | The user must keep the page open; cancellation is unavailable. |
| Do not implement a sentiment threshold | This is a generative next-character model, not a classifier. Temperature and top-k affect sampling. | No classification metrics or threshold are applicable. |

No larger Medium training run or quality claim is justified by the CPU-only
environment and tiny initial learning runs. Actual results are recorded in
`outputs/` only after commands are executed.

The first measured ECB feed contained three press releases, one per
chronological split. Both starter models were trained for ten CPU steps with
the same seed/context/batch settings and were evaluated on the single held-out
test document after selection. Full measurements are published in
`MEASURED_RESULTS.md`; the tiny dataset is explicitly a limitation, not a
reason to imply representative quality.

A reproducibility test found that early pilot commands initially set the
training seed after model construction. The seed is now applied before weight
initialization and reset for training-window sampling; a test verifies equal
initial weights for equal seeds. Starter results were rerun and the published
measurements use those corrected runs. Earlier pilot records remain marked as
superseded in local experiment history rather than being presented as the
primary measured result.
