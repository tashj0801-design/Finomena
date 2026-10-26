# Finomena — A Financial Language Model Built From Scratch

Finomena is a small educational research project that teaches how a trainable
character-level bigram and a decoder-only GPT-style Transformer work. It
downloads attributable European Central Bank (ECB) press releases on demand,
trains from random initial weights, measures held-out loss and inference speed,
and presents real experiments in Streamlit.

This is not a financial adviser, a source of verified current news, or a
production financial service. Generated text can be inaccurate or nonsensical.

## Windows PowerShell quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts\inspect_environment.py
python scripts\prepare_data.py --max-documents 40
pytest
python scripts\train.py --model bigram --steps 20
python scripts\train.py --model gpt --profile tiny --steps 20
python scripts\evaluate.py --checkpoint checkpoints\gpt_latest.pt --split test
python scripts\benchmark.py --checkpoint checkpoints\gpt_latest.pt --repeats 20
streamlit run app.py
```

PowerShell may block activation scripts. In that case, call
`.\.venv\Scripts\python.exe` and `.\.venv\Scripts\pip.exe` directly.
Python 3.10+ works when compatible PyTorch wheels are available; this project
was exercised with Python 3.14.5 and CPU PyTorch. The default configuration is
CPU-safe. CUDA is used only when PyTorch detects it.

The offline smoke commands do not need a network connection:

```powershell
python scripts\train.py --model bigram --data data\sample\synthetic.jsonl --steps 5 --context-length 64 --batch-size 4 --eval-interval 2 --eval-batches 1
python scripts\train.py --model gpt --profile tiny --data data\sample\synthetic.jsonl --steps 5 --context-length 64 --batch-size 4 --eval-interval 2 --eval-batches 1
```

Training creates a timestamped best checkpoint and updates
`checkpoints\bigram_latest.pt` or `checkpoints\gpt_latest.pt`. Replace the
checkpoint argument in evaluation and benchmark commands to evaluate a
particular run. The test split is only evaluated by the explicit evaluate
command; do not use it repeatedly to tune a model.

## Data, artifacts, and limitations

`python scripts\prepare_data.py` fetches the latest English press releases from
the ECB RSS feed and follows only feed links under the press-release section.
It requests documents sequentially, with a delay, and stores the original
source URL, title, publication date, and attribution alongside each local
document. See [DATASET_REPORT.md](DATASET_REPORT.md) before downloading. No
corpus is included in this repository. Raw and processed data, checkpoints,
experiment logs, and benchmark results stay local and are Git-ignored.

To train a tiny learning run without network access, use
`python scripts\train.py --model gpt --profile tiny --steps 5 --data data\sample\synthetic.jsonl`;
this explicitly labelled fixture is synthetic, not financial news.
Training and evaluation metrics are saved under `outputs\`; checkpoint files
are under `checkpoints\`.

See the `docs\` guides for the data flow, tokenizer, attention math, training,
evaluation, benchmarking, dashboard, and limitations.

The first actual ECB run found only three press releases, one document per
chronological split. Verified starter metrics and benchmark conditions are
recorded in [docs/MEASURED_RESULTS.md](docs/MEASURED_RESULTS.md); they are
not claims of useful financial-language quality.
