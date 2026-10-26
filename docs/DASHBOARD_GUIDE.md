# Dashboard guide

Run `streamlit run app.py` from the project root. Importing `app` does not
start training or download data. The Overview shows observed hardware, corpus
preparation state, checkpoints and last experiment status. Dataset has an
explicit fetch/prepare action and displays the real cleaning/split report.

Model configuration displays the selected profile dimensions and actual
parameter count. Training supports baseline or GPT profile, validated
dimension overrides, context, batch size, learning rate, weight decay,
gradient clipping, evaluation interval, seed and device. A run is synchronous:
the page shows progress and does not start work on ordinary reruns. Leave the
page open until it finishes. Cancellation of a synchronous run is not
implemented; interruption is not recorded as a completed run.

Generation requires a real local checkpoint, supports prompt/empty prompt,
length, temperature, optional top-k, seed and safe device selection, and has
an output-clearing control. Output is labelled experimental and not financial
advice. Model comparison shows actual experiment measurements and offers an
explicit benchmark action. Experiment history displays run details and can
export JSON or CSV. With no run or benchmark, tables show honest empty states.
