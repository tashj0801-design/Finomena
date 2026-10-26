# Limitations and ethics

Finomena is an educational code project, not a financial service or adviser.
Generated statements are not verified reports, live market observations, or
investment recommendations. A small character-level model can hallucinate,
misstate numbers, produce broken words, and imitate biases in its small source
sample. It has no guaranteed knowledge of current events.

The ECB releases are official institutional commentary, not a representative
sample of financial journalism, investors, countries, or market periods. The
RSS feed is rolling; a small downloaded sample can omit relevant history.
Text extraction can retain page boilerplate or miss table meaning. A
chronological held-out split improves temporal separation, but does not make
the sample statistically representative. A short smoke run proves only that
the pipeline executes.

Raw and processed corpus files are excluded from Git. Users are responsible
for following the ECB's current terms and preserving attribution when
reproducing source material. Generated text is not reviewed by a person for
accuracy. No classification threshold exists for this generative model;
temperature and top-k are generation controls, not decision thresholds.

Model quality numbers are character-level and only meaningful under their
documented dataset, split, and training setup. P95 latency is omitted when
there are fewer than 20 repeats. GPU memory is unavailable on CPU. Hardware
comparisons across different devices are not direct speed comparisons.
