# Model architecture

## Bigram baseline

`BigramLanguageModel` is a trainable table implemented by an embedding of
shape `[vocabulary, vocabulary]`. For input IDs `[B,T]`, table lookup returns
logits `[B,T,V]`; each position's prediction depends only on the current
character, with no wider context.

## Decoder-only Transformer

For the default Tiny profile, the model embeds input characters into `C=64`
learned features and adds learned position embeddings for each of `T<=128`
positions. A stack of two pre-layer-normalised Transformer blocks processes
the sum. The final normalised features map to `V` vocabulary logits per
position.

Each attention layer projects hidden states `[B,T,C]` into query, key and
value tensors. With four heads, each projection is reshaped to
`[B,H,T,C/H]`. Queries describe what a position is looking for, keys describe
what each earlier position offers, and values carry the information to mix.
The score matrix is
`Q @ K.transpose(-2,-1) / sqrt(C/H)`, with shape `[B,H,T,T]`. Row-wise
softmax creates attention weights and their product with values combines
context.

The registered upper-triangular Boolean mask replaces future-position scores
with negative infinity before softmax. A position therefore assigns exactly
zero attention weight to future tokens. The test keeps an input prefix fixed,
changes only later characters, and checks that earlier logits are unchanged
in evaluation mode with dropout disabled.

Heads are concatenated and projected back to `C`. Each block applies residual
addition around attention and a position-wise MLP of width `4C`, GELU, and a
second residual addition; LayerNorm and dropout are explicit modules.

The output layer's unnormalised vocabulary scores are logits. Cross-entropy
compares `[B,T,V]` logits to next-character targets `[B,T]` after flattening
the first two axes. For `abcde`, one window is input `abcd`, target `bcde`:
the output at `a` is trained against `b`, not against `a`.

At inference the last position's logits are divided by temperature, optionally
filtered to top-k, softmaxed, and sampled. The sampled ID is appended; the
next call is cropped to the model's context window. Temperature/top-k are
generation controls, not classification thresholds.
