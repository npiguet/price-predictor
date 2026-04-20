# March 16, 2026 — Transformer Tuning and Tokenizer Design

**TL;DR:** I finished wrapping up feature 009 in the early hours, then
retrained the transformer on real data and compared results. The rest
of the day was split between understanding the transformer architecture
deeply and designing a custom MTG tokenizer (feature 010) from first
principles.

The session started just before 1 AM finishing off the tail of feature
009 — printing metadata fields, enrichment, training, and API plumbing.
After committing that work, I retrained both the sklearn and transformer
models on real data and pasted the evaluation results in for Claude to
analyze. The transformer had improved more than sklearn on a relative
basis, but was still worse in absolute terms. That gap prompted a
discussion about what the transformer could do better without changing
the input data.

Claude proposed three low-effort architectural changes: replacing the
CLS token extraction with masked mean pooling over all token outputs, a
deeper regression head (two linear layers with a ReLU in between), and
switching from MSE to Huber loss to reduce the outsized influence of
expensive cards. Before agreeing to any of it, I wanted to actually
understand what each change meant. I asked for a technical primer aimed
at a senior software engineer with rusty neural-network knowledge, and
we went back and forth on several conceptual questions: what the [CLS]
vector actually is, what dropout does mechanically, why it seems wasteful
to throw away all token vectors except one after the attention layers,
what information is encoded per-position in the attention output, and
whether a deeper regression head after pooling is contradictory to using
mean pooling in the first place. That last question surfaced a real
tension in Claude's earlier suggestions that we had to think through
carefully.

I resumed the session in the late afternoon and evening, and by around
9 PM I had approved the three changes. I asked Claude to update the spec
and tasks first before touching code, so the session could survive a
token limit without leaving half-implemented work. The implementation
commit landed around 21:06: mean pooling, the two-layer head, and Huber
loss, along with tests and spec updates. I then retrained and compared
against the old checkpoint. Loading the old model initially failed
because the code had already moved on, so I had to roll back temporarily
to run the old evaluation and get a fair comparison.

From there the focus shifted entirely to spec design for feature 010, the
custom MTG tokenizer. I asked Claude to review the approach I had in
mind and critique it. The main question was whether a BPE fallback was
actually needed on top of a domain-specific word list. Claude ran an OOV
analysis against the real corpus and found 98.1% coverage at a frequency
threshold of 5 — the tail was essentially proper nouns from card names,
not MTG mechanics vocabulary. That made the case for dropping BPE and
using a simple word-level tokenizer with an [UNK] token instead, which
I accepted.

Two specific tokenization decisions got settled in the subsequent
clarification pass. For multi-word keywords like "first strike," the
right approach is underscore normalization (first_strike) derived from
the Forge Keyword enum entries that contain spaces — treating them as
atomic tokens rather than two words. For generic mana costs like {3}, I
raised the question of whether decomposing them into {1}{1}{1} would be
better for the model than keeping distinct tokens per value. We recorded
the full reasoning in the spec: distinct tokens were chosen because the
model can learn the ordinality from co-occurrence, decomposition would
bloat sequences and confuse colored vs. colorless mana, and all distinct
values seen in the corpus ({0} through {16} plus a handful of outliers)
are few enough to include without a cap. The decision was recorded
explicitly so it can be revisited.

The late-evening speckit clarification session locked down the CLI
design: a standalone `vocabulary` subcommand (not `build-vocab`) builds
and persists the vocabulary to `models/price-predictor/transformer/vocab.txt`,
with a `--vocab-path` override available on all four transformer
commands. [PAD] (token ID 0) and [UNK] are the only special tokens;
no [CLS], [SEP], or [MASK] are needed for a regression task that uses
mean pooling.

*Note: reconstructed from prompt history + git log; full session
transcripts were auto-deleted after 30 days.*
