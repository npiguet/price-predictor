# March 19, 2026 — Transformer Tuning and Architecture Search

**TL;DR:** I started the day reviewing disappointing evaluation results and
spent the session iterating on the transformer's architecture and training
strategy. Two commits landed: one adding metadata side-channel inputs and
bucket-stratified oversampling, another making key architecture parameters
configurable for automated search.

The session opened around midnight with me pasting evaluation output and
noting the results were disappointing. The morning resumed at 07:37 with
another round of evaluation numbers. I wanted to understand the per-bucket
breakdown better and asked Claude to split the broad "<2€" bucket into finer
slices — first into <0.50€ and 0.50–2€, then further into <0.10€ and
0.10–0.50€. That incremental granularity request reflected a real diagnostic
need: cheap commons dominate the dataset and a coarse bucket would hide
whether the model was simply learning to predict "low" for everything in
that range.

Around 08:02 I explicitly asked Claude not to change any code yet and just
discuss the results — comparing sampler exponent 0.5 versus 1.0. That
discussion was the basis for a decision made later in the evening: at 18:59
I chose to move the default sampler exponent to 1.0, meaning full inverse
bucket weighting, and immediately asked for a commit. That landed as the
first of the two commits, a substantial one touching 28 files. It introduced
printing metadata (rarity, reserved-list flag, printings count, release year,
legalities) as a 15-dimensional float vector concatenated after the pooled
transformer outputs rather than tokenized into the text — a cleaner
separation between the language-like card description and the structured
numerical facts about a card's printing history. Release year replaced set
code, which had been too sparse to embed meaningfully. The oversampler change
meant that expensive cards, which are rare in the training set, got equal
gradient pressure to bulk commons.

Shortly after, at 19:01, I shifted to architecture understanding. I asked
for a diagram of the current architecture with dimensions and layer counts,
and then asked follow-up questions: what "batch" means in that diagram, what
the attention mask does, what position embedding is (distinct from token
embedding), and what role the feed-forward layer plays. These were conceptual
questions aimed at building enough intuition to reason about whether a
smaller or larger model would perform better. By 19:14 I had enough context
to ask for the architecture hyperparameters — `n_layers`, `d_model`, and
`ff_dim` — to be made configurable via CLI and stored in the saved model
artifact. The motivation was explicit: I wanted to run automated exploration
later without losing track of which checkpoint used which architecture. That
second commit followed at 19:16, a small surgical change adding four CLI
flags and routing them through `TransformerConfig` which was already being
serialized into the `.pt` file.

At 19:18 I invoked `/plan` and sketched a hyperparameter search plan: train
models across combinations of those parameters, evaluate each, store results
to a file, and print the top five by a weighted accuracy ranking that
prioritized the 2–50€ range, then 0.5–2€, then >50€, then <0.5€ — ordering
that reflects where model accuracy matters most for practical use. I asked
how to run the resulting script and then waited for results.

The evening continued with me reviewing the search output at 20:59 and asking
for Claude's interpretation, then at 21:03 asking whether exploring even
wider or deeper models would help, and at 21:05 whether the output suggested
overfitting. At 22:54 I pasted a large block of training output for four
large-model runs and again asked for interpretation, followed at 23:00 by
evaluation-command output for those same four models.

At 23:02 I flagged something suspicious: too many runs showed exactly 100%
for the median-percent-error column, which felt wrong — some variation would
be expected from noise, and round identical values suggested a bug or a
degenerate metric calculation rather than genuinely perfect performance.
The session ended with me exploring the idea of quantization and pruning
trained models (23:11), then lower-precision training (23:12), then FP8
(23:13), and finally asking about BF8 (23:14) — curiosity about numerical
precision as a way to train larger models without proportionally higher memory
cost, though no code changes followed from that thread.

*Note: reconstructed from prompt history + git log; full session transcripts
were auto-deleted after 30 days.*
