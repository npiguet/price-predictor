# April 22, 2026 — Self-play data quality decisions

**TL;DR:** The self-play pipeline went live today and immediately confirmed
the expected failure mode: the scorer's greedy decks lose most matches because
they pack too many high-CMC cards. Most of the day was spent deciding how to
improve the quality of training data going forward: richer match-outcome
metadata, Bo7 instead of Bo3, and a diagnosis of the throughput decline.

The day started with a brief discussion about attention pooling as a
replacement for mean pooling in the transformer encoder. I wasn't looking to
code anything — I wanted to understand the tradeoffs. The conclusion was that
`concat(attention_pool, max_pool)` is a defensible upgrade: attention pooling
fixes mean pooling's dilution problem on long cards while max pooling retains
the discrete feature-detection signal. The modest upside is mainly for
embedding quality and fine-tuning ergonomics rather than price prediction
accuracy, and Claude's recommendation was to defer it unless a retraining is
already planned for another reason.

The bulk of the day was the `speckit.implement` run for spec 014, which built
out the full self-play pipeline: random-set pool generation, the new
`build-decks` subcommand, and `match-outcomes --generated-decks-path` for
self-play. All of that landed in a single commit. Then immediately a domain
bug surfaced: old small-set boosters like DRK and FEM (8 cards per pack) were
slipping through as eligible sets. It turned out Forge itself already has a
`> 11 cards expected` rule in `AdventureEventData.isValidDraftBlock()` that
our mirror had skipped. I decided to apply the same filter on both the Java
and Python sides, with a `MIN_BOOSTER_CARDS = 12` threshold.

Once the matches were actually running, I noticed the build-decks step was
slow and asked whether GPU was being used. It was — close to 100% on the
RTX 3060 Ti — so the bottleneck was the forward-pass math, not I/O. Claude
identified some redundant per-iteration work: constant tensors being
recomputed every greedy step, and `deck_idx`/`rem_idx` doing a CPU→GPU
transfer on every iteration. I decided to apply those fixes along with fp16
autocast, and to add timestamps to the progress messages. I then ran a real
10k-pool batch and shared the output, which showed a steady ~60s per 100
pools with no sign of the time interval growing in the first 500 pools.

The most substantial design work of the day was rethinking the match-outcome
file format. I had already been collecting matches without set code, without
the build method per deck, without timestamps, and without game-level detail.
The conversation made it clear that all four gaps would matter for future
dataset management. I decided to restart from scratch rather than add
backward-compatibility shims — the cost is a couple of days of regeneration
and the benefit is a clean format going forward. The final 10-field schema is:
`timestamp;run_id;set_code;method_A;method_B;deck_A;deck_B;games;play;
duration_s`. I also decided to make `--self-play-label` both mandatory in
self-play mode and forbidden in phase-0 mode, to prevent accidentally mixing
generations under a generic "scorer" tag. That all went through spec review
first, then implementation.

The Bo7 decision came from a direct question about whether more games or more
matches would give better training data. Claude's reasoning was that label
noise from Bo3 wrong-flips is a real training-quality problem, and that near-
duplicate deck correlation already erodes the independent-information value of
raw match counts. The math worked out: 15k Bo7 matches lands at ~60 forge-
best decks per eligible set (200 sets), which is in the middle of the
estimated 50-80 sweet spot for archetype coverage without near-duplicate
saturation. I decided to go with 15k Bo7.

The Bo7 decision was later confirmed empirically. I shared live match-outcomes
data and Claude ran diagnostic scripts. The key finding: **25.5% of Bo1
labels would flip relative to Bo7, and 16.5% of Bo3 labels would flip**. That
16.5% figure is the empirical case for the switch — the previous Bo3 default
would have mislabeled roughly 1 in 6 matches. The per-method win rates also
showed a clean strict ordering (forge-best 65%, forge-3sub 52%, forge-8sub
36%, random 15%), and the skill gap between methods widens monotonically with
match length, which is exactly the Bradley-Terry prediction.

Late in the day I noticed throughput declining over the course of a multi-hour
run. The duration diagnostic script showed the run-wide average rate dropping
~14% after the first 90 minutes. Claude identified two contributors: individual
matches getting slightly longer on average (fatter long tail) and worker
utilization dropping. My hypothesis was worker log files — Forge is verbose,
12 workers in append mode accumulate hundreds of MB of logs, and Windows
Defender scans on every write. I deleted the logs and restarted. Throughput
recovered immediately. I then decided to route all worker output to
`/dev/null` going forward, since I only look at supervisor logs anyway.

The discussion about corpus management was useful context for the gen-2
training plan. I laid out a table showing the intended mix: ~29k phase-0 rows
plus ~23k self-play rows for a 52k total, with gen-1 scorer decks at ~28% of
the combined set. Claude flagged two things: random decks are diluting from
10% to 7% of the new data (they serve as a low-end anchor), and since I plan
to retrain from scratch every generation rather than fine-tune, keeping old-
generation self-play data is actively necessary — without it, the model has no
evidence that the old failure modes are bad, and will likely reinvent them.
