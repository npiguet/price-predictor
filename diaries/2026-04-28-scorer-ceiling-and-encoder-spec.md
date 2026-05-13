# April 28, 2026 — Scorer ceiling and encoder spec

**TL;DR:** A dropout sweep confirmed ~0.70 val_acc is the real
ceiling for the current scorer, not a tuning problem. I spent the
rest of the day specifying encoder fine-tuning as the next
meaningful lever.

The day started with a look at 20K match outcomes to decide whether
the training data was ready for gen2. Claude's read was that it was:
the critical matchup — gen1 vs forge-best — had about 1,640 samples,
enough corrective signal for the reward-hacking issue logged earlier.
The thin diagonals (random vs random, forge-8sub vs forge-8sub)
don't carry much information anyway.

I asked whether dropout was in the code yet. It wasn't. Claude
walked me through why it would help here: without it, the scorer was
free to memorize narrow co-adapted patterns from training matches
rather than broad generalizable signals about deck quality. I asked
for the same explanation for AdamW weight decay, and Claude was
careful to distinguish the two — dropout breaks co-adaptation,
weight decay caps weight magnitude growth. It also clarified that the
current code uses plain `Adam` with no weight decay at all, after
briefly getting this wrong and correcting itself when I pushed back.

I made a plan to add dropout and ran a sweep across four settings on
the 6-layer model. The result was a textbook inverted-U: dropout 0.2
peaked at val_acc=0.7015, dropout 0.1 and 0.3 were slightly worse,
dropout 0.4 clearly worse. I then tried 5 layers with dropout 0.2
and got 0.6948 — slightly below the 6-layer no-dropout baseline of
0.7017. The summary: 6-layer + dropout 0.2 and 5-layer + no dropout
land at exactly the same ceiling. I decided on 6-layer + dropout 0.2
as the gen2 default, reasoning that the regularized model should be
more robust on out-of-distribution inputs like gen1's own greedy
decks.

On the question of whether dropout had really done anything: Claude
pointed out that the key question is not val_acc improvement (which
was negligible) but whether the model's scores are tighter and its
OOD behavior is better. The acid test is head-to-head against
forge-best in `evaluate-scorer`, not val_acc numbers.

I decided to skip AdamW weight decay. The dropout sweep already
showed regularization knobs can't move the ceiling; the
implementation cost wasn't worth the marginal upside vs. other
work.

Then came the multi-pooling and deck-stats experiment. Claude's
intuitions about sum pooling's theoretical advantage for counting
features were solid, and I was persuaded. We added PMA + max + mean
pooling and 23 hand-computed deck-level statistics (mana curve
histogram, color counts, pip totals, creature/noncreature split —
with the right colorless-handling special cases). The discussion
around colorless was detailed: color count and pips use the `{C}`-pip
rule for the colorless slot, while cards-per-color uses the
`is_colorless` flag. I also pushed back on Claude's "sum pooling is
more expressive" argument for the fixed-size-deck case, and Claude
acknowledged I was right — for a fixed 23-card deck, sum and mean
carry identical information, just at different scale, so mean is
preferred for training stability.

Claude then found that the deck-stats were effectively being ignored
by the model at runtime: 23 features out of 3,287 MLP inputs means
their contribution to any output unit is about 800× smaller than the
pooled features at initialization. We tried scaling the deck-stats by
50× and 200× to force the model to notice them. The epoch-1
train_acc bumped by 3 pp, but peak val_acc was unchanged at ~0.695
across all three scaling settings. Claude concluded — and I agreed —
that PMA is a learned aggregator capable of deriving the same
information from cards, so the deck stats just provided a faster path
to the same answer, not new generalizable signal.

The conclusion from these experiments was that I should revert the
architectural changes. That's now four independent interventions
(depth, dropout, multi-pooling, deck stats) all converging at ~0.70,
which makes a strong case that this is the irreducible Bo7-noise
floor rather than a model deficiency.

I asked Claude to explain the Bo7 noise floor math in detail. The
number that landed was: at a per-game win probability of 0.65, the
better deck still loses a Bo7 about 20% of the time. Crucially,
Claude pointed out that Bo7 amplifies a per-game edge far less than I
expected — going from Bo1 to Bo7 at p=0.55 only moves the match-win
rate from 55% to 61%. I raised the question of whether the Bo7 match
actually plays all 7 games (it doesn't — it stops at 4 wins), and
Claude showed the math: the stop-at-4 and play-all-7 formulations
give identical match-win probabilities because whoever reached 4
wins first would still have ≥4 wins after 7 hypothetical games.

I asked the sharper question: if my scoring model can't reliably
distinguish close decks, and self-play will eventually make gen_N ≈
forge-best, won't the labels become coin flips and stall progress?
Claude's answer separated two things the scorer does: binary
prediction (bounded by the noise floor) vs. continuous ranking (not
bounded, because the score difference encodes a calibrated
probability that averages out over many training examples). The
greedy search uses ranks, not binary calls, so val_acc undersizes the
scorer's deployment usefulness.

That led to the main question of the day: what's actually left to
try? Claude identified embedding unfreeze as the highest-leverage
remaining lever, for a reason that had been stated implicitly but
became clear here: all the failed interventions targeted the
*aggregation* over per-card features; embedding unfreeze targets the
per-card features themselves. I raised the concern that the current
`--unfreeze-embeddings` flag just makes the lookup-table rows
trainable — each card gets about 47 gradient updates across the
entire corpus. Claude agreed this is too few: with 26K unique cards
and 27K matches, the long-tail of rare cards barely shifts at all,
and there's no parameter sharing across similar cards.

Real encoder fine-tuning — keeping the price-predictor's SAB stack
in the training graph — addresses this because every one of the 1.24M
card references in the corpus contributes gradient to all encoder
parameters via shared weights. I confirmed the encoder is only 2
layers, which makes the compute cost much more reasonable than Claude
initially assumed: maybe 2–4× slower per epoch, not 5–15×.

I asked Claude to write a spec for this in `specs/2026-04-27-encoder-fine-tuning.md`.
The discussion that shaped the spec covered several important points:
the two-run structure (Phase A with frozen encoder, Phase B resuming
from Phase A's best checkpoint with a non-zero `--embedding-lr`),
why the embedding LR needs to be far lower than the main LR (each
training example triggers 46 encoder forward passes — 23 cards ×
2 decks — so autograd accumulates 46× the gradient into encoder
parameters compared to scorer-specific parameters; to actually make
the encoder move 10× *slower* than the scorer, you need
approximately `lr / (10 × 46) ≈ 2e-8`), where the fine-tuned encoder
weights live (bundled into the sealed scorer checkpoint, not a
separate file; the price-predictor checkpoint is never modified), and
how `encode-cards` needs a `--scorer-checkpoint` flag so it can
re-generate `.npz` files from the Phase B encoder after training.

The spec went through multiple review passes during the day. I
pointed out that the document read more like an instruction manual
than a formal specification, so Claude restructured it into a tight
Specification section (tables, numbered lists, CLI flag tables, a
checkpoint format table, a monitoring table) followed by a Rationale
section that explains the decisions without restating them. The final
session ended with me starting another review pass asking for
redundancy and verbosity cuts — that one got interrupted before
completion.
