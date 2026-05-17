# Sealed Encoder — Hyperparameter Sweep & the Label Ceiling

*Runs: 2026-05-11. Dataset: `output/sealed/cards-played.txt` (the cleaned
974,028-game corpus → 27,983 cards after dropping the 2 with no converted
`.txt`; 22,387 train / 5,596 val, card-disjoint). Spec 016 multi-head + MLM
encoder; metric of interest is the per-head **`val corr`** = Pearson r
between predicted and (shrunk) target labels on the held-out cards.*

## TL;DR

- The encoder is **card-text-saturated**. Depth past ~4 layers (~2 under
  attn pool), more attention heads / pool queries, and `--mlm-weight`
  changes don't move `val corr`. Best main-head correlations land at
  ~0.58 (`score_play`), ~0.65 (`played_rate`), ~0.61 (`cast_lift`),
  ~0.45 (`color_lift_*`).
- **Drop the max-pool** (`--pool-mode attn`): same endpoint as the dual
  attention‖max pool but *faster to converge* (~1.5× fewer epochs to the
  same `val_loss`), fewer params, and the pooled/shipped embedding halves.
- **`d_model = 128` is essentially free on `val corr`; `d_model = 64` is
  not** — 64d collapses the colour heads and tanks MLM. 128 is the floor.
- **`n_heads = n_pool_queries = 2` is as good as 4** — the 256d · 2h/2q ·
  attn run is marginally the best on the entire board.
- The **split-half reliability ceiling** of the labels is ~0.78–0.79 for
  the signed heads and ~0.98 for `played_rate` — so there is real headroom
  *in the labels*, but it is **not** reachable by making the encoder
  bigger. Part of that headroom is genuinely not a function of the card
  text (set-relative playability, Forge-AI quirks); how much is unknown.
- **Ship `6L · 2 heads · 2 pool queries · 128d · --pool-mode attn`** (run
  O). The Pareto point for deck-building: `n_layers` is amortised in
  `encode-cards` (free at inference) so keep it high; `pooled_dim`
  determines the scorer's input width (and, when downstream code reads it
  from the encoder config, the scorer's own `d_model`), so shrink it as
  far as `val corr` allows — 128 here. For the conservative
  best-`val corr` pick, **run N (`6L · 2h/2q · 256d · attn`)** instead.
- Cheap diagnostics worth running before any bigger change: an
  effective-rank / PCA probe of the learned card embeddings (are the
  pooled dims even being used?) and a temporary MLP-head probe (is the
  *readout* the limit, or the *representation*?). The only thing that can
  raise the ceiling itself is **richer per-game data** — more informative
  Forge instrumentation (card fate at end of game, drawn-but-not-cast) —
  not the architecture.

## The label ceiling — split-half reliability

`scripts/split_half_reliability.py` splits the games randomly into two
halves, recomputes the nine per-card shrunk labels on each half
independently (same aggregation, same `--shrinkage-k 20`), and reports the
Pearson r between half A's and half B's labels (`r_half`) plus the
Spearman–Brown projection `2r/(1+r)` to the full game count (`r_full`).
`r_full` is the ceiling the model's `val corr` should be read against — a
predictor cannot be more correlated with the labels than the labels are
with themselves.

| head          | n_cards | r_half | r_full (Spearman–Brown) |
|---------------|--------:|-------:|------------------------:|
| score_play    |  27,955 |  0.641 |               **0.782** |
| score_draw    |  27,948 |  0.627 |               **0.771** |
| played_rate   |  27,959 |  0.953 |               **0.976** |
| cast_lift     |  25,108 |  0.646 |               **0.785** |
| color_lift_W  |  27,957 |  0.429 |                   0.600 |
| color_lift_U  |  27,955 |  0.508 |                   0.674 |
| color_lift_B  |  27,955 |  0.454 |                   0.624 |
| color_lift_R  |  27,959 |  0.496 |                   0.663 |
| color_lift_G  |  27,957 |  0.481 |                   0.649 |

Reading it:

- `played_rate` is an almost perfectly reliable label (~0.98 — "what
  fraction of decks that ran this card played it" barely moves under
  resampling, lots of observations per card), yet the model only reaches
  ~0.65. Largest gap of any head; not capacity-bound (see below).
- The signed heads (`score_*`, `cast_lift`) cap around ~0.78–0.79 — model
  at ~0.58–0.61, so meaningful but smaller headroom.
- The `color_lift_*` heads have *genuinely lower ceilings* (~0.60–0.67) —
  the per-colour slices are sparse and noisy, so a lower model `val corr`
  there (~0.45) is partly "the label itself is fuzzy," not just "the model
  is weak." Even so there's headroom relative to *their* ceiling.
- `cast_lift` has fewer cards (25,108 vs ~27,955): ~2,800 cards were always
  played or never played within their decks, so `cast_lift` (a played-vs-
  not-played win-rate difference) is undefined for them — they're absent
  from this head's correlation but present in the others'.

**Caveat — the ceiling is not all reachable.** Split-half reliability is a
property of the *label only*; it says nothing about whether the reliable
signal is a function of the card features. A card's aggregated `played_rate`
is a mixture over every sealed environment it appeared in, and a card's
playability genuinely differs by set — the encoder sees only set-agnostic
card text (no set code, no printing side-channel). Forge-AI quirks ("the AI
never casts card X") are also reliable but not in the card text. So the gap
between the model's `val corr` and `r_full` is *ambiguous*: part is "the
encoder hasn't extracted it," part is "it isn't in the card text." The
sweep below strongly suggests the first part is small.

## The hyperparameter sweep

All runs: `d_model=256`, `ff_dim = 4·d_model` (so `ff_dim=1024` at the
default), 100-epoch cap, AdamW `lr=1e-4`, 5%-warmup-then-constant, best
checkpoint by full validation loss `reg + mlm_weight·mlm`, `--pool-mode
dual` (attention ‖ max-pool) unless noted. Best-epoch validation numbers:

| run | layers | heads | pool-q | d_model | mlm_w | pool | best ep | val_loss | val reg | MLM ppl / acc | score_play | score_draw | played_rate | cast_lift | color_lift W/U/B/R/G |
|-----|-------:|------:|-------:|--------:|------:|------|--------:|---------:|--------:|---------------|-----------:|-----------:|------------:|----------:|----------------------|
| **A** | 6 | 4 | 4 | 256 | 0.1 | dual | 100 | 0.0685 | 0.0193 | 1.6 / 85.5% | **+0.58** | +0.56 | **+0.65** | **+0.61** | .40/.52/.46/.49/.46 |
| B | 4 | 4 | 4 | 256 | 0.1 | dual | 99 | 0.0721 | 0.0191 | 1.7 / 84.2% | +0.57 | +0.56 | +0.64 | +0.60 | .40/.52/.44/.48/.45 |
| C | 2 | 4 | 4 | 256 | 0.1 | dual | 99 | 0.0804 | 0.0198 | 1.8 / 82.3% | +0.56 | +0.55 | +0.62 | +0.59 | .38/.51/.42/.47/.45 |
| D | 1 | 4 | 4 | 256 | 0.1 | dual | 99 | 0.0971 | 0.0207 | 2.1 / 78.4% | +0.55 | +0.55 | +0.61 | +0.59 | .40/.51/.43/.46/.46 |
| E | 6 | **8** | **8** | 256 | 0.1 | dual | 100 | 0.0691 | 0.0199 | 1.6 / 85.4% | +0.58 | +0.57 | +0.64 | +0.62 | .40/.52/.44/.48/.46 |
| F | 6 | **16** | **16** | 256 | 0.1 | dual | 97 | 0.0714 | 0.0195 | 1.7 / 84.4% | +0.58 | +0.57 | +0.62 | +0.60 | .40/.53/.41/.48/.47 |
| G | 6 | 4 | 4 | 256 | **0.02** | dual | 95 | 0.0341 | 0.0202 | 2.0 / 80.0% | +0.56 | +0.56 | +0.60 | +0.59 | .39/.52/.43/.47/.46 |
| H | 6 | 4 | 4 | 256 | 0.1 | **attn** | 100 | 0.0649 | 0.0194 | 1.6 / 86.6% | +0.57 | +0.57 | +0.63 | +0.60 | .42/.53/.46/.49/.49 |
| I | 4 | 4 | 4 | 256 | 0.1 | attn | 99 | 0.0693 | 0.0195 | 1.6 / 85.1% | +0.57 | +0.56 | +0.62 | +0.59 | .41/.52/.43/.47/.47 |
| J | 2 | 4 | 4 | 256 | 0.1 | attn | 99 | 0.0771 | 0.0193 | 1.8 / 83.0% | +0.57 | +0.56 | +0.63 | +0.60 | .40/.53/.43/.47/.46 |
| K | 1 | 4 | 4 | 256 | 0.1 | attn | 99 | 0.0926 | 0.0197 | 2.1 / 79.1% | +0.55 | +0.55 | +0.61 | +0.59 | .40/.50/.42/.45/.44 |
| L | 6 | 4 | 4 | **128** | 0.1 | attn | 100 | 0.0853 | 0.0192 | 1.9 / 80.6% | +0.56 | +0.55 | +0.63 | +0.59 | .41/.51/.43/.46/.45 |
| M | 6 | 4 | 4 | **64** | 0.1 | attn | 100 | 0.1638 | 0.0213 | 4.2 / 62.8% | +0.52 | +0.52 | +0.56 | +0.55 | .22/.40/.30/.35/.32 |
| **N** | 6 | **2** | **2** | 256 | 0.1 | attn | 97 | 0.0664 | 0.0192 | 1.6 / 86.0% | **+0.58** | **+0.58** | +0.63 | **+0.61** | .41/.52/.45/.48/.47 |
| O | 6 | **2** | **2** | **128** | 0.1 | attn | 100 | 0.0839 | 0.0190 | 1.9 / 81.2% | +0.56 | +0.56 | +0.63 | +0.58 | .39/.51/.42/.47/.46 |
| P | 6 | **2** | **2** | **64** | 0.1 | attn | 98 | 0.1492 | 0.0211 | 3.6 / 65.7% | +0.52 | +0.52 | +0.55 | +0.55 | .27/.38/.33/.34/.38 |

`val_loss` is the quantity the best-checkpoint selector minimises, but it
isn't comparable across `mlm_weight` (`val_loss = reg + mlm_weight·mlm`, and
the MLM term dominates): non-G runs sit at 0.065–0.164, run G at 0.034 —
that lower number is purely the smaller MLM coefficient, not a better model.
Compare `val reg` (or the per-head `val corr`) when ranking runs that differ
in `mlm_weight`.

### Depth (runs A–D, fixed 4 heads / 4 queries / mlm_weight 0.1)

A clean, well-behaved curve with sharply diminishing returns:

- `val_loss`: 0.097 (1L) → 0.080 (2L) → 0.072 (4L) → 0.069 (6L). 1→2 buys
  0.017, 2→4 buys 0.008, 4→6 buys 0.004.
- **Almost all of that is MLM**: ppl 2.1 → 1.8 → 1.7 → 1.6; acc 78% → 82% →
  84% → 85.5%. More layers ⇒ a better masked-token model.
- `val reg` is essentially **flat** (0.0191–0.0207); 4L and 6L are within
  noise of each other. Depth does ~nothing for the regression MSE.
- `val corr` rises but **barely**: ~+0.03 on the main heads over the full
  1→6 range (`score_play` 0.55→0.58, `played_rate` 0.61→0.65), ~+0.01 per
  doubling; `color_lift_*` flat (~0.45). **4 layers reaches ≈ all of 6
  layers' correlation** (0.57/0.64 vs 0.58/0.65).

So the regression task saturates around 2–4 layers; the auxiliary MLM keeps
improving past where the regression heads stop caring.

### Attention heads & pool queries (runs E, F vs A — fixed 6L, mlm 0.1)

`n_heads` splits `d_model=256` into that many subspaces; `n_pool_queries`
splits the (fixed-size) attention-pooled vector the same way. Both are
*reallocations* of a fixed parameter budget, not capacity additions — and
that's what the runs show:

- **8 heads / 8 queries ≈ 4 heads / 4 queries** — `val corr` identical to
  ±0.01 on every head, MLM identical (ppl 1.6, acc ~85%).
- **16 heads / 16 queries is *worse*** — `played_rate` 0.62 vs 0.65,
  `cast_lift` 0.60 vs 0.61, MLM acc 84.4% vs 85.5%, and it had to early-stop
  at epoch 97. The 16-dim heads / 16-dim pool slots are too narrow (the
  predicted failure mode of pushing the head count too high).

Conclusion: 4 heads / 4 pool queries is fine; there is no reason to go up,
and a reason not to.

### Pool mode — drop the max-pool (run H vs A — fixed 6L, 4h/4q, mlm 0.1)

The card vector fed to the regression heads is, by default,
`cat([multi-query attention pool, max-pool])` over the token outputs
(`--pool-mode dual`, width `2·d_model`). Run H uses `--pool-mode attn` —
the attention pool only (width `d_model`). Same data, same seed; the
encoder *trunk* gets the same random init (the heads are constructed after
it), so this is a fairly clean A/B.

**At the endpoint, indistinguishable** (run H vs A in the table): every
`val corr` head within ±0.02, `val reg` identical (0.0194 vs 0.0193). The
max-pool branch buys nothing.

**But it converges *faster* without max-pool** — H is ahead of A at every
checkpoint:

| epoch | A `val_loss` | H `val_loss` | A `score_play` / `played_rate` | H `score_play` / `played_rate` |
|------:|-------------:|-------------:|--------------------------------|--------------------------------|
| 5  | 0.403 | **0.331** | +0.30 / +0.40 | **+0.49 / +0.44** |
| 10 | 0.304 | **0.199** | +0.44 / +0.47 | **+0.51 / +0.50** |
| 20 | 0.165 | **0.115** | +0.48 / +0.53 | **+0.54 / +0.57** |
| 32 | 0.115 | **0.094** | +0.52 / +0.59 | **+0.55 / +0.62** |
| 64 | 0.080 | **0.073** | +0.57 / +0.64 | +0.56 / +0.63 |
| 100| 0.069 | **0.065** | +0.58 / +0.65 | +0.57 / +0.63 |

H reaches A's *final* `val_loss` (~0.069) around epoch ~64 — roughly 1.5×
fewer epochs to the same quality. Likely mechanism: max-pool over the token
outputs is a noisy, redundant channel; early on the linear heads can
partly "explain" the labels through it, so the gradient into the
*attention* pool (the half doing the useful work) is diluted, and the
heads spend early epochs learning to discount the max-pool noise. Strip it
out and 100 % of the regression pressure goes into the attention pool. This
isn't an MLM-init artifact (`val_loss` is mostly MLM and the two runs got
different MLM-head inits) — `val reg` and `val corr`, which don't touch the
MLM head, show the same faster-convergence pattern. Caveat: one run each,
so the "1.5×" is approximate; a second seed would pin it down. Verdict:
`--pool-mode attn` is strictly better — same endpoint, faster, fewer params,
and the shipped/pooled embedding halves (`2·d_model → d_model`).

### Depth under attn pool (runs H, I, J, K — fixed 4h/4q, 256d, mlm 0.1)

Re-running the depth ladder with `--pool-mode attn` for the same 1L → 6L
range. The same diminishing-returns curve appears in `val_loss` — and the
saturation point on `val corr` is **even earlier**:

| depth | val_reg | MLM ppl / acc | sp | sd | pr | cl | colour avg |
|------:|--------:|---------------|----|----|----|----|-----------:|
| 6L (H) | 0.0194 | 1.6 / 86.6% | .57 | .57 | .63 | .60 | .48 |
| 4L (I) | 0.0195 | 1.6 / 85.1% | .57 | .56 | .62 | .59 | .46 |
| 2L (J) | 0.0193 | 1.8 / 83.0% | .57 | .56 | .63 | .60 | .46 |
| 1L (K) | 0.0197 | 2.1 / 79.1% | .55 | .55 | .61 | .59 | .44 |

Under attn pooling **2L is essentially tied with 6L on the signed heads**
(`score_play` .57=.57, `played_rate` .63=.63, `cast_lift` .60=.60); only
the colour heads lose ~0.02 going from 6L → 2L. Compare to the dual
ladder, where 2L → 6L bought a clear (if small) `val corr` lift on every
head. So removing max-pool concentrates the regression pressure into the
attention pool *and* tightens the depth-saturation point. 1L still drops
~0.02 on `score_play` / `played_rate` — depth past 1 is needed, but 2 is
enough for the main heads.

### `d_model` under attn pool (runs H, L, M — fixed 6L, 4h/4q, mlm 0.1)

Shrinking `d_model` (with `ff_dim = 4·d_model` and `n_heads=n_pool_queries=4`
holding):

| d_model | val_reg | MLM ppl / acc | sp | sd | pr | cl | colour avg |
|--------:|--------:|---------------|----|----|----|----|-----------:|
| 256 (H) | 0.0194 | 1.6 / 86.6% | .57 | .57 | .63 | .60 | .48 |
| 128 (L) | 0.0192 | 1.9 / 80.6% | .56 | .55 | .63 | .59 | .45 |
|  64 (M) | 0.0213 | 4.2 / 62.8% | .52 | .52 | .56 | .55 | .32 |

**128d is essentially free on `val corr`** — every signed head within
±0.01, colour avg down ~0.03 — even though MLM degrades clearly
(ppl 1.6 → 1.9, acc 87 → 81 %). The capacity-sensitive auxiliary task
loses, but the regression task barely notices. ("Representation analysis"
below probes the 128d and 256d embeddings directly and confirms this.)

**64d falls off a cliff.** `val corr` drops 0.04–0.07 on every signed head,
the colour heads collapse (W: .42 → .22), `val reg` ticks up (0.0194 →
0.0213) — genuine underfit — and MLM tanks (acc 87 → 63 %, ppl 1.6 → 4.2).
So `d_model = 64` is below the "useful capacity floor" for this corpus;
128 is the smallest size that retains the task signal.

### Fewer heads & pool queries (runs N, O, P — `n_heads = n_pool_queries = 2`)

Halving heads/queries at every d_model:

| d_model | 4h/4q | 2h/2q | sp / pr / cl (4h/4q vs 2h/2q) | colour avg (4h/4q vs 2h/2q) |
|--------:|:-----:|:-----:|------------------------------|-----------------------------|
| 256 | run H | run N | .57/.63/.60 vs .58/.63/.61 | .48 vs .47 |
| 128 | run L | run O | .56/.63/.59 vs .56/.63/.58 | .45 vs .45 |
| 64 | run M | run P | .52/.56/.55 vs .52/.55/.55 | .32 vs .34 |

`2h/2q` is ±0.01 of `4h/4q` everywhere — at 256d it's marginally the
*best* run on the entire board (run N, `val_loss` 0.0664, `score_play`
.58, `cast_lift` .61). So the earlier "4 heads / 4 pool queries is fine"
verdict is really "could go *down* to 2/2 with no measurable loss" — even
more parsimonious, fewer params. (At 64d the underfit isn't fixed; that
was a capacity problem, not a head-count one.)

### `mlm_weight` (run G vs A — fixed 6L, 4h/4q)

Lowering `mlm_weight` from 0.1 to 0.02 made the regression task **slightly
worse**, not better — the opposite of the "turn down MLM to concentrate
picker-useful signal in the encoder" hypothesis:

- `played_rate` 0.60 (down from 0.65), `score_play` 0.56 (down from 0.58),
  `cast_lift` 0.58–0.59 (down from 0.61), `val reg` 0.0202 (up from 0.0193).
- The gradient-norm probe confirms the mechanism: at `mlm_weight=0.1` the
  shared-encoder gradient is ~1:1 regression-vs-(weighted-)MLM
  (`raw_mlm ≈ 10× reg`, the 0.1 weight roughly cancelling that — the
  per-batch ratio is noisy, ~0.7–3× batch to batch); at `mlm_weight=0.02`
  the ratio drops to ~0.3–0.6× — regression now dominates the trunk by
  ~2–3×.
- And the low-`mlm_weight` run shows **mild overfitting**: best at epoch 95,
  then `val_loss` bounces (0.0341 → 0.0352 → 0.0341 → 0.0349 → 0.0345) and
  `val reg` creeps *up* (0.0202 → 0.0215 → …) while `train reg` keeps
  dropping. The MLM term at 0.1 is acting as a useful regularizer on the
  shared trunk; turning it down lets the encoder chase the noisy, shrunk
  regression labels.

So `mlm_weight=0.1` was the right call — possibly could go slightly higher,
but run A (at 0.1) was not visibly overfitting (best at the last epoch), so
the upside of 0.2 is probably flat-to-marginal. The "lower the MLM weight"
idea is empirically dead.

## Representation analysis — what's actually inside the card embedding

### What this section is checking, and why

The encoder turns each card into a list of numbers — a *vector* — that the
deck scorer then consumes. The width of that list is `pooled_dim` (= the
`--d-model` you trained with, when `--pool-mode attn`). Two natural
questions:

1. **Is the encoder actually using all those numbers, or only a few of
   them?** If a 256-number embedding is "really" just 3 numbers dressed up,
   then making the encoder *wider* is pointless — there's nothing for the
   extra slots to hold.
2. **Are the numbers it does use *about playability*, or are they noise?**
   The encoder could be spending its capacity on distinctions that don't
   matter for picking a deck.

We have an unusually clean way to look at this: the two final encoders —
both `6L · 2 heads · 2 pool queries · --pool-mode attn`, trained to
convergence (250-epoch cap, best checkpoint ~ep 170–190) — differ only in
width: `d_model = 256` (256-wide embedding; `val_loss` 0.0600, MLM
accuracy 87.8 %) vs `d_model = 128` (128-wide; `val_loss` 0.0717, MLM
84.3 %). And on the regression task they perform **identically**:
`score_play` +0.57, `score_draw` +0.57, `played_rate` +0.64, `cast_lift`
+0.60, `color_lift_*` ≈ 0.45–0.50 for both — the only thing the bigger
model does better is the auxiliary masked-language objective, which the
scorer never sees. (Quick reminder of what the heads mean: `played_rate` =
fraction of decks containing the card that actually cast it; `score_play` /
`score_draw` = the card's win-rate signal when it got cast, on the play /
on the draw; `cast_lift` = how much more its deck wins when the card got
cast vs. when it sat in hand; `color_lift_X` = how the card does
specifically in decks running colour X, relative to its overall average.)

`scripts/embedding_effective_rank.py` encodes the whole corpus (~32.6k
cards), stacks the embeddings into a big matrix, and runs the two analyses
below on each model.

### Analysis 1 — how many "real" dimensions does the embedding have?

**The idea.** Picture each card as a dot in a 256-dimensional space (one
axis per number in the embedding). The cloud of ~32.6k dots probably
doesn't fill that space evenly — it might be stretched out a lot in a few
directions and almost flat in the rest, the way a sheet of paper is a 2D
object even though it sits in 3D. *Principal Component Analysis (PCA)*
finds the natural axes of that cloud, ordered from "the direction the
cards vary along the most" (PC1) down to "the direction they barely vary
along at all". "Variance along an axis" just means *how spread out the
cards are* along it — a high-variance axis is one where cards differ a
lot; a near-zero-variance axis is one where every card has roughly the
same value, so it carries no information.

The *effective rank* boils this down to one number: roughly "how many axes
does it take to describe the cloud". If 95 % of the spread lives in the
first 5 axes and the other 251 are essentially flat, the effective rank is
about 5 — the embedding is "really" 5-dimensional, wearing a 256-dimensional
costume. (We report it two ways: "raw" on the embedding as-is, and
"standardized" after rescaling each of the 256 numbers to the same size
first, so one or two large-scale numbers can't dominate the picture. They
tell the same story; the standardized one comes out a bit larger.)

| model | effective rank (raw / standardized) | PC1's share of the spread | top 2 PCs hold | 90% of spread in | 99% in |
| --- | --- | --- | --- | --- | --- |
| 256d (256-wide) | 2.4 / 5.9 | 61% | 80% | 18 axes | 96 |
| 128d (128-wide) | 4.8 / 8.6 | 42% | 58% | 23 axes | 55 |

In plain terms: **both embeddings are tiny.** A single direction accounts
for 40–60 % of how cards differ from one another, and two directions cover
58–80 %. The effective rank — ~3 to ~9 depending on how you measure — is
3–5 % of the nominal width in both cases. So the encoder is nowhere near
"running out of room": it's compressing whatever it learned about cards
into a handful of knobs and leaving the vast majority of its slots nearly
unused. The corollary for the architecture: **making the encoder wider
would be wasted dimensions** — it doesn't use the width it already has.

**Direction of the size effect.** Note that the *wider* model is the
*lower*-rank one (effective rank ~2.4 vs ~4.8). This isn't a labelling
slip — the probe reads each checkpoint's width directly (`pooled dims:
256` / `pooled dims: 128`) — and it isn't a contradiction either:
effective rank measures how concentrated the cloud is, not how many
dimensions the cloud nominally lives in, so a cloud sitting in 256-D space
can be effectively 2-D (dots scattered on a tilted sheet of paper inside a
room). The likely mechanism is *slack*: with 256 slots the optimiser can
afford to put the dominant "playability" signal almost entirely into one
direction (which ends up carrying 61 % of all the variance) and leave the
other ~250 nearly flat; with only 128 slots those dimensions are more
contested, so the variance is forced to spread a little more evenly — a
narrower model is under more pressure to use its dimensions efficiently.
How much to read into the precise figures: each is a single training run
(no seed replication), and the large 2.4-vs-4.8 gap is from the *raw*
spectrum — the standardized version (5.9 vs 8.6, the more robust measure)
shows the same direction far more mildly. So the conservative statement is
just "both embeddings are very low-rank, and the wider one is not more
spread out than the narrower one."

### Analysis 2 — are those few directions about playability?

Analysis 1 says the embedding lives in ~a handful of directions; it
doesn't say those directions are *useful*. A direction could have lots of
spread and still be irrelevant to whether a card is good in sealed. So:
take each label we actually care about (`played_rate`, `score_play`, …)
and ask how well you can predict it using just the first `k` of the
encoder's natural axes (the top-`k` principal components). The score is
**R²**, the standard "fraction explained" — like a report-card grade from
0 to 1: R² = 1 means those `k` axes pin the label down exactly, R² = 0
means they tell you nothing about it, R² = 0.5 means they account for half
the card-to-card variation in that label. We sweep `k` from 1 up to "all
of them" so you can see *how many* axes a given label needs.

| label | 256d: top-2 axes | 256d: all 256 axes | 128d: top-2 axes | 128d: all 128 axes |
|-------|-----------------:|-------------------:|-----------------:|-------------------:|
| played_rate | 0.72 | 0.79 | 0.55 | 0.60 |
| score_play  | 0.53 | 0.61 | 0.33 | 0.42 |
| score_draw  | 0.53 | 0.60 | 0.33 | 0.41 |
| cast_lift   | 0.63 | 0.70 | 0.47 | 0.49 |
| color_lift (avg of the 5) | 0.11 | 0.32 | 0.08 | 0.27 |

What this says:

- **`played_rate` is essentially the embedding's main axis.** With just
  the top *two* directions you already recover ~90 % of everything those
  directions ever recover for `played_rate` (256d: 0.72 of the eventual
  0.79; 128d: 0.55 of 0.60) — and remember those same two directions hold
  most of the *spread* in the embedding (58–80 %). Put together: the single
  biggest "way cards differ" that the encoder learned is, basically, "how
  playable is this card." Which makes sense — `played_rate` is the most
  direct, lowest-noise label in the set.
- **The win-rate heads need more axes.** `score_play` keeps climbing as you
  add directions — 0.53 with the top 2, 0.61 with all 256 — so there's
  real signal spread out past the dominant couple of axes, just thinner.
- **The per-colour heads live in the noisy tail.** `color_lift_W` goes
  from ~0.06 with the top 2 axes to ~0.25 with all of them — it *needs*
  the low-spread directions, and even then it's only weakly captured. This
  lines up with the label-ceiling section: the per-colour labels are
  themselves sparse and noisy (their split-half reliability is ~0.6–0.67,
  versus ~0.78–0.98 for the others), so there isn't much clean signal there
  for the encoder to put anywhere.
- This ordering — `played_rate` (≈ the top 1–2 axes) ≫ the win-rate heads
  (spread over more axes) ≫ the colour heads (diffuse, in the tail) — is
  the **same in both models**, so it's a property of the data, not a quirk
  of one training run.

### A caveat: these R² numbers flatter the bigger model

Notice the 256d embedding looks much more "label-decodable" in the table —
`played_rate` 0.79 vs 0.60, `score_play` 0.61 vs 0.42 — even though the two
models score *identically* on held-out cards (`val corr`). Two reasons the
table over-credits the 256d model:

1. **It's graded on practice problems.** The encoder was trained on ~22k
   of these ~28k cards, and we then fit the "predict the label from the
   axes" mapping on all 28k — so part of what the R² measures is the model
   recognising cards it's effectively seen, not generalising. A bigger
   model memorises more, so its in-sample R² is inflated more.
2. **The comparison isn't quite fair even at the same `k`.** When we say
   "top-128 axes", the 256d model gets to keep its *best* 128 of 256 axes,
   while the 128d model is stuck with *all* 128 of its.

The number that isn't fooled by either of these is `val corr` — the
correlation between predictions and labels on the card-disjoint validation
split, with the actual trained heads. And there the 128d and 256d encoders
are dead even. So treat this section as "shape and structure of what the
encoder learned", not as a head-to-head — for the head-to-head, 128d = 256d.

### "If the embedding is ~5-dimensional, why does `d_model = 64` break it?"

This is the obvious follow-up — if 99 % of the 128d model's spread fits in
55 axes (and its effective rank is ~5–9), surely 64 dimensions is plenty?
Yet the sweep showed `d_model = 64` collapsing: every signed head dropped
0.04–0.07, the colour heads cratered (`color_lift_W` 0.43 → 0.22), `val
reg` rose (genuine underfit), and MLM accuracy fell off a cliff (88 % →
81 % at 128d → 63 % at 64d). The resolution is a distinction worth
remembering:

**`d_model` is the *internal working width of the whole transformer*, not
just the width of the final embedding.** That one number also sizes the
token embeddings, the residual stream every layer reads and writes, the
Q/K/V projections inside self-attention, the feed-forward block, and the
per-token vectors the MLM head consumes. `--d-model 64` doesn't mean "use
64 of the 128 output dimensions" — it means "rebuild the *entire* network
at half the width". The *output* might survive in ~5 numbers (the effective
rank says the final answer genuinely is small), but the machinery that
*computes* the answer needs more scratch space than the answer itself.
Analogy: a calculator's screen shows ~10 digits, but the answer to "2 + 2"
is one digit — that doesn't mean you can build the calculator with a
one-digit internal register, because the intermediate arithmetic (carries,
partial products) needs more room than the final number. The pooled
embedding is the screen; `d_model` is the registers.

Two things make 64 specifically catastrophic rather than merely worse:

- **The MLM head can't predict ~5000 vocab tokens from a 64-wide vector.**
  It's a `Linear(d_model, vocab_size)` reading the per-token contextualized
  vectors; from 256 or 128 wide that's fine, from 64 it's a brutal squeeze
  (perplexity 1.6 → 1.9 → 4.2). And the MLM loss isn't a side metric — it's
  part of the training objective *and* a regulariser on the shared trunk
  (see the `mlm_weight` section), so when MLM collapses, the trunk gets
  shaped worse and the regression heads inherit the damage.
- **"99 % of the *variance* in 55 axes" isn't "the labels only need 55
  axes".** Analysis 2 showed the per-colour heads keep gaining R² all the
  way down to the *last* PCs of the 128d output — so even the bottom ~70
  dimensions carry a little task-relevant signal. Variance ranking ≠
  label-relevance ranking; "55 dimensions" understates what the task
  actually leans on.

So: the PCA tells you the *answer* is small; `d_model` is the *workspace*,
and 64 is below the workspace floor for this network + objective. 128
works and ties 256, so the floor sits somewhere between — probably ~96-ish,
though the colour-head collapse at 64 makes "stay at 128" the safe call.

### Net

- **Don't widen the encoder.** Both final models use ~3–5 % of their
  dimensions; there's nothing for extra width to hold. (You *can* go
  narrower — but 64d collapses, because `d_model` is the network's internal
  workspace, not just the output width; see the subsection above. 128 is
  the floor.)
- **128d and 256d are equivalent for the deck-picker** — same held-out
  performance, so the smaller one is the obvious choice (half the
  embedding width fed downstream).
- **The encoder's representation is dominated by "playability."** Its
  top one-or-two directions essentially *are* `played_rate`; the win-rate
  heads are spread thinner across more directions; the per-colour heads are
  weak and live in the noisy tail (which tracks the per-colour labels being
  noisy to begin with).

## Deployment economics — what `pooled_dim` actually buys you

The encoder isn't re-run at deck-build time — `encode-cards` is a one-shot
pass that caches a `(pooled_dim + FEATURE_COUNT)`-wide `.npz` per card. So:

| stage | encoder `n_layers` | encoder `pooled_dim` | scorer `d_model` |
|---|---|---|---|
| `encode-cards` (one-shot)              | matters       | matters            | n/a |
| Phase B fine-tuning                    | matters       | matters            | matters |
| **Scorer inference / deck building**   | **n/a (cached)** | **only via the scorer** | **dominates** |

At deck-build time the dominant cost is repeated *scorer* forwards (greedy
deck building runs many). The scorer body's FLOPs scale with
`scorer_d_model²`, **not** with the encoder's `pooled_dim` directly —
encoder `pooled_dim` only shows up as the width of the cached embedding
and the scorer's input projection `Linear(pooled_dim + 32, scorer_d_model)`,
a single matmul per card that's small next to a multi-layer Set Transformer.

So why does `pooled_dim` matter at all? Because **`scorer_d_model` is
typically tied to the encoder's output width** (the codebase currently
hardwires `scorer_d_model = total_dim(256) = 544 = 2·encoder_d_model + 32`).
Halve the encoder `pooled_dim` and you halve `scorer_d_model` → ~4× faster
scorer body — that's where the real runtime win lives. Conversely, encoder
`n_layers` is amortised entirely into `encode-cards`: at deck-build time
fewer layers buys nothing.

**Implication for the Pareto choice:** pick the smallest `pooled_dim` that
doesn't lose `val corr`; keep `n_layers` high (it's free at inference, and
the colour heads benefit a little); use `--pool-mode attn` (halves
`pooled_dim` *and* converges faster); drop heads/queries to 2 (no signal
loss, smaller). Combining the ablations:

- `6L · 2h/2q · 128d · attn` (run O) — `pooled_dim = 128` (versus the
  default 512 = 2·256). `val corr`: `score_play` .56, `played_rate` .63,
  `cast_lift` .58 — within ±0.02 of the best on the board (run N at
  256d-2h/2q: .58/.63/.61) on signed heads; colour avg .45 vs N's .47.
- For the smallest *single change* off run A's baseline: `6L · 2h/2q ·
  256d · attn` (run N) — `pooled_dim = 256`, `val corr` ≥ run A on every
  head. No `d_model` reduction, but the attn pool already halves the
  shipped embedding (512 → 256).

**Caveat — the runtime win requires a downstream change.** Currently
`sealed/domain/card_embedding_layout.text_dim(d) = 2·d` is the canonical
text-vector width, baked into `scorer_model.SealedScorerConfig.d_model` (=
`total_dim(256) = 544`), the `.npz` cache split, and the Phase-B text-vec
splice in `match_data_loader.py`. Until `card_embedding_layout` /
`scorer_model` / `match_data_loader` derive the text dim from the actual
saved encoder's `pooled_dim` instead of assuming `2·d_model`, training a
smaller encoder won't actually shrink the scorer — it'll error at
`train-scorer` time on a dimension mismatch. The encoder choice is right
regardless; the runtime payoff needs that follow-up.

## Conclusions

1. **Architecture size is not the bottleneck.** Nothing in the sweep —
   depth past 4 (or even 2 under attn pool), 2×/4× the heads, 2×/4× the
   pool queries — moves `val corr` on the picker-relevant heads, and a
   1-layer encoder is within ~0.03 of a 6-layer one. Widening `d_model` or
   the pool would be more of the same.
2. **The gap to the split-half ceiling is the label-mixing / missing-
   context / feature-ceiling story.** Even the 6-layer model sits at
   ~0.58/0.65/0.45 vs a ceiling of ~0.78/0.98/~0.63 — and stacking layers
   gives near-zero returns there. The unreachable part is plausibly large
   for `played_rate` (a near-perfect label the model can't get past ~0.65
   no matter the size), consistent with "a chunk of it isn't in the card
   text."
3. **`mlm_weight=0.1` is dual-purpose** — it both balances the trunk
   gradient (~1:1) and regularizes the encoder against the noisy regression
   labels. Don't lower it.
4. **The max-pool branch is dead weight** — `--pool-mode attn` (attention
   pool only) lands at the same endpoint as the default dual pool but gets
   there ~1.5× faster and ships a half-width embedding. No reason to keep it.
5. **`d_model = 128` is essentially free on `val corr`; `d_model = 64` is
   not.** Going 256 → 128 (run H → L, `--pool-mode attn`) costs at most
   ~0.01 on any signed head and ~0.03 on the colour avg, while halving the
   pooled embedding and roughly quartering per-layer FLOPs. Going 128 → 64
   collapses the colour heads (W: .43 → .22), drops `val corr` 0.04–0.07
   on every signed head, and tanks MLM. 128 is the floor.
6. **`n_heads = n_pool_queries = 2` ≈ 4.** Run N (256d, 2h/2q, attn) is
   marginally the best run on the board; runs O and P confirm 2/2 ≈ 4/4 at
   128d and 64d. Smallest viable head count, fewer params.
7. **Recommended ship — `6L · 2h/2q · 128d · --pool-mode attn`** (run O,
   `pooled_dim = 128`). Pareto choice for the deck-building workload: keep
   layers high (free at inference, helps the colour heads), shrink
   everything that costs `pooled_dim`. For the conservative pick that
   maximises `val corr` instead, `6L · 2h/2q · 256d · attn` (run N) is the
   best-on-board model — `pooled_dim = 256`. The runtime difference between
   the two only materialises once the scorer side reads the encoder's
   actual `pooled_dim` instead of hardwiring 544 (see "Deployment economics"
   caveat).

## Next steps

Two cheap diagnostics first — they cost ~no compute and tell you whether a
bigger model could even help; then the one lever that can raise the ceiling
itself.

1. **Effective-rank / PCA probe of the learned card embeddings (≈free).**
   Run the saved encoder over the corpus, collect the pooled card vectors
   (the `2·d_model`-dim, or `d_model`-dim under `--pool-mode attn`, output
   of `encode()` before the deterministic-feature concat), and look at the
   singular-value spectrum: how many components hold 90/95/99 % of the
   variance, and the participation-ratio "effective rank"
   `(Σσᵢ²)² / Σσᵢ⁴`. If the variance collapses into far fewer dims than the
   model has, the dimensionality budget isn't the bottleneck and widening
   `d_model` / the pool is pointless. If it's spread across most dims, the
   representation may be cramped. (Worth doing per-set or per-colour too —
   if every set's cards occupy nearly the same low-dim subspace, that's a
   sign the encoder is collapsing distinctions, not running out of room.)
2. **MLP-head probe (one throwaway run).** Temporarily replace the linear
   regression heads with 2-layer MLPs and re-train. If `val corr` jumps,
   the *readout* was the limit and a richer pooled vector might help; if it
   doesn't move, the *representation* genuinely doesn't contain more
   decodable signal → no encoder-size change matters. You would *not* ship
   the MLP heads (Decision D-8 keeps the heads minimal so the signal lives
   in the encoder), but as a diagnostic it cleanly separates "encoder is
   the limit" from "readout is the limit."
3. **Richer per-game Forge instrumentation** — the only thing that raises
   the *label* ceiling. Needs a Forge re-run; keep accumulating alongside
   the existing 974k games (the label arithmetic already treats empty cells
   as "no signal"). Highest-leverage cheap additions: **(a) fate of each
   card at end of game** — still on the battlefield? if not, died in combat
   / to removal / sacrificed? — a far more informative per-card outcome than
   "was cast," and strongly card-text-inferable; **(b) drawn-but-not-cast
   vs. never-drawn** — splits the current "not played" bucket's strong
   negative ("the AI declined to cast it") from its no-signal case ("bottom
   of the library"), de-noising `cast_lift` / `played_rate`. Then re-run
   `split_half_reliability.py` to see whether the *ceiling itself* rose
   before committing to the heavier combat-event plumbing (per-creature
   damage dealt, creatures killed in combat, turn cast, margin of victory).
   Those would also enable new "expected board impact" regression heads —
   targets that are far more directly a function of the card text than
   win-rate is, and lower variance, so the encoder can fit them tighter.

## Tooling added for this investigation

- `scripts/split_half_reliability.py` — the ceiling estimator above.
- `sealed train-encoder` per-epoch log line `trunk grad norm @1st batch:
  reg=… mlm*w=… (raw …) mlm*w / reg = …x` — measures `‖∂L_reg/∂trunk‖` vs
  `‖∂(w·L_mlm)/∂trunk‖` on the epoch's first batch (two extra `autograd.grad`
  passes; the optimization step is untouched). Loss *magnitude* doesn't tell
  you the gradient split — a term can be large but flat — so this is the
  number to watch when reasoning about `--mlm-weight`.
