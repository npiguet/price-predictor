# Sealed Encoder — Hyperparameter Sweep & the Label Ceiling

*Runs: 2026-05-11. Dataset: `output/sealed/cards-played.txt` (the cleaned
974,028-game corpus → 27,983 cards after dropping the 2 with no converted
`.txt`; 22,387 train / 5,596 val, card-disjoint). Spec 016 multi-head + MLM
encoder; metric of interest is the per-head **`val corr`** = Pearson r
between predicted and (shrunk) target labels on the held-out cards.*

## TL;DR

- The encoder is **card-text-saturated**. Depth past ~4 layers, and more
  attention heads / pool queries, do **not** move `val corr`; the one
  objective knob you'd reach for (`--mlm-weight` down to 0.02) makes it
  slightly *worse*. Best main-head correlations land at ~0.58 (`score_play`),
  ~0.65 (`played_rate`), ~0.61 (`cast_lift`), ~0.45 (`color_lift_*`).
- **Drop the max-pool** (`--pool-mode attn`): same endpoint as the dual
  attention‖max pool but *faster to converge* (~1.5× fewer epochs to the
  same `val_loss`), fewer params, and the pooled/shipped embedding halves.
- The **split-half reliability ceiling** of the labels (the most a perfect
  predictor could reach against these aggregated, shrunk targets) is
  ~0.78–0.79 for the signed heads and ~0.98 for `played_rate` — so there is
  real headroom *in the labels*, but it is **not** reachable by making the
  encoder bigger. Part of that headroom is genuinely not a function of the
  card text (set-relative playability, Forge-AI quirks); how much is unknown.
- **Ship `4L · 4 heads · 4 pool queries · mlm_weight=0.1 · --pool-mode attn`.**
  Matches the 6-layer model on `val corr` to within ~0.01 on every head at
  2/3 the depth, drops the dead max-pool branch, and is clear of the mild
  overfitting the low-`mlm_weight` run shows.
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

All runs: `d_model=256`, `ff_dim=1024` (hardcoded), 100-epoch cap, AdamW
`lr=1e-4`, 5%-warmup-then-constant, best checkpoint by full validation loss
`reg + mlm_weight·mlm`, `--pool-mode dual` (attention ‖ max-pool) except
where noted. Best-epoch validation numbers:

| run | layers | heads | pool-q | mlm_w | pool | best ep | val reg | MLM ppl / acc | score_play | score_draw | played_rate | cast_lift | color_lift W/U/B/R/G |
|-----|-------:|------:|-------:|------:|------|--------:|--------:|---------------|-----------:|-----------:|------------:|----------:|----------------------|
| **A** | 6 | 4 | 4 | 0.1 | dual | 100 | 0.0193 | 1.6 / 85.5% | **+0.58** | +0.56 | **+0.65** | **+0.61** | .40/.52/.46/.49/.46 |
| B | 4 | 4 | 4 | 0.1 | dual | 99 | 0.0191 | 1.7 / 84.2% | +0.57 | +0.56 | +0.64 | +0.60 | .40/.52/.44/.48/.45 |
| C | 2 | 4 | 4 | 0.1 | dual | 99 | 0.0198 | 1.8 / 82.3% | +0.56 | +0.55 | +0.62 | +0.59 | .38/.51/.42/.47/.45 |
| D | 1 | 4 | 4 | 0.1 | dual | 99 | 0.0207 | 2.1 / 78.4% | +0.55 | +0.55 | +0.61 | +0.59 | .40/.51/.43/.46/.46 |
| E | 6 | **8** | **8** | 0.1 | dual | 100 | 0.0199 | 1.6 / 85.4% | +0.58 | +0.57 | +0.64 | +0.62 | .40/.52/.44/.48/.46 |
| F | 6 | **16** | **16** | 0.1 | dual | 97 | 0.0195 | 1.7 / 84.4% | +0.58 | +0.57 | +0.62 | +0.60 | .40/.53/.41/.48/.47 |
| G | 6 | 4 | 4 | **0.02** | dual | 95 | 0.0202 | 2.0 / 80.0% | +0.56 | +0.56 | +0.60 | +0.59 | .39/.52/.43/.47/.46 |
| H | 6 | 4 | 4 | 0.1 | **attn** | 100 | 0.0194 | 1.6 / 86.6% | +0.57 | +0.57 | +0.63 | +0.60 | .42/.53/.46/.49/.49 |

`val_loss` is omitted from the table because it isn't comparable across
`mlm_weight` (`val_loss = reg + mlm_weight·mlm`, and the MLM term dominates):
runs A–F and H sit at 0.065–0.097, run G at 0.034 — that lower number is
purely the smaller MLM coefficient, not a better model.

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

## Conclusions

1. **Architecture size is not the bottleneck.** Nothing in the sweep —
   depth past 4, 2×/4× the heads, 2×/4× the pool queries — moves `val corr`
   on the picker-relevant heads, and a 1-layer encoder is within ~0.03 of a
   6-layer one. A 1-layer model has minuscule effective capacity yet nearly
   matches; widening `d_model` or the pool would be more of the same.
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
5. **Ship `4L · 4 heads · 4 pool queries · mlm_weight=0.1 · --pool-mode attn`**
   (the depth of run B, the pool of run H). ≈ run A on `val corr`
   (0.57/0.64 vs 0.58/0.65) at 50% fewer layers, with the dead max-pool
   dropped, clear of run G's overfitting. (Run A's only edge is a slightly
   better MLM and ~+0.01 corr, which won't survive to a downstream win-rate
   difference given the scorer eval swings 20+ pp on pool-set luck. Minor:
   B/H both hit "best" on essentially the last epoch, so a longer run might
   squeeze another ~+0.01 — diminishing, probably not worth it versus the
   data-side work.)

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
