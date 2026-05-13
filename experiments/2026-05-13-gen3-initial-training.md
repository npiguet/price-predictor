# Gen3 Scorer — Initial Training Runs

## Background

"Gen3" is the next scorer iteration after gen2. The architectural and training
recipe is similar to gen2's best configuration (6-layer Set Transformer, dropout
0.2, `--lr 1e-5`, AdamW with grad-norm-1.0 clipping) but **gen3's input
representation is different**: the per-card embedding is supplied by a new
sealed-trained encoder built per the spec at
[`specs/2026-05-03-card-winnability-pretraining.md`](../specs/2026-05-03-card-winnability-pretraining.md).

That spec trains a dedicated encoder from random init on per-card targets
derived from per-game play data (nine regression heads — net winning influence
on the play and on the draw, played rate, cast-lift, and a per-color affinity
for each of WUBRG — plus a masked-token reconstruction auxiliary). It replaces
the 512 price-predictor encoder dims that gen1/gen2 used as their per-card
representation. The motivation, established in
[`experiments/2026-05-02-deterministic-feature-reliance.md`](2026-05-02-deterministic-feature-reliance.md),
was that the price-predictor encoder is loosely aligned with sealed playability
and confounded by collector/reserved-list effects, leaving per-card quality
discrimination as the dominant remaining bottleneck.

The gen3 scorer evaluated below is `256-best_l6_h4_s4_ff1088_mlp256_lr1e-05.pt`
(saved under `models/sealed/scorer/gen-3/`), trained on cards encoded with the
new 256-d-trunk sealed encoder.

## Headline result

Two independent `evaluate-scorer` runs, same checkpoint, same configuration
(`--pools 24 --best-of 7 --workers 12`, set randomly chosen each run):

| Run     | Aggregate (scorer / Forge) | Games | Mean per-pool delta | Per-pool min / max | Pools positive |
|---------|----------------------------|-------|---------------------|--------------------|----------------|
| 1       | 64.0% / 36.0%              | 3157  | +28.7 pp            | +2.0 / +52.4 pp    | 24 / 24        |
| 2       | 58.4% / 41.6%              | 3174  | +17.3 pp            | −3.5 / +41.1 pp    | 23 / 24        |
| **Combined** | **~61% / ~39%**       | 6331  | **~+23 pp**         | −3.5 / +52.4 pp    | **47 / 48**    |

The result is unambiguously positive: the gen3 scorer's greedy-built deck
beats the Forge-built deck (`forge-best`) on the same pool in 47 of 48 pools
across the two runs, with the only negative pool losing by just 3.5 pp.

The per-pool deck dumps are saved alongside the checkpoint as
`...-decks.txt` / `...-decks-forge.txt` (run 1) and
`...-decks2.txt` / `...-decks-forge2.txt` (run 2).

## Training behavior

Two gen3 scorers were trained side-by-side on different encoder widths
(128-d trunk and 256-d trunk). Both use the same recipe — `--epochs 200
--patience 20`, batch size 64, AdamW with per-parameter-group max-norm 1.0
grad clipping, 6 SAB layers, dropout 0.2, `--lr 1e-5`, on the same
`match-outcomes-all.txt` corpus (41,359 matches → 33,087 train + 8,272 val,
27,119 unique cards). The scorer input dim is
`encoder_trunk + 32 deterministic features`, giving `d_model = 288` for
the 256-d run and `d_model = 160` for the 128-d run. **Both checkpoints
are retained** as deck-builders for the next round of self-play match
generation — having two scorers with different per-card priors is a
feature for self-play, not a wart, since the match corpus benefits from
deck diversity beyond what a single "best" scorer would produce.

| Encoder | d_model | Peak val_acc | Peak epoch | Min val_loss | Min-loss epoch | Stopped at | Wall time |
|---------|---------|--------------|------------|--------------|----------------|------------|-----------|
| 128-d   | 160     | 0.7170       | 23         | 0.5454       | 25             | e43        | ~21 min   |
| 256-d   | 288     | 0.7337       | 16         | 0.5304       | 14             | e36        | ~20 min   |

256-d is the stronger scorer on val_acc and was the model evaluated against
`forge-best` above. Both checkpoints feed the next self-play generation as
deck-builders alongside the existing Forge methods.

### Both encoders push above gen2's ~0.70 val_acc plateau

Every gen2 architecture and regularization combination converged at
val_acc ≈ 0.695 — see the "four orthogonal interventions failing" table in
[`2026-04-26-gen2-initial-training.md`](2026-04-26-gen2-initial-training.md). Gen2's analysis
concluded that ceiling was the irreducible Bo7 label-noise floor, estimated
the oracle ceiling at ~0.72–0.78, and identified per-card features (encoder
unfreezing or replacement) as the only remaining lever. Gen3 confirms that
prediction:

| Configuration                | Encoder source  | Corpus (matches) | Peak val_acc |
|------------------------------|-----------------|------------------|--------------|
| gen2 6L + dropout 0.2 (best) | price-predictor | ~27K             | 0.7015       |
| gen3 128-d                   | sealed-trained  | 41K              | 0.7170       |
| **gen3 256-d**               | sealed-trained  | 41K              | **0.7337**   |

The 256-d gen3 result lands in the middle of the gen2-predicted oracle
ceiling band, which is consistent with the gen2 plateau being a per-card
encoder limit rather than the irreducible noise floor — but the gen2→gen3
val_acc gap has **three** confounded effects, not the two an encoder-only
read of it would assume:

1. **Encoder swap** (price-predictor → sealed-trained).
2. **Corpus size growth** (~27K → 41K matches).
3. **Corpus composition shift.** The 14K new matches added since gen2 are
   not a representative sample of the original mix — they came from new
   self-play sources (e.g., gen2-vs-Forge-method matches) that change the
   matchup-difficulty distribution. Per the gen2 oracle-ceiling analysis,
   val_acc is bounded by the average over per-matchup Bo7 oracle accuracies,
   which directly depends on what fraction of the corpus is easy
   (forge-best-vs-random ≈ 97% oracle) vs hard (mirror, gen-vs-forge-best
   in the ~47–50% range). A corpus that has shifted toward easier matchups
   will lift val_acc even with the model held fixed; one that has shifted
   toward harder ones (more near-mirror gen-self-play) will drop it. The
   composition shift's *direction* is not measured in this writeup, so the
   confounder is undetermined-in-sign, not bounded-from-below.

The +3.2 pp gen2→gen3 lift therefore cannot be attributed cleanly to the
encoder. What the logs *do* isolate is the **+1.67 pp 128-d → 256-d effect**
on a fixed corpus: encoder width contributes a meaningful slice of the
total improvement, and that slice at least is clean. The remaining +1.55 pp
(gen2 best → gen3 128-d) sits in the three-way confound and would need a
gen2-recipe rerun on the 41K corpus to disentangle.

### Min val_loss and peak val_acc nearly coincide

In every gen2 run, min val_loss preceded peak val_acc by 5–10 epochs, and
gen2 explicitly recommended switching the early-stop criterion to val_loss
because val_acc kept climbing past the loss-optimal point. In gen3, the two
metrics travel together — 2 epochs apart in either direction (256-d:
loss-min e14, acc-peak e16; 128-d: acc-peak e23, loss-min e25).

The earlier signature gen2 warned about — "model becomes more confident on
the training set in ways that flip a few borderline validation pairs the
right way (raising val_acc) while making the average validation
log-likelihood worse" — is essentially absent in gen3. Whether the
val_acc-based early-stop policy stays defensible long-term depends on this
holding under other configurations; for these two runs it's a non-issue.

### Score magnitudes stay bounded

Gen2 documented the score std growing ~10× over training (e.g. its 5-layer
no-dropout run: winner std 0.50 → 5.07 by epoch 29) and called it "the
typical signature of an unconstrained scoring head that AdamW with weight
decay would rein in." Gen3 256-d's winner/loser stds stay below 2.0
throughout (e36: winner std 1.47, loser std 1.95). Grad-norm means grow from
~1.4 → ~4.0 over the run with max never exceeding ~8 — gen2's sab0 spikes
of 12–18 at 2 layers and 27–46 at 6 layers (without dropout) are gone.

The per-parameter-group max-norm 1.0 clipping (added to `train-scorer`
after the gen2 analysis recommended it, alongside the dropout default) is
doing the work it was meant to.

### Faster epoch-1 learning

Gen3 256-d reaches val_acc 0.7101 at epoch 1; the gen2 sweep reported
"every run reaches val_acc ≥ 0.665 at epoch 1 and ≥ 0.68 by epoch 4."
That's +4–5 pp at epoch 1, consistent with the new encoder giving the
scorer more useful per-card information from the start rather than the
gain coming purely from longer training.

## What this resolves about the gen1/gen2 reward-hacking concern

Gen1's known failure mode was "rates its own greedy decks highly, but those
decks lose to `forge-best` ~57% of the time at Bo7" — the over-confidence /
reward-hacking pattern documented in `2026-04-26-gen2-initial-training.md`. Gen3 reverses
the sign of that gap by a wide margin. The downstream story (the in-play
performance of greedy decks built from scorer scores) is no longer the
limiting factor for scorer deployment.

The new encoder is the most plausible reason: gen2 made the scorer better at
discriminating *deck shape* (color count, curve), but per-card quality
discrimination kept the ceiling near `forge-best` on matched-shape decks
(see `2026-05-02-deterministic-feature-reliance.md`). The sealed-trained encoder targets
exactly that gap, and the eval result is consistent with that gap closing.

## Variance: pool-set luck is still substantial at 24 pools × Bo7

The two runs differ by **11.4 pp of mean per-pool delta** (28.7 vs 17.3),
with the same scorer and same configuration. The only thing that changed
between runs is which set was randomly selected (`--set` was not pinned) and
the random pool draws. That fixes a useful number for future eval planning:

- Even at 24 pools × Bo7 × 576 matches, a single evaluation's mean delta
  carries roughly a ±5 pp uncertainty.
- Reporting only one run's headline number is misleading. The combined view
  across both runs (47/48 pools, ~+23 pp mean delta) is more representative
  than either run alone.
- The 12-pool variance flagged in earlier work (`project_scorer_training_diagnosis`
  memory note: ~20+ pp swings on pool-set luck) tightens roughly ~2× at
  24 pools but does not collapse.

## Score-delta predicts win-rate-delta per pool

For each of the 48 pools we have both the win-rate delta from match outcomes
and the scorer-assigned score of each of the two decks (the headers on the
`=== Deck N  score=±X.XXXX ===` lines in the dumped deck files). The two
quantities — `score_dlt = scorer_score(scorer_deck) − scorer_score(forge_deck)`
and `wr_dlt = scorer_wr − forge_wr` — correlate non-trivially across pools:

| Subset        | Pearson r | Spearman | OLS slope (pp wr_dlt per 1.0 score) |
|---------------|-----------|----------|-------------------------------------|
| Run 1 (24)    | +0.288    | +0.269   | —                                   |
| Run 2 (24)    | +0.609    | +0.565   | —                                   |
| Combined (48) | +0.522    | +0.532   | +7.80                               |

Combined OLS fit: `wr_dlt(pp) ≈ +7.81 + 7.80 · score_dlt`.

The score gap explains roughly a quarter of the win-delta variance — useful
signal, not deterministic. Concretely:

- **Largest score gap and clearest in-play correlate.** Run 2 Pool 8 has the
  most-negative Forge-deck score in the whole dataset (−2.43) and the largest
  score gap (+3.82); the in-play win delta lands at +31.1 pp. When the scorer
  rates the Forge deck deeply negative, it's flagging a real mana-base or
  composition pathology that shows up in matches.
- **Only negative-delta pool fits the trend.** Run 2 Pool 9 (wr_dlt −3.5 pp,
  the only pool the scorer loses) has the smallest score gap of any pool
  in either run (+0.54). It's not an exception to the relationship — it's the
  low-confidence end of it.
- **Outliers go in both directions.** Run 1 Pool 8 (score gap +1.48, wr_dlt
  only +2.0 pp) and Run 2 Pool 15 (+1.04, +1.8 pp) are moderate-confidence
  predictions that didn't pay off in play; consistent with the noise level
  the r=0.52 fit implies.

### Why run 2's correlation is stronger than run 1's

Run 2's per-run r (+0.61) is more than 2× run 1's (+0.29). The mechanical
reason is range: run 1's score deltas span +1.13 to +3.20 (a 2.07-wide
window), while run 2 spans +0.17 to +4.02 (3.85-wide). Wider input range
makes a linear relationship easier to detect at fixed noise level. There's
nothing about run 2 the model "did better" — the random pool draw simply
produced more variety in deck-pair quality.

### Why this is useful

The score gap can be read off a deck dump in milliseconds, while a single
24-pool Bo7 eval takes hours. The gap is a useful first-order indicator of
expected win-rate delta when inspecting candidate decks or comparing
scorer variants, with these caveats:

- Treat the relationship as a "moderate predictor," not a substitute for
  match play. r=0.52 leaves three-quarters of the variance unexplained.
- Small score gaps (<1.0) carry weak signal; expect noisy outcomes.
- Strongly negative scores on a `forge-best` deck are the most informative
  signal — they correspond to the largest observed win deltas.

## Decisions / next steps

- **Use both gen3 checkpoints (128-d and 256-d) as deck-builders in the
  next round of self-play match generation.** The `forge-best`-vs-scorer
  head-to-head on the 256-d checkpoint is decisively positive, and gen3
  as a whole supersedes gen2 as the default scorer family. Keeping both
  widths in the deck-builder rotation adds diversity to the training
  corpus that the next scorer iteration will consume.
- **Expand evaluation to more pools or pin the set.** A single 24-pool eval's
  ±5 pp uncertainty is uncomfortably wide for ranking gen3 variants against
  each other. Either run two independent 24-pool evals per candidate, or pin
  `--set` and use the same pool seeds across candidates to remove pool-set
  luck as a confounder.
- **Use the score-delta-vs-win-delta correlation as a sanity check, not a
  benchmark.** It's a cheap diagnostic on the scorer's internal calibration,
  but not a substitute for actual match play when comparing two scorers.
- **Reassess the data-noise-floor reasoning from gen2.** Gen2's analysis
  concluded the val_acc ceiling at ~0.70 was data-limited rather than
  model-limited, with the encoder identified as the upstream constraint. Gen3
  is consistent with that diagnosis — replacing the encoder is what moved the
  in-play result. The val_acc ceiling for gen3 itself has not been measured
  against the gen2 ceiling yet; that's the next experiment.
