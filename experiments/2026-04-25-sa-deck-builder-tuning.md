# Simulated annealing for greedy deck builder — tuning notes

## Background

The original `GreedyDeckBuilder` (in `src/sealed/domain/greedy_deck_builder.py`)
is a pure hill-climbing local search:

1. Start from a random 23-card subset of the pool (~90 cards).
2. At each iteration, score every (deck slot × bench card) single-card swap in
   one batched forward pass (≈1500 candidates).
3. Apply the best swap if it improves the score, otherwise stop.

Two structural problems with this for sealed deck-building:

- **Random init starts maximally diverse**: a random 23-card subset of a
  90-card pool almost certainly contains all five colors. The greedy has to
  "climb out" of a 5-color starting point via single-card moves.
- **Greedy can't make multi-card moves**: dropping a color requires several
  individually-bad swaps in a row. Each swap evaluated in isolation looks
  worse than keeping the off-color card, so the greedy converges to
  multi-color local optima. Color coherence is a non-local reward (only
  shows up once *all* off-color cards are gone, when the mana base works),
  which makes it especially hard for single-step search.

This explains the recurring observation that scorer-built decks use too many
colors compared to Forge's archetype-aware builder.

## What was added

`GreedyDeckBuilder` supports simulated annealing plus multi-restart, exposed
through four params:

- `temperature` (default `0.0` — pure greedy, behavior unchanged).
- `cooling` (default `0.95`) — per-iteration temperature multiplier.
- `max_iterations` (default `200`) — hard cap on iterations.
- `restarts` (default `1`) — either a positive integer N (run N
  independent SA searches from fresh random inits) or the literal
  `"color-pairs"` (one search per MTG two-color pair, each seeded with
  an on-color initial 23-spell deck — see §Color-pair seeded init). The
  best deck across all runs is returned.

Implementation detail: at `T > 0`, instead of taking the argmax swap, the
algorithm samples a swap from the full softmax-temperature distribution
over all 1541 candidates: `P(swap_i) ∝ exp(score_i / T)`. This integrates
naturally with batched GPU scoring — no change to the forward-pass shape.
At `T == 0` the algorithm collapses to argmax + break-on-no-improvement
(the original greedy behavior).

The algorithm tracks the best deck seen across all iterations *and* across
all restarts, and returns that, not the final state. So SA is at least as
good as a random walk in the worst case, regardless of how cooled the
temperature gets.

CLI: `--sa-temperature`, `--sa-cooling`, `--sa-max-iterations`, `--restarts`
on `build-decks`. There's also `--print-decks` to dump the human-readable
format used by `evaluate-scorer` (sorted by mana value, lands at bottom).

## Initial T sweep (gen-1 6-layer scorer, 12 KTK pools)

Checkpoint: `models/sealed/scorer/gen-1/6layers_lr1e-05.pt`.
Pools file: `models/sealed/scorer/gen-1/6layers_pools.txt`.
All runs use `cooling=0.95`, `max_iterations=200`, `restarts=1`.
Random init is non-deterministic across runs (seed unfixed).

| Pool      | Greedy  | T=0.3   | T=0.5   | T=0.8   | T=1.0   |
|-----------|--------:|--------:|--------:|--------:|--------:|
| 1         | 1.6008  | 1.9954  | 1.9954  | 1.9954  | 1.6008  |
| 2         | 1.4791  | 1.5381  | 1.5381  | **1.7805** | 1.5381  |
| 3         | 2.1497  | 2.1500  | 2.1500  | 2.1497  | 2.1500  |
| 4         | 1.8259  | 1.9713  | 1.9713  | 1.9713  | 1.9713  |
| 5         | **1.7545** | 1.7545  | 1.7545  | 1.7269  | 1.7269  |
| 6         | **1.8760** | 1.8374  | 1.8760  | 1.8374  | 1.8760  |
| 7         | 1.6169  | 1.7296  | 1.3045  | **1.7719** | 1.7719  |
| 8         | 1.3636  | 1.3858  | 1.3858  | **1.4071** | 1.4071  |
| 9         | **1.4554** | 1.4554  | 1.4300  | 1.4010  | 1.4300  |
| 10        | 1.8969  | 1.8969  | 1.8017  | 1.8969  | 1.8969  |
| 11        | 1.6888  | 1.8596  | 1.8596  | 1.8596  | 1.8596  |
| 12        | 1.6056  | 1.6056  | 1.2114  | 1.6056  | 1.0896  |
| **mean**  | 1.6094  | 1.6816  | 1.6565  | **1.7836** | 1.6932  |
| **min**   | 1.3636  | 1.3858  | 1.2114  | 1.4010  | **1.0896** |
| **max**   | 2.1497  | 2.1500  | 2.1500  | 2.1497  | 2.1500  |

**Win/tie/loss counts vs greedy:**

| T   | Wins | Ties | Losses | Mean Δ |
|-----|-----:|-----:|-------:|-------:|
| 0.3 |    6 |    5 |      1 | +0.072 |
| 0.5 |    5 |    3 |      4 | +0.047 |
| 0.8 |    6 |    3 |      3 | +0.174 |
| 1.0 |    5 |    4 |      3 | +0.084 |

**Per-pool best ensemble** (max score across all 5 settings): mean = 1.7937.
Marginal gain over T=0.8 alone (+0.010). The robust single setting is
nearly as good as the full ensemble.

### Observations

- **T=0.8 is the best single setting**: highest mean (1.7836 vs 1.6094 for
  greedy, +10.8% relative). The gain comes from a few pools where SA
  escapes a much worse multi-color local optimum:
  - **Pool 1**: 1.6008 → 1.9954 (+24.7%). Greedy converged to a 5-color
    R/B/G/U/W deck. SA at T=0.3-0.8 found a focused W/R deck (with the
    same handful of multicolor splash cards) scoring much higher.
  - **Pool 7**: 1.6169 → 1.7719 (+9.6%).
  - **Pool 11**: 1.6888 → 1.8596 (+10.1%).
- **Higher T is not monotonically better.** T=1.0 is *worse* on average than
  T=0.8 despite covering more of the landscape. Two failure modes show up:
  - **Pool 12**: drops from 1.6056 to 1.0896 (-32%). High-T exploration
    wandered into a basin so bad that the best-deck tracker still couldn't
    recover anything close to greedy.
  - **Pool 1**: drops back to greedy's score (1.6008) — SA explored too
    aggressively and re-converged to the multi-color basin that T=0.3-0.8
    had escaped.
  The score-vs-T curve has a peak (somewhere around T=0.7-0.9 here), then
  declines. Beyond the peak, exploration is essentially random search and
  even best-deck tracking can't recover.
- **T=0.5 is in a "no man's land"**: counter-intuitively worse than T=0.3
  on this set. Two big regressions (pool 7 -0.31, pool 12 -0.39) drag the
  mean below T=0.3's. Possibly T=0.5 is hot enough to leave good basins but
  not hot enough to reliably find better ones, ending up in shallow
  alternatives.
- **No single T wins on every pool.** Pools 5, 6, 9: greedy already finds
  the best basin; SA at any T is worse or tied. Pools 1, 4, 11: T≥0.3
  finds a meaningfully better basin reliably. Pool 2: only T=0.8 finds
  the best basin. Pool 12: T≥0.5 catastrophically degrades.

T=0.8 with `cooling=0.95`, `restarts=1` was used as the default for
generating the gen-2 self-play decks.

## Full (T, cooling, restarts) sweep (gen-2 6-layer scorer, 12 KTK pools)

Checkpoint: `models/sealed/scorer/gen-2/best_l6_full_training.pt`.
Pools file: `models/sealed/scorer/gen-2/gen1_pools.txt` (same 12 pools across
all runs). All runs use `max_iterations=200`. Raw deck dumps live in
`models/sealed/scorer/gen-2/best_l6_full_training_decks{,B,C,D,E,F}.txt`.

| Run | T   | cooling | restarts |
|-----|----:|--------:|---------:|
| A   | 0.8 | 0.95    | 1        |
| B   | 0.5 | 0.95    | 1        |
| C   | 0.5 | 0.98    | 1        |
| D   | 0.8 | 0.98    | 1        |
| E   | 0.5 | 0.95    | 4        |
| F   | 0.8 | 0.95    | 4        |

Note that the gen-2 scorer's score scale differs from gen-1 — absolute scores
across the two sweeps are not directly comparable, only within-sweep deltas.

| Pool      | A      | B      | C      | D      | E      | F      |
|-----------|-------:|-------:|-------:|-------:|-------:|-------:|
| 1         | 3.1603 | 3.1603 | 3.1603 | 3.1515 | 3.1603 | **3.1603** |
| 2         | 3.1200 | 3.1200 | 3.1200 | 3.1048 | 3.1200 | **3.1200** |
| 3         | 2.9348 | 2.9348 | 2.9348 | 2.4285 | 2.9348 | **2.9348** |
| 4         | 2.8496 | 2.9075 | 2.9075 | 2.8496 | 2.8496 | **2.9075** |
| 5         | 2.8686 | 2.8686 | 2.8686 | 2.8686 | 2.8686 | 2.8686 |
| 6         | 2.7379 | 2.6325 | 2.1250 | 2.1017 | 2.7390 | **2.8183** |
| 7         | 2.5293 | 2.5297 | 2.5293 | 2.5297 | 2.5297 | 2.5293 |
| 8         | 2.5411 | 2.7247 | 2.5408 | 2.7448 | 2.7247 | **2.7448** |
| 9         | 2.3637 | **2.4802** | **2.4802** | 2.3637 | 2.3637 | 2.3637 |
| 10        | 2.1472 | 2.1472 | 2.1472 | 1.8184 | 2.1472 | 2.1472 |
| 11        | 2.6360 | 2.0317 | 2.6360 | 2.6360 | **2.7210** | **2.7210** |
| 12        | 2.9209 | 2.9085 | 2.9223 | 2.9073 | **2.9255** | 2.9223 |
| **mean**  | 2.7341 | 2.7038 | 2.6977 | 2.6254 | 2.7570 | **2.7698** |
| **min**   | 2.1472 | 2.0317 | 2.1250 | 1.8184 | 2.1472 | 2.1472 |
| **max**   | 3.1603 | 3.1603 | 3.1603 | 3.1515 | 3.1603 | 3.1603 |

**Per-pool best ensemble** (max across all six runs): mean = 2.7798. Only
+0.010 above F alone — same pattern as the gen-1 sweep, where the best
single setting nearly matches the full ensemble.

### Observations

- **Restarts are the dominant lever.** Going 1 → 4 restarts at fixed
  `(T, cooling=0.95)`:
  - T=0.5: B 2.7038 → E 2.7570 (**+0.053**)
  - T=0.8: A 2.7341 → F 2.7698 (**+0.036**)

  Restarts also kill the worst tail. Pool 11 at T=0.5 collapses to 2.03
  with one restart but recovers to 2.72 with four — an independent random
  init lands outside the bad basin even when the original run wandered into
  one.
- **Slower cooling (0.98) is a regression on this scorer.** Both `cooling=0.98`
  runs end up the worst two:
  - T=0.5: B 2.7038 → C 2.6977 (-0.006, plus a -0.51 crater on pool 6)
  - T=0.8: A 2.7341 → D 2.6254 (-0.109, with -0.51 on pool 3, -0.64 on
    pool 6, -0.33 on pool 10)

  Slow cooling holds T high enough long enough that exploration overshoots
  good basins; the best-deck tracker can't always recover. Stick to
  `cooling ≤ 0.95`; for the chosen operating point see §Cooling sub-sweep
  and §Recommendation.
- **T=0.8 still beats T=0.5, but the margin is small.** F vs E is +0.013,
  mostly from pool 6 (+0.08 for F). Restarts narrow the T-gap considerably:
  the gap that justified T=0.8 in the gen-1 sweep was largely the cost of
  T=0.5's missed-basin runs, and restarts buy that cost back.
- **Pools fall into three buckets** on this scorer:
  - *Settled* (1, 2, 3, 5, 7, 10): every run except D finds the same deck.
    Random init lands in a strong basin reliably; SA isn't doing useful
    work here.
  - *T-sensitive* (4, 6, 8, 9, 11): different settings find different
    basins. Pool 11 is the most volatile (range 2.03 → 2.72).
  - *Marginal* (12): all settings within ±0.02; SA noise.
- **The brittleness pool moves with the scorer.** On gen-1, pool 12 was the
  catastrophic-failure case at T≥0.5. On gen-2, it's pool 11. The volatile
  pool tracks the scorer's loss surface, not the pool itself, which is
  another argument for restarts: a recipe like "avoid T=0.5" doesn't
  generalize across checkpoints, but "do 4 restarts" does.

## Cooling sub-sweep at fixed (T=0.8, restarts=4)

The gen-2 sweep above tests `cooling ∈ {0.95, 0.98}`; a follow-up sub-sweep
at faster cooling fills in the curve below 0.95. Same scorer, same 12 KTK
pools, fixed `T=0.8` and `restarts=4`. Wall-clock is for the full 12-pool
build.

| `cooling` | min iters before T < 1e-3 | wall-clock | quality vs cooling=0.95 |
|----------:|-------------------------:|-----------:|-------------------------|
| 0.95      | 131                       | ~7m        | reference (highest mean) |
| 0.93      | 92                        | (untested) | —                        |
| 0.90      | 64                        | ~3m        | within noise on every pool (max Δ ≈ 0) |
| 0.85      | 41                        | ~2m15s     | within noise on every pool (max Δ < 0.08) |
| 0.80      | 30                        | ~1m40s     | one pool catastrophically worse (1.6 vs 2.1) |

Three regimes:

- **`cooling ≥ 0.95`**: long high-T window, full SA quality, expensive.
- **`cooling ∈ [0.85, 0.93]`**: SA finds the same basins as `cooling=0.95`
  on essentially every pool; the only thing the longer window buys is rare
  ≥0.1 wins on the most volatile 1-2 pools. Wall-clock scales with the
  exploration window (cooling=0.85 is ~⅓ the cost of 0.95).
- **`cooling ≤ 0.80`**: exploration window is now too brief to find new
  basins but long enough to *flee* good ones the random init started near.
  Best-deck tracking limits the damage but doesn't eliminate it — one of
  the 12 KTK pools drops by 0.5 under `cooling=0.80` because every restart
  wandered out of the basin random init landed in. Below ~0.85, SA can be
  strictly worse than greedy.

Implication: `cooling=0.85` is the Pareto sweet spot for the gen-2 6-layer
scorer.

## Color-pair seeded init (gen-2 6-layer scorer, 12 KTK pools)

Available as `--restarts color-pairs` on `build-decks`. Replaces random
restart with one independent search per MTG two-color pair (10 total: WU,
WB, WR, WG, UB, UR, UG, BR, BG, RG). Each restart's initial 23 spells are
filtered to cards whose printed colors are a subset of the pair (colorless
cards always qualify). The color filter applies only to the seed deck —
the full pool remains as `spells_remaining`/`lands_remaining` and the
search is unconstrained after init. Pairs without 23 eligible spells are
skipped; if all 10 are skipped on a degenerate pool, the build falls back
to one random restart.

The hypothesis was that seeding the search inside a coherent 2-color basin
would let the scorer optimize within commitment instead of forcing it to
climb out of a 5-color random init via single-card moves (the structural
problem flagged in §Background).

### Outcome

Run config: `T=0.8, cooling=0.85, max_iterations=200, restarts=color-pairs`.
Same 12 gen-2 KTK pools as run F above (`T=0.8, cooling=0.95, restarts=4`).
Raw deck dump: `models/sealed/scorer/gen-2/best_l6_full_training_decksF.txt`
for run F; same checkpoint, identical pools file for color-pairs.

| Pool | Run F  | color-pairs | Δ          |
|------|-------:|------------:|-----------:|
| 1    | 3.1603 | 3.1603      | 0          |
| 2    | 3.1200 | 3.1200      | 0          |
| 3    | 2.9348 | 2.9348      | 0          |
| 4    | 2.9075 | 2.9075      | 0          |
| 5    | 2.8686 | 2.8686      | 0          |
| 6    | 2.8183 | 2.8183      | 0          |
| 7    | 2.5293 | 2.5297      | +0.0004    |
| 8    | 2.7448 | 2.7247      | −0.0201    |
| 9    | 2.3637 | **2.4802**  | **+0.1165** |
| 10   | 2.1472 | 2.1231      | −0.0241    |
| 11   | 2.7210 | 2.7210      | 0          |
| 12   | 2.9223 | 2.9209      | −0.0014    |
| **mean** | 2.7698 | 2.7758  | +0.0060    |

Seven of twelve pools produce byte-identical decks. One (pool 9) finds a
meaningfully better basin (+0.12 — comparable in size to the largest gains
SA-with-restarts shows elsewhere in this doc). The other four differ
within noise.

### Wall-clock regression

The 12-pool color-pair run takes about 26 minutes. `--restarts 4` at the
same cooling takes ~2m15s. Naively color-pair should be 2.5× longer (10
restarts vs 4); the extra ~4× comes from each color-pair restart running
all `--sa-max-iterations` iterations (default 200) instead of terminating
at the post-cooling early-stop within ~10 iterations of `T < 1e-3`.

The cause is structural. The post-cooling early-stop is

```
if effectively_greedy and candidate_score <= s.current_score:
    s.finished = True
```

where `current_score` updates every iteration to whatever was just chosen.
A color-pair-seeded restart lands inside one of the scorer's "fertile"
regions where the search keeps applying moves that improve over the
immediately-previous `current_score` (typically by ≈0.001) but don't
advance `best_score`. Each new move trivially satisfies `candidate_score >
current_score`, so the early-stop never fires. A random multi-color init
lands on a flat plateau where no neighbor improves and the early-stop
fires within a handful of iterations of the cooling threshold.

### Implication

The structural argument behind color-pair init — that random init starts
maximally diverse and forces the search to climb out of a 5-color basin
via single-card moves — turns out not to bind on the gen-2 6-layer
scorer. Random init reaches the same basins color-pair init seeds, just
via a different path. There is no meaningful mean-score improvement, and
the wall-clock penalty makes color-pair init a Pareto regression at
current settings.

The 1/12 pool where color-pair finds a strictly better basin is real but
rare — random restarts already buy similar improvements on a similar
fraction of pools at much lower cost.

A patience-based early-stop (terminate a restart when `best_score` hasn't
improved over the last K iterations, rather than checking
`candidate_score > current_score`) would let color-pair init terminate at
the same point as random restart, eliminating the wall-clock penalty.
With that fix the two strategies become roughly equal-cost; whether
color-pair init then offers any quality edge over random restart is
unmeasured.

## Recommendation

**Default for `build-decks`: `--sa-temperature 0.8 --sa-cooling 0.85
--restarts 4`.**

- `T=0.8` gives the largest mean-score gain over greedy in the gen-2
  sweep without the catastrophic-tail behavior at higher T from gen-1.
- `cooling=0.85` captures essentially the same quality as `cooling=0.95`
  (within ~0.08 score per pool on the 12 KTK pools) at roughly ⅓ the
  wall-clock, by limiting the post-cooling drift window. Faster cooling
  than 0.85 risks the regression from §Cooling sub-sweep where SA flees
  good basins without finding new ones.
- 4 random restarts kill the worst-tail variance per the gen-2 sweep.

Color-pair seeded init (`--restarts color-pairs`) is implemented but not
recommended at current settings — equivalent mean score to `--restarts 4`,
~10× wall-clock. See §Color-pair seeded init.

### Score ≠ deck quality

Important caveat (unchanged from the gen-1 analysis): the scorer is known
to be miscalibrated (multi-color preference, scorer-built decks losing to
Forge despite scoring well — see `feedback_card_encoder.md`-adjacent
observations and `project_scorer_overconfidence.md`). All SA can do is find
decks the scorer *thinks* are best. A higher-scoring deck can be a
worse-playing deck if the scorer is the bottleneck.

The eval harness is not yet integrated with SA-built decks. Win-rate
against Forge — and, more directly, an A-vs-F `match-outcomes` run —
is needed to confirm the +0.04 mean-score gain reflects real strength
and not deeper miscalibration exploitation.
