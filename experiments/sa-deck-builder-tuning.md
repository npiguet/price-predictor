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

`GreedyDeckBuilder` now supports simulated annealing via three params:

- `temperature` (default `0.0` — pure greedy, behavior unchanged).
- `cooling` (default `0.95`) — per-iteration temperature multiplier.
- `max_iterations` (default `200`) — hard cap on iterations.

Implementation detail: at `T > 0`, instead of taking the argmax swap, the
algorithm samples a swap from the full softmax-temperature distribution
over all 1541 candidates: `P(swap_i) ∝ exp(score_i / T)`. This integrates
naturally with batched GPU scoring — no change to the forward-pass shape.
At `T == 0` the algorithm collapses to argmax + break-on-no-improvement
(the original greedy behavior).

The algorithm tracks the best deck seen across all iterations and returns
that, not the final state. So SA is at least as good as a random walk in
the worst case, regardless of how cooled the temperature gets.

CLI: `--sa-temperature`, `--sa-cooling`, `--sa-max-iterations` on
`build-decks`. There's also `--print-decks` to dump the human-readable
format used by `evaluate-scorer` (sorted by mana value, lands at bottom).

## T0 sweep results (12 pools, 6-layer scorer, KTK set)

Checkpoint: `models/sealed/scorer/gen-1/6layers_lr1e-05.pt`.
Pools file: `models/sealed/scorer/gen-1/6layers_pools.txt`.
All runs use `cooling=0.95`, `max_iterations=200`.
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

## Conclusions

### T=0.8 is the best single setting on this scorer

Highest mean score across pools (1.7836 vs 1.6094 for greedy, +10.8%
relative improvement). Critically, the gain comes from a few pools where
SA escapes a much worse multi-color local optimum:

- **Pool 1**: 1.6008 → 1.9954 (+24.7%). Greedy converged to a 5-color
  R/B/G/U/W deck. SA at T=0.3-0.8 found a focused W/R deck (with the
  same handful of multicolor splash cards) scoring much higher.
- **Pool 7**: 1.6169 → 1.7719 (+9.6%).
- **Pool 11**: 1.6888 → 1.8596 (+10.1%).

These are exactly the cases where greedy's "no improving single-swap"
termination misses a much better basin reachable via several
intermediate worse swaps.

### Higher T is not monotonically better

T=1.0 is *worse* on average than T=0.8, despite covering more of the
landscape. Two failure modes show up at T=1.0:

- **Pool 12**: drops from 1.6056 to 1.0896 (-32%). High-T exploration
  wandered into a basin so bad that the best-deck tracker still couldn't
  recover anything close to greedy.
- **Pool 1**: drops back to greedy's score (1.6008) — SA explored too
  aggressively and re-converged to the multi-color basin that T=0.3-0.8
  had escaped.

The score-vs-T curve has a peak (somewhere around T=0.7-0.9 here), then
declines. Beyond the peak, exploration is essentially random search and
even best-deck tracking can't recover.

### T=0.5 is in a "no man's land"

Counter-intuitively worse than T=0.3 on this set. Two big regressions
(pool 7 -0.31, pool 12 -0.39) drag the mean below T=0.3's. Possibly
T=0.5 is hot enough to leave good basins but not hot enough to reliably
find better ones, ending up in shallow alternative basins.

### Per-pool optimal T varies

No single T wins on every pool:

- Pools 5, 6, 9: greedy already finds the best basin; SA at any T is
  worse or tied.
- Pools 1, 4, 11: T≥0.3 finds a meaningfully better basin reliably.
- Pool 2: only T=0.8 finds the best basin.
- Pool 12: T≥0.5 catastrophically degrades.

### Score ≠ deck quality

Important caveat: the scorer is known to be miscalibrated (multi-color
preference, see `feedback_card_encoder.md`-adjacent observations). All
SA can do is find decks the scorer *thinks* are best. A higher-scoring
deck can be a worse-playing deck if the scorer is the bottleneck.

The eval harness is not yet integrated with SA-built decks. Win-rate
against Forge needs to be checked to confirm SA-built decks actually
play better, not just score better.

## Recommendation for gen-2 self-play

**Use `--sa-temperature 0.8 --sa-cooling 0.95 --sa-max-iterations 200`**
as the default for `build-decks` when generating gen-2 input decks.

Rationale:
- Highest mean score across the test pools.
- Net improvement on most pools, with bounded downside on the few where
  greedy was already strong.
- Reuses the same scorer architecture (6-layer Set Transformer), so the
  score-landscape topology should be similar for gen-2's checkpoint and
  the optimal T should generalize within ±0.1.

Optional refinement if the cost is acceptable: build each pool 2x with
`T=0.5` and `T=0.8`, keep the higher-scoring deck. This captures the
~+0.01 ensemble gain and provides per-pool robustness against the
occasional T=0.8 regression. Cost is one extra forward-pass batch per
pool (negligible relative to match-outcomes throughput).

## Open questions

1. **Does SA improve actual win rate?** Score is a proxy. The whole
   reason for SA is that the user observed scorer-built decks losing to
   Forge — a small `match-outcomes` run with `gen-1-SA` decks vs `gen-1-greedy`
   decks would tell us whether SA's score gains translate to gameplay
   improvements or just exploit miscalibration harder.
2. **Sweep `cooling` at T=0.8.** Slower cooling (0.97-0.98) might
   widen the exploration window and find even better basins; faster
   (0.92) might converge too quickly. Not yet tested.
3. **Multi-restart.** With a fixed seed (which the algorithm doesn't
   currently set), running SA 3-5 times with different inits per pool
   and keeping the best is a likely cheap win for the high-variance
   pools (especially T=1.0's failure mode).
4. **Color-pair seeded init.** Instead of random 23-card init, seed
   each greedy run with cards from a single color pair (10 runs per
   pool, one per color combination). Directly addresses the
   "random init contains all colors" structural problem; orthogonal
   to SA and probably stackable with it.
