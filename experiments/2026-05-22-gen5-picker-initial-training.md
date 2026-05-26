# Gen-5 Picker — Initial Training

## Background

"Gen-5" is not a new scorer or encoder — it keeps gen-4's frozen 512d
sealed encoder and Set Transformer scorer (the production build selected in
[`2026-05-15-gen4-initial-training.md`](2026-05-15-gen4-initial-training.md),
checkpoint `512-best_l6_h4_s4_ff2176_mlp512_lr1e-05_mwlog.pt`) and changes
the **deck builder**. Where gen-1 through gen-4 built decks with an explicit
per-pool search (`build-decks`: greedy + simulated annealing against the
scorer, up to 200 swap iterations with restarts), gen-5 trains a one-shot
learned policy — the **picker** — that ranks a whole pool in a single forward
pass and reads off a deck via a deterministic walk (spec
[`017-one-shot-deck-picker`](../specs/017-one-shot-deck-picker/spec.md)).

The framing that organizes everything below: the picker is an **amortized
optimizer** and the SA builder is an **explicit** one. The builder refines
per pool at inference; the picker pays that cost once at training time and
then builds in one shot. A one-shot policy is not expected to beat an
explicit per-instance search outright — the bar is to come close while being
far cheaper, and ideally to generalize patterns the search rediscovers from
scratch on every pool.

Gen-5 retrains nothing upstream: the scorer and the `.npz` embedding cache
are frozen. That makes the picker's quality **upper-bounded by the gen-4
scorer** — the picker maximizes the scorer's deck score, so it can be no
better than the scorer's judgment of what a good deck is. This bound matters
for reading every metric below.

## Headline result

The picker trains stably from random init under REINFORCE to a validation
reward (the frozen gen-4 scorer's mean rating of the picker's deterministic
decks on held-out pools) of **2.234**, up from ~1.96 at the first epoch.
Switching to a reward-ranked **top-k** objective and chaining four runs of it
(deeper sample pools, greedier selection, gentler LR) then climbs the picker to
**2.426** on scorer-score — past the gen4-512 greedy/SA builder's mean (2.36) by
~0.07. So as a *scorer-maximizer* the single-forward policy out-optimizes the
explicit per-pool SA search.

**Match-play settles the real question:** in 4,442 best-of-7 Forge games the
picker plays at **parity with the SA builder** (head-to-head 52.3%, ~1.1σ — not
significant) and dominates Forge's builders (beats `forge-best` 69%). So the
above-SA *scorer-score* margin did **not** show up as a win-rate edge, but the
deployable result is exactly what an amortized optimizer is for: **search-level
deck quality at one forward pass per pool**, equalling a 200-iteration search in
real games at a fraction of the cost. The frontier is now the scorer,
not the policy.

## Setup

- **Model.** Policy transformer over a pool: an input projection (identity
  when the picker's internal width equals the 544-wide cache), a stack of SAB
  layers (the scorer's self-attention block, reused), a per-card head (one
  logit per pool card), and an auxiliary pool-quality head (one scalar per
  pool). Default 4 layers / 8 heads; internal width inherits the 544-wide
  `.npz` cache.
- **Training.** REINFORCE from random init against the frozen scorer.
  Per step: sample `--n-samples` (64) decks per pool with a GPU-batched
  without-replacement sampler, score them with the frozen scorer, and
  optimize a policy-gradient + entropy + auxiliary loss with a **per-pool
  empirical-mean baseline**. The policy-gradient term uses the Plackett-Luce
  log-probability of each sampled deck; the advantage (reward − per-pool mean)
  is detached; the aux head regresses the per-pool mean reward. Entropy bonus
  on a val-reward-driven decay schedule. AdamW, `--lr 3e-4`, per-group
  grad-norm clip 1.0, seed 42, 100-epoch cap, `--patience 10`.
- **Data.** 100k sealed pools (`pools-picker.txt`), split 80k train / 20k
  val (front 20% is the fixed val slice). ~12.5 min/epoch on the available
  GPU.
- **Inference.** Deterministic pick-decomposition walk: sort pool cards by
  logit, take 23 spells in ranked order plus any nonbasic lands encountered
  before the quota fills, then fill basics. This is what `pick-decks`
  deploys and what validation scores.

## Baseline run — REINFORCE from random init

The first healthy run used defaults (4 layers, 8 heads, no dropout, no
advantage normalization).

| epoch | val_reward | note |
|------:|-----------:|------|
| 0 | 1.964 | |
| 5 | 2.117 | |
| 13 | 2.182 | plateau begins |
| 31 | **2.234** | best |
| 41 | 2.209 | early stop (10 epochs w/o improvement) |

Most of the learning happened in the first ~13 epochs; epochs 13–41 wandered
in a 2.15–2.23 noise band, and "best at 31" is only +0.02 over epoch 17. The
+0.27 total gain maps — with the usual caveats — to very roughly +2 pp of
scorer-believed win rate over the policy's own early self (using the prior
pipeline calibration of ~7.8 pp win rate per 1.0 of scorer-score gap).

### Cold-start was not realized

The spec's main risk was a degenerate reward landscape at random init: if the
frozen scorer rated every random 23-card subset of a pool near-identically,
the per-pool advantage would collapse to ~0 and there would be no gradient.
That did not happen — val_reward climbed off random init and the aux head's
loss fell (0.22 → 0.08), both of which require structured within-pool reward
variance. The scorer assigns enough spread across a pool's subsets to produce
a usable policy gradient.

### The entropy decay schedule never fired

The entropy coefficient is held constant until val reward improves for
`--entropy-decay-after` (5) **consecutive** epochs, then decays on plateaus.
The val curve is too noisy to ever deliver 5 improving epochs in a row (its
longest run is ~2–3), so the schedule never armed and the coefficient stayed
at its initial 0.01 for the entire run. The policy was therefore held at
roughly constant exploration start to finish: entropy dipped to ~1.3 nats
around epoch 5, then rose back to a ~2.5-nat equilibrium and stayed there.
The intended "anneal exploration late so the policy sharpens" behavior never
engaged — a real tuning lever (a time/step-based decay, or a
"best-not-improved-for-N" trigger instead of "N-consecutive-improvements"
would actually fire against this noise).

### Deck shapes are sane and in-distribution

Across the run the deterministic decks held steady at ~2.9 colors, ~18
creatures, ~0.77 creature share, and a front-loaded curve — no drift toward
over-splashing or degenerate builds. That sits right on top of gen-3's greedy
builder (~2.72 colors / 18.15 creatures), i.e. the picker stays inside the
scorer's training distribution of decks rather than wandering into novel
deck-space where the scorer's score would be untrustworthy. This is the
reassuring read on the upper-bound caveat: the picker is maximizing the
scorer without exploiting it into regions the scorer never saw.

### What val_reward represents

It is the frozen gen-4 scorer's mean rating of the picker's deterministic
decks on the 20k held-out pools. The scorer is a pairwise-ranking model, so
the absolute scale is arbitrary (only score *differences* are calibrated);
the number is meaningful as (a) a within-run progress meter for the optimizer
and (b) a win-rate proxy *relative to another builder scored by the same
scorer*. Because the picker is trained to maximize exactly this quantity,
chasing a rising val_reward is the intended behavior, not reward-hacking to be
suppressed — the deployment goal is to out-maximize the greedy builder at the
shared objective, and the residual risk is only whether the picker reaches
score regions where the scorer is poorly calibrated (the in-distribution
shape check above argues it does not).

## Deck-quality comparison — picker vs build-decks vs forge-best

`scripts/compare_deck_builders.py` builds one deck per shared pool three ways
(forge-best, the `build-decks` greedy/SA search, and the picker), then rates
all of them with the same gen-4 scorer so the score headers are directly
comparable pool-by-pool. At this point — the REINFORCE-baseline picker, before
top-k — it scored a notch below the SA builder on most pools, edged it on a
few, and stayed well above `forge-best` throughout. (The later top-k runs
closed and reversed this gap; see the top-k section.)

This is the cheap, no-Forge-games version of the validation: the question
"is the picker a better scorer-maximizer than greedy?" is answered directly
by these scores, since both maximize the same scorer. At the baseline the
answer was "close, slightly behind," which is already a strong result for a
one-shot policy against an explicit search and motivated the improvement work
below. The
remaining job for actual match-play is only to confirm the score gap reflects
real win rate, which — per the gen-4 match-play evaluation — it reliably does
for in-distribution decks.

### Score ceiling: the gen4-512 builder's mean as the picker's target

The picker amortizes the `build-decks` greedy/SA search, so the **mean deck
score that search achieves is the natural single-number target** the picker is
trying to match (parity) and ideally exceed. Scoring all 10,000 gen4-512
builder decks (`output/sealed/generated-decks-gen4-512.txt`) with the gen4-512
scorer gives that reference distribution:

| stat | value |
|---|---|
| n | 10,000 |
| mean | **2.36** |
| std | 0.74 |
| min / max | −1.05 / 4.01 |
| p5 / p25 / p50 / p75 / p95 | 1.03 / 1.93 / 2.44 / 2.90 / 3.44 |

Read as a stopping gauge: the picker's REINFORCE-baseline val_reward (2.234, a
mean over its 20k val pools) sits **~0.13 below the builder mean (2.36)** —
i.e. the picker is already near builder parity, with only ~0.13 of headroom to
the search it amortizes. The reward-ranked top-k runs have since closed that
gap and gone past it — best val_reward 2.426, ~0.07 *above* the SA mean (see
the top-k section). Training is "good enough" once the picker's
mean val_reward approaches
2.36; pushing well past it would mean out-optimizing the explicit search,
which is the most a frozen-scorer picker can deliver.

Why the mean-to-mean is precise despite the wide per-deck spread: every deck
in both samples is built from its own fresh random pool drawn by the same
generation method, so the two are i.i.d. samples from the same pool
distribution. The per-deck std (0.74) is dominated by pool difficulty, but
that averages out of the **mean** by √n — the standard error of the builder
mean is ~0.74/√10000 ≈ 0.007 and of the picker mean ~0.74/√20000 ≈ 0.005, so
the standard error of the 0.08 gap is ~0.01 (the gap is ~8–9σ). Pool
variability does not contaminate the comparison; it is averaged away. The only
assumption the gauge rests on is that both pool sets were generated from the
**same distribution** (same set-eligibility / method) — given that, the
mean-to-mean is a tight single-number headroom estimate. (A same-pool paired
comparison would additionally reveal per-pool *consistency* — how often, not
just on average, the picker matches the builder — but is not needed to
estimate the average headroom at this sample size.)

### Score-to-win-rate: the builder ladder

To translate score gaps into win rates, three builders that are all in the
gen-4 match-play tournament were scored with the gen4-512 scorer (their
generated-decks files), pairing each builder's mean deck score with its
measured Bo7 match win rate:

| builder | mean score | std | Bo7 WR (vs field) |
|---|---|---|---|
| gen4-512 | 2.36 | 0.74 | 57.8% |
| gen4-256 | 1.98 | 0.80 | 54.4% |
| forge-best | 0.90 | 1.12 | 38.2% |

This places the gen4-512 ceiling (2.36) in context: the builder ladder spans
~0.9 → 2.0 → 2.36, so the picker at ~2.28 sits just shy of the gen4-512 builder
and well above gen4-256 and forge-best — its 0.08 shortfall is tiny against the
~1.5-point forge→gen4-512 span. (Weaker builders also show a larger std —
forge-best 1.12, min −7.4 — because they occasionally build genuinely bad
decks; and the forge-best decks are an older set mix than the gen-4 decks, so
that anchor carries the distribution-mismatch caveat. The gen4-512-vs-256 pair
is the clean comparison.)

**Empirical conversion: ~10–13 pp of Bo7 win rate per 1.0 of scorer score**
near this operating range. gen4-512 vs gen4-256 is Δscore 0.38 ↔ +3.4 pp (or
head-to-head 54.6% = +4.6 pp) → ~9–12 pp/1.0; the wider forge→gen-4 range gives
~13–15 pp/1.0. Consistent with the pipeline's earlier ~7.8 pp/1.0 estimate.

The conversion is **flatter than the scorer's own score scale implies.** The
scorer is a Bradley-Terry ranking model, so if a score gap Δ were a calibrated
win-probability logit, the predicted edge would be ≈ logistic(Δ) — ~25 pp/1.0
near even. The measured ~10–13 pp/1.0 is roughly half that, i.e. the scorer's
score scale is **over-dispersed (overconfident) by ~2×** relative to real win
rates: a score gap corresponds to a smaller real-game edge than its magnitude
suggests. Use the empirical slope, not logistic(Δ).

Cashing this out for the picker: the final top-k picker (2.426) is now ~0.07
*above* the gen4-512 builder (2.36) ≈ **~0.7–0.9 pp** of win rate, and the
total gain over plain REINFORCE (2.234 → 2.426, +0.19) ≈ **~2 pp**. These are
modest in absolute terms and — above the SA baseline — increasingly
extrapolation (the calibration was fit in the normal-builder regime). Match-play
later checked exactly this and found **no measurable above-SA win-rate edge**
(parity with the SA builder; see *Match-play validation*) — the extrapolated
above-SA score did not cash out.

## Improvement attempts that did not work

The plateau at ~2.23 prompted two changes. Both destabilized training; both
are instructive about why.

### GRPO-style advantage normalization

**Rationale.** Within-pool reward variance is small (the 64 sampled decks
share most cards), so the raw advantages are tiny (~0.01), `policy_loss` is
~0.001, and the gradient is weak — the likely cause of the slow plateau.
Dividing the centered reward by the per-pool reward std (GRPO group
normalization) rescales the signal to unit variance per pool. Added as the
opt-in `--normalize-advantage` flag.

**Result.** Collapse. val_reward stuck at ~1.97 (roughly the init level) for
all logged epochs, never climbing; entropy crashed from ~1.4 nats to ~0.3
nats within five epochs; `policy_loss` jumped to ~−0.5 (≈100× the baseline),
confirming the ~100× larger effective step.

**Diagnosis.** Unit-variance advantages multiplied the effective
policy-gradient step by ~10–100× while `--lr` (3e-4) and `--entropy-coef`
(0.01) were left unchanged, so the policy collapsed onto a mediocre
near-deterministic solution before exploring. A feedback loop compounds it:
as the policy sharpens, within-pool reward std → 0, which *amplifies* the
normalized advantage further, accelerating the collapse.

**Lesson.** Normalization is not free — it needs the LR dropped by roughly
the rescale factor (≈10×) and likely a larger entropy coefficient. Given the
un-normalized baseline already reaches 2.234 healthily, retuning
normalization is a "revisit later" lever, not a first bet.

### Dropout inside the policy

**Rationale.** The scorer (gen-4) trains with dropout 0.2 and benefits from
it; by analogy dropout might regularize the picker. Run with
`--n-layers 6 --n-heads 4 --dropout 0.1`.

**Result.** Catastrophic. val_reward sat at **−3.4** and never improved
(early stop at epoch 10); the policy entropy was pinned near its **maximum**
(~4.19 nats vs ln(90) ≈ 4.5), i.e. ~uniform over the pool; `aux_loss` *grew*
(0.77 → 1.87) where the healthy baseline fell; and the deterministic decks
were random-looking (≈4.9 colors — nearly all five — ~12.6 creatures).

**Diagnosis.** This is a conceptual incompatibility between dropout and
on-policy policy gradients, not a tuning issue. Dropout inside an on-policy
REINFORCE network perturbs the action distribution on every forward,
scrambling which cards get high logits. REINFORCE is already high-variance;
dropout injects a second large variance source directly into the policy, the
sharpening gradient cannot stay consistent across steps, the entropy bonus
wins, and the policy settles at the uniform equilibrium. The scorer tolerates
dropout because it is **supervised** (fixed win/loss targets); the picker
does not because it is **on-policy RL** (its targets are generated by its own
dropout-perturbed policy). This is why dropout is generally avoided inside
policy-gradient networks; RL's regularizer is the entropy bonus, which the
picker already has.

**Caveat.** This run also changed layers (4 → 6) and heads (8 → 4), so a
dropout-off rerun at 6L/4H is still pending to fully isolate dropout — but the
near-maximum-entropy signature points squarely at dropout, and depth past 6L
was already shown dead / data-limited at the scorer stage in gen-4.

## Analysis — the central challenge is REINFORCE variance

Both failures share a root: the policy-gradient signal here is weak (tiny
within-pool reward spread) *and* fragile (high variance), and the two
interventions either amplified the variance (normalization) or added more of
it (dropout), tipping the policy into collapse. Tuning REINFORCE knobs is
fighting the symptom. The more promising direction is to change the
**objective type** to something with a stable, scale-invariant gradient.

## Reward-ranked (top-k / RAFT–ReST) objective — results

The fix that worked: change the *objective type*. Instead of advantage-
weighting all 64 samples, keep only the **top-k by reward per pool** and train
by plain maximum likelihood on their log-probs (RAFT / ReST / reward-weighted-
regression family — the policy imitating its own best samples). Why it fits
the picker's failure mode:

- **Scale-invariant.** Depends only on the *ranking* of the samples, not on
  the magnitude of their reward differences — full-strength gradient even when
  decks score nearly identically, sidestepping the tiny-advantage problem
  without the normalization blow-up.
- **No negative gradient.** Only pushes *up* toward good decks; the softmax
  handles "push down the rest" for free, removing the high-variance term that
  collapsed both prior attempts.
- **Self-improvement loop.** Sample → score → keep best → imitate → repeat
  ratchets the policy up (ReST exactly; the cheap self-expert cousin of
  search-distillation).

All runs warm-start from the 2.234 REINFORCE checkpoint (the objective is
resumable and not architecture-locked, FR-039), switching only the policy
loss. Top-16 of 64 holds `--n-samples` at 64 so the only change vs the 2.234
baseline is the objective.

### The crux is the entropy/diversity interaction

Top-k *sharpens* the policy (it chases its own best samples), so the entropy
bonus must be strong enough — and the LR gentle enough — to keep the 64
sampled candidates diverse, or the top-k filter starves on near-identical
decks. Two runs at top-16-of-64 made this concrete:

| run | LR | entropy-coef | temp | entropy held | best val_reward |
|---|---|---|---|---|---|
| default | 3e-4 | 0.01 | 1.0 | collapsed to ~0.5–0.7 nats | 2.281 |
| gentle | 1e-4 | 0.03 | 1.2 | healthy ~1.7–2.5 nats | **2.3126** |

The default-LR run over-sharpened: ~5,000 top-k updates per epoch at LR 3e-4
overwhelmed the 0.01 entropy bonus, entropy crashed from 2.65 to ~0.6 nats in
the first epoch, candidate diversity starved, and val_reward only crept to
2.281. The gentle run (1e-4 LR, 3× entropy, temperature 1.2) held entropy at
1.7–2.5 nats, kept improving for ~28 epochs, and reached **best val_reward
2.3126 at epoch 60** (early-stopped at 70). So: k controls *target greediness*;
temperature/entropy controls *candidate diversity*; they must move together.

### The top-k ladder: from parity to measurably above the SA search

Four top-k runs were chained, each warm-started from the previous best, with
the sample pool deepened, selection made greedier, and the learning rate
lowered at each step. Entropy was held healthy throughout (via elevated
`--entropy-coef` + `--temperature 1.2`); the lone failure was the first run,
left at the default LR, where entropy collapsed.

| run (chained) | config | entropy held | peak val_reward |
|---|---|---|---|
| top-16/64, default | lr 3e-4, ec 0.01, T 1.0 | collapsed ~0.6 nats | 2.281 |
| top-16/64, gentle | lr 1e-4, ec 0.03, T 1.2 | ~2.2 nats | 2.313 |
| top-16/128 | lr 1e-4, ec 0.03, T 1.2, 20 evals/epoch | ~2.4 nats | 2.373 |
| top-4/256 | lr 1e-5, ec 0.04, T 1.2, 20 evals/epoch | ~2.6 nats | **2.426** |

Against the REINFORCE baseline (2.234) and the SA builder mean (2.36), the
picker **crossed parity at top-16/128 (2.373) and pulled clearly above the SA
search at top-4/256 (2.426)** — ~+0.07 over the SA mean, several σ on the
mean-to-mean comparison. So the amortized one-shot picker is now a
**measurably better scorer-maximizer than the explicit per-pool SA search**,
at one forward pass versus a 200-iteration search. The per-stage gains shrank
as compute grew (+0.079 → +0.060 → +0.053, each stage ~2× the sampling cost
and the last needing ~40 epochs of slow creep): clear diminishing returns.

### The recipe that worked

- **Change the objective, not the REINFORCE knobs.** The two REINFORCE patches
  (advantage normalization, dropout) collapsed; top-k — a scale-invariant,
  no-negative-gradient supervised loss — is what broke the plateau.
- **Hold candidate diversity or the greedy filter starves.** Gentle LR
  (1e-4 → 1e-5) + elevated `--entropy-coef` (0.03 → 0.04) + `--temperature 1.2`
  kept entropy at ~2.2–2.6 nats. The one run left at the default LR
  over-sharpened (entropy → 0.6 nats) and stalled at 2.281.
- **Deepen the pool and tighten the cut for more headroom.** Larger N (64 → 256)
  gives higher-quality targets; greedier k (16 → 4) raises selection pressure.
  Together they took the picker above SA — at ~2× compute per step.
- **Frequent eval + best-checkpoint banks intra-epoch peaks** (FR-019): the
  best states landed mid-epoch and would have been missed at epoch boundaries.

### Two caveats on the above-SA gains

1. **Maximization bias.** With ~20 evals/epoch over 30+ epochs (~600 evals) and
   the best-checkpoint keeping the max, late "new highs" are partly
   extreme-value statistics on a roughly-plateaued oscillation, not continued
   learning. The early climb was real (the oscillation *center* rose); the last
   ~0.02–0.03 is inflated by max-of-many on a fixed val set.
2. **Colors drift — ambiguous, not clearly Goodhart.** `colors_mean` fell
   monotonically along the ladder: REINFORCE ~2.9 → 16/64 ~2.79 → 16/128 ~2.7 →
   4/256 ~2.64 (dipping to ~2.5). This *could* be the scorer over-rewarding low
   color count (a quirk the greedy picker would exploit), but it is at least as
   plausibly **real 2-color discipline**: tighter sealed decks are genuinely
   more consistent, the gen2→gen3 step cut colors (3.27 → 2.7) as a documented
   *improvement*, and the encoder's `cast_lift` / `color_lift` heads were
   built to value castability. **Match-play resolved this in favor of "real
   discipline":** the picker's 2-color decks won *more* than the SA builder's
   (64.9% vs 62.2%; see *Match-play validation*), so the drift was a genuine
   improvement, not scorer exploitation.

### Conclusion

Top-k is the objective that unlocked the picker: REINFORCE plateaued at 2.234;
chained top-k reached **2.426**, surpassing both the REINFORCE policy and the
explicit SA search on scorer-score. The recipe — gentle LR, diversity held via
entropy-coef/temperature, progressively deeper pools and greedier selection,
frequent eval — is the reusable finding. **What is settled:** the picker
matches and exceeds the SA search as a scorer-*maximizer* at one forward pass.
**What match-play then resolved** (see *Match-play validation* below): the
above-SA *scorer-score* margin did **not** translate into a measurable
win-rate edge — the picker plays at **parity** with the SA search in real games
(head-to-head 52.3%, ~1.1σ). So top-k's deployable achievement is *matching*
the explicit search at one forward pass, not beating it; the last above-SA
scorer-score climb was score that didn't cash out (the maximization-bias caveat
biting). The picker has saturated this scorer; further deck-quality gains
require a better reward model (gen-6 scorer), not more policy tuning.

## Match-play validation — gen-5 picker vs the SA builder

Round-robin self-play, best-of-7, **4,442 matches**, between the gen-5 picker
(`gen5`), the gen-4 SA builders (`gen4-512`, `gen4-256`), Forge's own builders
(`forge-best`, `forge-3sub`, `forge-8sub`), and `random`. This is the decisive
test of whether the picker's above-SA *scorer-score* is real *win rate*.

### Ranking (Bo7 win rate vs the field)

| builder | Bo7 WR |
|---|---|
| gen5 (picker) | 61.6% |
| gen4-512 (SA) | 60.2% |
| gen4-256 (SA) | 58.4% |
| forge-best | 30.4% |
| forge-3sub | 16.2% |
| forge-8sub | 7.6% |
| random | 2.0% |

The three learned builders are a class apart from Forge's; gen5 sits nominally
on top.

### The decisive comparison: gen5 ≈ SA builder

Head-to-head, **gen5 beats gen4-512 (the SA search) 52.3% (n=560)** — `z ≈
1.1σ` from 50%, **not significant**. gen5 vs gen4-256 is 51.7% (n=524, ~0.8σ);
gen4-512 vs gen4-256 is 54.2% (n=498, ~1.9σ). So the picker plays at
**statistical parity with the explicit per-pool SA search in real games** — the
amortized-optimizer goal reached: one forward pass equals a 200-iteration
search on win rate. Against Forge it dominates exactly as the SA builders do
(gen5 beats forge-best 69.0%, forge-3sub 86.5%, forge-8sub 95.2%, random 98.9%).

### The above-SA scorer-score did not cash out (as expected)

The picker's +0.07 *scorer-score* margin over SA (2.426 vs 2.36) predicted only
~0.6 pp of win rate, and head-to-head shows no measurable edge — consistent
with the maximization-bias caveat. **Caveat on the caveat:** at n=560 this eval
only resolves edges ≥ ~4 pp at 2σ, and the predicted edge (~0.6 pp) is below
that floor. So the result **confirms there is no *large* gen5-over-SA edge
(parity)** and cannot confirm or refute the sub-pp one — either way the
practical verdict is parity, and the late scorer-score climb bought no
demonstrable wins.

### The colors drift was real discipline, not Goodhart

gen5 shifted toward 2-color (1,154 two-color decks vs gen4-512's 760, fewer
3-color) — the `colors_mean` → ~2.5 drift we flagged during training. It did
**not** hurt: gen5's 2-color win rate is **64.9% vs gen4-512's 62.2%**. So the
drift was the picker finding tighter, more consistent decks that win at least
as well — vindicating the read that 2-color discipline is a genuine sealed
virtue, not a scorer quirk being exploited.

### Sanity

Mirror diagonals (a builder vs itself) sit within ~1σ of 50% (gen5 52.3%,
gen4-512 47.7%, gen4-256 53.4%), so no systematic harness bias. Bo7 is the
right unit: 26.9% of Bo1 outcomes would flip under Bo7 (9.6% for Bo5).

### Verdict

**Gen-5 is validated and deployable.** The one-shot picker equals the gen-4 SA
search's deck quality in real Forge games — at a fraction of the inference cost
— and crushes every weaker builder. The "beating SA on scorer-score" was score
that did not translate to wins; the real, confirmed achievement is *parity at
one forward pass*. 
