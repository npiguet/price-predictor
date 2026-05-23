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
Switching to a reward-ranked **top-k** objective then pushes it to **2.3126** —
within **~0.047** of the gen4-512 greedy/SA builder's mean (2.36), i.e. **~0.5
pp of win rate from builder parity**. So a single-forward policy now
essentially matches an explicit per-pool search, and clears Forge's own
builder by a wide margin (forge-best mean 0.90).

That is the amortized-optimizer goal reached: search-level deck quality at one
forward pass per pool. The remaining open work is match-play confirmation that
the scorer-score parity holds up in real games — not further scorer-chasing.
The rest of this file is the record of how it got there, including two
improvement attempts that failed before top-k succeeded.

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
comparable pool-by-pool. The picker scored a notch below the SA builder on
most pools, edged it on a few, and stayed well above `forge-best` throughout.

This is the cheap, no-Forge-games version of the validation: the question
"is the picker a better scorer-maximizer than greedy?" is answered directly
by these scores, since both maximize the same scorer. The current answer is
"close, slightly behind," which is a strong result for a one-shot policy
against an explicit search and motivates the improvement work below. The
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
the search it amortizes. The reward-ranked top-k runs have since narrowed this
to ~0.047 (best val_reward 2.3126). Training is "good enough" once the picker's
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

Cashing this out for the picker: after top-k the remaining 0.047 gap to the
gen4-512 builder ≈ **~0.5 pp** of win rate, and the total top-k gain over plain
REINFORCE (+0.079) ≈ **~0.8–1 pp**. Small but real — confirming "near builder
parity," with roughly half a point of win rate left to extract against this
scorer.

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

### Outcome — near builder parity

2.3126 vs the gen4-512 builder ceiling (2.36) is a gap of **~0.047** (≈0.5 pp
of win rate). The picker closed the headroom from 0.13 (REINFORCE) → 0.047
(~64% of it), with the total top-k gain over REINFORCE (+0.079) worth ~0.8–1
pp. Deck shapes stayed in-distribution throughout (~2.85 colors, ~18
creatures). The picker now essentially **matches the explicit greedy/SA search
at one forward pass per pool** — the goal of an amortized picker.

A `top-16 of 128` warm-start (deeper sample pool → higher-quality targets,
same gentle settings) is running to test the last sliver of headroom, but the
remaining ~0.047 is small enough that the higher-value next step is match-play
validation rather than more scorer-chasing. The ReST k-annealing schedule
(16 → 8 → 4 across resume-from-best stages) remains available if more is
wanted.

## Open questions / next steps

- **Match-play confirmation (highest value).** The top-k picker (2.3126) is at
  ~builder parity on scorer score; run self-play (`pick-decks` decks vs
  `build-decks` decks, same pools) to confirm that translates to real win rate
  in Forge games. This, not more scorer-chasing, is the gating test for
  whether the picker is deployable.
- **`top-16 of 128` run (in progress).** Warm-started from 2.3126 with the
  gentle settings; testing whether a deeper sample pool extracts the last
  ~0.047 toward the 2.36 ceiling.
- **ReST k-annealing schedule** (16 → 8 → 4, resume-from-best per stage) — the
  remaining lever if 128-sample stalls and more headroom is still wanted; keep
  temperature ≥ 1.2 so the deeper-cut filter doesn't starve.
- **Fix the entropy decay schedule** so it actually anneals against a noisy
  val curve (time/step-based or best-not-improved-for-N trigger) rather than
  the never-firing "N consecutive improvements" condition. Less urgent now
  that the manual entropy-coef/temperature tuning works.
- **Builder distillation (expert iteration)** — train the picker to imitate
  the SA builder's chosen decks directly (the explicit search as expert), the
  search-based counterpart to the self-expert top-k loop. A fallback if the
  picker needs to exceed (not just match) the builder.
