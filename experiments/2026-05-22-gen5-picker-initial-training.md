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

The picker trains stably from random init and reaches a validation reward
(the frozen gen-4 scorer's mean rating of the picker's deterministic decks on
held-out pools) of **2.234**, up from ~1.96 at the first epoch. On the
deck-quality comparison against the other builders on shared pools
(`scripts/compare_deck_builders.py`, all decks rated by the same gen-4
scorer), the picker lands **a notch below the greedy/SA builder, occasionally
above it, and consistently and substantially above `forge-best`**.

That is a working amortized optimizer: a single-forward policy reproducing
most of an explicit search's deck quality, and clearing Forge's own builder
comfortably. The open work is closing the small remaining gap to the search —
and the rest of this file is the record of what was tried.

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

## Planned next step — reward-ranked (top-k / RAFT–ReST) objective

Instead of advantage-weighting all 64 samples, keep only the **top-k by
reward per pool** and train by plain maximum likelihood to make those decks
more likely (RAFT / ReST / reward-weighted-regression family — the policy
imitating its own best samples). Why this fits the picker's failure mode:

- **Scale-invariant.** Depends only on the *ranking* of the 64 samples, not
  on the magnitude of their reward differences — so it gives a full-strength
  gradient even when decks score nearly identically, sidestepping the tiny-
  advantage problem without the normalization blow-up.
- **No negative gradient.** Only pushes *up* toward good decks; the softmax
  normalization handles "push down the rest" for free. This removes the
  high-variance term that amplified into collapse under both prior attempts.
- **Supervised within a step**, so optimizers (and dropout) behave — the
  policy-gradient pathologies do not apply.
- **Self-improvement loop.** Sample → score → keep best → imitate → repeat
  ratchets the policy up; this is ReST exactly, and the cheap (self-expert)
  cousin of search-distillation.

**Recommended configs.** Start with **top-16 of 64** (25% kept), holding
`--n-samples` at 64 so the only change vs the 2.234 baseline is the objective
(clean A/B). Then try **top-16 of 128** (deeper sample pool → higher-quality
targets while keeping a stable 16-target gradient). Avoid very small k early:
the scorer is only ~72% pairwise-accurate, so the single "best" sample is
often best by scorer noise — keeping the top quarter hedges against imitating
noise. k controls *target greediness*; sampling temperature / entropy controls
*candidate diversity* — they must move together (keep temperature ≥ 1 so the
top-k filter has varied candidates to choose from; if diversity collapses,
top-k starves).

**Cheap first test.** Warm-start: resume from the 2.234 checkpoint with
top-16-of-64 immediately (`--objective topk --topk 16`, lower `--lr` ~1e-4),
which asks the directly-useful question "can top-k push past where REINFORCE
plateaued?" for a fraction of the compute. A positive result is clean
evidence top-k helps; a null is ambiguous (could be at the scorer ceiling
where REINFORCE also stalled). The schedule (anneal k 16 → 8 → 4 across
resume-from-best stages, ReST's rising threshold) can be run manually once
fixed-k is validated.

## Open questions / next steps

- **Validate fixed-k top-k** (top-16 of 64) against the 2.234 baseline —
  warm-started first for a fast read, from-scratch if the warm-start is
  ambiguous. Then top-16 of 128.
- **Manual k-annealing schedule** (16 → 8 → 4, resume-from-best per stage,
  dual patience), if fixed-k validates and then plateaus.
- **Fix the entropy decay schedule** so it actually anneals against a noisy
  val curve (time/step-based or best-not-improved-for-N trigger), letting the
  policy sharpen late instead of holding constant exploration.
- **Builder distillation (expert iteration).** Train the picker to imitate
  the SA builder's chosen decks (the explicit search as expert), optionally
  followed by RL fine-tuning — the search-based counterpart to the self-expert
  top-k loop.
- **Isolate dropout from depth** with a `--n-layers 6 --n-heads 4 --dropout 0`
  rerun, to confirm the 6L/4H architecture trains normally without dropout.
- **Match-play confirmation.** Once a picker beats the greedy builder on
  scorer score, run self-play (`pick-decks` decks vs `build-decks` decks, same
  pools) to confirm the score advantage is real win rate.
