# Draft agent (gen-1) — design rationale

## Background

This document collects the design discussions that shaped the gen-1 draft
agent. The spec ([`../specs/2026-05-28-draft-agent.md`](../specs/2026-05-28-draft-agent.md))
is the normative description of *what* to build; this file is the record of
*why* — alternatives considered, trade-offs weighed, and the project context
that selected the chosen options.

The gen-1 draft agent is an offline-trained foundation: an imitation policy
that picks like a competent Forge drafter, plus a critic that predicts the
final deck's quality from any mid-draft state, trained jointly on a shared
transformer body from Forge-generated draft data. No RL, no live integration
of the model into the draft loop. Gen 2 (a future spec) is the RL self-play
stage that builds on this foundation; many design choices below are
constrained by "build the gen-1 artefact that slots cleanly into gen-2."

The objective throughout is win rate *when the Forge AI pilots the deck the
agent drafts*. Forge's piloting tendencies are the target distribution by
design, not a bias to correct.

## Why draft is hard

Booster draft is a partially observable sequential game. Eight drafters sit in
a pod; each opens a 15-card booster, takes one card, passes the rest; packs
pass left in packs 1 and 3 and right in pack 2; this repeats until all packs
are empty (3 packs × 15 picks = 45 cards per drafter). A drafter sees only the
packs that reach it, never the other seats' pools. Three classic difficulties:

1. **Hidden information** — seven other drafters consume the same cards
   simultaneously; you never observe their pools directly.
2. **Open colours / signal reading** — to avoid fighting over contested
   colours you must infer what is "open" from what reaches you late.
3. **Sparse, hard-to-attribute reward** — deck quality is only known at the
   end and is hard to attribute to any single pick.

The agent conditions on its whole observation history (pool so far + cards
seen-and-passed + pack/pick counters). That history is an implicit belief
about hidden state, so hidden information and open-colour reading emerge from
the input representation without an explicit opponent model.

## Two-stage plan (imitation → RL)

Accepted. Stage 1 (imitation) is supervised, stable, and cheap, and yields
most of a working drafter. Stage 2 (RL self-play) refines it. Generation 1 is
Stage 1 plus the critic that Stage 2 will need; the RL itself is generation 2.
This is the "bootstrapping ladder": climb the cheap, certain rungs first
(imitation, then critic on the same data) so the actor and critic are
pre-aligned on one play distribution before any RL.

## Imitation vs straight to actor-critic — why imitation first

Imitation is not an *alternative* to RL — it ceilings at Forge's skill (the
agent can only copy what it's trained on). It's a launchpad that makes the
subsequent RL stage tractable. Going straight to actor-critic from random
init is possible in principle but brutal here:

- **Sparse reward + long horizon.** A draft is ~45 sequential picks with
  reward only at the end. Random-init RL almost never stumbles onto a coherent
  deck by chance, so positive outcomes are essentially never reinforced.
  Imitation drops the policy into a region where competent drafts (and
  therefore informative reward) happen routinely.
- **Critic cold-start is circular.** `advantage = return − V(s)` needs a
  meaningful critic, but `V` is only defined relative to the continuation
  policy. From-random both pieces are useless and chase a violently moving
  target. Imitation gives a competent, stable policy so the critic learns
  sensible values from the start; gen-2's actor↔critic coupling then starts
  from sanity rather than noise.
- **Sample economy.** Imitation is cheap supervised learning on Forge data
  generated as a byproduct of running the draft environment at all. RL
  samples are expensive (draft + build + score). Most picks (take the bomb,
  stay in your two colours) are obvious-ish and imitation nails them for
  free; RL is reserved for the genuinely pivotal decisions where it adds
  value.
- **Stability and anchoring.** Random-policy + random-critic RL is prone to
  collapse and to *reward-hacking* the (imperfect) scorer. Imitation gives a
  baseline on the manifold of reasonable drafts and a reference to KL-anchor
  RL against.
- **Working measurable product early.** Imitation alone produces a competent,
  evaluable drafter before committing GPU to the finicky RL stage.

Same recipe as RLHF (SFT → PPO) and AlphaGo (supervised → self-play): the
hard task gets a supervised warm-start so the RL stage has somewhere to stand.

### The picker cold-start counterexample, addressed

A reasonable challenge: the one-shot picker (spec 017) was feared to suffer a
similar cold-start under REINFORCE from random init, and it worked fine. Why
expect the draft agent to be different?

The decisive difference is **horizon**. The picker is effectively a
single-step contextual bandit: one forward pass computes all per-card logits
*once*, the sequential without-replacement sampling just draws a set from
those fixed logits, and the reward attaches directly to the whole set. No
temporal credit assignment, no evolving environment. That's about the easiest
RL setting, and cold-start fears there were indeed overblown.

The draft is a genuine 45-step sequential POMDP: the policy re-decides at
every pick on an evolving state, with the only reward at the end. The
single-step bandit lesson — RL-from-scratch on dense decomposable reward is
easy — does not generalise to horizon-45, sparse-terminal-reward, with a
from-scratch-useless critic. That's the hard regime, and the cold-start risk
there is well-founded.

Even granting the uncertainty, imitation is nearly *free* in this project
because the Forge draft environment must run anyway (Forge bots have to draft
to generate the critic data and to host the draft topology), and once Forge
bots are drafting, their picks are demonstrations sitting right there.
Imitation-first is a free pass over data that already exists rather than a
costly hedge. The picker had no analogous free demonstrations — only
pre-generated pools.

So: imitation-first is cheap, the draft's long horizon makes the cold-start
risk real (unlike the picker's bandit), and the demonstrations are produced
for free — do it regardless of how cold-start would have shaken out.

## State representation choices

Accepted: a typed token sequence (`POOL` / `PACK` / `PASSED` / `TAKEN` /
`CONTEXT`) over the reused `.npz` card embeddings. Open-colour reading is left
**implicit** — the `PASSED`/`TAKEN` tokens plus pack/pick counters carry the
signal. `TAKEN` records *observed* opponent picks read off the wheel, which is
factual observation, not a learned model of any neighbour, so the policy
still builds no explicit opponent model. An auxiliary head predicting each
neighbour's archetype was considered and deferred to gen 2 (it needs self-play
ground truth to supervise).

**Rejected — pick-as-attention-op.** Treating the pick distribution *as* the
attention weights between a pool/context query and the pack keys is
parameter-efficient and interpretable, but forces the policy through a single
attention bottleneck that struggles to represent context-dependent value
("good *because* I lack fixing"). The per-card head over a jointly-attended
trunk (the picker's validated pattern) is more expressive at negligible extra
cost. The attention formulation remains available as an ablation, not the
default.

## Token types, the wheel, and the round-end flush

`PASSED` and `TAKEN` are the two halves of the seat's "saw but doesn't own"
history. The split exists because the seat's *knowledge* of these cards
evolves over the draft:

- A card moves `PASSED → TAKEN` at two times:
  1. **The wheel** (pick 9 in an 8-pod): the seat sees a pack it already saw
     8 picks earlier, and the cards missing on return (less the seat's own
     pick) were taken by the seven intervening opponents. Diff resolves
     `PASSED → TAKEN` for those, `PASSED → PACK` for the survivors.
  2. **Pack end**: once the pack is exhausted every card from it is in some
     pool; cards still in `PASSED` are necessarily in an opponent's pool, so
     the remaining `PASSED` set flushes wholesale to `TAKEN`.

This makes `PASSED` a within-pack transient (it empties at every boundary)
and `TAKEN` the complete record of cards opponents took. The wheel gives
*early* within-pack detection (informing picks 9–15); the flush resolves
the rest for use in subsequent packs.

The *positive* half of the wheel — a strong card that came back to you,
signalling that colour is open — needs no separate type. It re-enters `PACK`
carrying a recency `pick_ago ≈ 8`, and that recency *is* the wheel signal on
the PACK token side (see below).

Each card *instance* is in **at most one** of the four sets at a time. That
mutual exclusivity matters for reconstruction: when an instance transitions
(PASSED → TAKEN, etc.) it leaves its old set.

## Duplicates, by role

Whether a type keeps duplicate cards follows from what the type *is*:

- **`PACK`** is the action space. The pick is one card *name*; two physical
  copies are the same choice. Deduped to one token per name. Keeping both
  would inflate that name's selection probability by its copy count, which
  has nothing to do with the card's pick value (summing softmax mass over
  identical-logit tokens just doubles the name's probability for no
  modelling reason).
- **`POOL`** is the deck the seat holds. A second copy is a materially
  different build state (you can run a playset). Multiset.
- **`PASSED` / `TAKEN`** are observation history. Distinct physical copies
  passed/taken from *different* packs are distinct events and tell the model
  something different from one copy alone (two `TAKEN` instances of the same
  name across two packs is a *stronger* contested signal than one). One
  token per card instance — but re-observing the *same* instance (a card
  that wheels back) updates that instance's status and recency rather than
  spawning a second token.

The rule generalises: dedup follows from the set's role — action / state /
observation.

## Recency: relative split + last-in-pack semantics

Recency captures both **staleness** of an old observation and the **wheel**
(a card from this pack returning ~8 picks later). Three design choices were
made:

### Split into `packs_ago` + `pick_ago`, not a single global scalar

A single global "picks ago" conflates within-pack and cross-pack distance.
"8 ago, same round" is the wheel (high-value signal); "8 ago, across the
P1→P2 boundary" is a dead pack (irrelevant). A single scalar can't even
represent whether a round boundary was crossed — both look identical. So
split into `packs_ago` (round-distance, captures stale vs live) and
`pick_ago` (within-pack timing, captures the wheel).

### Relative split over absolute `(pack_number, pick_number)`

Both forms carry the same information given the CONTEXT token (which has
absolute current `pack/pick`), so it's a choice of which one is *easier for
the model to use*.

- The wheel sits at a fixed *relative* offset (~8). With relative encoding
  the learned `pick_ago` table can make that value special directly. With
  absolute encoding the model would have to *derive* a difference through
  attention.
- "Now" is constant within a forward pass, so absolute per-token coordinates
  partly duplicate what the CONTEXT token already carries. Relative
  pre-bakes the age into the token.
- Relative generalises better: the wheel-at-8 pattern is identical at every
  "now" (whereas absolute encoding would re-learn it per position).

### `pick_ago` = "since last in the seat's pack", frozen at the boundary

Two refinements stacked here:

- *Last-in-pack vs first-observed.* "First observed" is monotone-in-time for
  any single observation, but doesn't reset when the same card is re-seen
  (e.g., wheel re-pass). The user-requested behaviour was: when a card moves
  PACK → PASSED, `pick_ago` resets to 1. Defining `pick_ago` as "picks since
  the card was last in the seat's pack (prior to the current pick); 0 if
  never in the pack before now" gives that reset *automatically* for any
  re-passing, including wheel re-pass. It also produces all the desired
  per-token values (fresh PACK = 0; wheeled PACK = ~8; just-passed = 1;
  growing for sit-in-PASSED).
- *Freeze at the pack boundary.* When `packs_ago` ticks from 0 to ≥1,
  `pick_ago` stops advancing and holds its end-of-pack value. This keeps
  `pick_ago` in the natural `[0, P−1]` range (so the embedding table stays
  small and densely trained on every row), and prevents `pick_ago` from
  re-encoding what `packs_ago` already says about cross-pack distance. The
  end-of-pack frozen value also carries a faint "how early in its pack was
  this card seen" signal for free, which is mild colour-history information.

### Learned table vs scalar concatenation for `pick_ago`

A scalar concat (one input dim per token) is mathematically just `s · w` —
one learned direction scaled linearly by the value, so the contribution is
forced monotone in `pick_ago`. The wheel is the canonical *non-monotone*
structure (value 8 is special in a way 7 and 9 are not), so a learned
embedding table (each `pick_ago` value gets its own vector) is genuinely
more expressive than a scalar here. The table itself is tiny (P rows ×
`d(pick_ago)`), so the cost is negligible.

The same argument applies to `packs_ago` (3 rows, even more trivially).

### No stop-gradient for frozen indices

"Frozen" means the *scalar value* the card looks up stops changing once it
crosses a pack boundary; it does not mean the *embedding row* the value
indexes stops training. The lookup is non-differentiable w.r.t. the index
anyway, so there's no gradient flowing into the index. The selected row
trains normally from every token that selects it (live or stale), and a
shared row's meaning is disambiguated by the `packs_ago` it is paired with
— the same way a polysemous word embedding handles multiple senses.

## Concat-into-`d_model` (vs add-at-fixed-width)

The scorer and picker already concatenate their `FEATURE_COUNT` deterministic
block onto the pooled-text vector and let `d_model = pooled_dim +
FEATURE_COUNT` (no input projection). Following that convention for the
draft features (type one-hot + recency embeddings) means `d_model` grows by
the feature widths, and those dimensions persist through the entire residual
stream — they get dedicated capacity, not a bottleneck through a projection.

This is genuinely different from "add a `d_model`-wide embedding at fixed
`d_model`" because there's no projection back down. The transformer is
slightly wider, the draft features occupy persistent non-interfering
dimensions, and the param count is the same as the equivalent add-at-fixed-
width formulation (the input projection's extra columns *are* the embedding
table — the algebra is identical).

The 4-dim type one-hot is fine as a raw one-hot because there's no input
projection; the first SAB layer's Q/K/V projections are the first learned
linear map the token hits, and they learn the per-type interpretation. No
separate learned type table is needed.

The constraint that ties everything together: `d_model` must remain divisible
by `n_heads`. Since `embedding_dim` already is, the *sum* of the added
feature widths must be a multiple of `n_heads`.

## What each head trains on: policy on whitelist, critic on all

The two heads train on **different seat subsets**, which is a deliberate
asymmetry:

**Policy: whitelisted agents only.** Cross-entropy *copies* the demonstrator
— it has no notion of good or bad, just maximises the probability of the
demonstrated action. Feeding it a weak agent's picks teaches it to reproduce
those bad picks. "Steer away from this pick" can't be expressed in
cross-entropy; it requires an advantage signal that signs the gradient by
outcome, which is gen-2 RL. So the policy learns from competent demonstrations
only.

**Critic: all seats.** The critic's job is to value states, including bad
ones. Training it only on competent seats would leave it blind to incoherent
pools and prone to wild extrapolation there. Weaker-agent seats supply
exactly that coverage. The apparent objection — that a weak agent's
final-deck label reflects a *bad* continuation, not the competent one the
policy will actually use — is mostly benign here:

- *As a baseline, bias is harmless.* The critic is consumed as a baseline for
  advantages (and as the regression target in gen-1 MC training); *any*
  state-dependent value is a valid baseline regardless of bias.
- *Genuinely-bad pools score low whoever finishes them.* The continuation
  confound mainly bites for *middling* pools finished sloppily, not for the
  incoherent ones (which have low score for structural reasons).
- *Gen-2 removes the bias outright.* On-policy actor-critic regresses the
  critic on the policy's own rollouts, so the continuation is the policy's
  by construction — `V^π` for free, no skill tag needed.

So we accept the mild bias rather than reintroducing the agent tag (dropped
as artificial). Together this gives the gen-1 foundation for gen-2's "steer
away from bad picks": imitation establishes competent play, the
broadly-trained critic learns to recognise bad states, and gen-2 actor-critic
wires the critic's advantage into the policy gradient so it actually steers.

### Why no agent (skill) tag in the model input

An earlier design fed a per-seat agent tag (`full` / `r30` / `r100`) to the
model so the critic could learn an agent-aware `V` and be queried with the
policy's tag for `V^π`. We dropped the tag:

- It's an artefact of the synthetic data — not a property of the real draft
  state.
- The actor never needs it (it always plays as itself).
- The critic gets a competent-continuation-ish target more simply by training
  on all seats and accepting the mild bias (or by training on whitelisted
  seats only, restricting coverage). The tag was a third option that adds an
  artificial feature for marginal benefit.
- Gen-2 on-policy training yields `V^π` directly anyway, so the tag was only
  ever a gen-1 crutch.

## Pool evaluator for a dense early signal

The originating discussion weighed four ways to evaluate an incomplete pool
for per-pick reward shaping:

- **Opt 1 — separate pool-quality regression model** (partial pool → final
  win rate).
- **Opt 2 — synthetic completion** (sample completions to 45 cards, run the
  sealed model, average). Rejected: expensive and noisy early.
- **Opt 3 — mask-trained sealed model** (retrain to accept partial pools).
- **Opt 4 — terminal reward only** (no dense signal).

Resolution: the **learned critic** subsumes the pool-evaluator role and gives
a value from pick 1 without touching the sealed scorer. It is effectively
Opt 1, but as a head on the shared trunk rather than a separate model, so the
policy and the value estimate share representation. The sealed scorer cannot
itself serve as the dense critic because it needs a full ~23-spell deck and
so only becomes meaningful late in the draft; the critic learns to predict
that eventual score from any state instead.

### Greedy pool-maximisation is the wrong decision rule

Picking the card that maximises current pool quality each step is a greedy
climb that gets stuck:

- It won't lower pool quality temporarily to switch out of a contested lane.
- It undervalues speculative enablers and fixing.
- It ignores signalling/hate picks.
- Deck quality is lumpy — a coherent 18-card archetype beats 23 scattered
  strong cards across four colours.

The objective is **expected final return**, not current pool value; the
critic is a shaping signal, never the objective.

## Reward / fitness

Pod-relative reward — `deck_score − mean({other seats' deck_score})` —
chosen for several reasons that compound:

### Why pod-relative, not absolute

Subtracting a pod reference is a variance-reducing baseline (drafts of
strong or weak pools yield wildly different absolute scores, but relative
performance within a pod is consistent). In draft it's more than a baseline:
drafters fight over the same cards, so pod-relative reward has a genuine
two-sided gradient — improving yours degrades theirs.

The clinching case is the **hate pick**: slightly lowering your own deck to
deny a key card to a competitor. Under absolute reward that looks bad (your
`deck_score` drops); under pod-relative it is correctly valued positive
whenever the competitor's deck loses more than yours does. The objective
gameplay actually optimises is "win the pod," not "maximise my deck's
absolute quality," and pod-relative reward is what aligns the gradient with
that objective.

### Leave-one-out

The reference is the mean of *other* seats — leave-one-out — so the seat
doesn't subtract itself. For the *mean* form this is just a constant ×7/8
scaling of include-self and mathematically near-equivalent, but cosmetically
cleaner. (For a *max* form it would be essential — see below.)

### Mean over opponents, not max

`pod_mean(others)` aggregates over all 7 opponents and is differentiable in
each. `pod_best(others)` is a single opponent's score with all that
opponent's per-pool noise, plus a hard threshold (a tiny score change can
flip the argmax and discontinuously jump the reward). Three problems with
max:

1. **Smoothness/variance.** Mean averages noise over 7; max rides on one.
   Mean gives smoother gradients and lower-variance labels.
2. **Strategic coverage of denial.** Under mean, hate picks against any
   opponent improve your reward. Under max, only denial against the *current
   argmax* matters — but you can't reliably predict in advance who the
   eventual leader will be, so a signal that only values denial against the
   eventual winner wastes information.
3. **Alignment with the tournament objective.** A draft event is a series of
   pairwise matches; what you accumulate is wins, and `E[wins] = Σ_i P(beat
   i)`, which scales linearly with `deck_score − mean(others)` in any
   Bradley-Terry-ish setting. So `pod_mean` *is* the gradient of "expected
   match wins." `pod_best` rewards going from second to first but doesn't
   distinguish "first by a hair" from "first by a mile," and undervalues
   going from fourth to third — wrong shape for a tournament.
4. **Self-play stability.** In gen-2 self-play, argmax identity is unstable
   as policies improve; mean is stable.

### Aggregation encodes the goal (gen 2)

- Mean of `P(beat i)` = expected match win rate.
- Product of `P(beat i)` = probability of beating everyone, whose gradient
  saturates and concentrates on the closest matchups.
- Linear margin against the top deck rewards overkill and is blind to a
  second shaky matchup — not equivalent to "beat everyone."

### Calibration is cheap here

The scorer is a Bradley-Terry model (`binary_cross_entropy_with_logits` on
the score delta), so scores are already logits: `sigmoid((S_A − S_B)/T)` is
a win probability up to one temperature `T`, fit in an afternoon on held-out
matches. Gen-1's critic regresses pod-relative reward in raw score space;
the temperature fit unlocks switching gen-2 to the product objective if the
goal becomes "robustly beat the whole pod" rather than "crush one deck."

Transitivity of the scalar score is treated as a working assumption,
supported by this project's finding that per-pool score deltas track actual
win-rate deltas.

## Builder for critic labels (picker default, SA fallback)

The picker is the default label-builder over the SA builder purely for
**throughput** — one build per seat per draft (eight per draft); the picker's
~5 ms forward vs the SA builder's ~5 s is the difference between labeling
taking minutes and taking days at corpus scale.

The picker is out-of-distribution on a 45-card pool (it was trained on
~60–90), but that matters far less for *labeling* than it would for
production deckbuilding, for two reasons:

- **The critic needs only rank-consistent labels.** Advantages are value
  *jumps* — a uniformly-slightly-worse builder is a near-monotone transform
  of pool value, which the advantage is invariant to. A constant absolute
  gap between picker and SA builds doesn't bias the critic.
- **On a focused drafted pool, deck score is relatively builder-insensitive.**
  A drafted pool is already ~2 colours and mostly playables. The high-impact
  decisions (play the bombs, stay in colour) are easy and any decent ranker
  gets them; the decisions the picker might flub (the 23rd card, a marginal
  splash) move the score least. So the picker's degradation translates into
  small score error here, plausibly smaller than on a wide sealed pool.

The residual risk is *pool-composition-dependent* divergence — the picker
building well on some pool types and badly on others — which would inject
structured label noise. That's exactly what the §5.3 validation gates on,
with SA as the higher-fidelity fallback.

### Gen-1 validation outcome (2026-06-01): picker confirmed

Ran the §5.3 builder-validation script on 300 freshly-drafted 45-card pools
(`validate_builder --fresh-pools --n-pools 300`, 512-d scorer + the gen5
`4top256` picker):

- picker-vs-SA Spearman ≈ **0.945** (gating),
- SA-vs-SA reference ceiling ≈ **0.995** (SA is near-deterministic on draft
  pools, so almost none of the picker's disagreement is SA noise),
- SA − picker score-gap **median ≈ 0.19**, **IQR ≈ 0.44**.

**Decision: keep `--build-method picker` for gen 1.** The deciding fact is that
the ≈0.19 median gap matches what the picker showed against SA at its *initial*
training, before the (time-consuming) fine-tuning runs
(`experiments/2026-05-22-gen5-picker-initial-training.md`). So the
~60–90-card-sealed → 45-card-draft distribution shift did **not** meaningfully
degrade the picker — the gap is its intrinsic one-shot-vs-search deficit, not
draft-specific — and pod-relative leave-one-out reward absorbs the roughly
uniform component of that gap. Good enough for the first training round.

A dedicated picker fine-tuning run on 45-card draft pools is a possible later
improvement, but it does not gate gen 1.

This is the same continuation-matching idea that recurs elsewhere: label
with whatever builder will actually materialise the agent's decks
downstream.

### How the picker handles 45 cards

No padding, resizing, or special handling. The picker is a set transformer
(SAB layers, no positional encoding, attention masked to real cards) with a
per-card head, so its forward pass is intrinsically length-agnostic. The 45
drafted cards run through the exact same code path a ~80-card sealed pool
takes. The only difference from training is that 45 is below the picker's
~60–90 training range — extrapolation in set size, not a different input
format. Set attention extrapolates gracefully in length, and the dominant
term in each per-card logit is the card's own frozen embedding (pool-size
invariant); the §5.3 validation confirms this empirically before the picker
is trusted as the labeler.

## Critic training (MC vs TD vs warm-start)

Accepted: **Monte-Carlo regression** — label every state with its draft's
final pod-relative reward and fit by MSE. Stable, stationary target, cannot
really fail, and reuses the same drafts as the imitation head (one dataset,
two heads on the shared encoder).

- **Rejected for now — TD / bootstrapping** (regress `V(s_t)` toward
  `V(s_{t+1}) + reward`): lower variance but biased with a moving target;
  not worth the instability on a solo / single-GPU budget. GAE(λ→1) leans on
  MC anyway; add TD only if measured variance forces it.
- **Optional — warm-start from human/17lands data**: pretrain the critic by
  regressing intermediate pool-states against final results. Cheap if the
  data is available; not required.

## Critic-only greedy actor (rejected) → actor-critic

A "try each card, keep the biggest value jump" greedy actor against the
critic is a fine *bootstrap* but not the endpoint, for three reasons:

1. **Off-distribution exploitation.** Greedy argmax over an imperfect critic
   actively hunts its largest overestimate.
2. **Inference cost.** One critic pass per candidate per pick is far costlier
   than one policy forward, compounding over millions of training picks.
3. **Continuation mismatch.** The critic's value assumes a continuation
   policy; if the greedy actor differs, it drifts off-distribution and the
   guidance degrades.

The resolution is **actor-critic**: the policy produces picks and the
rollouts the critic learns from, so the critic's assumed continuation stays
locked to the actual one. Gen 1 already trains both on the same body; gen 2
closes the loop.

## Credit assignment (gen 2)

The originating discussion's recursive-forking ("vine") idea and the
root-level GRPO/RLOO idea are recognised as the **same baseline trick at
different granularities** (pod-level, pick-level, trajectory-level) and
compose. The textbook answer to the heterogeneous per-pick importance they
target is a **learned value function + GAE(λ)**: dense, cheap per-step signal
at every pick, with λ near 1 leaning on Monte-Carlo. That is the gen-2
workhorse, and it is exactly why gen 1 builds the critic. **Forking** stays
surgical — concentrated in pack 2 where picks are pivotal and the critic is
least trustworthy, with common random numbers (fixed unopened packs +
opponent seeds) across paired branches — used to sharpen and audit the
critic, not as the main estimator. Root-only GRPO (one scalar advantage per
full draft) remains the trivial fallback. None of this is built in gen 1.

## Agent-mixed pods (probabilistic, opponent-diversity + critic coverage)

The `--agent-mix` is interpreted as probability weights, with each seat
sampled independently from the categorical distribution per pod. Two
reasons:

- **Pod-level diversity.** Fixed counts (every pod 6 + 1 + 1) make every pod
  structurally identical. Independent per-seat sampling produces varied pod
  compositions and surfaces a real signal the policy should learn: "if a
  good card reaches me late, take it" — the late-pick signal of opponents
  being weak. With a fixed mix, that scenario is under-represented.
- **Critic coverage.** Since the critic trains on all seats, weaker-agent
  seats supply incoherent-pool examples that anchor the low end of the
  value scale.

Heavily-random pods are kept a minority via the default weights, because
all-random dynamics (colours wide open, signals meaningless) are unrealistic
and would teach the policy patterns that competent pods punish. Opponent-seat
randomness stays near pure upside (varied signals + robustness);
heavy own-seat randomness is the delicate dial because it pushes the
imitation distribution out of competent territory, and is kept a minority.

## Self-play regeneration (gen 2)

The OOD fix is to regenerate data from a mix of the current policy + Forge +
laddered/random bots and retrain a next generation on old + new data
concatenated. Mixing preserves opponent diversity; concatenation prevents
catastrophic forgetting and bad-region coverage loss. A steady minority of
dumb/random bots is kept every generation — as the policy improves it stops
visiting incoherent states and that coverage would otherwise evaporate from
fresh data. A frozen held-out set (carved out now) gives a cross-generation
yardstick. All deferred to gen 2.

## Worker → supervisor transport: stdout with a sentinel prefix

For gen 1, the model is *not* a draft participant — Forge's draft AI does
all the picking (with random overrides for the degraded variants). So the
Java worker emits the draft transcript without `deck`/`deck_score`, the
Python supervisor runs the picker and scorer to fill those in and writes
the complete JSON record. The transport question is: how does the Java
worker get its transcript to the Python supervisor robustly, given that
Forge prints random log noise to stdout?

The chosen approach is **sentinel-prefixed stdout**: the Java worker prints
each transcript as one line starting with `<<DRAFT-EVENT-JSON>>` followed by
the compact JSON; the Python supervisor filters for that prefix and
defensively parses the suffix as JSON (silently skipping anything that
fails). Forge's incidental output is ignored, and the worker's own
diagnostics go to stderr.

The simpler-but-less-robust alternative was a sentinel-only contract
without parsing fallback (silently drop unprefixed lines, hard-fail
prefixed lines that don't parse). The slightly more complex alternatives
were a file-based pending area (one JSON file per draft, watched by the
supervisor — fully avoids stdout but adds filesystem coordination), or
configuring Forge's logging to stderr only (invasive and fragile against
libraries that bypass the logger).

The trade-off accepted: if the supervisor crashes mid-record the in-flight
draft is lost (the worker has no idea to re-emit). For gen-1's data-gen
scale that's noise; drafts are cheap. Worker JVM crashes are handled
transparently (supervisor restarts a fresh worker and continues).

### Java integration for shipping (gen 2+)

The picker and scorer (and eventually the trained draft agent) are PyTorch
modules. For gen 1 they run on the Python side — the model isn't in the
draft loop, so there's no per-pick IPC. For eventually shipping the trained
agent with the Forge game, the standard paths are:

- **TorchScript + DJL (Deep Java Library).** `torch.jit.script` the model in
  Python → save as `.pt` → load via DJL's libtorch backend in Java. Standard
  PyTorch family.
- **ONNX + ONNX Runtime (Java bindings).** Export to ONNX, load via
  ONNX Runtime in Java. More portable; export is finickier for non-standard
  ops.

Both work for SAB / multihead attention / linear / LayerNorm / softmax —
every op the picker, scorer, and draft agent use. Worth keeping in mind
during gen-1 development: stick to standard `nn.Module` building blocks
(no custom CUDA kernels or exotic Python control flow that won't trace).
Eventual TorchScript export becomes mechanical.

## First training run (2026-06-03)

First full `train-draft-agent` run, on the greedy-labelled corpus
(`drafts-greedy.jsonl`, `--cards-path output/cardsfolder-512/`). Killed
partway through epoch 1 by hand — read below for why.

**Setup.** 10 623 drafts → 3 793 008 train + 9 240 val examples (draft-disjoint
split), `embedding_dim=544`, `packs=3`, `P=20`. Critic raw target (pod-relative
reward) `mean≈0, std≈1.4712`; targets are standardised before the MSE, so the
meaningful critic baseline is "predict the mean → MSE ≈ 1.0". Schedule: up to
100 epochs, eval+checkpoint every 1185 steps (≈100 mini-epochs/epoch), patience
30 mini-epochs. ~10.5 steps/s, drifting down to ~8.7 over the run (likely host/
GPU contention or throttling, not the eval pauses).

**Where it got to (~1.8 epochs):**

- **Imitation head — learned well.** top1 (model's #1 == Forge's actual pick)
  0.35 → ~0.78; top3 0.67 → ~0.98; val_imit (CE) 1.70 → ~0.55. Top-3 of ~0.98
  is the headline: the demonstrated pick is almost always in the model's top
  three.
- **Critic head — good, plateaued early.** val_crit ≈ 0.27–0.28 on the
  standardised scale ⇒ roughly **75 % of reward variance explained**. It was
  already there by mid-epoch 0 and barely moved afterwards. `per_pack_mse`
  ordered p1 ≈ 0.44 > p2 ≈ 0.23 > p3 ≈ 0.14 — exactly right: final-deck quality
  is hardest to predict at pack 1 (deck maximally undetermined) and easiest at
  pack 3 (deck nearly fixed).
- **No overfitting.** train and val track each other tightly throughout
  (3.8 M examples vs this model size), so the gap is negligible.

**Why it was killed.** The run *looked* like it was bouncing in a fixed range,
and the apparent best (val_loss ≈ 0.804 at epoch 1 step 52140) looked like
chance rather than a durable level. That read is half right:

- *The 0.804 is a lucky eval.* Its neighbours sit at 0.84–0.86, so it's a ~0.04
  downward spike off a local level of ~0.85. Best-checkpoint-by-val_loss takes
  the **minimum over ~180 noisy eval estimates** (≈100/epoch × ~1.8 epochs);
  with an eval-noise band of ±0.03–0.04 the min of that many draws sits well
  below the true level by order-statistics alone. So the saved "best" is partly
  selection bias.
- *But there was real epoch-over-epoch improvement.* The val_loss **envelope**
  went from ~0.91 (end of epoch 0) to ~0.83 typical in epoch 1 — a genuine
  ~0.07–0.08 drop, not noise. The improvement is **decelerating**, and the
  per-epoch gain is now comparable to the eval noise, which is exactly why it
  reads as "flat" to the eye.
- *The moving part is imitation, not the critic.* Almost all of the epoch-0→1
  val gain is in val_imit (~0.63 → ~0.55); val_crit was flat from early epoch 0.
  So the critic was effectively done; pick-prediction was still slowly
  improving.

**Caveats / follow-ups.**

- This run used **greedy**-built deck labels, whereas gen-1's chosen labeller is
  `--build-method picker` (see the 2026-06-01 validation entry). The `imit`
  numbers are builder-independent (pick labels don't depend on the builder), but
  the `crit` numbers are **not** comparable to a picker-labelled run.
- Next run: judge convergence on a **trailing average** of evals (or a larger
  val set) so the trend isn't drowned by ±0.03 jitter — and so best-checkpoint
  selection doesn't lock onto a lucky dip. If maximum imitation top-1 is wanted,
  another epoch or two probably still had a little to give; the critic did not.

### Lowering the LR — the 3e-4 → 3e-5 → 3e-6 decay ladder (2026-06-03)

Dropping the LR one decade at a time at each plateau is a clean, monotone win —
textbook decay-at-plateau, and exactly the behaviour the `--lr-decay-patience`
annealing automates in a single run:

| stage            | resumed from | best val_loss | val_imit | val_crit | top1  |
|------------------|--------------|---------------|----------|----------|-------|
| `3e-4` (initial) | scratch      | ~0.80         | ~0.53    | ~0.27    | 0.79  |
| `3e-5`           | 3e-4 plateau | **0.635**     | 0.363    | 0.272    | 0.860 |
| `3e-6`           | 3e-5 best    | **0.615**     | 0.338    | 0.277    | 0.870 |

- **`3e-5` is the big win.** The `3e-4` run plateaus around 0.80 / top1 0.79;
  resumed at `3e-5` it descends cleanly and monotonically to **0.635 / top1
  0.860** over one epoch. `3e-4` is too hot to settle the near-converged policy;
  a decade lower unsticks it.
- **`3e-6` is a small polish.** Resumed from the `3e-5` best, val_loss sits nearly
  flat (~0.628–0.635) for most of the epoch, then dips late to **0.6148 / top1
  0.870** (step 100725) and early-stops 30 mini-epochs later. ~0.02 val_loss and
  +1 pt top1 — sharply diminishing returns, i.e. `3e-6` is near the floor.
- **All of it is the policy head.** Across the two stages val_imit falls
  0.51 → 0.338 while val_crit stays pinned ~0.27 (the critic converged early and
  even drifts slightly *up* at the lower LRs as the policy keeps fitting); top1
  climbs 0.79 → 0.87.

Net: the gen-1 best is **val_loss 0.6148 / top1 0.870** (`20260603_234318.pt`),
reached by the decay ladder — each decade lower settles the policy a notch finer
with diminishing returns, which is what the annealing feature drives end-to-end.

## Open / future questions

- **Gen-2 RL spec.** Actor-critic with GAE, the choice between REINFORCE
  (simpler) and PPO (more sample-efficient), the KL anchor to the imitation
  policy, the reward calibration choice (mean vs product after temperature
  fitting), the role of surgical forking, and the self-play regeneration
  loop.
- **Java integration for shipping** — see above.
- **Cross-format generalisation.** The model is set-agnostic and
  format-agnostic by construction (no set embedding); how well it
  generalises across sets and to Chaos draft is an empirical question for
  later.

## Glossary

- **POMDP** — partially observable decision process; the agent acts on its
  observation history (its belief about hidden state).
- **Critic / value function `V(s)`** — predicted expected final return from a
  state; here, the predicted final pod-relative reward.
- **Advantage** — how much better an action is than the state's average; the
  per-pick value jump `V(s_{t+1}) − V(s_t)`.
- **GAE(λ)** — generalised advantage estimation; λ interpolates Monte-Carlo
  (unbiased, high variance) and TD/critic (biased, low variance).
- **Baseline** — a reference subtracted from reward to cut variance (the
  leave-one-out pod mean here).
- **GRPO / RLOO** — sample a group of trajectories from a state; advantage =
  reward minus the group mean.
- **Vine (TRPO)** — branch at states and compare rollout returns for a
  low-variance advantage.
- **CRN** — common random numbers; hold the future fixed across compared
  branches.
- **`V^π`** — the value assuming the current policy continues (the target RL
  actually wants); in gen 1 only loosely approximated (the critic regresses
  on a mix of Forge continuations), and reached directly by gen-2 on-policy
  training.
- **Bradley-Terry model** — a logistic model where pairwise outcomes are
  `P(A beats B) = sigmoid(S_A − S_B)`; the sealed scorer was trained this
  way, so its scalar scores are already logits.
