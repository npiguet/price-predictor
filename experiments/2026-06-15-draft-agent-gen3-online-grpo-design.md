# Draft agent (gen-3) — online GRPO self-play design

## Context — what gen-2 was, and why we are moving off it

Gen-2 is an **offline actor-critic**: freeze a snapshot πₖ, generate a self-play
corpus with it, then train **many passes over that fixed corpus** with
`REINFORCE + GAE baseline + KL anchor + entropy bonus`. The policy gradient term
is `−A·logπ(a)`.

That term is **unbounded below**: for a negative-advantage pick, minimising
`−A·logπ = |A|·logπ` drives `π(a) → 0` and `logπ → −∞`. On a fixed corpus reused
over many epochs the policy drifts off the snapshot, its advantages go stale, and
the optimiser **reward-hacks the stale numbers** — with only the KL/entropy leash
bounding it.

Empirically the gen-2 runs bore this out: a near-noop, a catastrophic collapse
(`val_loss → −114`) once a decaying KL schedule cut the leash, and — even with a
*fixed* KL — the same collapse merely slowed (a soft KL penalty bounds average
divergence, not the per-pick surrogate). The full run-by-run results live in
the gen-2 experiment file
([`2026-06-10-draft-agent-gen2-design.md`](2026-06-10-draft-agent-gen2-design.md),
*Results* section). The structural lesson carried into gen-3: **offline reuse is
what makes the surrogate hackable**, and a decaying trust-region leash over a
stale corpus is a fragile way to contain it.

## Decision

Gen-3 switches to an **online, critic-free, GRPO-style** loop and strips the
auxiliary machinery to the minimum:

1. **Online (streaming) self-play.** Generate ~10 fresh drafts from the *current*
   policy, take **one pass** over them (~56 minibatch steps at batch 32),
   update, throw the corpus away, regenerate, repeat. Data is effectively
   infinite; nothing is ever shown twice.
2. **Critic-free GRPO advantage.** The advantage is the **pod-relative
   leave-one-out `deck_score`, standardised over the round** — the same scalar for
   all 45 of a seat's picks. No learned critic, no GAE.
3. **No KL anchor.** Freshness is the trust region.
4. **No entropy bonus.** Exploration comes from the rollout temperature `T`.

The loss collapses to a single term over the learner picks:

```
L = −A · logπ_T(a | s)          (averaged over learner picks)
A = standardise_over_round( pod-relative leave-one-out deck_score )
```

## Why each piece

### Why online instead of offline

Online regenerates the corpus every ~56 steps from the policy being trained, so
the advantages are **always fresh** — computed against essentially the same
policy that produced the data. The surrogate never gets a stale target to hack,
so the KL/entropy leash that gen-2 leaned on becomes optional rather than
load-bearing. The freshness *is* the trust region. The whole gen-2 failure mode
(stale-corpus reward-hacking, fragile decaying leash) cannot occur because there
is no stale corpus.

The cost of online is rollout generation, but in this project generation and
training take **about the same wall-clock time** (an even split), and drafts are
cheap (several per minute, ~180 learner examples each at a 50% learner seat mix).
Regenerating every round is affordable, so the central objection to online —
rollout cost — does not bind here.

### Why critic-free (GRPO) instead of actor-critic + GAE

- **The group baseline is already in the reward.** The pod-relative
  leave-one-out `deck_score` *is* the RLOO/group baseline (each seat measured
  against the mean of the other seats in its pod). The main variance reducer lives
  in the reward construction, not in the critic; the gen-2 critic was a *second*
  baseline layered on an already-centred signal.
- **The reward is coarse, so fine credit assignment buys little.** `deck_score`
  is a build-filtered terminal reward that barely moves with any single pick, so
  GAE's per-pick credit had little real signal to assign. Monte-Carlo return
  (one shared advantage per seat) loses almost nothing here.
- **Online would make the critic harder, not easier.** A learned critic online
  must track a *moving* `V^π` and carry standardisation/refresh bookkeeping every
  round. Dropping it removes that whole problem.
- **It matches where the field has gone.** PPO with a learned value model + GAE
  was the long-standing default (and gen-2's lineage). For **terminal/outcome
  rewards with a natural per-prompt group**, the modern choice is critic-free —
  GRPO / RLOO — precisely because a separate value model is costly and finicky and
  a group baseline is simpler and often lower-variance. Our setting (terminal
  `deck_score`, the pod as the group, free generation) sits squarely in that
  column; gen-2's offline reuse with an inherited critic head sat in the other.

Concretely, going critic-free deletes `value_weight`, `gae_lambda`, the critic's
training, the standardisation bookkeeping, and the per-round advantage-precompute
pass.

### Why drop the KL anchor

The KL leash was only ever needed to bound the unbounded surrogate over *stale*
advantages. Online removes staleness, so the leash is no longer load-bearing.
Dropping it removes `kl_coef` **and** the val-keyed decay scheduler — the exact
mechanism that caused gen-2's collapse.

### Why drop the entropy bonus

Exploration is already provided by the rollout temperature `T` (sample-mode
generation). An entropy bonus is a redundant *second* exploration dial. We drop
it; if the policy later collapses to monotonous decks we can add a small **fixed**
one back (never a decaying schedule).

## What is left to tune

Three knobs, each fully understood:

- **`lr`** — step size. Online tolerates a *larger* lr than gen-2's 3e-5, because
  each round resets the policy on-policy, so a too-large step is corrected by the
  next fresh batch rather than compounding into runaway.
- **`temperature T`** — exploration, applied at generation time. The single
  exploration lever. See *Exploration and the residual failure mode* below for how
  high it needs to be and how to read that off the logs.
- **`drafts-per-round`** — the corpus size per update (10).

Fixed-and-forget: batch size 32 (8 GB VRAM budget), grad-clip max-norm 1.0,
**one pass per round**, the `learner_agents` whitelist.

## Exploration and the residual failure mode

Gen-2 had **two** distinct failures, and online only cures one of them:

1. **Meaningless learning on stale / hacked advantages.** Online cures this by
   construction — there is no stale corpus.
2. **The gradient reshapes the probability tail without ever flipping the argmax
   top pick.** Online does **not** automatically cure this. Whether gen-3 escapes
   it depends on two things:
   - **Exploration** — the policy must sample non-argmax picks often enough that
     the gradient can *see* a stronger alternative and reinforce it past the
     current top pick. If every rollout takes the argmax, the top pick is the only
     action ever credited, so it can only be reinforced, never displaced.
   - **Reward discrimination** — `deck_score` must actually differ between the
     argmax pick and its challenger for the advantage to point anywhere. This is a
     property of the reward, not of `T`; we can only watch it, not tune it.

Temperature `T` is the one lever on exploration, so "is `T` high enough?" really
means "is gen-3 exploring enough to have a *chance* of moving the top pick?"

**The crucial contrast with gen-2:** even if gen-3 *also* turns out to be a no-op
on the objective, its no-op is **transparent** — a flat anchor margin says so
directly. Gen-2's no-op was masked by a training loss diving to −3 while the
argmax deck quality never moved. Whatever gen-3 does, we will be able to read it.

### How high does T need to be, and how to read it off the logs

At `T = 1.0` the policy is too sharp to explore: the reference's mean
`log π_ref ≈ −0.324` and gen-2's training entropy `H ≈ 0.33` correspond to a
**perplexity `exp(H) ≈ 1.4`** — only ~1.4 effective choices per pick, i.e. it
almost always takes the argmax. Raising `T` flattens the sampling distribution.

Aim the sweep at:

- **perplexity `exp(H) ≈ 2–3`** effective choices per pick,
- **off-argmax sampling rate ≈ 25–40 %** (fraction of picks where the sampled
  card ≠ the argmax card).

Sweep **`T ∈ {1.0, 1.5, 2.0, 2.5}`** and take the *lowest* `T` that reaches those
bands — higher `T` than needed only adds gradient variance.

Two cheap diagnostics to add to the per-round log (both computed from rollout
distributions we already have):

- **perplexity `exp(H)`** of the sampling policy — the readable form of entropy,
- **off-argmax rate** — fraction of learner picks where the sampled action ≠ argmax.

And one thing to watch *across* rounds: **entropy decays even at fixed `T`**,
because REINFORCE sharpens the logits as it learns (gen-2 drifted `0.33 → 0.10`
over a run). So a `T` chosen at round 0 can silently fall out of the exploration
band by round 20. Watch the perplexity / off-argmax curves over the whole run, not
just at the start; if they sag below the band, raise `T` (or, later, add a small
**fixed** entropy bonus).

## Rejected / dropped (and the trigger to revisit)

- **Offline multi-epoch training over a frozen corpus** (the gen-2 recipe) —
  the staleness/collapse source. Not revisited.
- **Learned critic + GAE** — deferred. Revisit if gradient variance visibly
  stalls learning, or if a denser/shaped reward ever replaces `deck_score` (where
  per-pick credit would start to matter).
- **KL anchor** (to frozen gen-1 or to the previous round) — deferred. Revisit by
  adding a *small previous-round* KL if the policy lurches too far in a single
  round.
- **Entropy bonus** — deferred. Revisit with a small **fixed** coefficient if the
  policy's entropy crashes / decks go monotonous.
- **Any val-keyed coefficient decay schedule** — rejected outright; this was the
  proximate cause of the gen-2 collapse.
- **In-the-Forge-loop per-pick online** (live IPC for value/logπ at every pick) —
  rejected (carried over from gen-2): it reintroduces per-pick IPC, gives tiny
  high-variance batches, and *still* needs a trust region. Our "online" is at the
  granularity of whole drafts, with log-probs recomputed post-hoc from the fresh
  corpus — no live value/logπ side-channel is needed.

## How we will know it worked

Unchanged from gen-2 — the **cross-generation yardstick**: one greedy
`generate-draft-data --pick-mode argmax` run with a single fixed agent mix that
co-seats the generations being compared at random, then
`analyze-generated-decks --agent <each>` and compare **mean `deck_score`**.
Random co-seating balances the opponent-strength confound so the per-agent means
are directly comparable. Promotion is a **manual judgement**, no fixed rule.

### Live progress signal — the anchor margin

The yardstick needs training to stop. For a *continuous* read on progress without
stopping, we exploit the fact that the self-play generation mix already contains
**fixed-strength anchor seats** — a frozen gen-1 (~30–35 % of seats) plus the
Forge bots (`forge-r30`, `forge-r100`). The learner (gen-3) is only ~50 % of each
pod; the rest is a stable reference field. So we log, over a sliding window of the
last ~100 drafts:

```
anchor_margin = mean( gen-3-seat deck_score ) − mean( frozen-gen-1-seat deck_score )
```

Because the anchor never moves, this margin is a genuine **absolute** progress
curve — it climbs as gen-3 pulls away from its starting point and plateaus as the
improvement saturates. (`gen-3 − forge-r30/-r100` margins come for free and read
as absolute strength vs the gen-0 Forge baseline.) A ~100-draft window over a
30–35 % anchor gives ~250+ frozen-anchor decks, so the windowed mean is precise,
not coin-flip noise.

Two disciplines keep this valid:

- **The anchor stays frozen for the whole gen-3 run** — never swapped to a later
  generation or to "previous round." The instant the anchor moves, the baseline
  moves and the curve stops meaning "improvement over a fixed point." (Same
  frozen-reference discipline as the yardstick, applied live.)
- The margin is read on **sample-mode** rollout decks while promotion is judged on
  **argmax**; since sample-vs-argmax `deck_score` deltas are nearly identical
  (exploration among near-equal picks doesn't change the built 40-card deck), the
  live margin tracks the eventual yardstick closely.

Treat a **plateau in the anchor margin** (over the last ~100 drafts) as the
trigger to *pause and run the real argmax yardstick* — not as an automatic stop
and not as the promotion decision. Plateaus in RL can be temporary shelves, and
promotion is judged on the yardstick regardless.

Other in-run health signals to watch (cheap, no held-out val set needed):

- **perplexity `exp(H)`** and **off-argmax rate** staying inside the exploration
  band (see *Exploration and the residual failure mode*) — these decay across
  rounds even at fixed `T`, so watch the curve, not just round 0,
- **policy entropy** not crashing toward zero (premature collapse; the same signal
  as perplexity, in nats),
- **KL-to-previous-round** (logged even though unpenalised) staying modest — a
  large per-round KL means `lr` is too high for the round size.

(Only the on-policy gen-3 seats feed the policy gradient — `--learner-agents
gen-3`; the anchor seats serve purely as opponents and as this progress readout.)

## Open questions

- **Base checkpoint:** gen-2 or gen-1. Currently leaning gen-1, since gen-2 looks
  marginal; settle this against the yardstick before the first gen-3 round.
- **`lr` and `T` starting values**, and how aggressive a single round can be
  before per-round KL gets uncomfortably large.
- **Orchestration:** the generate→train→repeat loop is many small rounds. Driving
  it by hand is painful enough that it likely justifies a thin orchestration
  script around the existing commands — to decide once the manual loop has proven
  the recipe.
- **Stopping:** how many rounds. The live **anchor margin** is the in-run trigger
  to pause and check; promotion is judged on the argmax yardstick.

## Glossary (gen-3 additions; see the gen-2 doc for the rest)

- **GRPO (Group Relative Policy Optimization)** — a critic-free policy gradient
  that uses the *mean reward over a group of rollouts sharing a prompt* as the
  baseline, instead of a learned value function. Here the **pod is the group**.
- **RLOO (REINFORCE Leave-One-Out)** — the same idea with each sample's baseline
  being the mean of the *other* group members. Our pod-relative leave-one-out
  reward is exactly this estimator.
- **Online / streaming policy gradient** — generate fresh rollouts from the
  current policy, take a small number of steps, discard, regenerate. Keeps every
  update on-policy by construction.
- **Group baseline** — subtracting a per-group mean reward to centre the
  advantage; a variance reducer that needs no separate model when comparable
  rollouts per prompt are cheap.

---

## Outcome / Result

_(to be filled in as gen-3 rounds run)_

- **Base / config used:**
- **Rounds run:**
- **Yardstick result (mean `deck_score`, gen-3 vs gen-2 vs gen-1 vs Forge):**
- **In-run behaviour (reward trend, entropy, per-round KL):**
- **Verdict (promote / iterate / abandon):**
- **Follow-ups (which dropped piece, if any, had to come back):**
