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

The cost of online is rollout generation, but drafts are cheap: several per minute,
~180 learner examples each at a 50% learner seat mix. Regenerating every round is
affordable, so the central objection to online — rollout cost — does not bind here.

The runs put a number on that cost, and it is not the even split assumed when this
was written. Generation is 81–86 % of each round, 44–58 s against 8–12 s of training
at 10 drafts/round. Regenerating every round remains affordable. Speedup work belongs
on the Forge side, and `--drafts-per-round` is close to free in training time.

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

Sweep `T` and take the lowest value that reaches those bands, since higher `T` than
needed only adds gradient variance. The sweep proposed here, `T ∈ {1.0, 1.5, 2.0,
2.5}`, was aimed too low: nothing below `T = 3` reaches the band on this policy.

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

Because the anchor never moves, this margin was expected to be a genuine absolute
progress curve, climbing as gen-3 pulls away from its starting point and plateauing
as the improvement saturates. A ~100-draft window over a 30–35 % anchor gives ~250+
frozen-anchor decks, so the windowed mean is precise, not coin-flip noise.

Frozen weights did not give a frozen baseline. Denial drags the whole field down as
the learner improves: an improving learner takes cards its podmates would otherwise
have had, so their decks get worse even though their weights never change. Windowed
means over the 2026-08-08 run:

| Label | round 9 | round 1268 |
|---|---|---|
| `gen3a`, the anchor | 1.665 | 1.308 |
| `gen3c` | 1.625 | 0.417 |
| `forge-full` | 1.115 | 0.764 |
| `gen4`, the learner | 1.883 | 1.776 |

The learner's own raw mean fell too, on a steeper fitted slope than the anchor's. So
the margin is a competitive result against a stated field, not a measurement of
absolute strength. A rising margin can mean pulling away from a field that is itself
sinking. Whether the absolute decline is denial, the three learner seats competing
for the same cards, or genuine degradation is unresolved and worth its own run.

Two disciplines keep this valid:

- **The anchor stays frozen for the whole gen-3 run** — never swapped to a later
  generation or to "previous round." The instant the anchor moves, the baseline
  moves and the curve stops meaning "improvement over a fixed point." (Same
  frozen-reference discipline as the yardstick, applied live.)
- The margin is read on **sample-mode** rollout decks while promotion is judged on
  **argmax**; since sample-vs-argmax `deck_score` deltas are nearly identical
  (exploration among near-equal picks doesn't change the built 40-card deck), the
  live margin tracks the eventual yardstick closely.

The trigger to pause and run the real argmax yardstick was meant to be a plateau in
the anchor margin over the last ~100 drafts. The runs do not plateau, they peak and
decline, so the trigger is a decline from a tracked best. Best-checkpoint selection
and LR annealing now handle it. Either way it is not an automatic stop and not the
promotion decision; promotion is judged on the yardstick regardless.

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

Gen-3 clears the objective gen-2 failed to move. Every candidate taken to the
yardstick beats both the frozen gen-1 it was fine-tuned from and `forge-full`. Gen-2
had tied Forge at 1.49.

Config: learner and anchor both warm-started from the same gen-1 checkpoint; reward
from the frozen sealed scorer with `--build-method greedy`; 10 drafts/round, batch 32,
clip 1.0, 200 warmup steps, 100-draft anchor window, random set, seed 42.

The raw reward turned out to be a weak progress read. It rose in three of the four
`lr 1e-5` runs and stayed flat in the fourth. That is what a pod-relative
leave-one-out quantity does when about 3 of 8 seats are learners.

Best-checkpoint selection, patience and LR annealing (spec 021 phase 7) were added
during the sweep, in response to the peak-and-decline behaviour below.

### The runs

Two families, differing in two respects:

- **Field at T** (2026-08-05) — one `-T` applied to every ML agent, so the frozen
  anchor sampled too. Mix `gen3:3,gen1:5,forge-r30:1,forge-r100:1`.
- **Field at argmax** (2026-08-06) — `-T` on the learner only (spec 021 phase 8).
  Mix `gen3:3,gen1:3,forge-full:2,forge-r30:1,forge-r100:1`: two anchor seats
  traded for two pure-Forge seats.

Each yardstick is a separate 500-draft (4000-seat) argmax run under the fixed mix
`gen3:2,gen1:1,forge-full:1`, giving ~2000 gen-3 decks and ~1000 per reference.
Scores are mean (median).


| Family | lr | T | Rounds | Best margin (round) | Final margin | gen-3 (ys)  | gen-1 (ys)  | forge-full (ys) | gen3 − gen1 (ys) |
|---|---|---|---|---|---|-------------|-------------|-----------------|------------------|
| field at T | 1e-6 | 2.0 | 89 | +0.255 (r87) | +0.255 | —           | —           | —               | —                |
| field at T | 1e-5 | 2.0 | 121 | +0.887 (r101) | +0.707 | 1.98 (2.08) | 1.25 (1.32) | 1.14 (1.21)     | +0.73            |
| field at T | 1e-5 | 3.0 | 28 | +0.504 (r27) | +0.504 | 1.76 (1.87) | 1.51 (1.62) | 1.31 (1.39)     | +0.25            |
| field at T | 1e-4 | 2.0 | 33 | +0.444 (r15) | +0.363 | —           | —           | —               | —                |
| field at argmax | 1e-5 | 2.0 | 89 | +0.374 (r47) | +0.081 | 1.79 (1.92) | 1.41 (1.52) | 1.23 (1.40)     | +0.38            |
| field at argmax | 1e-5 | 3.0 | 199 | +0.439 (r179) | +0.114 | 1.73 (2.12) | 1.17 (1.31) | 1.02 (1.16)     | +0.56            |

Read the margin columns with three limits in mind. Margins do not compare across the
two families, which trained against different fields. Run lengths are not controlled:
the field-at-T `T = 3` run was stopped while its margin was still climbing, so its
yardstick figure is a lower bound. And every yardsticked row used its run's
best-margin checkpoint except field-at-T `T = 2`, which predates best-checkpoint
selection and used the nearest snapshot, round 105.

The `gen3 − gen1` column is the reliable ranking. Within a corpus the agents share
pods, so the comparison is paired and pod luck cancels. The field is fixed, so the
gen-3 checkpoint is the only variable. Set and booster luck averages out over 500
drafts by the law of large numbers. Ranking on `forge-full` instead of gen-1 gives the
same order.

### Exploration

| Family / run | T | ppl @r0 | ppl mean | ppl last 10 | off-arg @r0 | off-arg mean | off-arg last 10 | in band? |
|---|---|---|---|---|---|---|---|---|
| field at T, lr 1e-5 | 2.0 | 1.75 | 1.56 | 1.53 | 20.3 % | 17.0 % | 15.2 % | no — declines out |
| field at argmax, lr 1e-5 | 2.0 | 1.97 | 1.71 | 2.02 | 27.5 % | 19.9 % | 25.2 % | no — sags mid-run |
| field at T, lr 1e-5 | 3.0 | 2.67 | 2.29 | 2.06 | 34.7 % | 28.9 % | 25.2 % | yes |
| field at argmax, lr 1e-5 | 3.0 | 2.51 | 2.15 | 2.10 | 34.2 % | 27.5 % | 29.5 % | yes |
| field at T, lr 1e-6 | 2.0 | 1.77 | 1.81 | 1.72 | 21.4 % | 22.3 % | 20.5 % | no — flat |
| field at T, lr 1e-4 | 2.0 | 1.80 | 2.11 | 2.30 | 20.3 % | 29.3 % | 34.2 % | nominally, but thrashing (1.35–3.96) |

`T = 2` starts below the band and declines further, bottoming at 9.1 % off-argmax in
the field-at-T run. `T = 3` starts inside the band, and the 199-round run was still
inside it at round 198. So the entropy decay predicted above at fixed `T` is present
at `T = 2` and absent at `T = 3` over three hours. `T = 3` also ranks above `T = 2`
within the field-at-argmax pair. The best agent overall still came from `T = 2`, in
the other family, over more rounds.

### Movement, and the learning-rate sweep

| Family / run | KL(prev) median | p90 | max | KL(π₀‖πₖ) final | grad-norm mean | max |
|---|---|---|---|---|---|---|
| field at T, lr 1e-6 | 0.0000 | 0.0001 | 0.0001 | *not logged* | 5.4 | 6.9 |
| field at T, lr 1e-5, T 2.0 | 0.0034 | 0.0100 | 1.349 | *not logged* | 6.2 | 15.8 |
| field at T, lr 1e-5, T 3.0 | 0.0022 | 0.0044 | 0.0071 | 0.140 | 4.7 | 5.8 |
| field at T, lr 1e-4 | 0.974 | 2.019 | 3.568 | *not logged* | 13.1 | 36.3 |
| field at argmax, lr 1e-5, T 2.0 | 0.0054 | 0.171 | 0.500 | 1.181 | 7.6 | 13.9 |
| field at argmax, lr 1e-5, T 3.0 | 0.0297 | 0.154 | 0.376 | 0.924 | 7.0 | 12.0 |

(KL-to-run-start was added mid-sweep on 2026-08-05; the three earlier runs lack it.)

`lr 1e-6` did not train: its per-round KL is indistinguishable from zero, its
learner score barely moved over 89 rounds, and most of its small best margin came
from anchor drift. That margin is in any case a maximum over ~80 overlapping
windows whose per-window standard error is on the order of 0.1.

`lr 1e-4` was unstable. Its per-round KL runs two orders of magnitude above the
`lr 1e-5` runs, with gradient norms and perplexity swinging in step. Its margin
nonetheless reads healthy in the run table, peaking and ending well above zero. The
component means show why that reading misleads. The learner's own score peaked near
round 15 and then returned to its starting level, while the anchor kept falling. After
round 15 the margin was sustained by the field declining. That is a legitimate result
against that field, but not one the yardstick's field rewards, and the run was not
carried to a yardstick. The learner's raw windowed score, printed on the same
`progress` line as the margin, separates the two cases.

`lr 1e-5` is the operating point. Its cumulative KL runs two orders of magnitude
above its per-round median, as anticipated above.

### Deck composition

From the yardstick corpora, where gen-3 and gen-1 share pods:

| Candidate | creatures g3 / g1 | 2-colour % g3 / g1 | avg MV g3 / g1 | rares g3 / g1 |
|---|---|---|---|---|
| field at T, T 2.0 | 17.72 / 15.92 | 73.4 / 64.3 | 3.17 / 3.11 | 1.31 / 1.57 |
| field at T, T 3.0 | 17.11 / 16.55 | 73.0 / 70.4 | 3.13 / 3.13 | 1.44 / 1.52 |
| field at argmax, T 2.0 | 17.13 / 16.10 | 64.7 / 68.9 | 3.14 / 3.10 | 1.40 / 1.50 |
| field at argmax, T 3.0 | 17.35 / 15.67 | 65.7 / 69.0 | 3.14 / 3.08 | 1.39 / 1.63 |

Every gen-3 drafts more creatures than the gen-1 it came from, on an unchanged curve,
with slightly fewer rares and uncommons: on-colour commons in place of higher-rarity
cards. This is the objective as specified. Forge pilots creatures better than spells,
and the scorer was fitted to Forge-piloted outcomes.

Colour discipline is where the families separate, and they separate in opposite
directions from the frozen gen-1 the learners started from. Counting distinct basic
land types in each built deck's mana base, paired within corpus:

| Candidate | share ≥ 4 types, g3 / g1 / forge | g3 − g1 | score of the ≥ 4 decks, g3 / g1 | learner picks trained |
|---|---|---|---|---|
| field at T, T 2.0 | 5.5 / 9.1 / 10.4 % | −3.6 pp | 1.45 / 0.52 | ~110k |
| field at T, T 3.0 | 6.4 / 7.3 / 7.7 % | −0.9 pp | 1.19 / 0.66 | ~30k |
| field at argmax, T 2.0 | 8.4 / 7.1 / 7.9 % | +1.3 pp | 1.15 / 0.33 | ~50k |
| field at argmax, T 3.0 | 13.3 / 6.1 / 8.0 % | +7.2 pp | −0.48 / +0.14 | ~195k |

Both field-at-T policies build narrower mana bases than the frozen gen-1 sitting in
the same pods; both field-at-argmax policies build wider ones. Within each family
the deviation grows with the number of learner picks trained on, so the field sets
the direction of the drift and training length sets its size.

Going wide is not itself the fault. For three of the four candidates the wide decks
comfortably outscore the references' wide decks. That is what going wide only when the
pool supports it looks like. The field-at-argmax `T = 3` candidate inverts it: its
wide decks score below those of the gen-1 seats beside it. It goes wide when the pool
does not support it.

The record preserves every pick and everything passed at it, so why that happens is
answerable from the corpus rather than by conjecture. See *What the pick record says*
below.

### Mean against median

The candidates differ in the shape of their score distribution, not only its level. A
median above the mean means a left tail: a minority of bad decks pulling the average
down while the median stays in the bulk. The table below splits that gap in two. Drag
is the part four- and five-colour mana bases account for, measured as how much the
mean would rise if those decks scored like the same policy's three-colour decks. The
residual is everything else.

| Candidate | med − mean | wide-deck drag | residual | margin over gen-1, mean / median |
|---|---|---|---|---|
| field at T, T 2 | +0.10 | +0.015 | +0.087 | +0.73 / +0.76 |
| field at T, T 3 | +0.11 | +0.013 | +0.100 | +0.25 / +0.25 |
| field at argmax, T 2 | +0.13 | +0.025 | +0.105 | +0.38 / +0.40 |
| field at argmax, T 3 | +0.38 | +0.283 | +0.101 | +0.56 / +0.81 |
| *gen-1 and forge-full, all corpora* | +0.07 … +0.17 | +0.020 … +0.054 | +0.034 … +0.136 | — |

Two separate things live in that gap. The residual is flat across all four candidates
and the references carry the same skew, so it belongs to `deck_score` itself. A sealed
deck can be far worse than average more easily than it can be far better: no
playables, a broken curve, colour screw. Nothing about the training explains it.

Everything that separates the families sits in the drag column, and subtracting it
leaves the four candidates indistinguishable in shape. Bucketing every gen-3 deck by
the number of distinct basic land types in its mana base shows what the tail is
made of:

| Candidate | 2 types | 3 types | 4 types | 5 types |
|---|---|---|---|---|
| field at T, T 2.0 | 2.12 (68 %) | 1.72 (26 %) | 1.51 (4.8 %) | 1.02 (0.7 %) |
| field at T, T 3.0 | 1.94 (68 %) | 1.40 (25 %) | 1.20 (5.2 %) | 1.13 (1.2 %) |
| field at argmax, T 2.0 | 2.04 (61 %) | 1.46 (30 %) | 1.17 (7.4 %) | 1.02 (0.9 %) |
| field at argmax, T 3.0 | 2.25 (63 %) | 1.59 (23 %) | 0.09 (9.4 %) | −1.88 (3.9 %) |

Each cell is the bucket's mean `deck_score`, with the share of that candidate's
decks falling in it.

Two factors multiply, and the `T = 3` argmax candidate is worse on both. It builds
four- and five-colour decks 2.4× more often than the field-at-T candidates, and its
wide decks are much worse than theirs. Its five-colour decks average −1.88, where
every other candidate's average around +1.0. Its two-colour decks are the best
two-colour play in the sweep, which is why it leads on median margin while placing
second on mean.

The checkpoints trained against the sampled fields are the ones with the better and
tighter score distributions, which is not what the design expected. The pick record
below is where a mechanism has to come from.

### What the pick record says

The corpus stores every booster in pick order, so each seat's 45 picks reconstruct
exactly, along with every card it passed at each of them. Why the colours drift is
therefore answerable from the record rather than by conjecture. (Data extracted with
`scripts/analyze_draft_lanes.py` and `scripts/analyze_pick_quality.py`.)

Each pick gets two readings, both anchored on the seat's eventual top-2 colours. Was
the pick off-lane? And if it was, was an on-colour card still in the pack, meaning the
agent chose rather than was forced? Anchoring on the top two rather than on the
finished deck's colours matters, since a five-colour deck would otherwise score as
perfectly on-colour against itself.

Off-lane rate at picks 6–10 of pack 1, with the share of those picks that were
voluntary in brackets. 

| Corpus | gen-3 | gen-1 | forge-full |
|---|---|---|---|
| field at T, T 2 | 9.6 % (34.2) | 10.8 % (45.4) | 11.5 % (50.4) |
| field at T, T 3 | 9.2 % (44.4) | 8.9 % (44.9) | 10.0 % (49.5) |
| field at argmax, T 2 | 12.3 % (56.7) | 9.5 % (35.8) | 10.8 % (44.8) |
| field at argmax, T 3 | 14.6 % (63.3) | 9.6 % (48.9) | 10.9 % (57.8) |

Take the two readings one at a time.

**Voluntary share (the bracketed number).** Both field-at-argmax candidates are above
their gen1 and forge-full references. Both field-at-T candidates are below their references. 
Training against a sampled field makes the agent *choose* to go off-lane less often 
than its ancestors did; training against argmax makes it choose to go off-lane more often.
This is somewhat counter-intuitive: a sampled field passes more good off-colour cards 
downstream, so it ought to tempt the learner to splash *more*. The measurement says the reverse.

**Off-lane rate.** Both field-at-argmax candidates go off-lane more often than their gen1 and 
forge-full references. Both field-at-T candidates go off-lane less often.

Picks 6–10 are tabled because they are the one window where both readings say
something. At the start of a pack the voluntary share saturates: the pack is fresh and
nearly always still holds something on-colour, so almost every off-lane pick is a
choice, for every checkpoint alike. At the end of a pack the off-lane rate saturates
the other way: the pack is down to dregs, so every checkpoint is off-lane close to
half the time. At both ends the pack decides rather than the agent, and the
checkpoints are indistinguishable.

In each of those windows the reading that has *not* saturated splits the same way it
does at picks 6–10. From the opening picks the argmax `T = 3` policy already goes
off-lane at nearly twice gen-1's rate. Deep in the pack, where both are usually
forced, it still turns down an available on-colour card more than twice as often as
gen-1. The split runs the whole length of the pack, so it is not an artefact of the
window we tabled.


#### Hypothesis 1 — lane starvation. Not what the policy does, but maybe what taught it.

The hypothesis: an argmax field takes the good cards in the learner's colours
consistently, so the learner keeps facing packs with nothing playable on-colour, takes
the off-colour card because it has to, and ends with a pool that never found a deep
pair.

At the yardstick the `T = 3` argmax policy is choosing its off-lane picks, not being
forced into them. It declines an available on-colour card in close to two thirds of
its off-lane picks, where gen-1 declines in about half, and the excess sits early in
the pack. Both policies decline far less often once the pack is down to dregs, but
even there this one declines more than twice as often as gen-1. Starvation predicts
the opposite on both counts: picks that are forced, and concentrated late once the
on-colour cards are gone.

The yardstick can only show what the deployed policy does, not what taught it, and
starvation is a claim about training. The yardstick field is identical for all four
candidates by construction, so it cannot show a difference in starvation at all. The fields
differed during training, and there starvation is entirely plausible.

- Under field at argmax the opponents play their best and take their colours
  correctly, while the learner samples at `T` and plays below its own best. Its
  in-training pools are poorer, and it sits in lane-starved positions far more often
  than it ever does at the yardstick.
- In a starved position the only available pick is off-colour. That is not a
  preference, it is the position.
- The update cannot tell the two apart. The seat receives one score for the whole
  draft, shared by all 45 picks, so when a draft finishes above the pod average every
  pick in it is reinforced together, forced ones included. Nothing in the gradient
  encodes "that pick was not a choice."

That is a mechanism for the behaviour measured above. A policy can acquire a general
taste for off-colour cards in positions where it never had a choice, then carry that
taste into positions where it does.

#### Hypothesis 2 — card power over lane fit. It is not taking bombs.

The hypothesis: it is taking bombs. Breaking colour is correct when the card is enough
better than the on-colour alternative, and a policy trained against a strong field
might sensibly learn to grab power and sort the colours out later.

That predicts a positive quality premium: when the agent breaks colour, the card it
takes should beat the best on-colour card it passed. Every card is scored by
`shrunk_score_play` from the sealed win-rate labels. That scale is independent of the
question, being built from real game outcomes rather than from anyone's pick
behaviour, and it covers 98 % of drafted card slots.

| Corpus | took best card in pack, g3 / g1 / forge | mean off-lane premium | share of off-lane picks beating the on-colour option |
|---|---|---|---|
| field at T, T 2 | 22.9 / 19.7 / 19.5 % | +0.024 / +0.013 / +0.011 | 63.0 / 56.3 / 55.0 % |
| field at T, T 3 | 21.7 / 20.2 / 19.8 % | +0.016 / +0.012 / +0.010 | 59.3 / 55.0 / 54.4 % |
| field at argmax, T 2 | 22.8 / 20.0 / 19.5 % | +0.022 / +0.013 / +0.009 | 61.8 / 58.0 / 55.7 % |
| field at argmax, T 3 | 28.5 / 21.7 / 21.2 % | +0.005 / +0.015 / +0.011 | 51.9 / 58.7 / 55.8 % |

The `T = 3` argmax policy breaks colour at a coin flip. Barely half its off-lane picks
beat the on-colour card they passed, and the median premium is essentially zero. It
makes voluntary off-lane picks about twice as often as gen-1 and gains nothing by
them. A second, unrelated quality scale agrees: ranking cards by pick rate when
available, estimated from the frozen reference seats alone, still leaves it at a coin
flip.

The other three gen-3 policies break colour more selectively than either reference.
Their off-lane picks beat the passed on-colour card about three times in five, at a
larger premium than gen-1 or Forge manage.

The first column says the failure is narrow. The `T = 3` argmax policy takes the
highest-win-rate card in the pack more often than any other agent in the sweep, its
own references included. Its card evaluation is not damaged. What it has lost is the
other half of a draft pick: how the card fits the pool it already holds.

#### Hypothesis 3 — a colour prior learned from Forge. Real, and not the family difference.

The hypothesis: Forge pilots green, black and white better than blue and red,
because blue and red lean on instants and sorceries and Forge plays those worst.
`deck_score` is fitted to Forge-piloted outcomes, so a policy trained on it should
acquire a taste for those three colours, and that taste should show when it breaks
lane.

The test compares, at each off-lane pick, the colour of the card taken against the
colour mix of the off-lane cards available in that pack at that moment. The
availability baseline carries the whole measurement. Off-lane is defined against the
seat's own eventual top-2, so a seat in black-green can never make a black off-lane
pick, and raw colour counts would only re-describe the lane distribution. Each pick
contributes weight 1 to both sides, and gold cards split their weight across their
colours.

Mean per off-lane pick of (green-black-white taken − green-black-white available),
with the learner picks each candidate trained on:

| Corpus | gen-3 | gen-1 | forge-full | learner picks trained |
|---|---|---|---|---|
| field at T, T 2 | +2.90 pp | +0.27 | −0.58 | ~110k |
| field at T, T 3 | +0.70 | −0.19 | −0.10 | ~30k |
| field at argmax, T 2 | +1.73 | +0.40 | +0.61 | ~50k |
| field at argmax, T 3 | +2.77 | +0.10 | −0.13 | ~195k |

Standard errors are 0.23–0.34 pp, so every gen-3 figure is many standard errors from
zero and no reference figure is more than two.

Every gen-3 candidate leans towards green, black and white when it breaks lane.
Neither reference leans at all: Forge and the agent that imitated it take whatever
the pack offers. The size of the lean tracks how long a candidate trained, not which
field it trained against — ordered by learner picks, the four run +0.70, +1.73,
+2.90, +2.77. Both families lean the same way. That is the opposite of the mana-base
result above, where the field sets the direction of the drift and training length
only sets its size.

The lean shows in the lanes as well. The most-trained candidate takes white in over
half its seats and red in under a third, where the gen-1 seats in the same pods sit
near 38 % and 45 %.

None of this explains why the two families diverge, because all four candidates do
it. The largest lean belongs to field at T, `T = 2` — the candidate with the
narrowest mana bases and the best yardstick score. Colour lean and colour discipline
move independently here, if anything in opposite directions.

The lean is correct play against this opponent. Forge wins more with green, black and
white, because blue and red lean on instants and sorceries and Forge plays those
worst. `deck_score` measures Forge-piloted outcomes, so a policy that learns which
colours win those games is doing exactly what it was asked. Against human opponents
the same preference would be miscalibrated, but human opponents are not what this
stack optimises for.

Where the candidates part company is what they do with the preference off-lane. Three
of them break colour towards these colours and earn a quality premium for it, by
Hypothesis 2's table. The `T = 3` argmax policy breaks colour towards them and earns
nothing: no premium, and a mana base that widens instead of converging. Its off-lane
picks are not arbitrary, then, but colour-directed. When it strays, it strays towards
white and green, rather than towards the card that was worth straying for.

### Field at T against field at argmax

The spec got this backwards (§ 8.1). Under field at T the frozen agents sample too, so
they sometimes pass a good card they should have kept. The learner then trains against
packs that a properly-playing field would never have handed it. The spec expected that
to transfer badly. It transferred best: field at T produced the strongest candidate on
the yardstick.

Each choice is better at a different thing. Field at argmax trains against a field
that resembles the yardstick's, so its live margin predicts the yardstick result well.
Field at T's margin runs well above what the yardstick later gives it, so a field-at-T
run's quality cannot be read off its own log. Field at T drafts better, though:
narrower mana bases, and more selective off-lane picks. That is the property worth
optimising, so field at T is the method to keep for gen-4.

### The reward and the run-control metric are different numbers

The trainer maximises one number and run control keys off a different one. The reward
is pod-relative leave-one-out: a seat's `deck_score` minus the mean of the other
scored seats in its pod, the learner's own other seats included. The anchor margin is
the learner's windowed mean minus one frozen label's, and best-checkpoint selection,
`--lr-decay-patience` and `--patience` all key off it. So the trainer maximises "beat
the pod" while selection asks "beat `gen3a`".

Over the 1269-round run of 2026-08-08 the candidates track each other closely and
still disagree on which round to keep.

| Criterion | Selects | correlation with the anchor margin | noise |
|---|---|---|---|
| anchor margin against `gen3a`, the incumbent | round 312 | — | 0.059 |
| field margin, all frozen labels pooled | round 653 | +0.91 | 0.056 |
| windowed `R`, the reward itself | round 640 | +0.77 | 0.037 |

Noise is the standard deviation of the round-to-round change.

Policy loss is not a third candidate, despite being the number the trainer descends.
`R` is standardised within each round before it becomes the advantage, which bounds
the loss by the spread of `logπ`. It therefore measures how spread out the policy is
rather than how good it is: it correlates −0.85 with entropy, and +0.34 with the
margin, which is the wrong sign for something being minimised.

Decision: keep the anchor margin. The field margin is better motivated, because its
baseline excludes the learner's own seats where the reward's includes them and
attenuates any uniform gain. But it agrees with the incumbent at +0.91, the incumbent
produced a promotable candidate, and no in-run metric can settle the 340-round
disagreement in any case. The argmax yardstick owns that decision. Revisit if a
yardstick run ranks a margin-selected checkpoint below one the field margin preferred.

Read either number as relative, not absolute; see *Live progress signal* above for
why the frozen field does not give a fixed baseline.

### Where this leaves gen-3

Promote the field-at-T `lr 1e-5`, `T = 2` run's round-105 snapshot. It has the best
mean margin of the four candidates, a median margin within 0.05 of the best, and the
tightest score distribution.

Carry two settings into gen-4. Field at T is the training field, for the reasons
above. The temperature is unsettled: `T = 3` is the only value that holds the
exploration band, but it ran for just 28 rounds under field at T, and the `T = 2` run
lost its band mid-run and still produced the best candidate. The band and the outcome
disagree, so the question is open.

Instrument two things that were missing this time. Both are cheap.

- Keep each run's rollouts. The trainer appends to the shared
  `output/draft/drafts.jsonl`; give every run its own `--output-path` and keep the
  file. Those records are the only view of what the learner was seeing while it
  trained, and they are what would decide between the two candidate causes of the
  colour drift. Losing them cost the analysis that would have closed this out.
- Log the wide-mana-base rate per round, and read it beside the anchor margin. It is
  the failure mode that separated the candidates, the margin nets it away, and
  best-checkpoint selection is blind to it.

One run settles the open question: `lr 1e-5`, `T = 3`, field at T, on the field-at-T
mix, at least 150 rounds, with `--patience` and `--lr-decay-patience` armed. It
matches the incumbent on field, mix and round count, and changes only the temperature.
