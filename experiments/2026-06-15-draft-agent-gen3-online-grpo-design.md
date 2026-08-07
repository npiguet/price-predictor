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

Gen-3 clears the objective gen-2 failed to move: every candidate taken to the
yardstick beats both the frozen gen-1 it was fine-tuned from and `forge-full`,
where gen-2 had tied Forge at 1.49. 

- **Base / config:** learner and anchor both warm-started from the same gen-1
  checkpoint; reward from the frozen sealed scorer with `--build-method greedy`;
  10 drafts/round, batch 32, clip 1.0, 200 warmup steps, 100-draft anchor window,
  random set, seed 42.
- **Reward trend:** the raw reward is a weak progress read — it rose in three of
  the four `lr 1e-5` runs and stayed flat (+0.29 → +0.28) in the fourth, as
  expected of a pod-relative leave-one-out quantity when ~3 of 8 seats are
  learners.
- **Follow-ups:** best-checkpoint selection, patience and LR annealing (spec 021
  phase 7), added in response to the peak-and-decline behaviour below.

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

| Family | lr | T | Rounds | Best margin (round) | Final margin | gen-3 | gen-1 | forge-full | gen3 − gen1 |
|---|---|---|---|---|---|---|---|---|---|
| field at T | 1e-6 | 2.0 | 89 | +0.255 (r87) | +0.255 | — | — | — | — |
| field at T | 1e-5 | 2.0 | 121 | +0.887 (r101) | +0.707 | 1.98 (2.08) | 1.25 (1.32) | 1.14 (1.21) | +0.73 |
| field at T | 1e-5 | 3.0 | 28 | +0.504 (r27) | +0.504 | 1.76 (1.87) | 1.51 (1.62) | 1.31 (1.39) | +0.25 |
| field at T | 1e-4 | 2.0 | 33 | +0.444 (r15) | +0.363 | — | — | — | — |
| field at argmax | 1e-5 | 2.0 | 89 | +0.374 (r47) | +0.081 | 1.79 (1.92) | 1.41 (1.52) | 1.23 (1.40) | +0.38 |
| field at argmax | 1e-5 | 3.0 | 199 | +0.439 (r179) | +0.114 | 1.73 (2.12) | 1.17 (1.31) | 1.02 (1.16) | +0.56 |

Each yardsticked row used its run's best-margin checkpoint, except the field-at-T
`T = 2` row, which predates best-checkpoint selection and used the nearest snapshot
(round 105).

Margins do not compare across the two families, which are different fields, and run
lengths are not controlled: the field-at-T `T = 3` run was stopped while its margin
was still climbing, so its yardstick figure is a lower bound. (A 7-round
field-at-argmax probe on the field-at-T mix was discarded.) The yardstick columns
compare only within a row — each is its own corpus over randomly drawn sets, and the
level moves with the draw, the same frozen gen-1 spanning a range across the four
corpora wider than several of the effects being measured. Within a corpus the agents
share pods, so the comparison is paired, and ranking on `forge-full` instead of
gen-1 gives the same order.

### Composition of the margin

Every label's raw windowed mean is printed beside the margin, so the learner's rise
and the anchor's fall separate. Measured from the first full window (r9) to each
run's best round:

| Family / run | learner r9 → best | anchor r9 → best | Δ learner | Δ anchor | anchor's share |
|---|---|---|---|---|---|
| field at T, lr 1e-5, T 2.0 | 1.58 → 2.13 | 1.57 → 1.24 | +0.55 | −0.33 | 37 % |
| field at T, lr 1e-5, T 3.0 | 1.38 → 1.67 | 1.34 → 1.17 | +0.30 | −0.17 | 36 % |
| field at T, lr 1e-4 | 1.61 → 1.86 | 1.50 → 1.42 | +0.25 | −0.08 | 24 % |
| field at T, lr 1e-6 | 1.65 → 1.75 | 1.62 → 1.49 | +0.10 | −0.13 | 57 % |
| field at argmax, lr 1e-5, T 2.0 | 1.53 → 1.92 | 1.81 → 1.54 | +0.39 | −0.27 | 41 % |
| field at argmax, lr 1e-5, T 3.0 | 1.47 → 2.00 | 1.89 → 1.56 | +0.54 | −0.33 | 38 % |

Across the four `lr 1e-5` runs the share is near-constant at about 60/40, and
sampling the field does not change it. The two runs departing from that share are
the two failed learning rates below.

### Exploration

| Family / run | T | ppl @r0 | ppl mean | ppl last 10 | off-arg @r0 | off-arg mean | off-arg last 10 | in band? |
|---|---|---|---|---|---|---|---|---|
| field at T, lr 1e-5 | 2.0 | 1.75 | 1.56 | 1.53 | 20.3 % | 17.0 % | 15.2 % | no — declines out |
| field at argmax, lr 1e-5 | 2.0 | 1.97 | 1.71 | 2.02 | 27.5 % | 19.9 % | 25.2 % | no — sags mid-run |
| field at T, lr 1e-5 | 3.0 | 2.67 | 2.29 | 2.06 | 34.7 % | 28.9 % | 25.2 % | yes |
| field at argmax, lr 1e-5 | 3.0 | 2.51 | 2.15 | 2.10 | 34.2 % | 27.5 % | 29.5 % | yes |
| field at T, lr 1e-6 | 2.0 | 1.77 | 1.81 | 1.72 | 21.4 % | 22.3 % | 20.5 % | no — flat |
| field at T, lr 1e-4 | 2.0 | 1.80 | 2.11 | 2.30 | 20.3 % | 29.3 % | 34.2 % | nominally, but thrashing (1.35–3.96) |

`T = 2` starts below the band and declines further, bottoming at 9.1 % off-argmax
in the field-at-T run; `T = 3` starts inside it, and the 199-round run was still
inside it at round 198. The entropy decay predicted above at fixed `T` is thus
present at `T = 2` and absent at `T = 3` over three hours. `T = 3` also ranks above
`T = 2` within the field-at-argmax pair, though the best agent overall came from
`T = 2` in the other family over more rounds.

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

`lr 1e-4` was unstable — per-round KL two orders of magnitude above the `lr 1e-5`
runs, with gradient norms and perplexity swinging in step — yet its margin reads
healthy in the run table, peaking and ending well above zero. The component means
show why that reading misleads: the learner's own score peaked near round 15 and
then returned to its starting level, while the anchor kept falling. After round 15
the margin was sustained by the field declining. That is a legitimate result
against that field but not one the yardstick's field rewards, and the run was not
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

Every gen-3 drafts more creatures than the gen-1 it came from, with slightly fewer
rares and uncommons — on-colour commons in place of higher-rarity cards — and an
unchanged curve. This is the objective as specified: Forge pilots creatures better
than spells, and the scorer was fitted to Forge-piloted outcomes.

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
comfortably outscore the references' wide decks, which is what going wide only when
the pool supports it looks like. The field-at-argmax `T = 3` candidate inverts that
as well: its wide decks score below those of the gen-1 seats beside it. It is going
wide when the pool does not support it.

Why that happens is answerable from the corpus rather than by conjecture, since the
record preserves every pick and everything passed at it — see *What the pick record
says* below.

### Mean against median

The candidates differ in the shape of their score distribution, not only its level.
A median above the mean means a left tail: a minority of bad decks pulling the
average down while the median stays in the bulk. Splitting each candidate's gap
into the part four- and five-colour mana bases account for — "drag", how much the
mean would rise if those decks scored like the same policy's three-colour decks —
and everything else:

| Candidate | med − mean | wide-deck drag | residual | margin over gen-1, mean / median |
|---|---|---|---|---|
| field at T, T 2 | +0.10 | +0.015 | +0.087 | +0.73 / +0.76 |
| field at T, T 3 | +0.11 | +0.013 | +0.100 | +0.25 / +0.25 |
| field at argmax, T 2 | +0.13 | +0.025 | +0.105 | +0.38 / +0.40 |
| field at argmax, T 3 | +0.38 | +0.283 | +0.101 | +0.56 / +0.81 |
| *gen-1 and forge-full, all corpora* | +0.07 … +0.17 | +0.020 … +0.054 | +0.034 … +0.136 | — |

Two separate things live in that gap. The residual is flat across all four
candidates, and the references carry the same skew, so it belongs to `deck_score`
itself: a sealed deck can be far *worse* than average — no playables, broken curve,
colour-screwed — more easily than it can be far better. Nothing about the training
explains it.

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

Two factors multiply, and the `T = 3` argmax candidate is worse on both — 2.4× the
tail size of the field-at-T candidates, and roughly four times the depth, its
five-colour decks averaging −1.88 where every other candidate's average around
+1.0. Its own two-colour decks are the best two-colour play in the sweep, which is
why it leads on median margin while placing second on mean.

Removing the tail entirely — scoring each candidate's wide decks like its own
three-colour decks — would put the field-at-T `T = 2` candidate at 2.00 and the
field-at-argmax `T = 3` one at 2.01, or +0.75 and +0.84 against their respective
anchors. On that basis the second would be ahead: its entire deficit is the 13 % of
drafts where it loses the plot. The counterfactual is hypothetical, since a
genuinely incoherent pool cannot be built into a good deck by any builder, so the
drag is an upper bound on what is recoverable.

That the sampled field is the one producing the tighter distribution partly inverts
the expectation in § 8.1 of the spec: its weakness in the wheeling dimension acts as
a regulariser and not only as a handicap.

### What the pick record says

The corpus stores every booster in pick order, so each seat's 45 picks — and every
card it passed at each of them — reconstruct exactly. The question of *why* the
colours drift is therefore answerable from the record rather than by conjecture.
(Data extracted with `scripts/analyze_draft_lanes.py` and
`scripts/analyze_pick_quality.py`.)

Two readings per pick, both anchored on the seat's eventual top-2 colours: whether
the pick was **off-lane**, and if so whether an on-colour card was **still in the
pack** — whether the agent was forced or chose. Anchoring on the top two rather than
on the finished deck's colours matters, since a five-colour deck would otherwise
score as perfectly on-colour against itself.

Off-lane rate at picks 6–10 of pack 1, with the share of those picks that were
voluntary in brackets:

| Corpus | gen-3 | gen-1 | forge-full |
|---|---|---|---|
| field at T, T 2 | 9.6 % (34.2) | 10.8 % (45.4) | 11.5 % (50.4) |
| field at T, T 3 | 9.2 % (44.4) | 8.9 % (44.9) | 10.0 % (49.5) |
| field at argmax, T 2 | 12.3 % (56.7) | 9.5 % (35.8) | 10.8 % (44.8) |
| field at argmax, T 3 | 14.6 % (63.3) | 9.6 % (48.9) | 10.9 % (57.8) |

On the voluntary share the ordering holds in all four corpora: field-at-T gen-3
below both references, field-at-argmax gen-3 above both. On the off-lane rate it
holds in three, the exception being the field-at-T `T = 3` policy, which is level
with gen-1 — the least-trained candidate of the four at ~30k learner picks, so it has
barely moved off its ancestor at all. The pattern repeats in two further windows: at
picks 1–5 for the off-lane rate (the `T = 3` argmax policy opens at 11.5 % against
gen-1's 6.4 %) and at picks 11–15 for the voluntary share (14.5 % against 6.0 %).

The two cells where it does not appear are the ones where the game removes the
choice, and they read as a control. At picks 11–15 all twelve agent-rows are
off-lane 39–44 % of the time, because the pack is down to dregs; at picks 1–5 all of
them are 86–97 % voluntary, because a fresh pack nearly always still holds something
on-colour. The agents separate exactly where a decision exists and nowhere
else.

The `forge-full` column earns its place here, because **gen-1 was trained by
imitating `forge-full`**. Forge is the ancestral behaviour, and gen-1 had already
tightened slightly on its teacher. Field at T continues that line; field at argmax
reverses it and overshoots Forge itself.

Worth recording what the obvious guess would have been: a sampled field passes more
good off-colour cards downstream, so it ought to tempt the learner to splash *more*.
The measurement says the reverse.

#### Hypothesis 1 — lane starvation. Falsified as behaviour, revived as cause.

The first explanation we reached for. An argmax field takes the good cards in the
learner's colours correctly and consistently, so the learner keeps facing packs with
nothing playable on-colour, takes the off-colour card because it has to, and ends
with a pool that never found a deep pair.

As a description of the *trained* policy it is wrong. It predicts off-lane picks
that are **forced**, and concentrated **late** in a pack once the on-colour cards are
gone. The excess is voluntary — the `T = 3` argmax policy declines an available
on-colour card in 63 % of its off-lane picks, against gen-1's 49 % — and it sits in
the *early* picks, fading to nothing by picks 11–15. Even deep in a pack it declines
an available on-colour card more than twice as often as gen-1 does. At the yardstick
it is not being starved.

That test was mis-scoped, though, and the distinction matters: what the deployed
policy *does* is a different question from what *taught* it. The yardstick field is
identical for all four candidates by construction, so it cannot show a difference in
starvation at all. The place the fields differed was training — and there starvation
is entirely plausible.

- Under field at argmax the opponents play their best and take their colours
  correctly, while the learner samples at `T` and plays below its own best. Its
  in-training pools are poorer, and it sits in lane-starved positions far more often
  than it ever does at the yardstick.
- In a starved position the only available pick *is* off-colour. That is not a
  preference, it is the position.
- The update cannot tell the two apart. The seat receives **one score for the whole
  draft**, shared by all 45 picks, so when a draft finishes above the pod average
  every pick in it is reinforced together — forced ones included. Nothing in the
  gradient encodes "that pick was not a choice."

A policy can therefore acquire a general taste for off-colour cards from positions
where it never had one, which is exactly the deployed behaviour measured above.
Starvation is dead as a description of the agent and live as a candidate cause of
it.

#### Hypothesis 2 — card power over lane fit. Falsified.

The natural replacement: it is taking bombs. Breaking colour is correct when the
card is enough better than the on-colour alternative, and a policy trained against a
strong field might sensibly learn to grab power and sort the colours out later.

That predicts a **positive quality premium** — when the agent breaks colour, the card
it takes should beat the best on-colour card it passed. Scoring every card by
`shrunk_score_play` from the sealed win-rate labels (an independent measure, built
from real game outcomes rather than from anyone's pick behaviour, covering 98 % of
drafted card slots):

| Corpus | took best card in pack, g3 / g1 / forge | mean off-lane premium | share of off-lane picks beating the on-colour option |
|---|---|---|---|
| field at T, T 2 | 22.9 / 19.7 / 19.5 % | +0.024 / +0.013 / +0.011 | 63.0 / 56.3 / 55.0 % |
| field at T, T 3 | 21.7 / 20.2 / 19.8 % | +0.016 / +0.012 / +0.010 | 59.3 / 55.0 / 54.4 % |
| field at argmax, T 2 | 22.8 / 20.0 / 19.5 % | +0.022 / +0.013 / +0.009 | 61.8 / 58.0 / 55.7 % |
| field at argmax, T 3 | 28.5 / 21.7 / 21.2 % | +0.005 / +0.015 / +0.011 | 51.9 / 58.7 / 55.8 % |

The `T = 3` argmax policy's off-lane picks carry no premium at all: 51.9 % of them
beat the on-colour card they passed, which is close to a coin flip, and the median
premium is +0.004. It is not taking bombs. Combining the two rates, it makes
voluntary off-lane picks on about 9 % of its coloured picks against gen-1's 5 % —
twice as many, for nothing. Repeating the measurement on an unrelated quality scale
— pick rate when available, estimated from the frozen reference seats alone — puts
the same figure at 50.9 %.

The three healthy gen-3 policies do the opposite, and beat both references: they
break colour *more* selectively than gen-1 or Forge, at +0.016 to +0.024 premium and
59–63 % above zero.

#### What survives

The same table that kills the second hypothesis points at a narrower reading. The
`T = 3` argmax policy takes the highest-win-rate card in the pack **more** often than
anything else here — 28.5 % against 19.5–21.7 % for every reference. Its card
evaluation is not damaged; by that measure it is the best of the twelve. What it has
lost is the other half of a draft pick: **how the card fits what is already in the
pool**. It takes the best card when the best card happens to be on-colour and wanders
when it is not — a coin-flip premium on its off-lane picks, a mana base that widens
through the draft instead of converging, and a pool the builder can only assemble
into a five-colour pile.

Two mechanisms could produce that, and they are different claims:

| Mechanism | What it says | What it predicts in the *training* rollouts |
|---|---|---|
| Starvation generalises | Forced off-colour picks are reinforced along with everything else in an above-average draft, and the taste transfers to unstarved positions | The learner's *forced* off-lane rate is far higher under field at argmax than under field at T |
| The fit signal is too weak | Fit is learnable only if the accumulating pool predicts the final score; against a strong field the learner's pool is thin and fragmented, so it predicts poorly and the gradient falls back on the component that always predicts — card power | No forced-rate difference; instead a weaker correlation between pool coherence and final `deck_score` under the argmax field |

Either would also explain why the field-at-T policies came out *more* fit-aware than
the references rather than merely no worse. Separating them takes one pass of
`scripts/analyze_draft_lanes.py` over the training rollouts — which no longer exist,
because the online trainer appends every run to the shared
`output/draft/drafts.jsonl` and that file is gone. Keeping them is a gen-4 action.

A third factor is not a cause but an amplifier, and it cuts the other way. The
learner is ~30 % of seats in training and 50 % at the yardstick, where four identical
copies share a pod and contest the same colours as hard as it is possible to contest
them. A policy that learned "move when your lane is cut" moves far more often there
than it ever did in training, so the yardstick probably overstates this failure
relative to a realistic pod. It applies equally to all four candidates, so it does
not explain the family difference.

None of it was visible to the selection machinery. `best_*.pt` is chosen on the
anchor margin, a windowed **mean**, which nets the wide-deck losses against the gains
from that policy's excellent two-colour play — and at a 13 % tail the net was still
positive. The checkpoint we yardsticked is round 179 of 199, so if colour discipline
decays while the margin climbs, margin-based selection actively prefers the more
degraded policy.

One further check needs no retraining at all: hide the accumulated pool from the
state the model sees, and measure how much each checkpoint's pick changes. If
fit-awareness is what differs, the field-at-argmax `T = 3` policy's picks should move
least.

Two caveats stand over all of this. The mix changed together with the temperature
scope, so a stronger field and a more deterministic field cannot be separated by
these runs. And the greedy builder sits inside the measurement — a mana-aware builder
might rescue some of these pools, which would narrow the gap without any change in
drafting behaviour.

### Field at T against field at argmax

The spec's argument (§ 8.1) is that a sampled frozen agent passes downstream cards a
properly-playing agent would have kept, weakening the training field in the
dimension the yardstick tests; the predicted consequence is worse transfer. The runs
do not show that. Field at T produced the best candidate on the yardstick's own
metric, and the distribution-shape evidence above suggests the weakened field is
doing useful work rather than only costing accuracy.

The two choices trade off differently on measurement and on the policy they train:

| | Field at T | Field at argmax |
|---|---|---|
| Best margin vs yardstick margin | overshoots | tracks closely |
| Training field | weak in the wheeling dimension | matches the evaluation field |
| Wide mana bases (≥ 4 colours) | 5.5–6.4 % | 8.3–13.3 % |
| Off-lane picks | more selective than the references | less selective; no quality premium at `T = 3` |
| Score distribution | tight; no catastrophic tail | left tail, severe at `T = 3` |

**Field at T is the training method to keep for gen-4.** It wins on the mean, which
is both the yardstick metric and the quantity the GRPO reward optimises; it is
within 0.05 of the leader on median margin; and its output carries no catastrophic
tail, so the gain is uniform across pools rather than an average over "usually
excellent, occasionally broken". The wide-deck rate is lower at both temperatures.

The cost is accepted rather than solved: under field at T the live anchor margin
overshoots the yardstick, so a run cannot be promoted on its log. Promotion stays on
the argmax yardstick, as it already does.

### Corrections to the design above

- Generation and training are not an even split: generation is 81–86 % of each round
  (44–58 s against 8–12 s at 10 drafts/round). Regenerating every round remains
  affordable, but speedup work belongs on the Forge side, and `--drafts-per-round`
  is close to free in training time.
- The stopping trigger is not a plateau. These runs peak and decline, so the trigger
  is a decline from a tracked best — what best-checkpoint selection and LR annealing
  now handle.
- The proposed sweep `T ∈ {1.0, 1.5, 2.0, 2.5}` is aimed too low: the target band is
  not reached below `T = 3` on this policy.

### Where this leaves gen-3

The promotable candidate is the field-at-T `lr 1e-5`, `T = 2` run's round-105
snapshot: the best mean margin of the four, a median margin within 0.05 of the best,
and the tightest score distribution.

Two settings to carry into gen-4:

- **Field at T** as the training field, per the section above.
- **`T = 3`** is the temperature that holds the exploration band, but it was only
  run for 28 rounds under field at T. The field-at-T `T = 2` run lost its
  exploration band mid-run and still produced the best candidate, so the band and
  the outcome disagree here and the question is open.

And two things to instrument, both cheap and both missing this time:

- **Keep each run's rollouts.** The trainer appends to the shared
  `output/draft/drafts.jsonl`; give every run its own `--output-path` and retain the
  file. Those records are the only view of what the learner was actually seeing while
  it trained, and they are what would decide between the two candidate causes above.
  Losing them cost the analysis that would otherwise have closed this out.
- **Log the wide-mana-base rate per round** and read it beside the anchor margin.
  It is the failure mode that separated the candidates, the margin nets it away, and
  best-checkpoint selection is blind to it.

The run that settles it: `lr 1e-5`, `T = 3`, field at T, on the field-at-T mix, at
least 150 rounds, with `--patience` and `--lr-decay-patience` armed. That matches the
incumbent on field, mix and round count and changes only the temperature.
