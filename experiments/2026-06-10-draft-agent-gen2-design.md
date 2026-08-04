# Draft agent (gen-2) — RL self-play design rationale

## Background

This document is the design rationale for **generation 2** of the draft agent:
the RL self-play stage that the gen-1 work was explicitly built to enable.

**Motivation.** The 2026-06-10 live-play analysis found gen-1 ≈ Forge on
scorer-judged deck quality (deck_score 1.52 vs 1.48). That is the designed
ceiling of imitation: cross-entropy can only copy Forge, so it cannot exceed it.
Generation 2's whole purpose is to push past that ceiling by **using the critic
to steer picks** — the learning half that gen-1 deliberately left unbuilt.

## What gen-2 adds

1. An **on-policy actor-critic trainer** that turns the critic's advantage into a
   policy-gradient update, KL-anchored to the frozen gen-1 policy.
2. The **self-play regeneration loop**: freeze → roll out → update → repeat,
   across generations, with coverage and forgetting controls.

Nothing in the architecture changes; gen-2 is a new *training procedure* over the
same model, plus orchestration.

## The bootstrapping payoff: warm-start both heads

The actor is initialised from the gen-1 policy and the critic from the gen-1
critic — co-trained on the same trunk, so the actor↔critic loop opens already
pre-aligned, from a competent policy and a critic that already explains ~73 % of
reward variance rather than from noise. That warm start is what makes RL on this
45-step, sparse-terminal POMDP tractable at all.

## On-policy actor-critic — the core, and how it reuses live play

The core is an **on-policy actor-critic**, rollouts produced by the current
policy through the existing live-play command.

The decisive implementation choice: **during a live draft the policy only has to
output a pick — nothing else.** The RL update needs two extra numbers at each
pick: the policy's log-probability of the action it took (`log π`, how likely the
policy was to make that pick) and the critic's value estimate for that state
(`V`). The naive way to get them is to have the Java draft loop ask the Python
policy for those numbers at every pick, over the same process-to-process channel
(inter-process communication, "IPC") that already carries the pick request — an
extra round-trip per pick. We avoid that entirely: a recorded corpus plus the
*frozen snapshot that generated it* are enough to recompute, in a single batched
GPU pass after the draft is over:

- the behaviour log-prob `log π_snap(aₜ | sₜ)` at every pick the agent made —
  just the log of the softmax output for the card the agent actually picked,
- the critic value `V(sₜ)` at every state,
- hence the GAE advantages and returns,

The reward itself is not recomputed at all — it is read directly from the record
(the pod-relative leave-one-out `deck_score`). So the live-play data generator is
reused **unchanged**; gen-2 adds only a trainer that consumes its output.

This pins the generation cycle to:

```
freeze policy snapshot πₖ
  → generate a self-play corpus with πₖ (live play, sample mode)
  → recompute logπ, V, GAE advantages from the corpus + πₖ
  → policy + critic gradient update → πₖ₊₁
repeat
```

On-policy correctness is automatic: the actions were drawn from `πₖ` and the
log-probs are recomputed from `πₖ`, so the gradient is genuinely on-policy as
long as each corpus is trained against the snapshot that produced it.

**Rejected — live per-pick value/log-prob IPC.** Unnecessary given the post-hoc
recompute, and it would couple the Python trainer to the Java draft loop for no
benefit. (A critic-only greedy actor — argmax over the critic each pick — is also
rejected: it hunts the critic's largest overestimate.)

## Algorithm: REINFORCE + GAE baseline

**REINFORCE + a GAE baseline + KL anchor + entropy bonus.** The strong
warm-start plus the KL anchor make destructive updates unlikely — exactly the
regime where plain REINFORCE+baseline is adequate and simplest, and its
policy-gradient + entropy + baseline machinery already exists in the repo.

## Credit assignment: GAE(λ→1)

**A learned value function + GAE(λ) with λ near 1** — a dense, cheap
per-pick advantage `≈ V(sₜ₊₁) − V(sₜ)` at every one of the 45 picks, with λ→1
leaning on the unbiased (higher-variance) Monte-Carlo return. λ is a
variance/bias dial to tune.

## The critic in gen-2: on-policy V^π

Training the critic on the policy's **own** rollouts yields `V^π`
by construction — the continuation is the policy's — removing the
continuation-mix bias the offline critic carried.

The critic estimates "from this half-built pool, how good will the final deck
be?" But the value of a partial pool depends on *who finishes the picks* — the
same pool is worth more if a strong drafter completes it, less if a weak one
does. So a state's value is only well-defined relative to a continuation policy;
`V^π` is its value assuming the **current** policy π drafts the rest. The gen-1
critic was trained offline on the Forge corpus, whose drafts were finished by a
*mixture* of agents (`forge-full`/`forge-r30`/`forge-r100`), so it learned the
value under that mixture, not under π — the "continuation-mix bias." In gen-2 the
training states come from π's own self-play rollouts, so every recorded
continuation *is* π's; regressing the critic on those outcomes gives `V^π`
directly, with no mixture to average over. This matters because the critic is the
baseline GAE subtracts to form the advantage: a biased critic biases the
advantages and nudges the policy gradient off-target.

Same context-token head and MC/GAE(λ→1) target; only the training states change
(now on-policy).

## Reward and objective: pod-relative leave-one-out mean

**Pod-relative leave-one-out reward, mean over opponents** — the form
`mean(P(beat i))` *is* the gradient of expected match wins, which is the
tournament objective.

The reward source stays the frozen scorer over picker-built decks; the failure
modes that raises — and why neither needs active defence — are discussed below.

## KL anchor to gen-1, and entropy for exploration

**A KL penalty `KL(π ‖ π_gen1)` against the frozen gen-1 policy.** It
(a) keeps the policy on the manifold of reasonable drafts and (b) stabilises
updates (a trust region against destructive steps).
Its coefficient follows a schedule — heavier early, relaxed as the critic proves
trustworthy. An **entropy bonus** adds local exploration on top of the
temperature-sampled rollouts.

## Exploration: temperature-sampled rollouts

Rollouts use `--pick-mode sample` at a temperature `T_explore`
(seeded for reproducibility); behaviour log-probs are recomputed at the same
temperature, so the policy gradient is exact. Argmax rollouts would give the
policy zero stochastic exploration (only Forge's booster/seat randomness varies
the data) — too little signal for the gradient to improve picks. Temperature and
entropy coefficient decay as the policy sharpens.

## Off-policy data: the on-policy constraint on combining generations

A policy gradient is **on-policy by definition** — a REINFORCE/PPO update over a
*previous* generation's rollouts optimises the wrong objective unless
importance-corrected. This constrains how generations of data may be combined:

- The **policy gradient uses only current-generation, on-policy rollouts.** PPO's
  importance ratios permit limited reuse *within* a generation (several epochs
  over one batch), but not across generations.
- **Old generations are retained for three off-policy-safe roles**: (a) critic
  coverage of incoherent / low-end states (MC value targets are valid off-policy,
  though a clean `V^π` is dominated by the on-policy rollouts), (b) a steady
  **minority of dumb/random-bot pods** every generation so that coverage does not
  evaporate as the policy improves, and (c) the **cross-generation yardstick**
  below.

This is the subtlety the spec must get right: "more data is better" silently
breaks the gradient if old generations are folded into it.

## Self-play regeneration loop and the cross-generation yardstick

Each generation regenerates data from a mix of the **current
policy + Forge + laddered/random bots**, trains `πₖ₊₁`, and iterates. The mix
preserves opponent diversity and the late-pick "opponents are weak, take the good
card" signal; the random-bot minority preserves low-end critic coverage.

Because the scorer is **frozen**, `deck_score` is on a fixed absolute scale, so
"is gen-2 better than gen-1" is answered by comparing mean `deck_score` — with
one condition on *how* the comparison is run. The trap is comparing each
generation's mean over its *own training corpus*: those corpora use different
mixes (gen-1 drafts against a gen-1-strength field, gen-2 against a gen-2-strength
field), and since stronger tablemates contest cards harder and depress the pool a
drafter can assemble, that cross-corpus contrast confounds drafting skill with
opponent strength. That offset is a fixed difference between the two populations,
so more pods do not remove it.

The fix is a single **evaluation run with one fixed agent mix** holding every
generation being compared, each seated into pods by drawing the other seats at
random from that mix. Then every agent faces the *same* distribution of opponents
(conditioning on which agent sits in a seat leaves the other seven i.i.d. from the
mix), so the opponent-strength effect enters each agent's mean equally and cancels
in the difference. This is ordinary randomization: it balances the confounder
across the agents being compared, so the contrast of marginal means is unbiased —
no matched pods, no reference battery, no opponent-strength model needed. Over a
large pod count the per-pod mix differences wash out and the per-agent mean
`deck_score` is directly comparable.

## Failure modes considered, and why neither needs active correction

Two concerns were weighed and judged not to warrant a dedicated defence.

**Reward-hacking the frozen scorer.** The picker is already trained by REINFORCE
directly against this same frozen scorer, over the *more* flexible 90-card sealed
pool, and produced good decks rather than degenerate scorer-exploiting ones. The
draft policy's reachable decks are a subset of what that picker can build from
real drafted pools, so it has less room to find an exploit, not more.

**Composition drift.** Self-play does shift the deck distribution generation over
generation (e.g. more creatures, heavier white), but that is the optimiser
correctly finding what Forge piloting wins with. Forge-piloted win rate *is* the
objective, so a drift toward decks Forge plays better is the target working as
intended, not a pathology to correct.

## Tooling / implementation surface (the part that does not exist yet)

gen-2 is gated on new code, not just a spec:

- **An RL trainer** that loads a prior-gen checkpoint as actor+critic, reads one
  or more self-play corpora, recomputes log-probs/values from the generating
  snapshot, computes GAE advantages, and applies the KL-anchored
  policy-gradient + value + entropy update.
- **Multi-corpus input** for the trainer, at least for the off-policy-safe
  critic/coverage roles.
- The **generation-loop orchestration** (freeze → generate → train → promote),
  which can start as a script around the existing commands.

Everything upstream — the gen-1 artefact, live rollout, reward, builder, state
reconstruction, analysis — already exists; the actor-critic update and the loop
are the build.

## Open questions and tuning for gen-2

- **λ for GAE**, KL-coefficient schedule, entropy schedule, exploration
  temperature schedule — all to tune.
- **How many generations**, and the stopping/promotion rule against the frozen
  yardstick.
- **Whether the post-hoc recompute is sufficient** or any per-pick live value is
  ever needed (current belief: recompute suffices).

## Results — gen-2 RL runs

Three offline RL runs were attempted from the gen-1 checkpoint. None produced a
clearly promotable gen-2; together they pin the failure to the **offline
reuse + decaying trust-region leash** structure rather than to data or critic
quality. (All used `lr 3e-5`, GAE `λ 0.95`, and the same self-play corpus
generated by gen-1 in sample mode.)

**The advantage signal was healthy in every run.** The exact GAE advantage std
was ~0.27 each epoch and the critic's value MSE ~0.27–0.31 (≈70–73 % of reward
variance explained). The data and the critic were fine; the **policy
optimisation** was the failure.

**Run A — `entropy_coef 0.1`, `value_weight 1.0` — near-noop.** With value_weight
1.0 the held-out objective was critic-dominated (~0.22), so the best checkpoint
was selected very early (~step 2790) and the policy barely moved (KL ≈ 0.25). The
`entropy_coef` of 0.1 actively *flattened* the policy (val entropy rose 0.39 →
0.65). The coefficient-decay schedule never armed (val improved only a few times,
non-consecutively).

**Run B — `entropy_coef 0.01`, `value_weight 0.1` — catastrophic collapse.**
Lowering value_weight to 0.1 made the held-out objective **policy-dominated and
noisy**, which fed the val-keyed `_CoefSchedule`: it read a "non-improving" eval
on nearly every step and decayed the coefficients on almost all of them, driving
`kl_coef` from 0.1 to ~3e-11. With the KL leash gone, the unbounded surrogate ran
away: `val_loss → −114`, `KL → 5–9`, the total loss swinging by ±hundreds. The
"best" (most-negative-val_loss) checkpoint was the **most-collapsed** model —
selection was anti-correlated with quality.

**Run C — fixed `kl_coef = 0.1` (decay disabled) — a *slow* collapse, not a
bounded equilibrium.** Pinning the KL coefficient looked, early on, like it had
restored the trust region: Run C's first ~5000 steps were **byte-identical** to
Run B (same seed/data, and Run B's `kl_coef` had not yet decayed much that early),
after which Run B's KL exploded while Run C's KL stayed pinned ~0.5 (band 0.4–0.7)
and `val_loss` drifted only gently negative (~−0.2 to −0.6 through ~step 20k). At
that point it was tempting to call it a regularised equilibrium. **Run to a full
epoch (55,875 steps) it was not one.** `val_loss` kept descending without bound —
−0.6 (step 17k) → −1.0 (28k) → −1.4 (36k) → −2.3 (47k) → −2.9 (55k), still
dropping into epoch 1 — driven entirely by the **policy term** going from ~−0.4 to
~−3.0, with per-batch totals swinging ±10–15 and entropy sinking 0.33 → ~0.10. It
is the same reward-hacking collapse as Run B, merely **slowed** by the fixed leash.

The instructive part is *why a pinned KL ≈ 0.5 fails to stop it*:

- **The penalty became negligible.** `kl_coef · KL = 0.1 × 0.5 = 0.05`, against a
  policy term of ~−3.0 — the leash weighs essentially nothing in the loss.
- **Average KL does not bound the surrogate.** KL is a *mean over picks*; the
  unbounded `−A·logπ` runs away on the **tail** (a minority of negative-advantage
  picks crushed toward `π→0`, `logπ→−∞`), which drives the loss arbitrarily
  negative while the *average* divergence stays ~0.5. KL can look calm while the
  loss is hacked on a subset.

Throughout, the value head stayed healthy (~0.27–0.31) and the exact GAE advantage
std held at 0.27 — again confirming the **data and critic were fine; the policy
optimisation was the failure**. (Best-by-`val_loss` here selects the *most*
collapsed checkpoint — step ~55k — so nothing in this run is salvageable.)

**Diagnosis.** The `_CoefSchedule` (decaying *both* coefficients on a val-loss
signal that, at low value_weight, is itself the noisy unbounded policy surrogate)
was the proximate cause of Run B's blowup — and a KL trust region must never decay
to zero. But Run C shows that even a **fixed, non-zero** KL is not the cure: a soft
KL penalty bounds the *average* divergence, not the per-pick surrogate, so the loss
still collapses on the tail, just more slowly. The real culprit is **staleness from
offline reuse**: 55,875 gradient steps over a single frozen corpus is the
maximal-staleness regime, and no leash setting contains the unbounded surrogate
across that many off-policy steps. That is what motivates the gen-3 move to an
**online, leash-free** loop — regenerate the corpus every ~56 steps so the
advantages are always fresh and there is no stale target to hack
([`2026-06-15-draft-agent-gen3-online-grpo-design.md`](2026-06-15-draft-agent-gen3-online-grpo-design.md)).

**Yardstick outcome — a confirmed no-op.** The Run C best checkpoint was evaluated
on the cross-generation yardstick: a 100-draft `--pick-mode argmax` corpus with
gen2 and `forge-full` **co-seated** in the same pods (~50/50 of the 800 seats), so
the opponent-strength confound cancels. Mean `deck_score` is **identical** —
gen2 1.49 (median 1.63, n=426) vs forge-full 1.49 (median 1.62, n=374); at n≈400
the means are precise, so this is a genuine null, not a too-close-to-call. It also
matches the earlier live-play band (gen1 ≈ 1.52, forge ≈ 1.48): **Forge, gen1, and
gen2 are indistinguishable on deck quality.** Deck composition is near-identical
too (creatures 16.7 vs 16.7, curve 3.14 vs 3.13, rarity/pips within noise).

A second, independent yardstick corpus on the most-collapsed checkpoint
(`074224`, the one whose training loss reached ~−3 and entropy ~0.10) gives gen2
1.57 (median 1.66, n=411) vs forge-full 1.61 (median 1.72, n=389) — a −0.04 mean /
−0.06 median gap, on the order of one standard error of the difference, i.e. at
most a sliver of degradation. (The absolute level differs from the first corpus —
forge 1.49 vs 1.61 — purely from cross-corpus set/opponent variation; only the
within-corpus gap is meaningful.) The decks stay structurally sane (curve 3.13,
creatures 17.0 vs 16.8, mild colour drift toward R/G). So across both corpora the
collapsed gen2 lands at **tie-to-−1σ vs Forge**: even *maximal* loss-collapse
leaves argmax deck quality essentially at the Forge level — the strongest evidence
that the collapse was cosmetic to the objective, with at most a sliver of the
tail-hacking leaking into a few flipped (slightly worse) top picks.

Notably the training-loss collapse did **not** wreck the argmax decks: the
surrogate was hacked in probability-space on the *tail* (crushing a minority of
negative-advantage picks toward `π→0`), but argmax only reads the *top* pick — which
the sticky gen-1 imitation prior mostly preserves — and `deck_score` is coarse and
build-filtered, so the damage never reached the built deck. The collapse was real
but cosmetic to the objective. The bottom line stands: offline RL here is
**best-case a no-op, worst-case a hack**, because the surrogate it optimises is
decoupled from `deck_score` — which is exactly the dead end gen-3's online loop is
meant to escape.

## Glossary

- **POMDP** — partially observable Markov decision process; here a draft, where
  the agent acts on each pack it sees but never observes hidden state (opponents'
  pools, future packs) and the reward (deck quality) arrives only at the end.
- **π / πₖ** — π is the policy (the agent's pick function); the subscript `k`
  indexes the self-play round, so `π₀` is the frozen gen-1 starting policy and each
  round produces the next `πₖ₊₁`. One round *is* a full generate-then-train cycle:
  freeze `πₖ`, re-run the live-play generator with it (sample mode) to produce a
  fresh corpus, then train on that corpus to get `πₖ₊₁`. The corpus must be
  regenerated every round — once the update moves the policy, the previous round's
  data is drawn from a policy that no longer exists and is stale (the on-policy
  requirement) — so it is *not* produced automatically inside one trainer run; the
  generate→train loop is an outer orchestration you drive.
- **IPC (inter-process communication)** — passing data between two separate
  running processes; here, the Java draft loop and the Python policy exchanging
  pick requests/responses over a pipe. The "per-pick IPC" gen-2 rejects would mean
  one such round-trip at every pick.
- **On-policy** — the gradient is estimated from data generated by the *current*
  policy; required by REINFORCE/PPO (unlike imitation/MC-critic, which tolerate
  off-policy data).
- **Actor-critic** — the policy (actor) is updated by a policy gradient weighted
  by an advantage estimated from the value function (critic).
- **Advantage / GAE(λ)** — the *advantage* of a pick is how much better it turned
  out than the critic expected for that state ("did this pick beat expectations?");
  a positive advantage is the signal to do more of that pick. Estimating it means
  estimating how good the pick ultimately was, and there are two extremes. Use the
  *actual* final deck score (**Monte-Carlo**): correct on average, but very noisy,
  since one pick's share of the credit is buried in all 45 picks plus the matchup
  luck. Or use just the critic's value of the *next* state (**one-step TD**): much
  steadier, but only as trustworthy as the imperfect critic. **GAE(λ)** —
  generalised advantage estimation — blends the whole spectrum between them with a
  single knob λ ∈ [0,1]: λ=0 is pure TD (low variance, biased), λ=1 is pure
  Monte-Carlo (unbiased, high variance), and intermediate λ trades the two off.
- **PPO** — proximal policy optimisation; a clipped surrogate objective that
  permits several gradient epochs per batch without destructive updates.
- **REINFORCE** — the plain policy-gradient estimator `∇ log π(a|s) · advantage`.
- **KL anchor** — a penalty `KL(π ‖ π_ref)` keeping the policy near a reference
  (here, frozen gen-1) as a trust region against destructive updates.
- **V^π** — the value assuming the current policy continues; obtained directly by
  regressing the critic on the policy's own on-policy rollouts.
- **Reward hacking** — the policy exploiting flaws in the (frozen, imperfect)
  scorer to score well without genuinely better decks.
- **Cross-generation yardstick** — a single fixed-mix evaluation run with every
  generation randomly co-seated, compared on raw mean `deck_score` (the frozen
  scorer gives a stable absolute scale; randomization balances the
  opponent-strength confound).
