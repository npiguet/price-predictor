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
