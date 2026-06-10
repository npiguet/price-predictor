# Draft agent (gen-2) — RL self-play design rationale

## Background

This document is the design rationale for **generation 2** of the draft agent:
the RL self-play stage that the gen-1 work was explicitly built to enable. It is
the *why* companion to the normative spec
[`../specs/2026-06-10-draft-agent-gen2-rl.md`](../specs/2026-06-10-draft-agent-gen2-rl.md)
(which specifies *what* to build), in the same format as the gen-1 design log
([`2026-05-30-draft-agent-design.md`](./2026-05-30-draft-agent-design.md)) —
alternatives considered, trade-offs weighed, decisions and the reasons behind
them. Where gen-1 deferred a choice "to gen 2", this file resolves it or states
why it stays open.

Two prior artefacts are the substrate:

- **gen-1** (spec [`../specs/2026-05-28-draft-agent.md`](../specs/2026-05-28-draft-agent.md),
  design log above): an imitation policy + a Monte-Carlo critic, co-trained on a
  shared transformer trunk from Forge-generated drafts. Supervised only — the
  critic is trained but never steers the policy.
- **019 live play** (spec [`../specs/019-draft-live-play/spec.md`](../specs/019-draft-live-play/spec.md)):
  the trained agent can pilot live seats in a real Forge pod, emitting a
  self-play corpus (`drafts.jsonl`) with every seat built + scored. This is the
  **data half** of self-play.

**Motivation.** The 2026-06-10 live-play analysis found gen-1 ≈ Forge on
scorer-judged deck quality (deck_score 1.52 vs 1.48). That is the designed
ceiling of imitation: cross-entropy can only copy Forge, so it cannot exceed it.
Generation 2's whole purpose is to push past that ceiling by **using the critic
to steer picks** — the learning half that gen-1 deliberately left unbuilt.

The objective is unchanged from gen-1: win rate *when the Forge AI pilots the
deck the agent drafts*, measured pod-relatively. Forge's piloting tendencies
remain the target distribution, not a bias to correct.

## What gen-2 adds, and what it inherits

**Inherited unchanged** (do not re-litigate): the objective; the typed-token
state representation (`POOL`/`PACK`/`PASSED`/`TAKEN`/`CONTEXT` + split recency);
the model architecture (SAB trunk + per-card policy head + context-token critic
head, 6 layers — the gen-1 depth pick); the pod-relative leave-one-out reward
shape; the picker as the deck builder and the frozen sealed scorer as the
fitness function; the live-play data-generation path (019).

**Added by gen-2** — exactly the missing learning half:

1. An **on-policy actor-critic trainer** that turns the critic's advantage into a
   policy-gradient update, KL-anchored to the frozen gen-1 policy.
2. The **self-play regeneration loop**: freeze → roll out → update → repeat,
   across generations, with coverage and forgetting controls.

Nothing in gen-1's architecture changes; gen-2 is a new *training procedure* over
the same model, plus orchestration.

## The bootstrapping payoff: warm-start both heads

**Accepted.** The actor is initialised from the gen-1 policy and the critic from
the gen-1 critic — both already co-trained on the same trunk, so they are
pre-aligned on one play distribution. This is the entire reason gen-1 built the
critic alongside the policy (the "bootstrapping ladder" from the gen-1 log):
gen-2 starts from a competent policy and a critic that already explains ~73 % of
reward variance, so the actor↔critic loop opens from sanity rather than noise.
RL-from-scratch on this 45-step sparse-terminal POMDP was argued infeasible in
the gen-1 log; that argument is the premise here.

## On-policy actor-critic — the core, and how it reuses 019

**Accepted: on-policy actor-critic**, with rollouts produced by the current
policy through the 019 live-play path.

The decisive implementation choice is that **the policy never needs to run inside
the Forge draft loop with per-pick IPC for values/log-probs**. The 019 corpus
already records every seat's full pick sequence and which agent piloted it, and
`build_state` / `OnlineDraftStateTracker` reconstruct the exact typed-token state
at any `(seat, pack, pick)`. So given the *frozen snapshot that generated the
corpus*, a batched post-rollout GPU pass recomputes:

- the behaviour log-prob `log π_snap(aₜ | sₜ)` at every pick the agent made,
- the critic value `V(sₜ)` at every state,
- hence the GAE advantages and returns,

and the reward is the pod-relative leave-one-out `deck_score` already in the
record. **019 needs no change** — it stays a pure data generator; gen-2 adds a
trainer that consumes its output.

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

**Rejected — critic-in-the-loop greedy actor / live per-pick value queries.**
The gen-1 log already rejected a critic-only greedy actor (off-distribution
exploitation of critic overestimates, inference cost, continuation mismatch).
Live per-pick value IPC is also unnecessary given post-hoc recomputation, and
would couple the Python trainer to the Java draft loop for no benefit.

## Algorithm: REINFORCE+GAE first, PPO as the planned upgrade

**Decision: start with REINFORCE + a GAE baseline + KL anchor + entropy bonus;
upgrade to PPO if updates prove unstable or sample cost dominates.**

Rationale:

- The strong gen-1 warm-start plus a KL anchor to gen-1 makes destructive policy
  updates unlikely, which is exactly the regime where plain REINFORCE+baseline
  is adequate and simplest.
- The one-shot picker (spec 017) already validated a REINFORCE implementation in
  this codebase, so the policy-gradient + entropy + baseline machinery is
  familiar and partly reusable, even though the draft is multi-step where the
  picker was single-step.
- **PPO** is the planned upgrade, not the starting point: its clipped surrogate
  and multiple epochs per batch buy sample efficiency and guard against
  destructive steps over the 45-step horizon, at the cost of more moving parts
  (ratio clipping, value clipping, GAE buffers, minibatching). Given that draft
  rollouts are expensive (draft + build + score), PPO's data reuse will likely
  pay for itself — but only adopt it once the simpler loop is shown to work, so
  failures are attributable.

This is a deliberate start-simple-then-upgrade call; the spec should treat the
REINFORCE↔PPO switch as a measured decision, not a fixed commitment.

## Credit assignment: GAE(λ→1), with forking as an optional sharpener

**Accepted: a learned value function + GAE(λ) with λ near 1.** This is the gen-1
log's named "gen-2 workhorse" and the reason the critic exists: it gives a dense,
cheap per-pick advantage `≈ V(sₜ₊₁) − V(sₜ)` at every one of the 45 picks, with
λ→1 leaning on the (unbiased, higher-variance) Monte-Carlo return that gen-1's
critic was already trained against. λ is a variance/bias dial to tune.

**Milestone 0 — root-only GRPO/RLOO as the trivial first cut.** Before GAE, the
simplest possible gen-2 is one scalar advantage per draft (the pod-relative
reward, centred by the pod/group mean) applied uniformly to every pick — no
per-pick critic in the gradient. It is high-variance but trivially correct and a
good smoke test that the rollout→reward→gradient plumbing works end-to-end. GAE
then replaces the flat advantage with the per-pick value jumps.

**Forking (vine) — optional, surgical.** Branch at a state and compare rollout
returns for a low-variance advantage, **concentrated in pack 2** (where picks are
most pivotal and the critic is least trustworthy) and with common random numbers
(fixed unopened packs + opponent seeds) across paired branches. Used to *sharpen
and audit* the critic, never as the main estimator. Deferred unless GAE variance
demands it.

## The critic in gen-2: on-policy V^π

**Accepted.** gen-1's critic was a mild approximation to `V^π` — it regressed
final reward over a *mix* of Forge continuations, and the log accepted a small
continuation bias (and dropped the agent-skill tag as artificial). gen-2 removes
that bias for free: regressing the critic on the *policy's own* rollouts yields
`V^π` by construction, since the continuation is the policy's. The critic keeps
the same context-token head and MC/GAE(λ→1) target; the only change is that its
training states now come from on-policy rollouts.

## Reward and objective: pod-relative mean now, calibrated product later

**Accepted: pod-relative leave-one-out reward, mean over opponents** — unchanged
from gen-1, for the reasons established there (a genuine two-sided gradient that
correctly values hate picks; smoothness/variance of mean over max; and
`mean(P(beat i)) = expected match win rate`, so the mean *is* the gradient of
"expected match wins").

**Calibration unlocks the product objective.** The sealed scorer is a
Bradley-Terry model, so scores are logits and `sigmoid((S_A − S_B)/T)` is a win
probability up to one temperature `T`, fit cheaply on held-out matches. With `T`
in hand, gen-2 can switch the aggregation from **mean** (expected match wins) to
**product** of `P(beat i)` (probability of beating the *whole* pod, gradient
concentrating on the closest matchups) if the goal becomes "robustly beat
everyone" rather than "win the most pairwise matches." **Start with mean**;
treat the temperature fit + product objective as a deliberate later switch.

The reward source stays the frozen scorer over picker-built decks (same as
gen-1), which raises reward-hacking as a first-class concern (below).

## KL anchor to gen-1, and entropy for exploration

**Accepted: a KL penalty `KL(π ‖ π_gen1)` against the frozen gen-1 policy.** The
gen-1 log flagged that imitation provides "a reference to KL-anchor RL against";
gen-2 cashes that in. The anchor (a) keeps the policy on the manifold of
reasonable drafts, (b) is the primary defence against the policy reward-hacking
the imperfect scorer, and (c) stabilises updates. Its coefficient follows a
schedule (heavier early, relaxed as the critic proves trustworthy), analogous to
the picker's entropy schedule. An **entropy bonus** supplies local exploration on
top of the temperature-sampled rollouts.

## Exploration: temperature-sampled rollouts

**Accepted.** Rollouts are generated with 019's `--pick-mode sample` at a
temperature `T_explore` (seeded for reproducibility). The behaviour distribution
is then `softmax(logits / T_explore)`, and the recomputed behaviour log-probs use
the same temperature, so the policy gradient is exact. Argmax rollouts would give
the policy zero stochastic exploration (only Forge's booster/seat randomness
varies the data), which is too little signal for the gradient to improve picks —
the same reasoning that recommended `sample` for gen-2 data generation in the
019 discussion. Temperature and entropy coefficient are decayed as the policy
sharpens.

## Off-policy data: reconciling the on-policy constraint with "concatenate old+new"

The gen-1 log's self-play note says to "retrain a next generation on **old + new
data concatenated**" to prevent forgetting and preserve bad-region coverage.
That is correct for an **imitation + MC-critic** retrain (both are off-policy
tolerant — cross-entropy and value-of-realised-return don't care which policy
generated the state). **It is *not* valid for a policy gradient**, which is
on-policy by definition: a REINFORCE/PPO update over a previous generation's
rollouts optimises the wrong objective unless importance-corrected.

**Resolution:**

- The **policy gradient uses only current-generation, on-policy rollouts.** PPO's
  importance ratios permit limited reuse *within* a generation (multiple epochs
  over the same batch), but not across generations.
- **Old generations are retained for three off-policy-safe roles**: (a) critic
  regularisation / bad-region coverage (MC value targets are valid off-policy,
  though for a clean `V^π` the on-policy rollouts dominate), (b) a steady
  **minority of dumb/random-bot pods** every generation so coverage of incoherent
  states does not evaporate as the policy improves, and (c) the **frozen held-out
  yardstick** below.

This correction is the single most important thing the gen-2 spec must get
right, because it is the place the gen-1 prose is misleading for the RL case.

## Self-play regeneration loop and the cross-generation yardstick

**Accepted.** Each generation regenerates data from a mix of the **current
policy + Forge + laddered/random bots** (via the 019 `--agent-mix` /
`--agent-checkpoint` machinery), trains `πₖ₊₁` as above, and iterates. The mix
preserves opponent diversity and the late-pick "opponents are weak, take the good
card" signal; the random-bot minority preserves low-end critic coverage.

A **frozen held-out evaluation set is carved out now** (before gen-2 training
mutates anything) to give an unbiased cross-generation yardstick — the agent
piloting a fixed battery of pods, scored, and/or `match-outcomes` head-to-heads
against gen-1 and Forge. Without a frozen reference, "is gen-2 better than gen-1"
is unanswerable because each generation shifts its own data distribution.

## Reward-hacking and failure modes

The reward is a *frozen, imperfect* scorer, so the policy can in principle find
states the scorer over-rates. Defences, in order: the **KL anchor** to gen-1
(keeps play near the human/Forge manifold), the **frozen scorer** (a moving
scorer would chase the policy), and **composition monitoring** via
`draft analyze-generated-decks` across generations. The gen-1 sealed work
documented a self-play bias where Forge's W/G/creature preference *amplified*
generation over generation; gen-2 must watch the same drift (creature count,
colour balance, curve) as an early-warning signal that the policy is exploiting
the scorer or collapsing diversity rather than genuinely improving. A divergence
between scorer score and `match-outcomes` win rate is the ground-truth alarm.

## Tooling / implementation surface (the part that does not exist yet)

gen-2 is gated on new code, not just a spec:

- **An RL trainer** (a new `train-draft-agent` mode or sibling subcommand) that:
  loads a gen-1 (or prior-gen) checkpoint as actor+critic, reads one or more
  self-play corpora, recomputes log-probs/values from the generating snapshot,
  computes GAE advantages, and applies the KL-anchored policy-gradient +
  value-regression + entropy update. Reuses gen-1's checkpoint plumbing
  (`DraftAgentStore`), `build_state`, the reward computation, and the warmup/
  clip/anneal scaffolding.
- **Snapshot/provenance recording** so a corpus is unambiguously tied to the
  policy that generated it (required for on-policy correctness). 019 stamps a
  `run_id` and the per-seat agent label; gen-2 additionally needs the generating
  checkpoint identity per generation.
- **Multi-corpus input** for `train-draft-agent` (currently a single
  `--drafts-path`), at least for the off-policy-safe critic/coverage roles.
- The **generation-loop orchestration** (freeze → generate → train → promote),
  which can start as a script around the existing 019 + trainer commands.

Everything upstream — gen-1 artefact, live rollout (019), reward, builder,
state reconstruction, analysis — already exists; the actor-critic update and the
loop are the build.

## Open / future questions

- **REINFORCE vs PPO** — empirical; start REINFORCE+GAE, measure stability and
  sample cost before adopting PPO.
- **λ for GAE**, KL-coefficient schedule, entropy schedule, exploration
  temperature schedule — all to tune.
- **mean vs product objective** — needs the Bradley-Terry temperature fit; start
  mean.
- **Forking budget** — whether pack-2 vine sharpening is needed at all, or GAE
  variance is acceptable.
- **How many generations**, and the stopping/promotion rule against the frozen
  yardstick.
- **Whether the post-hoc recompute is sufficient** or any per-pick live value is
  ever needed (current belief: recompute suffices).
- **Java shipping** of the eventual agent (TorchScript+DJL / ONNX), inherited as
  a gen-2+ concern from the gen-1 log.

## Glossary

Extends the gen-1 design log's glossary; the entries below are the ones gen-2
leans on most.

- **On-policy** — the gradient is estimated from data generated by the *current*
  policy; required by REINFORCE/PPO (unlike imitation/MC-critic, which tolerate
  off-policy data).
- **Actor-critic** — the policy (actor) is updated by a policy gradient weighted
  by an advantage estimated from the value function (critic).
- **GAE(λ)** — generalised advantage estimation; λ interpolates Monte-Carlo
  (unbiased, high variance) and one-step TD (biased, low variance). λ→1 ≈ MC.
- **PPO** — proximal policy optimisation; a clipped surrogate objective that
  permits several gradient epochs per batch without destructive updates.
- **REINFORCE** — the plain policy-gradient estimator `∇ log π(a|s) · advantage`.
- **KL anchor** — a penalty `KL(π ‖ π_ref)` keeping the policy near a reference
  (here, frozen gen-1) to prevent drift and reward-hacking.
- **V^π** — the value assuming the current policy continues; obtained directly
  by regressing the critic on the policy's own on-policy rollouts.
- **Reward hacking** — the policy exploiting flaws in the (frozen, imperfect)
  scorer to score well without genuinely better decks.
- **Frozen yardstick** — a held-out evaluation battery fixed before training, so
  successive generations are comparable on a stable reference.
