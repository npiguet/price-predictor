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

**Accepted: on-policy actor-critic**, rollouts produced by the current policy
through the existing live-play command.

The decisive implementation choice: **the policy never runs inside the Forge
draft loop with per-pick IPC for values/log-probs**. A recorded corpus plus the
*frozen snapshot that generated it* are sufficient to recompute, in a single
batched post-rollout GPU pass:

- the behaviour log-prob `log π_snap(aₜ | sₜ)` at every pick the agent made,
- the critic value `V(sₜ)` at every state,
- hence the GAE advantages and returns,

with the reward read directly from the record (pod-relative leave-one-out
`deck_score`). So the live-play data generator is reused **unchanged**; gen-2
adds only a trainer that consumes its output.

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

## Algorithm: REINFORCE+GAE first, PPO as the planned upgrade

**Decision: start with REINFORCE + a GAE baseline + KL anchor + entropy bonus;
upgrade to PPO if updates prove unstable or sample cost dominates.**

- The strong warm-start plus the KL anchor make destructive updates unlikely —
  exactly the regime where plain REINFORCE+baseline is adequate and simplest, and
  its policy-gradient + entropy + baseline machinery already exists in the repo.
- **PPO** is the planned upgrade: its clipped surrogate and multiple epochs per
  batch buy sample efficiency and guard against destructive steps over the
  45-step horizon, at the cost of more moving parts (ratio/value clipping, GAE
  buffers, minibatching). Draft rollouts are expensive (draft + build + score),
  so PPO's data reuse will likely pay for itself — but adopt it only once the
  simpler loop works, so failures stay attributable.

A deliberate start-simple-then-upgrade call; the REINFORCE↔PPO switch is a
measured decision, not a fixed commitment.

## Credit assignment: GAE(λ→1), with forking as an optional sharpener

**Accepted: a learned value function + GAE(λ) with λ near 1** — a dense, cheap
per-pick advantage `≈ V(sₜ₊₁) − V(sₜ)` at every one of the 45 picks, with λ→1
leaning on the unbiased (higher-variance) Monte-Carlo return. λ is a
variance/bias dial to tune.

**Milestone 0 — root-only GRPO/RLOO as the trivial first cut.** Before GAE, the
simplest possible gen-2 is one scalar advantage per draft (the pod-relative
reward centred by the group mean) applied uniformly to every pick — no per-pick
critic in the gradient. High-variance but trivially correct, and a good end-to-end
smoke test that the rollout→reward→gradient plumbing works. GAE then replaces the
flat advantage with the per-pick value jumps.

**Forking (vine) — optional, surgical.** Branch at a state and compare rollout
returns for a low-variance advantage, **concentrated in pack 2** (picks most
pivotal, critic least trustworthy) with common random numbers across paired
branches. Used to *sharpen and audit* the critic, never as the main estimator.
Deferred unless GAE variance demands it.

## The critic in gen-2: on-policy V^π

**Accepted.** Training the critic on the policy's **own** rollouts yields `V^π`
by construction — the continuation is the policy's — removing the
continuation-mix bias the offline critic carried. Same context-token head and
MC/GAE(λ→1) target; only the training states change (now on-policy).

## Reward and objective: pod-relative mean now, calibrated product later

**Accepted: pod-relative leave-one-out reward, mean over opponents** — the form
`mean(P(beat i))` *is* the gradient of expected match wins, which is the
tournament objective.

**Calibration unlocks the product objective.** The scorer is a Bradley-Terry
model, so scores are logits and `sigmoid((S_A − S_B)/T)` is a win probability up
to one temperature `T`, fit cheaply on held-out matches. With `T` in hand, gen-2
can switch the aggregation from **mean** (expected match wins) to **product** of
`P(beat i)` (probability of beating the *whole* pod, gradient concentrating on the
closest matchups) if the goal becomes "robustly beat everyone." **Start with
mean**; treat the temperature fit + product objective as a deliberate later switch.

The reward source stays the frozen scorer over picker-built decks, which makes
reward-hacking a first-class concern (below).

## KL anchor to gen-1, and entropy for exploration

**Accepted: a KL penalty `KL(π ‖ π_gen1)` against the frozen gen-1 policy.** It
(a) keeps the policy on the manifold of reasonable drafts, (b) is the primary
defence against reward-hacking the imperfect scorer, and (c) stabilises updates.
Its coefficient follows a schedule — heavier early, relaxed as the critic proves
trustworthy. An **entropy bonus** adds local exploration on top of the
temperature-sampled rollouts.

## Exploration: temperature-sampled rollouts

**Accepted.** Rollouts use `--pick-mode sample` at a temperature `T_explore`
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
  evaporate as the policy improves, and (c) the **frozen held-out yardstick**
  below.

This is the subtlety the spec must get right: "more data is better" silently
breaks the gradient if old generations are folded into it.

## Self-play regeneration loop and the cross-generation yardstick

**Accepted.** Each generation regenerates data from a mix of the **current
policy + Forge + laddered/random bots**, trains `πₖ₊₁`, and iterates. The mix
preserves opponent diversity and the late-pick "opponents are weak, take the good
card" signal; the random-bot minority preserves low-end critic coverage.

A **frozen held-out evaluation set is carved out now** — before any gen-2
training mutates the distribution — to give an unbiased cross-generation
yardstick (the agent piloting a fixed battery of pods, scored, and/or
`match-outcomes` head-to-heads). Without a frozen reference, "is gen-2 better than
gen-1" is unanswerable, because each generation shifts its own data distribution.

## Reward-hacking and failure modes

The reward is a *frozen, imperfect* scorer, so the policy can in principle find
states the scorer over-rates. Defences, in order: the **KL anchor** (keeps play
near the competent manifold), the **frozen scorer** (a moving scorer would chase
the policy), and **composition monitoring** via `draft analyze-generated-decks`
across generations. A self-play colour/creature bias can amplify generation over
generation, so creature count, colour balance, and curve are watched as
early-warning signals that the policy is exploiting the scorer or collapsing
diversity rather than improving. A divergence between scorer score and
`match-outcomes` win rate is the ground-truth alarm.

## Tooling / implementation surface (the part that does not exist yet)

gen-2 is gated on new code, not just a spec:

- **An RL trainer** that loads a prior-gen checkpoint as actor+critic, reads one
  or more self-play corpora, recomputes log-probs/values from the generating
  snapshot, computes GAE advantages, and applies the KL-anchored
  policy-gradient + value + entropy update.
- **Snapshot/provenance recording** so a corpus is unambiguously tied to the
  policy that generated it (required for on-policy correctness) — the generating
  checkpoint identity per generation.
- **Multi-corpus input** for the trainer, at least for the off-policy-safe
  critic/coverage roles.
- The **generation-loop orchestration** (freeze → generate → train → promote),
  which can start as a script around the existing commands.

Everything upstream — the gen-1 artefact, live rollout, reward, builder, state
reconstruction, analysis — already exists; the actor-critic update and the loop
are the build.

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
- **Java shipping** of the eventual agent (TorchScript+DJL / ONNX).

## Glossary

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
- **V^π** — the value assuming the current policy continues; obtained directly by
  regressing the critic on the policy's own on-policy rollouts.
- **Reward hacking** — the policy exploiting flaws in the (frozen, imperfect)
  scorer to score well without genuinely better decks.
- **Frozen yardstick** — a held-out evaluation battery fixed before training, so
  successive generations are comparable on a stable reference.
