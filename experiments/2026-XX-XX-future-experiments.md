# Future experiments

A holding pen for ideas surfaced during gen-2 work that aren't worth chasing
right now (typically because the binding constraint is data quality / volume,
not the model or the optimizer). Each entry: what it is, why it might help,
estimated magnitude of the effect, cost to try, and dependencies that might
unblock or favor it.

## Deferred gen-2 design alternatives (from the gen-2 RL self-play design)

These are the "later / only if needed" branches split out of
`2026-06-10-draft-agent-gen2-design.md`, so that doc holds only the gen-2 plan
we execute now. Each is a contingency, not part of the first gen-2 build.

### PPO, as an upgrade from REINFORCE+GAE

**What.** Replace the plain REINFORCE+GAE policy-gradient update with PPO's
clipped surrogate objective (ratio/value clipping, GAE buffers, minibatched
multiple epochs per batch).

**Why it might help.** Sample efficiency — PPO permits several gradient epochs
over one rollout batch, and draft rollouts are expensive (draft + build +
score), so the data reuse likely pays for itself. The clip also guards against
destructive steps over the 45-step horizon.

**Cost.** More moving parts than REINFORCE; only worth it once the simpler loop
works, so failures stay attributable.

**Trigger.** Adopt only if gen-2's REINFORCE updates prove unstable or sample
cost dominates — an empirical call made after measuring REINFORCE stability.

### Forking (vine) advantage sharpener

**What.** Branch at a chosen state and compare rollout returns for a
low-variance advantage, concentrated in pack 2 (picks most pivotal, critic least
trustworthy), with common random numbers across paired branches.

**Why it might help.** Sharpens and audits the critic where GAE variance is
highest. Never the main estimator — a surgical supplement.

**Cost.** Extra rollouts at every branch point.

**Trigger.** Deferred unless GAE variance proves unacceptable in gen-2.

### Calibrated product objective

**What.** Switch the reward aggregation from the **mean** of `P(beat i)` over pod
opponents (expected match wins) to the **product** (probability of beating the
whole pod). Requires fitting the Bradley-Terry scorer temperature `T` on
held-out matches so `sigmoid((S_A − S_B)/T)` is a calibrated win probability.

**Why it might help.** The product's gradient concentrates on the closest
matchups — the right objective if the goal becomes "robustly beat everyone"
rather than "win on average."

**Cost.** A cheap temperature fit, plus revalidating that the product gradient is
well-behaved.

**Trigger.** Start gen-2 with the mean objective; switch only deliberately, once
`T` is fit and the mean objective is working.

### Java shipping of the trained agent

**What.** Export the eventual agent for in-process Forge use (TorchScript+DJL or
ONNX) instead of the Python pick side-channel.

**Trigger / dependency.** Only relevant once an agent is good enough to ship into
Forge's heuristics; not on the gen-2 critical path.
