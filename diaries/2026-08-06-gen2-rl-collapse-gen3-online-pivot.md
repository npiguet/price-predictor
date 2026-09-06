# August 6, 2026 — Gen-2 RL collapse, gen-3 pivot

**TL;DR:** Ran down exactly why the gen-2 offline RL run for the draft
agent collapsed during training, then proved with a real argmax yardstick
that even the "healthier" runs moved the deployed policy by zero. That
result killed the offline actor-critic approach outright and drove a
redesign toward a much simpler online, critic-free GRPO loop for gen-3.

The day started from a concrete symptom: `val_loss` on the gen-2 RL
trainer diving to around -114. I pushed to understand the actual
mechanism rather than just patch it, and it came down to the policy loss
term `-A*logπ` being unbounded below — for any pick the policy decides
was bad, the loss keeps shrinking forever by driving that action's
probability toward zero, with nothing to stop it once the KL and entropy
coefficients had decayed away. The KL term exists specifically to lease a
trust region around a frozen reference policy; once its schedule decayed
to near zero, that leash was gone and the optimizer chased the unbounded
minimum. Restoring a fixed (non-decaying) KL stopped the runaway, but a
closer look at that "fixed" run showed it wasn't a stable equilibrium
either — just a slower version of the same collapse, because a KL penalty
bounds the *average* divergence, not the per-pick tail.

That led to the harder question of what the actual fix should be. PPO's
clipped surrogate came up as the literature-standard answer, but the more
useful realization was that the deeper cause wasn't the missing KL at
all — it was reusing one frozen corpus for tens of thousands of gradient
steps, which lets any offline objective drift arbitrarily from the data
that generated it. My first assumption was that going more online was
infeasible because Forge rollout generation is the expensive part, but
after checking the real numbers — corpus generation and training were
roughly an even wall-clock split — that assumption didn't hold, and
online training became a real option rather than a nonstarter.

That opened the door to stripping the whole recipe down for gen-3:
dropping KL, dropping the entropy bonus (redundant with the rollout
temperature, which already controls exploration), and dropping the
learned critic entirely in favor of critic-free GRPO. The reward is
already the pod-relative leave-one-out deck score, which turned out to
already be the RLOO/group baseline the critic was duplicating — so
cutting it loses little. That left three knobs to actually understand:
learning rate, temperature, and drafts-per-round.

Before committing to any of this, I ran the actual test: a 100-draft
argmax corpus comparing gen-2 against Forge, co-seated in the same pods.
The result was a flat tie — 1.49 vs 1.49 mean deck score, medians equally
matched, deck composition nearly identical. Offline RL had moved the real
objective by exactly zero. The training-loss collapse turned out to be
cosmetic: it lived entirely in the probability tail, while argmax deck
generation only reads the top pick, which the gen-1 imitation prior kept
sticky throughout.

That result raised a further concern worth sitting with: would gen-3's
online loop hit the same wall — learning that stays confined to the tail
while the top pick never moves? The honest answer worked out to be that
online fixes the *meaningless-learning* problem (advantages are fresh and
honest instead of stale and hackable) but not automatically the
*top-pick-inertia* problem, which depends on whether the rollout
temperature actually samples the runner-up card often enough, and whether
the deck-score reward is fine-grained enough to tell a marginally better
top pick from the current one. The difference from gen-2 is that if gen-3
turns out to be a no-op too, it will be visible rather than disguised as
loss-collapse "progress" — which became the reasoning behind adding a
live anchor-margin signal (gen-3's mean deck score against a frozen gen-1
baseline already present in the agent mix) so a plateau shows up directly
in training rather than only at yardstick time. Plain self-play reward
alone was noted to be a bad proxy for this, since pod-relative reward
mechanically sits near zero once all seats are clones of the same
improving policy.

Later work shifted to actually specifying the online trainer as a new
feature. One cleanup worth noting: the CLI for wiring agents into a draft
pod had accumulated a confusing pile of `--checkpoint` /
`--agent-mix` / `--learner-agents` flags, which got collapsed into three
named categories — Forge-piloted (no binding needed), frozen (bound
checkpoints, one of which anchors the margin), and a single learner
(label plus warm-start checkpoint, updated live during training) — each
with its own flag prefix instead of everything sharing generic
`agent`/`checkpoint` names.

One incident during the day: a diagnostic script Claude kicked off to
measure sampling-temperature behavior ended up competing for GPU with an
in-progress training run; once flagged, it was killed along with a
background poll loop, and the training run returned to normal.
