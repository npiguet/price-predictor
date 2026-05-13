# April 7, 2026 — Sequential PPO abandoned for one-shot

**TL;DR:** Feature 016 shipped two shaping formulas in a row, both failed to
produce learning, and that failure forced a fundamental rethink of the
Stage 2 architecture — ending with a decision to scrap sequential PPO
entirely in favour of one-shot deck scoring.

The day started by picking up the tail end of feature 016: the
recoverability-based per-step reward shaping that was meant to guide the
model toward balanced mana bases during sealed deck construction. The
implementation was complete — 137 tests green, committed. What followed
was a long training run that showed `shaping=-0.07` locked in place across
every single batch for 1,248 episodes. Not drifting, not noisy. Constant.

Claude spotted a bug in `compute_per_step_rewards`: the `remaining -= 1`
decrement was happening before `ratio_after` was computed, so both the
numerator and denominator were shifting in the same direction every step,
creating a structural negative bias independent of what was actually
picked. The fix was one line — move the decrement after `ratio_after`.
Tests still passed. Training still didn't learn.

At that point the continuous PBRS approach was dropped in favour of the
simpler discrete signal I proposed: no shaping until both pip demand and
mana supply are non-zero, then ±0.5 when imbalance is below 3 and ±1.0
when it's 3 or above. The hyperparameters `--urgency-exponent` and
`--temperature` disappeared from the CLI, spec, and tests. Another 134
tests green. Another training run. Still flat.

Two failed reward formulations with identical symptoms — zero movement in
`mean_score` across hundreds of episodes — raised the question of whether
the problem was the reward or the architecture. I asked Claude what
fundamental assumption was baked into sequential picking that might be
wrong.

The diagnosis was credit assignment. Deck mana balance is a property of
the full 40-card combination, not of individual picks. Decomposing it into
40 per-step signals is asking the model to solve a holistic 6-dimensional
optimisation problem one card at a time, chasing a moving target that
shifts every time a new spell is picked. Spell/land balance had been easy
to learn because it's a 1D counting problem with an immediate ±1 signal.
Color balance is 6D and coupled — picking a blue spell changes the optimal
allocation for white, black, red, green, and colorless simultaneously.

The discussion moved through three researcher perspectives: the case for
one-shot scoring, a skeptical counter (non-differentiable top-k, variance
of single-step REINFORCE, sequential conditioning as a feature), and a
neutral verdict recommending PPO diagnostics first, then decomposed spell
selection with expert iteration.

The clean spell-then-land decomposition collapsed when I pointed out that
non-basic lands — dual lands, utility lands, MDFCs — compete with spells
for slots and need to be evaluated holistically. The revised proposal
became: score all 90 pool cards in one transformer forward pass, select
the top 23+x (where x is the count of non-basic lands that ranked above
the cutoff), then fill the remaining land slots deterministically with
basics using the existing ideal distribution formula.

One consequence I noticed during this discussion: if lands are filled
deterministically from the ideal distribution, the mana scoring problem is
solved by construction. The model no longer needs to learn mana balance
at all — it just needs to pick good spells. That eliminates the entire
motivation for feature 016's shaping rewards.

The conversation also surfaced that Forge's `LimitedDeckEvaluator` only
scores individual card power averaged across the deck, with no awareness
of mana curve, color spread, or fixing. That rules it out as a Stage 1
heuristic scorer for REINFORCE. The deeper question became whether a
heuristic Stage 1 is needed at all, or whether going straight to Forge
self-play from the start is viable. The cold-start problem (random decks
produce noise, not signal) favours a bootstrapping phase; game outcomes as
the sole training signal favours correctness over efficiency. A hybrid via
expert iteration on game outcomes — run random-deck tournaments first,
train the model to imitate the winners, iterate — was left as the leading
candidate.

The session ended with a reset of the master branch back to the feature
011 state (encode-cards and generate-pools only), discarding the entire
PPO training stack, and a rewrite of `specs/2026-03-28-sealed-deck-picker.md` to
reflect the one-shot architecture.
