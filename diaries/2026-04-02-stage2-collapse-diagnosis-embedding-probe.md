# April 2, 2026 — Stage 2 collapse diagnosis, embedding probe

**TL;DR:** Stage 2 PPO training had been implemented and ran overnight,
but the model had collapsed. A long diagnostic session traced the failure
to entropy collapse rather than a broken reward, and ended with speccing
a new embedding validation feature.

The session opened mid-stream — context had been compacted from an earlier
run. By that point the full 36-task implementation of feature
`013-stage2-heuristic-gate` was done: the mana scorer, the Stage 2
training loop, sampling output, CLI wiring, 181 tests all green. So this
session was really about what happened when the code actually ran.

The training log I pasted told a clear story: scores had plateaued around
0.72–0.73 for 22k episodes, then at episode 22784 all 32 scores in a
batch became exactly 0.176. That number — `3/17` — was not noise. The
model was outputting the same deck for every pool regardless of contents.
Claude identified it as a deterministic policy collapse: once logits
concentrate sharply enough, `torch.multinomial` just picks the same slots
every time. The entropy coefficient at 0.01 was too weak to resist it.

My first instinct was that the reward function itself was broken — why
else would the model have settled on picking exactly 2 lands per color?
Claude showed the arithmetic: `score = 1 - (l1_error + |n_lands - 17|) /
17`. With 5 colors and 9 lands, the minimum l1 error is 8 (actual
sum = 9, ideal sum = 17), so `score = 1 - 16/17 ≈ 0.059` — and that's
exactly what we were seeing. The model had correctly maximized the 2-
per-color attractor built into the ideal land distribution formula. That
formula guarantees a floor of 2 sources per color, and 2×5 = 10 is close
enough to 9 once rounding kicks in.

I then noted that before the collapse the distribution looked reasonable:
17 lands, sensible spell mix. Claude's first repair was to switch from
replacing step rewards with the mana score to adding the mana score on
top of the Stage 1 per-step signals. That would preserve the land-count
gradient. But I pushed back: the mana score formula already contains
`|n_lands - 17|`, so the Stage 1 signal should be redundant. Claude
reconsidered and agreed: the 17-land behavior had been working for 22k
episodes precisely because the score formula penalizes deviations. The
real fix was the entropy coefficient. Bumping it from 0.01 to 0.05 pushes
back much harder against the probability mass concentrating — it won't
stop convergence but will prevent premature lock-in.

I then reverted Claude's reward-additive change. The code went back to
pure mana score replacement, with only the entropy coefficient changed.

What the pre-collapse samples also revealed was a subtler problem: the
model had learned a fixed `[3,3,3,3,3,2]` basic-land formula across all
pools. Sample 10 was the clearest evidence — the pool had no green spells
at all but the model still picked 3 Forests. The model wasn't adapting to
pool composition; it was applying a canned strategy. The 2 Wastes per
deck were also a sign: Wastes produce `{C}` but most pools have no
colorless-pip spells, so `ideal[C] = 0` and every Wastes pick costs 1 in
l1 error. With 2 Wastes locked in, the score ceiling was `1 - 2/17 ≈
0.882` — mathematically below the 0.90 convergence threshold. The model
could never have converged anyway.

That last point raised a deeper question: can the embeddings even
distinguish Plains from Wastes? The policy sees only embeddings. If the
embedding for a dual land and a Wastes look similar, no reward signal will
teach the model to pick one over the other. That is when we shifted to
speccing a new feature — `014-validate-embeddings` — to probe whether the
frozen price-predictor encoder actually encodes the features Stage 2 needs:
is-land, card color, pip counts, mana value, mana produced by lands.

The probe design uses standard linear probing: train a lightweight linear
classifier or regressor on top of frozen embeddings, ground truth extracted
from card text using the same parsing code as the mana scorer. The table
of features, probe types, and pass thresholds went through a few edits —
I dropped User Story 2 (set-scoped validation), FR-005 (color skip logic
for sparse colors — irrelevant at 30k cards), and SC-01 (no time
constraints). The spec ended the session clean.

The key realization from the day: the entropy collapse was always the
primary failure mode; the reward formula and the credit assignment were
secondary concerns. The `|n_lands - 17|` term in the score was doing its
job for 22k episodes — it was the policy distribution that broke, not the
math.
