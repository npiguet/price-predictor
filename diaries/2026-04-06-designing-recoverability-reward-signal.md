# Sunday, April 6, 2026 — Designing recoverability reward signal

**TL;DR:** Most of the day went into designing a mathematically sound
per-step mana-coherence reward for Stage 2 training. Along the way
action masking replaced duplicate penalties, the `validate-embeddings`
command was fully implemented, and a `count_pips` bug on Baldur's Gate
companion cards was found and fixed.

The observation that pushed the day's main design work was watching
Stage 2 training oscillate: the model would learn mana coherence, then
duplicate-pick its favorite cards because the mana signal outweighed
the duplicate penalty, then partially recover, then repeat. The problem
was that "don't duplicate" and "build good mana" were competing for the
same weights. Action masking resolved the tug-of-war by making
already-picked booster slots structurally unselectable — `-1e9` in the
logit rather than a learned avoidance — so every training episode now
completes all 40 picks and generates a usable mana score. The
`float('-inf')` approach was briefly tried and immediately caused NaN in
the entropy term because IEEE 754 gives `-inf × 0 = NaN`; the large
finite substitute underflows cleanly to 0.0 in float32.

With duplicate instability removed, the remaining problem was that
Stage 2's reward is a single float emitted at episode end, broadcast
uniformly to all 40 steps. The model can't tell which pick caused a
bad outcome. The session turned into an extended design discussion
about a per-step shaping signal, converging on a recoverability ratio
ψ(s) = imbalance(s) / remaining_picks(s)^α, where imbalance is the
summed absolute deviation between ideal and actual mana sources across
all colors. The key properties that were worked through: the ideal
distribution uses the same formula as the final mana score (the
`2 + (17 - 2·n_colors) · pip[c] / total_pips` normalization), so
shaping and terminal reward are consistent; absolute value instead of
`max(0, ...)` makes over-sourced colors count just as much as
under-sourced ones, which also rewards picking spells of a color you
have too many lands for; using total remaining_picks (not just land
picks) in the denominator means any pick can fix the imbalance. A
`max(0, ...)` clamp on ψ was identified as a saturation trap —
once the deck is unrecoverable the signal goes flat — so the shaping
term was switched to the raw delta ψ(s_t) - ψ(s_{t+1}), then wrapped
in tanh to bound its magnitude to [-1, 1]. Combined with the Stage 1
budget signal (±1 per pick), the total per-step reward sits in [-2, 2],
which has a clean interpretation: +2 means a legal pick that maximally
improves mana, -2 means an off-budget pick that catastrophically hurts
it. α and a temperature k inside the tanh are the two remaining
hyperparameters.

The validate-embeddings work was a self-contained implementation sprint
for feature 014. Twenty linear probes — one per mana-relevant card
property — are trained on frozen card embeddings to validate that
useful structure survives encoding. The interesting result from running
the probes on the current model was that colorless-pip R² was 0.043,
which looks alarming until you remember that R² near zero when the mean
is near zero just means "no better than always predicting the mean" —
and since almost no cards have {C} pips, always predicting zero is
already almost always correct. The exact match column (fraction of
predictions rounded to the nearest valid discrete value that equal the
true value) was added precisely to expose this: pip count (C) showed
0.999 exact match alongside the 0.043 R². Mana value landed at 0.500
exact match, which was identified as a real problem — half the time the
embedding doesn't encode CMC precisely enough to round to the right
integer, and mana value is foundational to deck building.

The `count_pips` bug discovered during class-distribution analysis was
a good domain example: Baldur's Gate companion cards have six separate
`mana cost:` lines in their converted text (one per background color
pair), and the function was summing all of them. Fixing it to break
after the first line brought the red pip maximum from 13 down to 8
(Khalni Hydra).

The broader architectural thread — whether to drop the pip count and
mana value auxiliary training heads now that explicit mana features are
baked directly into the card embedding — was resolved in the affirmative.
The argument was that the heads were training the transformer to
re-derive information already provided exactly by rule, pulling gradient
away from the higher-value representations only the transformer can
produce: effect semantics, triggers, synergy signals, power calibration.
