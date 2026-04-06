# Goal

Replace Stage 2's uniform end-of-episode mana score with a per-step reward derived from a
**recoverability potential function**. Each pick gets a reward proportional to how much it
moved the deck toward or away from its ideal mana distribution, weighted by how little time
remains to correct the imbalance — while keeping the Stage 1 budget signal alive to prevent
the model from forgetting that off-budget picks are suboptimal.

# Motivation

Stage 2 currently assigns the same mana score to every step in an episode:

```
r_t = mana_score(final deck)    for all t in [0..39]
```

This means every pick — the first spell picked, the last land picked, and everything in
between — receives identical credit or blame. The model has to figure out through many
episodes which individual picks were actually responsible for a good or bad mana outcome.
That is a hard credit assignment problem: 40 picks, one reward, no signal about which step
caused what.

The symptom observed in practice is a training instability cycle: the model learns some
mana preference, entropy collapses as it starts obsessively picking favourite cards, the
spell/land budget skews, the Stage 1 penalty pushes back, the model partially recovers, and
the cycle repeats. The root cause is that the uniform reward can be overwhelmed by the mana
signal — the two compete rather than cooperate.

What the model actually needs is a reward that answers, step by step: *did this pick move
the deck closer to or further from its ideal mana distribution, given how many picks remain
to correct it?*

# The Ideal Distribution

The same ideal distribution used by the final mana score (feature 013) defines the target:

```
colors_present   = { c : pip_demand[c] > 0 }
n_colors         = len(colors_present)
total_pips       = Σ_{c ∈ colors_present}  pip_demand[c]

ideal[c] = 2 + (lands remaining after mandatory lands) × (proportion of pips for c) 
ideal[c] = 2 + (17 − 2 × n_colors) × pip_demand[c] / total_pips   | for c ∈ colors_present
ideal[c] = 0                                                      | otherwise
```

Each color in the deck gets a minimum floor of 2 sources, and the remaining
`17 − 2 × n_colors` sources are distributed proportionally to pip demand. This is the same
formula used to score the completed deck, so the per-step shaping and the terminal reward
are optimizing the exact same target.

`pip_demand[c]` is the accumulated pip count for color `c` from all non-land cards picked
so far. It is a moving quantity — it changes as spells are picked.

When no spells have been picked yet (`total_pips = 0`), `colors_present` is empty and
`ideal[c] = 0` for all colors. Imbalance is 0 and the shaping term is 0.

# Imbalance and Discrete Shaping

The **imbalance** at deck state `s` is the total absolute deviation from the ideal across
all colors:

```
imbalance(s) = Σ_c  |ideal[c] − actual_sources[c]|
```

where `actual_sources[c]` is the number of sources of color `c` among all land cards picked
so far.

Using absolute value makes the signal symmetric: both under-supplying a color (fewer sources
than ideal) and over-supplying it (more sources than ideal) move the deck away from optimal.
This also means picking a spell of an over-sourced color is rewarded — it shifts `ideal[c]`
upward, closing the surplus gap.

# Per-Step Reward

The shaping signal at step `t` is a discrete value based on how the pick changed imbalance:

```
shaping(t) = 0      if pip_counts empty OR actual_sources empty
shaping(t) = +0.5   if imbalance_before < 3 and imbalance decreased
shaping(t) = -0.5   if imbalance_before < 3 and imbalance increased
shaping(t) = +1.0   if imbalance_before >= 3 and imbalance decreased
shaping(t) = -1.0   if imbalance_before >= 3 and imbalance increased
shaping(t) = 0      if imbalance unchanged
```

The full per-step reward is then:

```
r_total(t) = r_stage1(t)  +  shaping(t)
```

This has bounds [−2, 2], interpretable as:

| r_total | Meaning |
|---------|---------|
| +2 | On-budget pick that fixes mana balance (imbalance was >= 3) |
| +1.5 | On-budget pick that fixes mana balance (imbalance was < 3) |
| +1 | On-budget pick, neutral for mana (no demand/supply yet, or unchanged) |
| 0 | On-budget pick that worsens mana balance (imbalance was >= 3) |
| −2 | Off-budget pick that worsens mana balance (imbalance was >= 3) |

The Stage 1 term `r_stage1(t)` is +1 for a pick within the spell/land budget, −1 for one
that exceeds it — at full strength, unchanged from Stage 1. Action masking ensures all picks
are structurally valid (no duplicates); the budget term still steers the spell/land ratio.

# Why This Works

**Clear directional signal.** The discrete values (+0.5, +1.0, -0.5, -1.0) are large enough
for PPO to learn from. The threshold at imbalance >= 3 amplifies the signal when the mana
base is seriously out of balance.

**Graceful activation.** Shaping is 0 until both pip demand and mana supply exist. This
prevents nonsensical imbalance calculations when the ideal distribution is empty or there
are no sources to compare against.

**Neutral picks are truly neutral.** A pick that leaves the imbalance unchanged receives
shaping = 0, not a spurious negative signal.

**Both directions of imbalance are captured.** Picking a land of an over-sourced color
increases imbalance (negative shaping). Picking a spell of an over-sourced color shifts
`ideal[c]` upward, reducing imbalance (positive shaping). The function rewards moves toward
balance regardless of whether that means adding supply or adding demand.

**Stage 1 stays at full strength.** Adding `shaping(t)` rather than replacing or scaling
`r_stage1(t)` means the spell/land budget constraint produces exactly the same gradient it
did in Stage 1. The two signals are complementary: Stage 1 governs the ratio of spells to
lands; the shaping term governs color coordination within that ratio.

**Bounded total reward.** shaping ∈ {-1, -0.5, 0, +0.5, +1}, so r_total ∈ [-2, 2].
No runaway gradients.

# Boundary Conditions

- **No spells yet** (`pip_counts` empty): shaping = 0. No pip demand exists.

- **No lands yet** (`actual_sources` empty): shaping = 0. No mana supply exists.

- **Colorless spells ({C}, generic)**: generic mana pips do not contribute to any color's
  `pip_demand`. Colorless pips ({C}) contribute to the C bucket and generate demand for
  colorless sources.

- **Single-color deck**: n_colors = 1, `ideal[c] = 17` for that color, 0 for all others.
  Imbalance equals `|17 − actual_sources[c]|`. The model is rewarded for picking sources of
  that color and penalized for picking lands of other colors.

# Sample Output Enhancement

The `sealed sample` command must print the mana cost of each non-land card before its name in
the pick list, e.g. `1. {U}{U} Counterspell` instead of `1. Counterspell`. Land cards are
printed without a mana cost prefix. This makes it easy to visually verify the color coordination
of picked decks at a glance.
