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

# The Recoverability Ratio

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

The **remaining capacity** to correct the imbalance is the total picks left in the episode,
regardless of type — both spell picks and land picks can move the deck toward the ideal:

```
remaining_picks(s) = 40 − picks_so_far
```

The **recoverability ratio** ψ measures how large the imbalance is relative to the
remaining capacity to fix it:

```
ψ(s) = imbalance(s) / remaining_picks(s)^α   | if remaining_picks(s) > 0
ψ(s) = imbalance(s)                          | if remaining_picks(s) = 0
```

where α > 1 is a tunable exponent (starting point: α = 2). The non-linearity is applied to
`remaining_picks`, not to `imbalance` — any imbalance is tolerable when many picks remain,
but the same imbalance becomes critical as picks run out. ψ ∈ [0, ∞).

# Per-Step Reward

The raw shaping signal for step `t` is the reduction in ψ:

```
shaping_raw(t) = ψ(s_t) − ψ(s_{t+1})
```

Positive when the pick reduced the ratio (good), negative when it increased it (bad). This
is unbounded in both directions: ψ(s_{t+1}) can exceed ψ(s_t) by an arbitrary amount if
picks are very bad, and `shaping_raw` grows without bound as remaining_picks shrinks.

To keep the shaping term on a comparable scale to the Stage 1 reward (±1), tanh is applied:

```
shaping(t) = tanh(k · shaping_raw(t))
```

where k is a temperature hyperparameter (starting point: k = 1). tanh preserves the sign
and keeps small values approximately linear, while clamping large values to ±1. It also
resolves the saturation problem that would occur with a clamped potential: even when the
deck is in a deeply unrecoverable state, any further worsening still produces shaping ≈ −1
rather than ≈ 0.

The full per-step reward is then:

```
r_total(t) = r_stage1(t)  +  shaping(t)
```

This has bounds [−2, 2], interpretable as:

| r_total | Meaning |
|---------|---------|
| +2 | On-budget pick that maximally fixes the mana balance |
| +1 | On-budget pick, neutral for mana |
| 0 | On-budget pick that is maximally bad for mana |
| −1 | Off-budget pick, neutral for mana |
| −2 | Off-budget pick that is maximally bad for mana |

The Stage 1 term `r_stage1(t)` is +1 for a pick within the spell/land budget, −1 for one
that exceeds it — at full strength, unchanged from Stage 1. Action masking ensures all picks
are structurally valid (no duplicates); the budget term still steers the spell/land ratio.

# Why This Works

**The urgency is automatic.** With α = 2, the denominator at step 0 is 39² = 1521; even a
large imbalance produces a tiny ψ and therefore a tiny shaping signal. By step 35 the
denominator is 25; the same imbalance dominates ψ and produces a strong signal. No manual
urgency schedule is needed.

**Neutral picks are gently penalized.** Even a pick that leaves the imbalance unchanged
produces a small negative shaping term, because the denominator shrinks by one step. Spending
a pick without improving anything costs a little recoverability — the right signal.

**Both directions of imbalance are captured.** Picking a land of an over-sourced color
increases imbalance (negative shaping). Picking a spell of an over-sourced color shifts
`ideal[c]` upward, reducing imbalance (positive shaping). The function rewards moves toward
balance regardless of whether that means adding supply or adding demand.

**Stage 1 stays at full strength.** Adding `shaping(t)` rather than replacing or scaling
`r_stage1(t)` means the spell/land budget constraint produces exactly the same gradient it
did in Stage 1. The two signals are complementary: Stage 1 governs the ratio of spells to
lands; the shaping term governs color coordination within that ratio.

**Bounded total reward.** tanh(k · shaping_raw) ∈ (−1, 1) always, so r_total ∈ (−2, 2).
No runaway gradients from late-game picks with tiny remaining_picks denominators.

# Hyperparameters

| Parameter | Role | Starting point |
|-----------|------|----------------|
| α | Exponent on remaining_picks in ψ. Controls when urgency kicks in. Higher α pushes the critical transition closer to the final picks. | 2 |
| k | Temperature on tanh. Higher k makes the shaping signal saturate at smaller raw values (more sensitive). Lower k keeps it linear over a wider range. | 1 |

# Boundary Conditions

- **Step 0** (no picks yet): `total_pips = 0`, `ideal[c] = 0`, `imbalance = 0`, `ψ = 0`,
  `shaping = 0`. The first pick will either create pip demand (spell) or zero-demand supply
  (land/artifact), both producing a near-zero shaping signal.

- **remaining_picks = 0**: `ψ(s) = imbalance(s)` (the ratio degenerates to the raw
  imbalance). This only occurs at the terminal state; no reward is computed there.

- **Colorless spells ({C}, generic)**: generic mana pips do not contribute to any color's
  `pip_demand`. Colorless pips ({C}) contribute to the C bucket and generate demand for
  colorless sources.

- **Single-color deck**: n_colors = 1, `ideal[c] = 17` for that color, 0 for all others.
  Imbalance equals `|17 − actual_sources[c]|`. The model is rewarded for picking sources of
  that color and penalized for picking lands of other colors.
