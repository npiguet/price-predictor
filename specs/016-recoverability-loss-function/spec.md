# Feature Specification: Recoverability-Based Per-Step Stage 2 Loss

**Feature Branch**: `016-recoverability-loss-function`
**Created**: 2026-04-06
**Status**: Draft
**Input**: User description: "Replace Stage 2's uniform end-of-episode mana score with a per-step reward derived from a recoverability potential function"

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Per-Step Recoverability Reward in Stage 2 Training (Priority: P1)

A researcher who has been training with Stage 2's uniform mana-score reward notices a training instability cycle:
the model learns a mana preference, entropy collapses as it obsessively picks favourite cards, the spell/land budget
skews, Stage 1's penalty pushes back, and the cycle repeats. They want a reward function that gives the model
step-by-step credit assignment — telling it after each pick whether that pick moved the deck closer to or further from
its ideal mana distribution, with urgency that increases as fewer picks remain to correct any imbalance. They launch
Stage 2 training and observe that each of the 40 picks in an episode receives its own reward value reflecting both the
Stage 1 budget signal and the recoverability-based mana shaping signal.

**Why this priority**: This is the core deliverable — replacing the uniform reward with per-step shaping. Without it,
the credit assignment problem remains unsolved and the instability cycle continues.

**Independent Test**: Can be fully tested by running Stage 2 training on a small dataset and verifying that each step
in an episode receives a distinct reward value (not a uniform value), that rewards fall within the expected (-2, 2)
range, and that picks that clearly improve mana balance receive higher rewards than picks that worsen it.

**Acceptance Scenarios**:

1. **Given** Stage 2 training is running with the recoverability reward, **When** an episode completes all 40 picks,
   **Then** each step has its own reward value (not all steps sharing the same value).

2. **Given** a deck state where white mana is under-supplied relative to the ideal, **When** the model picks a Plains
   (adding a white source), **Then** that step's shaping component is positive (the pick reduced the imbalance).

3. **Given** a deck state where white mana is already over-supplied, **When** the model picks another Plains, **Then**
   that step's shaping component is negative (the pick increased the imbalance).

4. **Given** Stage 2 training is running, **When** any step's total reward is computed, **Then** it falls within the
   range (-2, 2) — the sum of the Stage 1 budget signal (in {-1, +1}) and the bounded shaping term (in (-1, 1)).

5. **Given** Stage 2 training with the recoverability reward, **When** training progresses over many episodes, **Then**
   the model converges without the entropy-collapse/recovery instability cycle observed with the uniform reward.

---

### User Story 2 — Recoverability Ratio Computation (Priority: P1)

The system needs to compute how recoverable a deck's mana imbalance is at each step — quantifying how far the deck
is from its ideal mana distribution relative to how many picks remain to correct it. This computation is the
mathematical core of the per-step reward and must correctly track the moving ideal distribution, measure imbalance
symmetrically, and amplify urgency as picks run out.

**Why this priority**: The recoverability ratio is the foundation of the shaping signal. If the ratio is computed
incorrectly, the reward will send wrong signals to the model. Co-equal with Story 1 since neither works without
the other.

**Independent Test**: Can be tested independently by constructing known deck states at various points in an episode
and verifying that the recoverability ratio matches hand-calculated expected values — including early picks (low
urgency), late picks (high urgency), and the boundary conditions.

**Acceptance Scenarios**:

1. **Given** a deck state at step 5 (35 picks remaining) with an imbalance of 4.0, **When** the recoverability ratio
   is computed with exponent 2, **Then** the ratio is 4.0 / 35^2 = 0.00327 (approximately).

2. **Given** the same imbalance of 4.0 but at step 35 (5 picks remaining), **When** the recoverability ratio is
   computed with exponent 2, **Then** the ratio is 4.0 / 5^2 = 0.16 — dramatically higher than at step 5.

3. **Given** no spells have been picked yet (step 0), **When** the recoverability ratio is computed, **Then** the
   ratio is 0 because pip demand is zero, ideal distribution is zero for all colors, and imbalance is zero.

4. **Given** a deck state where actual sources exactly match the ideal distribution, **When** the recoverability
   ratio is computed, **Then** the ratio is 0 regardless of how many picks remain.

5. **Given** a deck state at the terminal step (remaining picks = 0), **When** the recoverability ratio is computed,
   **Then** the ratio equals the raw imbalance value (the denominator degenerates).

---

### User Story 3 — Stage 1 Budget Signal Preserved at Full Strength (Priority: P1)

A researcher wants assurance that adding the mana-shaping reward does not weaken or replace the Stage 1 budget
signal. The spell/land ratio constraint must produce exactly the same gradient it did in Stage 1 — the two signals
are complementary: Stage 1 governs the ratio of spells to lands; the shaping term governs color coordination within
that ratio.

**Why this priority**: If the Stage 1 signal is diluted, the model may lose its ability to maintain proper spell/land
ratios, undermining the foundation that Stage 1 established.

**Independent Test**: Can be tested by verifying that the Stage 1 component of the total reward is always exactly
+1 or -1 (unchanged from Stage 1 behaviour), and that the shaping term is purely additive.

**Acceptance Scenarios**:

1. **Given** a pick that is within the spell/land budget, **When** the total reward is computed, **Then** the Stage 1
   component is exactly +1 and the total reward is 1 + shaping(t).

2. **Given** a pick that exceeds the spell/land budget, **When** the total reward is computed, **Then** the Stage 1
   component is exactly -1 and the total reward is -1 + shaping(t).

3. **Given** training is running, **When** the reward for any step is examined, **Then** it is the arithmetic sum
   of the Stage 1 budget term and the shaping term — no scaling, weighting, or replacement is applied to either.

---

### User Story 4 — Configurable Hyperparameters (Priority: P2)

A researcher wants to tune the behaviour of the recoverability shaping signal by adjusting two hyperparameters:
the urgency exponent (controlling when the signal becomes strong) and the temperature (controlling how sensitive
the signal is to raw changes in the ratio). These should be adjustable without code changes.

**Why this priority**: Hyperparameter tuning is essential for getting the shaping signal to work well in practice,
but a reasonable default exists for initial training runs.

**Independent Test**: Can be tested by running training with different hyperparameter values and verifying that
the reward values change in the expected direction.

**Acceptance Scenarios**:

1. **Given** the user specifies the urgency exponent, **When** training runs, **Then** the recoverability ratio uses
   the specified exponent in the denominator.

2. **Given** the user specifies the temperature, **When** training runs, **Then** the shaping signal uses the
   specified temperature in the bounding function.

3. **Given** no hyperparameters are specified, **When** training runs, **Then** the system uses reasonable defaults
   (exponent = 2, temperature = 1).

---

### Edge Cases

- What happens when no spells have been picked yet (step 0)? Pip demand is zero for all colors, ideal distribution
  is zero for all colors, imbalance is zero, recoverability ratio is zero, and the shaping term is zero. The first
  pick produces a near-zero shaping signal.

- What happens when remaining picks = 0 (terminal state)? The recoverability ratio equals the raw imbalance. No
  reward is computed at the terminal state.

- What happens with colorless spells ({C}, generic)? Generic mana pips ({1}, {2}, {X}) do not contribute to any
  color's pip demand. Colorless pips ({C}) contribute to the C bucket and generate demand for colorless sources.

- What happens with a single-color deck? The ideal distribution allocates all 17 sources to that one color. The
  model is rewarded for picking sources of that color and penalized for picking lands of other colors.

- What happens when a pick leaves the imbalance unchanged? A small negative shaping signal results because the
  denominator (remaining picks) shrank by one step — spending a pick without improving anything costs a little
  recoverability.

- What happens when the deck is in a deeply unrecoverable state (extreme imbalance, few picks left)? The bounding
  function ensures that any further worsening still produces a shaping value near -1 rather than saturating at 0.
  The total reward is bounded and does not cause runaway gradients.

- What happens when a spell pick shifts the ideal distribution? Picking a spell of an over-sourced color shifts the
  ideal for that color upward, which can reduce imbalance and produce a positive shaping signal — the reward correctly
  captures that adding demand for an over-supplied color is a good move.

## Clarifications

### Session 2026-04-06

- Q: Which specific bounding function for the shaping signal? → A: `tanh(x / temperature)`
- Q: Should recoverability ratio, imbalance, and shaping signal be logged as training metrics? → A: Log batch-mean shaping signal and batch-mean final imbalance on the existing print line. Recoverability ratio is not logged (intermediate value, not useful at batch granularity).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST compute the ideal mana source distribution at each step using the same formula as feature
  013: each color present in the deck gets a minimum floor of 2 sources, and the remaining sources out of 17 total
  are distributed proportionally to accumulated pip demand. When no spells have been picked (total pips = 0), the
  ideal distribution is zero for all colors.

- **FR-002**: System MUST track pip demand as a running total that updates whenever a spell is picked. The pip
  counting rules from feature 013 apply: single-color pips ({W}, {U}, {B}, {R}, {G}) count +1.0, {C} counts +1.0
  to the colorless bucket, Phyrexian pips count +0.5 to their color, hybrid pips count +0.5 to each of their two
  colors, and generic mana is ignored.

- **FR-003**: System MUST compute the imbalance at each deck state as the sum of absolute differences between the
  ideal and actual source counts across all colors. Both under-supply and over-supply of a color contribute to
  the imbalance symmetrically.

- **FR-004**: System MUST compute the recoverability ratio as imbalance divided by remaining picks raised to a
  configurable exponent (default: 2). When remaining picks is zero, the ratio equals the raw imbalance.

- **FR-005**: System MUST compute the per-step shaping signal as the reduction in the recoverability ratio from
  before the pick to after the pick, passed through `tanh(delta / temperature)` where delta is the reduction and
  temperature is a configurable parameter (default: 1). This bounds the signal to the range (-1, 1), preserves
  sign, and keeps small values approximately linear.

- **FR-006**: System MUST compute the total per-step reward as the sum of the Stage 1 budget reward (+1 for on-budget
  picks, -1 for off-budget picks) and the shaping signal. The total reward falls within the range (-2, 2).

- **FR-007**: System MUST NOT scale, weight, or replace the Stage 1 budget reward. The budget signal must remain at
  full strength (+1 or -1) and be purely additively combined with the shaping term.

- **FR-008**: System MUST expose the urgency exponent and temperature as configurable hyperparameters with defaults
  of 2 and 1 respectively.

- **FR-009**: System MUST assign each step its own computed reward value. The uniform end-of-episode reward assignment
  from feature 013 is replaced by the per-step reward for the mana-shaping component.

- **FR-010**: System MUST correctly handle multi-face cards (transform, split, adventure) by counting pips from all
  faces when computing pip demand, consistent with feature 013 behaviour.

- **FR-011**: System MUST display the mana cost of each non-land card before its name in the sample output pick list
  (e.g. `1. {U}{U} Counterspell`). Land cards are printed without a mana cost prefix. This applies to the
  `sealed sample` command output.

- **FR-012**: System MUST log batch-mean shaping signal and batch-mean final imbalance on the existing per-batch
  print line alongside the current batch scores and timing metrics. These two values provide the primary diagnostic
  signals for tuning: whether shaping is net-positive and whether mana balance is converging.

### Key Entities

- **Pip Demand**: Per-color running total of mana requirements from all non-land cards picked so far in an episode.
  Tracks W, U, B, R, G, and C (colorless) as six independent dimensions. Updates as each spell is picked.

- **Ideal Source Distribution**: The target number of mana sources per color at a given deck state, computed from
  the current pip demand with a 2-source-per-color floor and proportional allocation of remaining sources out of
  17 total. Recalculated after each pick.

- **Imbalance**: The L1 distance between the ideal source distribution and the actual source counts across all
  colors. A scalar value that is zero when the deck perfectly matches its ideal and increases with any deviation
  in either direction.

- **Recoverability Ratio**: A measure of how critical the current imbalance is given how many picks remain. Small
  when many picks remain (imbalance is recoverable), large when few picks remain (imbalance is urgent). Drives
  the automatic urgency scaling of the shaping signal.

- **Shaping Signal**: The per-step reward component that measures whether a pick improved or worsened the
  recoverability of the deck's mana distribution. Bounded to (-1, 1) and added to the Stage 1 budget reward.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The recoverability ratio and shaping signal computations produce correct results for all pick types
  (spell adding pip demand, land adding sources, dual lands, colorless spells) as verified by unit tests with
  hand-calculated expected values.

- **SC-002**: A model trained with the per-step recoverability reward converges to producing decks with batch-wide
  mana scores above 0.90, meeting the Stage 2 completion criterion.

- **SC-003**: Training with the per-step reward exhibits stable convergence — no entropy-collapse/recovery cycles
  observed over a sustained training run (as monitored through logged entropy and reward metrics).

- **SC-004**: The per-step reward produces measurably different values across steps within a single episode, confirming
  that credit assignment is step-specific rather than uniform.

- **SC-005**: Late-episode picks that worsen mana balance receive stronger negative shaping signals than early-episode
  picks with the same imbalance change, confirming the urgency amplification is working.

## Assumptions

- A working Stage 2 training loop from feature 013 exists, including the heuristic mana score computation, pip
  counting, ideal distribution calculation, and actual source counting. This feature modifies the reward assignment
  from uniform to per-step — it does not rewrite the underlying mana analysis logic.
- The Stage 1 budget reward mechanism (+1/-1 for on/off-budget picks) is already in place and functioning correctly.
- Action masking from Stage 1 guarantees that all episodes complete all 40 picks. The recoverability reward does not
  need to handle early termination.
- The ideal distribution formula, pip counting rules, and source counting rules are identical to those defined in
  feature 013. Any changes to those rules would propagate to this feature.
- The default hyperparameters (exponent = 2, temperature = 1) are reasonable starting points based on the mathematical
  analysis in the feature description, but may need tuning after observing initial training runs.
