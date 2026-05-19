# Future experiments

A holding pen for ideas surfaced during gen-2 work that aren't worth chasing
right now (typically because the binding constraint is data quality / volume,
not the model or the optimizer). Each entry: what it is, why it might help,
estimated magnitude of the effect, cost to try, and dependencies that might
unblock or favor it.

## One-shot deck picker (no search)

### Idea

Replace `GreedyDeckBuilder`'s simulated-annealing loop with a model
that picks all 23 nonland cards in a single forward pass. Pool in →
23-card deck out, no iterative search at inference.

### Architecture

A **per-card classifier with pool context**:

- Encoder (Set Transformer or vanilla transformer with no positional
  encoding) reads the full ~80-card pool jointly, so each card's
  output token is contextualized by the rest of the pool via
  self-attention.
- One shared `Linear(d_model, 1)` projects each token output to a
  single logit — 80 logits total from one linear layer, not 80
  separate heads. Same weights for every card; the variation comes
  from the per-token transformer outputs being different.
- Inference: sort logits, take top 23.

Per-card decisions are not independent in practice — when scoring a
4-drop, the model can see how many 2-drops are in the pool and
adjust. Synergies are learned implicitly through correlations
between pool composition and high-scoring picks, not through any
explicit model of the chosen deck.

#### Alternative architectures considered and deprioritized

- **Query-decoder.** 23 learned "deck slot" query vectors
  cross-attend to the pool; each query emits a card index. Trained
  with Sinkhorn-normalized soft assignment to enforce distinct picks
  (a differentiable relaxation of the assignment problem); inference
  resolves to a hard assignment with the Hungarian algorithm. More
  expressive when slots cleanly specialize but susceptible to query
  collapse, and the role-specialization advantage is weaker for MTG
  decks than for problems with hard slot structure (e.g., sports
  lineups). Deprioritized.
- **Autoregressive pointer.** Pick one card at a time, 23 steps,
  conditioned on already-picked cards. Tried in earlier experiments
  without success. Excluded.

### Training approach

**Primary plan: REINFORCE from random init**, using the frozen
gen4-512 scorer as the reward function. No SA, no per-card labels,
no separate data-generation pass.

Per training step:

1. Generate a sealed pool (sub-millisecond via `generate-pools`).
2. Picker forward pass → 80 logits → softmax → sample 23 cards.
3. Score the sampled 23-card deck with the frozen gen4-512 scorer
   (single scorer forward, no SA) → scalar reward.
4. Gradient: `∇ log π(deck | pool) × (reward − baseline)`.
5. Backprop.

The teacher scorer is the entire training signal.

#### How sampling works (training vs inference)

The picker's 80 logits describe a distribution over cards, not a
deterministic ranking. At inference, take argmax / top-23 →
deterministic best deck. At training, sample 23 cards
stochastically from that distribution → can produce N different
decks from the same pool by drawing N times.

Two equivalent sampling schemes:

- **Sequential categorical.** Softmax → 80 probabilities. Draw
  one card, mask, renormalize, draw the next. 23 draws = one
  sampled deck. Run 64 times for 64 decks. Sequential per
  sample.
- **Gumbel top-K.** Add independent Gumbel(0,1) noise to each
  logit, take top-23 by noisy logits. Mathematically equivalent
  to categorical sampling without replacement, but vectorized:
  replicate logits 64 times, add 64 noise vectors, top-23 each
  row in one batched op. Standard trick for batched discrete
  sampling on GPU.

Diversity across the 64 samples is controlled by softmax
sharpness — flat distribution early in training → diverse
samples; sharp distribution late in training → similar samples,
exactly the RL exploration-vs-exploitation schedule that's
typically wanted. An optional temperature knob `logits / T`
gives manual control (T > 1 flattens, T < 1 sharpens), and the
entropy bonus `−β · H(π)` in the loss penalizes overly sharp
distributions and indirectly keeps sample diversity up.

REINFORCE needs `log π(deck | pool)` for each sampled deck —
the gradient is `∇ log π × (reward − baseline)`. For sampling
without replacement this is the Plackett-Luce log-probability
(sum over picked cards of `log(p_picked / sum_of_still_available)`),
which has a closed form and is differentiable in the logits.
The 64 samples per pool serve two roles simultaneously: 64
reward observations averaged into the per-pool baseline for
variance reduction, and 64 different `log π` terms summed into
one gradient step.

#### Why REINFORCE despite its standard sample-inefficiency reputation

The conventional wisdom "REINFORCE is sample-inefficient, use
supervised when labels are available" assumes labels are cheap.
Here they aren't — SA distillation costs ~12s per labeled pool on
a 3060ti. The relative throughput:

- **SA distillation**: ~0.083 labeled pools per second
  (single-stream), each yielding 23 clean per-card signals.
- **REINFORCE** (GPU-batched, e.g. 256 decks per step): plausibly
  ~5,000–10,000 deck samples per second, each yielding one noisy
  reward signal.

Even discounting REINFORCE samples by 100× for variance/sparsity
penalty, informational throughput is ~1000× higher. Over ~33h of
single-GPU compute, supervised distillation gets ~10k labeled
pools; REINFORCE gets ~600M sampled decks.

#### Why the ceiling also favors REINFORCE

Supervised distillation has ceiling = "what SA picks." SA is a
local search bounded by single-card-swap neighborhoods from a
random init — it finds high-scoring decks under the scorer but
not necessarily the highest. REINFORCE optimizes against the
scorer directly with no local-search constraint; its ceiling is
"what scores highest under the scorer," which is ≥ SA's by
construction. The picker can in principle find better decks than
SA does.

The true ceiling under both regimes is the scorer itself — the
picker can never beat what the scorer would rate as the best
possible deck. But REINFORCE gets closer to that ceiling than
supervised distillation from SA.

### Risks and mitigations

**Cold start.** REINFORCE's real risk is not sample efficiency
but the early-training reward landscape: at random init, all
sampled decks score similarly badly, gradient is near-zero noise,
and learning may stall before lifting off. Mitigations:

- **Batch baselines.** Sample N (e.g., 64) decks per pool, use
  mean reward as the per-pool baseline. The relative ranking of
  sampled decks always gives signal even when absolute rewards
  are uninformative.
- **Entropy bonus.** Add `−β · H(π)` to the loss to keep
  exploration alive — prevents premature collapse to a narrow
  distribution before the picker has explored enough deck space.

**Reward hacking.** SA's local-search constraint keeps it in
"reasonable deck space" — it never strays far from its random
init via single-card swaps. REINFORCE has no such inductive bias.
If the gen4-512 scorer has blind spots (decks it rates highly but
that play poorly), the picker will find them. Detection:
periodically run a small Forge match-outcomes batch against
forge-best and watch for divergence between scorer-reported score
and actual win rate. Mitigation if observed: KL penalty against a
reference distribution (e.g., a small SA-supervised picker) to
keep the picker from drifting too far from the data manifold the
scorer was trained on. This is the standard PPO recipe from RLHF.

**Failure mode of the architecture itself**: incoherent decks
(top-23 spread across 5 colors, no curve). Diagnosable by
comparing color/CMC distributions of the picker's picks vs SA's.
Under REINFORCE this should be partially self-correcting — if
incoherent decks score badly under the scorer, the reward signal
pushes the picker away from them.

### Fallback approaches

If REINFORCE from random init fails to lift off in the first day
or two of training:

- **Option A: SA warmstart.** Train supervised picker on ~5k
  SA-labeled pools (~17h compute), use it as REINFORCE
  initialization with KL penalty against itself. The canonical
  RLHF recipe (SFT → PPO).
- **Option C: Pure supervised distillation.** Train against SA's
  per-card picks. Two loss shapes worth combining:
  - **Pairwise ranking loss.** For each pool, every
    (teacher-picked, teacher-rejected) pair contributes
    `max(0, margin − (logit_picked − logit_rejected))`. 23 × 57
    = 1311 pairs per pool. Directly optimizes the ranking
    structure that top-K selection cares about (absolute logit
    values don't matter at inference).
  - **Per-card marginal contribution as soft label.** For each
    SA-built deck, compute "how much does the deck's score drop
    if I remove card X and replace it with the best
    remaining-in-pool?" Train via MSE against those marginals.
    Encodes teacher decisiveness in addition to the binary
    in/out signal.
  - Optionally an auxiliary deck-quality regression head: one
    extra scalar output, pooled across the trunk's tokens,
    supervised against the teacher scorer's score of the
    SA-built deck. Doesn't directly teach picking but pushes
    the encoder to learn pool-level features that correlate
    with deck strength.

  Ceiling capped at SA's quality but training is stable and
  well-understood.

### Why it might help

The current 512-d scorer + SA produces decks in ~12s on a 3060ti
(`restarts=1`, `sa-temperature=0.8`, `sa-cooling=0.85`, ~40
iterations of meaningful temperature × ~1100 candidate moves per
iteration ≈ 50k expensive forwards per deck). For a mobile
deployment target of ≥100× speedup, a one-shot architecture is the
most direct path: one encoder forward over the ~80-card pool plus
a tiny head, roughly 5 orders of magnitude fewer GPU operations
than the current search.

### Estimated magnitude

**Speed: 100×–1000× faster than the current builder**, well past
the mobile target.

**Quality** under REINFORCE: ceiling = "best deck under the
scorer," which may meet or exceed SA's local-search output.
Working hypothesis: 70–80% win rate vs forge-best (gen4-512's
measured ~78% as the reference point). Lower than that if
cold-start mitigation leaves residual instability or reward
hacking is uncontrolled; higher if the scorer's optimum is in
fact above SA's local-search reach.

Quality under fallback Option C (supervised only): ceiling
80–90% of SA's edge over forge-best — some compositional
reasoning is lost since the model never directly evaluates its
picked deck.

### Cost

**Low-medium under the primary plan (Option B).** New model
architecture (smaller than current scorer), one new training loop
combining picker forward + scorer reward, ~33h of single-GPU
training. No SA dependency, no separate data-generation pass, no
labeled dataset. Ongoing maintenance: one extra model artifact
type (`models/sealed/picker/`) and CLI subcommand
(`sealed pick-decks` or a `--picker` flag on `build-decks`).

Option A adds the ~17h SA labeling pass. Option C adds a ~33h
SA labeling pass and a more conventional supervised training
loop in place of REINFORCE.

### Dependencies / when to revisit

Blocked on having a teacher scorer worth distilling — currently
satisfied (gen-4 512d is the strongest scorer and the speed
problem this targets). Worth scheduling once mobile deployment
becomes a concrete goal, or alongside the policy/value approach
(below) to compare "skip search" vs "make search cheaper"
head-to-head. Both experiments answer independent questions and
the better answer ships.

## Learned move proposer for SA (policy/value split)

### Idea

Keep the SA loop, but cut the dominant cost — the ~1100 expensive
Set Transformer evaluations per iteration — by adding a cheap
"proposer" model that ranks moves and lets the scorer evaluate only
the top-K. AlphaGo-style policy/value split: the policy proposes,
the value scores.

Per iteration today: enumerate ~1100 candidate moves, score all of
them with the 512-d Set Transformer, pick the best (or sample by
softmax). Per iteration with a proposer:

1. One forward of a small proposer model over `(current_deck,
   pool)` produces a distribution over candidate moves.
2. Take top-K (e.g. K=20) by proposer score.
3. Evaluate those K with the expensive Set Transformer.
4. Apply the best as today (greedy or SA-sample by score).

Proposer architecture: a smaller Set Transformer (~64–128 d, 2
layers) over the concatenated deck and pool, with a per-pool-card
"swap-in priority" head and a per-deck-card "swap-out priority"
head. Pair scores are the outer product; add-land and remove-land
get special heads. This is a comparative ranking task, not an
absolute-quality task, so the model can be much smaller than the
scorer.

Training signal: log the existing SA's full per-iteration scored
move set across thousands of pool builds, then train the proposer
to match the scorer's softmax over moves via KL distillation. Each
iteration produces ~1100 labeled samples (one per scored move),
versus 1 label per iteration for argmax-only distillation —
~1000× denser signal.

### Why it might help

Most of the per-iteration cost is the scorer's 1100 forwards.
Replacing 1100 expensive forwards with 1 cheap proposer forward +
20 expensive scorer forwards cuts scorer work ~50×, with the
proposer overhead small enough to leave a net ~30–50× wall-clock
gain per iteration. Combined with a smaller scorer trunk or INT8
quantization (independent levers), the 100× mobile target is in
reach without giving up the SA's compositional search.

Unlike the one-shot approach, this keeps SA's failure-mode
recovery: large K rescues occasional proposer misranks, and the
temperature-sampling escape from local optima still functions.

### Estimated magnitude

**Speed: 30–50× on scorer work**, multiplicatively stackable with
scorer quantization or distillation to a smaller scorer trunk. Net
100× plausible. **Quality: close to the current SA's** since the
scorer still evaluates the chosen move and the proposer is trained
on the scorer's own ranking — failure mode is degradation
proportional to how often the true best move falls outside the
top-K, which scales gracefully with K.

### Cost

Medium. Second model architecture (smaller than the scorer),
training pipeline, and an instrumented `GreedyDeckBuilder` run
that logs `(deck_state, pool, scored_moves)` tuples for the
distillation corpus. The data collection pass piggybacks on
existing builds — every SA-driven `build-decks` run can optionally
emit the trajectory file as a side-effect, no extra GPU time.

### Dependencies / when to revisit

Same prerequisite as the one-shot picker: a teacher scorer worth
distilling. The two proposals are explicitly **complementary, not
competing** — one-shot skips search entirely (highest ceiling on
speedup, lowest ceiling on quality), policy/value makes search
cheaper (lower ceiling on speedup, quality ≈ current SA). Worth
prototyping both: the one-shot's quality determines whether the
search is structurally necessary, and the policy/value approach is
the conservative fallback if one-shot can't match the teacher.