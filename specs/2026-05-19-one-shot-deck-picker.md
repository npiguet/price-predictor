# Goal

A one-shot sealed deck picker: a model that takes a sealed pool and returns 23
nonland cards in a single forward pass, no iterative search at inference. The
picker replaces `GreedyDeckBuilder`'s simulated-annealing loop with a single
encoder forward over the pool plus a per-card scoring head; basic lands are
filled in by the existing `compute_basic_lands` manabase heuristic.

Inference cost drops from the current builder's tens of thousands of scorer
forwards per deck to one picker forward per deck. The target deployment is
mobile-class hardware where the current builder's wall-clock cost is
prohibitive.

# Background

`GreedyDeckBuilder` runs ~40–200 iterations of simulated annealing, each
iteration scoring ~1100 candidate single-card moves with the gen4-512 Set
Transformer scorer. The dominant cost is the scorer's per-iteration batched
forward pass; the search itself is doing real compositional work (escaping
4-color local optima, coordinating splash + dual-land moves) and cannot be
shortened without quality loss.

A one-shot architecture sidesteps this entirely. Instead of a teacher scorer
guiding an iterative search at inference, a learned picker emits all 23 picks
from a single pool forward. The teacher scorer's role moves from inference-
time guidance to training-time reward signal — the picker learns to maximize
the scorer's evaluation of its sampled decks.

The picker does not replace the scorer. The scorer remains the deck-quality
oracle used in `match-outcomes`, `evaluate-scorer`, and as the picker's
training reward. The picker is a new artifact in `models/sealed/picker/`,
trained against a frozen scorer.

# Specification

This section is prescriptive. Everything below is what an implementer builds;
rationale and tradeoffs are deferred to later sections.

## 1. Architecture

The picker is a **per-card classifier with pool context**. It consumes the
same per-card embedding format as the scorer (the `.npz` cache populated by
`sealed encode-cards`), so any encoder the scorer accepts also feeds the
picker.

Forward pass for a pool of N cards (typically N ≈ 60–90, depending on
set; the pool file already excludes basic lands per the
`generate-pools` output format):

1. Stack the N card embeddings into a single `(N, embedding_dim)` tensor,
   where `embedding_dim` matches the scorer's `ScorerConfig.d_model` (the
   width of the `.npz` cache — pooled encoder output plus the trailing
   `FEATURE_COUNT` deterministic features).
2. The picker's internal width `d_model` defaults to `embedding_dim`,
   in which case the transformer operates directly on the cache
   embeddings with no projection layer. A larger sealed encoder (or a
   wider `.npz` cache) therefore produces a proportionally wider — and
   more capable — picker by default. The default derivation happens
   at startup from the `--scorer-checkpoint`'s `ScorerConfig`. A
   `--d-model` CLI flag (§ 4.1) overrides this default for cases
   where the picker should be wider or narrower than its input; when
   set to a value other than `embedding_dim`, a single
   `Linear(embedding_dim, d_model)` projection layer is inserted
   between the input embeddings and the first SAB layer.
3. Run a stack of `n_layers` **Set Attention Blocks** (SAB, Lee et al.
   2019 — the same primitive used by the scorer and the sealed encoder,
   chosen for consistency with the rest of the project's set-input
   transformers). All N tokens attend to each other; no positional
   encoding, since the pool is an unordered set.
4. Apply a shared per-card head `Linear(d_model, 1)` to each of the N
   token outputs, producing N scalar logits.

**Output shape**: `(N,)` — one logit per pool card. Logits are unbounded
real numbers; their relative ordering is what matters.

The picker scores every pool card. Pool cards are either spells
(nonland) or nonbasic lands — both are rankable by the picker.
**Basic lands are not in the pool input** (excluded by
`generate-pools` upstream) and are not scored by the picker. They are
added after picking is complete by `compute_basic_lands` based on the
chosen spells' color requirements.

### 1.1 Pick decomposition: the spell quota

A finished sealed deck is 40 cards: 23 spells + some number of
nonbasic lands (typically 0–6 in practice, since a 6-booster pool
rarely contains more than 6 nonbasic lands total) + basic lands
filling the remainder. The picker's top-K selection accumulates
spells until the 23-spell quota is met; nonbasic lands appearing
above the 23rd spell in the sorted logit order are taken along the
way:

```
chosen_spells = []
chosen_nonbasic_lands = []
for idx in argsort(logits, descending=True):
    if len(chosen_spells) >= 23:
        break                                         # spell quota filled
    if is_land_embedding(pool[idx]):
        chosen_nonbasic_lands.append(idx)
    else:
        chosen_spells.append(idx)
chosen = chosen_spells + chosen_nonbasic_lands        # 23..29-ish cards
basic_lands = compute_basic_lands(card_texts_of(chosen))  # fills to 40
```

The loop terminates when the spell quota is met, not when a fixed
top-K is reached. The total number of cards the picker selects is
variable: 23 when no nonbasic land is picked, plus one extra pick
per nonbasic land that scored above the 23rd spell.

`is_land_embedding` operates on the embedding's land-flag slot in
the deterministic-feature block, already used by
`GreedyDeckBuilder._partition_pool` and identical in semantics.

The same `compute_basic_lands` call works regardless of how many
nonbasic lands were chosen: it computes `40 − len(chosen)` basic
lands, distributed across colors by the spell mana-pip histogram
(lands have no mana cost and are silently skipped by the pip
counter, so passing the combined `chosen` list is correct).

### 1.2 Auxiliary pool-quality head

In addition to the per-card head (§ 1 step 4), a second head produces a
single scalar prediction of the **mean reward** the picker achieves on
this pool — i.e., the average scorer score across decks sampled from
the picker's distribution at this pool:

```
pool_quality_pred = Linear(d_model, 1)(mean_pool(token_outputs))
```

Trained against the per-pool mean reward computed during the same
training step (§ 3.3 already computes this quantity as the policy-
gradient baseline; the same value is the aux head's MSE target,
detached so the aux loss does not flow back into the rewards). The aux
head is discarded at inference.

**Why include the aux head.** The aux head is structurally the same
shape as the existing gen4-512 scorer: card embeddings → SAB trunk →
scalar quality output. The scorer architecture has been validated on
deck quality from match outcomes in this exact project; the aux head
applies the same architectural pattern to a closely related task. The
substantive difference is that the aux head reasons over the ~80-card
pool instead of a 23-card deck — and to predict the pool's expected
reward accurately, the head's trunk must internally model *which
cards from this pool would be picked*. That subset-selection
reasoning is exactly the picker's task, so the representation pressure
the aux head exerts on the shared SAB trunk is directly aligned with
what the per-card head needs to learn. A trunk that can implicitly
score "the best 23-of-80 subset" is the same trunk that can rank
cards so the top-23 form a good subset.

(Prior project-internal experience also points the same direction:
the MLM auxiliary head used by `train-encoder` measurably improved
sealed encoder quality in prior generations. But the structural
argument above — same architecture as the scorer, same task family,
directly aligned representation pressure — is the load-bearing one;
the MLM experience is corroborating evidence, not the central
reason.)

**Known caveat.** The aux target `rewards.mean(dim=1)` is a
non-stationary on-policy quantity: it shifts as the picker improves.
This is the standard concern with value functions in
REINFORCE-with-baseline / A2C-style training; we're betting the
target moves slowly enough relative to optimizer steps for the head
to track it. Not free of risk, but a well-understood one.

**Default.** `--aux-weight` defaults to `0.1` (§ 4.1). The flag is
fully ablatable; setting it to `0` is documented as a reasonable
comparison run if post-training analysis suggests the aux head is
fighting the policy gradient rather than helping. The aux head
parameters are always *present* in the model; only the loss-term
coefficient varies.

## 2. Inference

Deterministic. Single forward pass per pool, followed by the pick
decomposition walk from § 1.1.

```
logits = picker(pool_embeddings)                            # shape (N,)
chosen = greedy_pick_decompose(logits, pool)                # § 1.1 loop
basic_lands = compute_basic_lands(card_texts_of(chosen))
full_deck = chosen + basic_lands                            # 40 cards
```

No sampling, no Gumbel noise — the loop walks the deterministic
sorted-logit order and applies the spell-quota / land-cap rule. The
number of cards in `chosen` is variable (23 spells + 0..17 nonbasic
lands); `compute_basic_lands` fills the rest to 40.

## 3. Training pipeline

**Primary plan: REINFORCE from random init**, using the frozen scorer
as the reward function. No SA, no per-card labels, no separate data-
generation pass.

**Why vanilla REINFORCE, not PPO.** The spec uses
REINFORCE-with-baseline rather than PPO or any other importance-
sampled / clipped variant. This is a deliberate trade of training
throughput for implementation simplicity. PPO would add a ratio
computation, a clip term, possibly an EMA reference distribution,
and the bookkeeping to do K gradient steps per sample batch — all of
which give a 4–10× sample-efficiency multiplier in standard RL
benchmarks. We are knowingly leaving that multiplier on the table,
on the grounds that the binding constraint for this project is "does
training lift off at all" (a question REINFORCE answers as well as
PPO does) rather than "does training converge with minimum compute."
Total training budget is expected to be a few hours of single-GPU
time; the simpler training loop pays for itself in implementation
and debugging effort. If REINFORCE-from-random stalls and the
contingency plan's Option A (SA warmstart + KL-regularized
REINFORCE) is adopted, PPO becomes a natural follow-up because the
KL-regularization complexity budget will have already been accepted
at that point.

### 3.1 Per-step loop

For each training step:

1. Pull `batch_size` sealed pools from the pool source (see "Pool
   source" below — amortized sub-millisecond per pool, but never
   synchronously spawned per-step).
2. For each pool, run the picker forward → N logits.
3. Sample `N_samples` decks per pool from the picker's distribution
   (see § 3.2). Total sampled decks per step: `batch_size × N_samples`.
4. Score every sampled deck with the frozen scorer in one batched
   forward → reward tensor of shape `(batch_size, N_samples)`.
5. Compute the per-pool baseline (§ 3.3) and the policy-gradient loss
   (§ 3.4).
6. Backprop and AdamW step on the picker parameters only. The scorer
   is frozen throughout.

**Pool source.** Pools are streamed from a pre-generated file
specified by the required `--pools-path` CLI flag (§ 4.1). A
one-shot `sealed generate-pools` run produces the file ahead of
training (e.g., 100k+ pools). The same corpus can be reused across
multiple training runs.

Pools are shuffled at the start of each epoch, and one epoch =
one full pass through the pool file. Training the picker against
a fixed pool corpus rather than freshly-generated pools removes
the ~15s Forge JVM startup cost from the training loop entirely
and gives a natural, reproducible definition of "epoch" that does
not require a separate `--steps-per-epoch` knob.

The encoder used to produce the `.npz` cache is also frozen at this
stage — the picker is trained against fixed card embeddings. A later
Phase B (analogous to scorer Phase B) could jointly fine-tune the
encoder with the picker, but is out of scope for the initial spec.

### 3.2 Sampling

Decks are sampled by **sequential categorical draws without
replacement** over the full pool (spells + nonbasic lands), with
the § 1.1 spell-quota stopping rule:

```python
probs = softmax(logits / temperature)
chosen_spells, chosen_lands = [], []
remaining = set(range(N))
while len(chosen_spells) < 23 and remaining:
    pick = categorical_sample(renormalize(probs, mask=remaining))
    remaining.remove(pick)
    bucket = chosen_lands if is_land_embedding(pool[pick]) else chosen_spells
    bucket.append(pick)
```

Each sampled deck takes ~23–29 iterations (one per pick). The
sampled "deck" passed to the scorer is `chosen +
compute_basic_lands(chosen)`.

The loop is implemented on GPU, batched across the `N_samples` and
`batch_size` dimensions: each iteration is one
`torch.multinomial` call (and one mask update) operating on the
full `(batch_size × N_samples, N)` probability tensor. ~25
vectorized GPU iterations per training step total — not per
sample. Per-sample Python loops dispatching individual GPU calls
are explicitly not what is meant. CPU sampling is not used; the
logits are already on GPU after the picker forward, and the
`log_prob` term (§ 3.5) needs to be on GPU for the backward pass
anyway.

A Gumbel-top-K vectorized sampler was considered and rejected.
Under the spell-quota walk the trajectory length is data-
dependent, so Gumbel would sort the full pool rather than taking a
fixed top-K — losing the vectorization advantage that makes Gumbel
attractive elsewhere. Since the sequential loop is bounded to ~25
short iterations per sampled deck, and is distributionally
identical to the Gumbel walk, the spec specifies the sequential
sampler as the single path.

Temperature defaults to 1.0; the picker's own logit scale controls
diversity by default. The `--temperature` flag is provided as an
exploration knob.

### 3.3 Per-pool baseline

For each pool, the baseline is the mean reward across its `N_samples`
sampled decks:

```python
baseline[i] = rewards[i].mean()          # shape (batch_size,)
advantage[i, j] = rewards[i, j] - baseline[i]
```

The advantage signal is the per-deck reward relative to its sibling
samples from the same pool. The picker is pushed toward decks that
beat the pool's average reward and away from decks that underperform.
This is the standard REINFORCE-with-baseline variance reduction; it
is essential here because absolute rewards across pools vary
substantially (a strong pool's worst deck may score above a weak
pool's best deck).

### 3.4 Loss

```
policy_loss  = -(advantage.detach() * log_prob).mean()
entropy_loss = -entropy_coef * entropy.mean()
aux_loss     = mse(pool_quality_pred, rewards.mean(dim=1).detach())
total_loss   = policy_loss + entropy_loss + aux_weight * aux_loss
```

Where:

- `log_prob` is the Plackett-Luce log-probability of the sampled
  deck under the current picker distribution (§ 3.5).
- `entropy` is the entropy of the picker's softmax over the
  pool, computed per pool, averaged over the batch.
- `entropy_coef` is the current value of the entropy coefficient,
  driven by the schedule described under § 4.1 "Entropy schedule":
  held constant at `--entropy-coef` (default `0.01`) until val
  reward shows monotonic improvement for `--entropy-decay-after`
  consecutive epochs, then decayed multiplicatively as val reward
  plateaus.
- `pool_quality_pred` is the auxiliary head's output (§ 1.2),
  one scalar per pool in the batch.
- `rewards.mean(dim=1)` is the per-pool mean reward across the
  `N_samples` sampled decks, the same quantity used as the
  policy-gradient baseline (§ 3.3). `.detach()` prevents the aux
  loss from flowing back into the reward computation.
- `aux_weight` is a CLI hyperparameter; default `0.1`.

`advantage` is `.detach()`-ed because it acts as a scalar weight
on the log-probability gradient, not as a quantity to differentiate
through. Backprop flows through `log_prob`, `entropy`, and
`pool_quality_pred`.

### 3.5 Plackett-Luce log-probability

For a sampled deck — between 23 picks (no nonbasic land chosen)
and 40 picks (all 17 land slots filled) per the § 1.1 walk —
drawn without replacement from a softmax over the pool, the
log-probability factorizes sequentially:

```
log P(deck | pool) = sum over picks of (logit_picked - logsumexp(remaining_logits))
```

Concretely, at pick step k (with `picked_so_far` already removed
from the pool):

```
remaining_logits = logits.clone()
remaining_logits[picked_so_far] = -inf
log_prob_step_k = logits[pick_k] - logsumexp(remaining_logits)
```

Summing across all picks gives the deck's log-probability. The sum
is differentiable in `logits`, which is what carries the gradient
back into the picker's parameters. The variable trajectory length
is not a problem: each step contributes its own term, and the sum
is taken over however many picks the walk produced.

The sequential sampler (§ 3.2) draws exactly from this distribution.

### 3.6 Cold-start sanity check (recommended pre-training procedure)

A documented manual diagnostic to run **once before the first full
training attempt against a given scorer checkpoint**. Not a CLI
subcommand and not a CLI flag — a one-off check expected to be
written as a ~30-line ad-hoc script and re-run only if the scorer
changes.

**What the check measures.** Whether the picker's within-pool reward
std at random init is large enough relative to the scorer's
discriminative range on similar-quality decks. The policy gradient
signal in § 3.4 is driven by the advantage
`rewards[i, j] - rewards[i].mean()`; if the per-pool reward std at
random init is much smaller than the score range the scorer reports
across nominally-similar decks, the picker's samples are
indistinguishable from the scorer's perspective, the advantage signal
is uninformative, and REINFORCE will stall regardless of how many
training steps it gets.

The scorer in `eval()` mode is deterministic — scoring the same deck
twice gives the same output — so there is no "intrinsic noise floor"
to compare against. The meaningful reference is the scorer's
observed score variation on decks of similar quality.

**Procedure.**

1. Compute reference statistics from the existing 70k-match
   corpus, one-off:
   - Score every deck in `match-outcomes-all.txt` with the gen4-512
     scorer (or load cached scores if available).
   - **Within-band std**: filter to one method tag — `random` is
     the relevant one, ~5500 deck-sides — and report the std of
     scores within that band. This is the scorer's natural
     discriminative spread on nominally-similar (random-built)
     decks: `sigma_random_band`.
   - **Cross-band gap** (for sanity): the difference of mean
     scores between `forge-best` and `random` decks. This is
     "what the scorer thinks a clearly-better deck is worth":
     `delta_forge_vs_random`.
2. Instantiate a fresh random-init picker (same architecture as
   intended for the full run).
3. Sample 100 pools from the intended training corpus.
4. For each pool, draw 1024 sampled decks via the § 3.2 sampler.
   This is essentially the planned full-training inner loop with
   `N_samples = 1024` and `batch_size = 100`, run for exactly one
   step.
5. Score every sampled deck with the frozen scorer; compute the
   per-pool reward std across the 1024 samples. Report the
   distribution across the 100 pools (median, p25, p75).
6. **Gating decision.** Compare the typical (median across pools)
   per-pool reward std at random init against `sigma_random_band`:
   - If the picker's per-pool std is on the same order as
     `sigma_random_band` (within ~3×, either direction), the
     picker's random-init samples span the scorer's natural
     within-band variation. Useful gradient signal exists; the
     full training run is justified.
   - If the picker's per-pool std is much smaller than
     `sigma_random_band` (e.g., < 1/10), the picker's samples are
     compressed into a region where the scorer barely
     differentiates them. REINFORCE will struggle to learn from
     this signal; consider the contingency plan's Option A
     instead of burning the full training run.
   - The `delta_forge_vs_random` reference is for interpreting
     the result, not gating it: if `sigma_random_band` itself is
     a large fraction of `delta_forge_vs_random`, the scorer is
     "fuzzy" within the random band and even a modest per-pool
     std is informative; if `sigma_random_band` is tiny relative
     to the gap, the scorer is sharp within bands and the picker
     needs proportionally larger per-pool std to be useful.

**Why this isn't a CLI feature.** The check is run a handful of
times across the project's lifetime (once per scorer architecture or
encoder choice). The CLI surface area, tests, and documentation that
would come with a `sealed probe-picker` subcommand or `--probe-only`
flag are not worth the ongoing maintenance burden for something that
runs so rarely. Treat this the same way as "before training, make
sure the pools file isn't empty" — a sanity procedure, not
infrastructure.

**N_samples ablation rides on this check.** The probe naturally
generates 1024 sampled decks per pool, which is enough to estimate
the variance reduction curve of the per-pool baseline as `N_samples`
varies. Bin the 1024 samples into groups of size N and report the
remaining std of the per-pool baseline as a function of N — this is
exactly the information needed to set `--n-samples` for the full
training run.

## 4. CLI

### 4.1 `train-picker` (new subcommand)

Trains a one-shot picker from scratch using REINFORCE against a
frozen scorer.

| Flag | Default | Meaning |
|------|---------|---------|
| `--scorer-checkpoint` | `models/sealed/scorer/latest.pt` | Frozen scorer used as the reward function. The picker's default input width is derived from this checkpoint's `ScorerConfig.d_model`. |
| `--cards-path` | `output/cardsfolder/` | Path to the `.npz` embedding cache. |
| `--pools-path` | _(required)_ | Pre-generated pools file (produced by a prior `sealed generate-pools` run). The training loop shuffles and streams from this file; one full pass = one epoch (§ 3.1 "Pool source"). |
| `--d-model` | _(derived = `embedding_dim`)_ | Picker internal width. When unset, defaults to the cache embedding width and no input projection is inserted (§ 1 step 2). When set to a value other than `embedding_dim`, a single `Linear(embedding_dim, d_model)` projection layer is inserted ahead of the first SAB layer, letting the picker be wider or narrower than its input. |
| `--n-layers` | `4` | Number of SAB layers. |
| `--n-heads` | `8` | Attention heads per layer. `d_model` (§ 1 step 2) must be divisible by this; the run fails fast at startup if not. |
| `--ff-dim` | `4 × d_model` | Feed-forward dimension. The default is computed from the resolved `d_model` at startup. |
| `--dropout` | `0.0` | Dropout in transformer layers. |
| `--aux-weight` | `0.1` | Coefficient on the auxiliary pool-quality MSE loss (§ 1.2, § 3.4). Setting this to `0` disables the aux loss while keeping the head parameters in the model; this is the documented ablation comparison. |
| `--batch-size` | `16` | Pools per gradient step. |
| `--n-samples` | `64` | Decks sampled per pool per step. Default is a placeholder, expected to be tuned by the cold-start sanity check (§ 3.6); the 16×64 vs 4×256 tradeoff (more pools/epoch vs more samples/pool for variance reduction) should be empirically resolved before a long training run. |
| `--temperature` | `1.0` | Softmax temperature for sampling. |
| `--entropy-coef` | `0.01` | Coefficient on the entropy bonus. Held constant until the val reward shows monotonic improvement for `--entropy-decay-after K` consecutive epochs, then decays toward 0 as val reward plateaus (see "Entropy schedule" below the table). The starting value itself is what `--entropy-coef` controls. |
| `--entropy-decay-after` | `5` | Number of consecutive epochs of monotonic val-reward improvement required before the entropy coefficient begins decaying from its initial value. Avoids premature decay when training is still in cold-start. |
| `--lr` | `3e-4` | AdamW learning rate. |
| `--max-grad-norm` | `1.0` | Per-parameter-group gradient norm cap. |
| `--epochs` | `100` | Maximum number of epochs. One epoch = one shuffled pass through the training portion of `--pools-path` (i.e., the file minus the held-out validation fraction). |
| `--val-fraction` | `0.2` | Fraction of `--pools-path` reserved as the validation set, taken from the front of the file. Excluded from training shuffles and reused identically across epochs (see "Validation reward during training" under Evaluation). Matches the existing convention used by `train-encoder` and `train-scorer`. |
| `--patience` | `10` | Early-stop after this many epochs without validation improvement. |
| `--resume` | _(none)_ | Continue a stopped run from this checkpoint. Loads picker weights, optimizer state, epoch counter, and best-validation-reward metadata. Architecture flags (`--n-layers`, `--n-heads`, `--ff-dim`, `--dropout`) are forbidden when this is set — architecture is inherited from the checkpoint. Mutually exclusive with `--picker-checkpoint`. |
| `--picker-checkpoint` | _(none)_ | Bootstrap a fresh run from this checkpoint's picker weights only. Optimizer state, epoch counter, and validation metadata are discarded. Architecture flags are forbidden (architecture is inherited from the checkpoint). Mutually exclusive with `--resume`. (Also the mechanism that would enable the Option A warmstart contingency — see "Contingency plans" — but is generally useful for any prior-picker bootstrap.) |
| `--kl-coef` | `0.0` | Coefficient on the KL penalty against the `--picker-checkpoint` reference distribution. `0.0` disables the penalty entirely (the default for a fresh REINFORCE-from-random-init run). Non-zero values are the Option A warmstart configuration; require `--picker-checkpoint`. |

**Entropy schedule.** The entropy coefficient stays constant at
`--entropy-coef` while training is still in cold-start, then decays
once the policy has demonstrated learning. Concretely: each epoch
records val reward; once val reward has improved monotonically for
`--entropy-decay-after` (default 5) consecutive epochs, the
coefficient is multiplied by 0.9 at the end of every subsequent
epoch in which val reward fails to improve on its previous best
(i.e., decay tracks val-reward plateaus, not wall-clock or step
count). This avoids the failure mode of a wall-clock-tied decay
collapsing the policy distribution before lift-off, while still
letting entropy fall away once the picker has converged and is in
fine-tuning mode.

### 4.2 `pick-decks` (new subcommand)

The inference counterpart to `build-decks`. Reads a pools file,
runs the picker once per pool, fills basic lands via
`compute_basic_lands`, writes `generated-decks.txt` in the existing
format (`LABEL;SET_CODE;Card1|Card2|...|Card40` per line).

| Flag | Default | Meaning |
|------|---------|---------|
| `--pools-path` | _(required)_ | Input pools file. |
| `--picker-checkpoint` | `models/sealed/picker/latest.pt` | Picker weights. |
| `--cards-path` | `output/cardsfolder/` | `.npz` embedding cache. |
| `--label` | _(required)_ | Generation-method tag (mirrors `build-decks --label`). Used as `method_A`/`method_B` when these decks feed `match-outcomes`. |
| `--output` | `output/sealed/generated-decks.txt` | Output deck file. |
| `--resume` | off | Append-and-skip semantics matching `build-decks --resume`. |

Output format is identical to `build-decks`, so picker-generated
decks are drop-in inputs for `match-outcomes --side-a-decks` /
`--side-b-decks` and any other downstream consumer.

## 5. Model artifacts

Checkpoints saved to `models/sealed/picker/`:

```
models/sealed/picker/
  {timestamp}.pt
  latest.pt
```

Each checkpoint contains:

- `model_state_dict` — picker weights only. Scorer and encoder
  weights are not embedded; the picker is paired at inference time
  with whichever scorer's `.npz` cache produced its training input
  width.
- `config` — `PickerConfig` (architecture hyperparameters,
  including the input width derived from the training scorer).
- `epoch`, `best_val_reward`, training metadata.

Picker checkpoints have no `encoder_state_dict` in the initial
spec — Phase A only. A future Phase B variant analogous to scorer
Phase B can carry encoder weights.

# Risks and mitigations

## Cold start

REINFORCE's failure mode at random init is a degenerate reward
landscape: if sampled decks score similarly, the per-pool
advantage signal is too small relative to the scorer's
within-band discriminative range, the gradient is uninformative,
and learning may stall before lifting off.

The **cold-start sanity check** (§ 3.6) is the spec's primary
defense: it measures the within-pool reward std at random init
directly and gates the full training run on that signal being
non-degenerate. The check runs a few hours of GPU as a manual
pre-training procedure and answers definitively whether
REINFORCE-from-random-init has a chance on this scorer.

Three secondary mitigations are built into the primary training
loop, all of which the cold-start check measures the efficacy of:

- **Per-pool baseline** (§ 3.3) — gives signal even when absolute
  rewards across pools are uninformative.
- **Multi-sample per pool** (`--n-samples`) — relative
  ranking of sibling samples carries information that single
  samples cannot.
- **Entropy bonus** (§ 3.4, § 4.1 "Entropy schedule") — keeps the
  picker's distribution from collapsing prematurely before
  exploration has surfaced high-reward regions of deck space.

If the cold-start check fails (per-pool reward std much smaller
than the scorer's within-band std on similar-quality decks), the
contingency-plans section below
describes Option A — the SA warmstart — which bootstraps the
picker into a region of policy space where the reward landscape
is no longer flat. Note that Option A is not part of this spec;
pivoting to it requires a new spec round.

## Reward hacking

The picker's training loss is the scorer's score, not actual
match win rate. SA's local-search constraint keeps it in
"reasonable deck space" — it never strays far from random init
via single-card swaps. The picker has no such inductive bias and
will exploit scorer blind spots if they exist.

Forge matches can't be the in-training detection mechanism. With
~30-minute matches and ~50-match batches needed for a directional
signal, every Forge check would take longer than the full expected
training run. The detection plan accordingly splits into a
**cheap in-training audit** that runs every epoch and a
**definitive end-of-training validation** (described under
Evaluation) that runs once.

**1. Per-epoch cross-scorer agreement.** The training reward is
the gen4-512 scorer, but reward-hacking failures are specific to
the gen4-512 model's quirks. Other trained scorers in the
project (gen3-256, gen3-128) have different architectures and
different blind spots; if the picker is genuinely producing
better decks, all scorers should agree they are better, and if
the picker is hacking gen4-512 specifically, the other scorers
will lag or diverge. Each epoch, score the picker's
deterministic-inference decks (the same set used for the
validation reward) with both gen4-512 (training reward) and
gen3-256 (auditor). Track `corr(gen4_score, gen3_score)` on
those decks across epochs. The baseline correlation can be
pre-computed once on the existing 70k-match corpus (just run
both scorers on every deck in `match-outcomes-all.txt` and take
the Spearman rank correlation); the alert threshold is "picker's
per-epoch correlation drops more than X std below the baseline."

This is essentially ensemble disagreement as a hacking detector.
Cost: one extra scorer forward per validation epoch. Negligible.

**2. Per-epoch distributional sanity checks.** The picker's decks
should look like sealed decks. Track per-epoch distributions on
the validation decks: color count, CMC histogram, creature
count, type balance. Compare against the established distribution
of the 70k-match corpus split by build method (forge-best,
gen3-256, etc., all already in `match-outcomes-all.txt`). If
the picker drifts to a region no human or Forge method would
produce — for example, 5-color decks averaging CMC 1.2 — that's
a soft reward-hacking signal even if cross-scorer agreement
holds. Cost: numpy aggregation on the same val-batch decks.
Free.

**3. End-of-training Forge validation.** See "End-of-training
Forge validation" under Evaluation. The few hours of Forge time
get spent once per completed training run rather than per
checkpoint, providing the definitive answer to "does the picker
actually win matches." Reward-hacking that slipped past the
in-training audits surfaces here.

**Mitigation if the in-training audits alert.** Restart training
from the most recent checkpoint with a larger `--entropy-coef`
and possibly an earlier checkpoint as the starting point. The
contingency-plans section's Option A (KL penalty against a
reference picker) is the heavier alternative if entropy
regularization alone does not stop the drift, but adopting it is
a spec-level decision rather than a flag flip.

## Incoherent decks

The picker has no explicit deck-coherence objective. Its top-23
might be 23 individually-strong cards in 5 colors with no
fixing — a card-level optimum that's a deck-level failure.

Under REINFORCE this is partially self-correcting: incoherent
decks score badly under the scorer, so the reward signal pushes
the picker away from them. The auxiliary pool-quality head
(§ 1.2) provides additional representation pressure on the SAB
trunk to learn pool-level features that correlate with deck
strength. The mitigation beyond that is mostly diagnostic:
compare the picker's color/CMC distributions against the SA
teacher's, and watch for persistent incoherence as a sign that
`--aux-weight` should be raised.

# Evaluation

Two evaluation regimes, both reusing existing infrastructure.

## Validation reward during training

At the end of every epoch, run the picker deterministically (the
§ 1.1 walk over argmax-sorted logits, no sampling) on a fixed
held-out set of pools, score the resulting decks with the frozen
scorer, and report mean reward. Best-checkpoint selection uses
this metric.

The validation pool set is carved off the front of
`--pools-path` at training start: the first `--val-fraction`
(default `0.2`, matching `train-encoder` and `train-scorer`) of
the file becomes the held-out validation set and is excluded
from the training shuffle. They are reused identically across
epochs to control variance. The remainder of the file is the
training corpus.

## End-to-end win rate vs forge-best

The canonical evaluation. Reuses `match-outcomes` with picker-built
decks as side A and forge-best as side B:

1. `sealed pick-decks --pools-path <fresh-pools> --label picker-{tag}`
   produces a `generated-decks.txt`.
2. `sealed match-outcomes --side-a-decks generated-decks.txt`
   runs Forge AI matches.
3. Compute aggregate win rate by `method_A == picker-{tag}` rows.

Sample size for a stable measurement: ≥ 200 matches at `--best-of 7`.
This is the same metric the gen4-512 scorer is evaluated on, so
picker results are directly comparable to scorer + SA results from
prior generations.

## End-of-training Forge validation

The definitive guard against reward-hacking that the in-training
cross-scorer audit (§ Reward hacking) cannot rule out on its own.
Forge matches are too expensive to run in-loop (a ~50-match batch
takes longer than the full expected training run), so the Forge
check is amortized to **once per completed training run**, on the
final and top-K-by-val-reward checkpoints rather than every epoch.

Procedure:

1. After training completes (early stop or `--epochs` reached),
   identify the final checkpoint and the top 2–3 checkpoints by
   validation reward.
2. For each candidate checkpoint, run the End-to-end win rate vs
   forge-best evaluation above (≥ 200 BO7 matches, fresh pools).
   3. **Decision rule.** If the best checkpoint's win rate vs
      forge-best is meaningfully above 50% and consistent with the
      validation reward ranking (i.e., higher val reward → higher
      actual win rate), the training run succeeded. If val reward
      was good but Forge win rate is at or below 50%, reward hacking
      slipped past the in-training audits and the run is discarded;
      the picker is not deployed and the failure mode is recorded for
      tuning the in-training audit thresholds on subsequent runs.

This sits structurally alongside the cold-start sanity check
(§ 3.6): one is a *pre*-training gate, this is a *post*-training
gate. Both are manual procedures rather than CLI infrastructure,
because both run a handful of times across the project's lifetime
and the existing `sealed match-outcomes` machinery already does
the heavy lifting.

## Per-pool comparison against SA

For a fixed pool set, build a deck with both the picker (one
forward) and `build-decks` + the gen4-512 scorer (12s of SA per
pool), then play picker-deck vs SA-deck head-to-head via
`match-outcomes`. This isolates the search-vs-policy question
within a controlled set, and is robust to set-mix shifts that
confound aggregate win-rate comparisons. Recommended sample size:
~50 pools, `--best-of 7`.

## Inference latency

Wall-clock time per deck on a fixed reference workload, measured
separately from quality. Same machine the scorer + SA baseline was
measured on, to support the speedup claim in the goal statement.

# Contingency plans (not part of this specification)

**The recipes below are NOT what this spec authorizes building.**
They are documented future-state options to consider **only if**
the primary plan (§ 3, REINFORCE from random init) fails to lift
off after a reasonable initial training attempt. The implementer
must build the spec as written first; any pivot to these
alternatives requires a new specification round, not just a flag
flip in this one.

The primary spec's CLI surface (§ 4.1) already includes
`--picker-checkpoint` and `--kl-coef` as generic capabilities
(resume / bootstrap / KL-regularize). These flags happen to be
exactly what Option A below would require — but the supervised
pretraining run that would produce a checkpoint suitable for
Option A is itself out of scope and would need its own spec. The
flags' inclusion in § 4.1 is not an authorization to build
Option A; they exist because they are generally useful for any
prior-picker bootstrap or KL-regularized continuation, not just
this contingency.

This entire section is preserved for the same reason a design
document keeps "rejected alternatives" — so that the contingency
shapes and rationale survive across iterations of the project
rather than being rediscovered from scratch under time pressure
if the primary plan turns out to need a fallback.

## Option A — SA warmstart

Run the existing `build-decks` on ~5k sealed pools with the
gen4-512 scorer to produce ~5k SA-built reference decks. Pretrain
the picker in a supervised mode against these reference decks
(see "Reference-deck dataset format" below), then continue with
REINFORCE (§ 3) using a KL penalty against the warmstarted picker
as a regularizer:

```
total_loss = policy_loss + entropy_loss + aux_weight * aux_loss
             + kl_coef * kl(picker || warmstart_picker)
```

The KL term keeps the picker close to the data manifold the scorer
was trained on, mitigating reward hacking ("Risks and mitigations"
above). This mirrors the standard SFT → PPO recipe from RLHF.

CLI configuration: pass the supervised pretraining checkpoint via
`--picker-checkpoint` (loads its weights as the REINFORCE init,
§ 4.1) and set `--kl-coef` to a non-zero value (e.g., 0.1) to
enable the penalty. The reference distribution for the KL is the
loaded `--picker-checkpoint`'s frozen copy.

## Option C — Pure supervised distillation

Same SA reference-deck collection as Option A, but train the
picker end-to-end supervised against the SA picks with no
REINFORCE stage. The loss combines two terms:

- **Pairwise margin loss.** For each pool, every
  (teacher-picked, teacher-rejected) pair contributes
  `max(0, margin − (logit_picked − logit_rejected))`. 23 × ~60 ≈
  1380 pairs per pool. Directly optimizes the ranking structure
  that top-K inference cares about.
- **Per-card BCE.** Standard `(in_deck / not_in_deck)`
  cross-entropy on each of the N pool cards. Regularizer that
  keeps logits calibrated to a meaningful absolute scale.

Optionally a per-card marginal-contribution MSE term: for each
SA-built deck, compute the score delta when each card is swapped
out for the best remaining-in-pool, and supervise the picker's
logits against those continuous decisiveness labels.

Quality ceiling under Option C is capped at SA's quality (the
picker learns to reproduce SA's picks, not exceed them).

## Reference-deck dataset format

Both contingency options consume the existing
`generated-decks.txt` format (`LABEL;SET_CODE;Card1|...|Card40`
per line) produced by `build-decks`. Label convention for the
SA-built reference set: `sa-reference-{scorer-tag}` (e.g.,
`sa-reference-gen4-512`).

Per-pool side information for Option C's marginal-contribution
target requires a richer log: at SA's last accepted move per
restart, record the full 23-card deck and the per-card score
delta. This would be recorded by an instrumentation flag on
`build-decks`; the resulting trajectory file would live alongside
`generated-decks.txt`.

# Out of scope

- **Phase B picker fine-tuning** (jointly training the picker + the
  underlying encoder). Analogous to the scorer's Phase B; can be
  added later as a separate spec.
- **Actor-critic baseline using the aux head.** The aux head
  (§ 1.2) is structurally a critic — it predicts expected
  reward — but is not used as the policy-gradient baseline (§ 3.3
  uses the empirical mean instead). A natural upgrade is to use
  `pool_quality_pred` directly as the baseline, supervised against
  the empirical mean as a target. The initial spec keeps the
  empirical mean baseline for simplicity; the upgrade is recorded
  here for a future spec round once basic lift-off is confirmed.
- **Multi-pool batching at inference.** The single-pool inference
  path is the operationally relevant one for the mobile deployment
  target; bulk evaluation can use the natural batch dimension across
  pools but no special CLI surface is needed.
- **Picker-as-deck-builder inside `match-outcomes` weighted
  rolls.** The natural integration is through
  `generated-decks.txt` files (§ 4.2), which `match-outcomes`
  already consumes via `--side-a-decks` / `--side-b-decks`. No
  changes to `match-outcomes` itself are required for picker decks
  to participate in self-play.
