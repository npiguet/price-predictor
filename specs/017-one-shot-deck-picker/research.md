# Research: One-Shot Sealed Deck Picker

**Branch**: `017-one-shot-deck-picker` | **Date**: 2026-05-20 | **Spec**: [spec.md](spec.md)

This document records the codebase survey (Constitution Principle VII) and the
Phase 0 design decisions that resolve the open questions left by the spec.

## Codebase Survey

The picker is the third major model in the `sealed/` package, after the
encoder (spec 010, 016) and the scorer (spec 011, 015). Survey scope: every
sub-problem the picker needs (transformer over a set, frozen-scorer reward,
pool I/O, deck output, manabase fill, checkpoint persistence, CLI registration,
training-loop scaffolding) is something the codebase already implements at
least once. Concrete pointers below.

### Overlapping domain vocabulary

| Existing concept | File / symbol | Overlap | Decision |
|---|---|---|---|
| `ScorerConfig` (Set Transformer scorer architecture) | `src/sealed/domain/scorer_model.py:14` | Same family of architecture knobs (`d_model`, `n_layers`, `n_heads`, `d_ff`, `dropout`) | **Mirror, do not extend.** The picker is a different model — different forward signature (returns N logits, not a scalar), no PMA, an extra aux head — so a parallel `PickerConfig` dataclass is warranted. Same flag names as the scorer for practitioner familiarity. |
| `SetTransformerScorer` (SAB stack + PMA + scoring MLP) | `src/sealed/domain/scorer_model.py:78` | Set-input transformer with SAB layers — exactly the trunk the picker needs | **Reuse SAB primitive.** The `SAB` class (`scorer_model.py:26`) is the unit set-transformer block; import and reuse it. No PMA in the picker (per-card head, not pooled). Aux head uses mean-pool (built inline, no new abstraction). |
| `GreedyDeckBuilder` (iterative SA picker) | `src/sealed/domain/greedy_deck_builder.py:103` | Same role (pool → 23 nonland cards) the picker plays, but via search instead of policy | **Parallel concept.** The picker replaces the search loop at inference. Keep `GreedyDeckBuilder` (still used by `build-decks`, `evaluate-scorer`); add a sibling `PickerModel` in `domain/`. `_partition_pool` and `is_land_embedding` semantics are reused identically. |
| `BuildDecksUseCase` (build-decks CLI use case) | `src/sealed/application/build_decks.py:80` | Same role `pick-decks` plays: pool file → generated-decks.txt | **Mirror, do not share.** The two diverge in the per-pool inner loop (SA vs. forward pass) and in checkpoint-loading types (scorer vs. picker). Factor common pieces (`_count_complete_lines_and_truncate_partial`, log line cadence, deck-line writer) only after the picker version is working — premature extraction would cross-couple two specs that are still settling. Survey-mandated follow-up: once both work, evaluate whether the resume/append loop is extractable. |
| `ScorerStore` (scorer .pt persistence) | `src/sealed/infrastructure/scorer_store.py:39` | Saves `(model_state_dict, optimizer_state_dict, epoch, best_val_*, config)` + optional encoder bits | **Mirror as `PickerStore`.** Same shape, different config dataclass (`PickerConfig`), different "best val" metric (`best_val_reward`, not `best_val_accuracy`). Reuse `torch_checkpoint.save_checkpoint` / `load_checkpoint` underneath (`src/price_predictor/infrastructure/torch_checkpoint.py:14`). |
| `EmbeddingTable` (live card-embedding lookup) | `src/sealed/infrastructure/match_data_loader.py:EmbeddingTable` | Wraps a `(N_cards, d_model)` tensor with name↔index lookup, supports per-batch text-vector splicing for Phase B | **Reuse.** The picker training loop needs `embeddings[pool_card_index]` lookups identical to what the scorer does. Phase B is out of scope; only the frozen-cache read path is needed. |
| `ConvertedCardLocator` (name → .txt / .npz lookup with prefix-fallback + sanitization) | `src/sealed/infrastructure/converted_card_locator.py:27` | Resolves card names to files, handles A-/double-faced/meld edge cases | **Reuse unmodified.** The picker reads embeddings through this; identical semantics to `build-decks`. |
| `parse_pools` / `GeneratedDeck` (pool file + generated-decks file readers/writers) | `src/sealed/infrastructure/pool_file_reader.py:32, 18` | The picker's input and output file formats are exactly these | **Reuse `parse_pools` for input; mirror `build_decks.py` line-by-line writer for output.** Output format identical (`LABEL;SET_CODE;Card1\|...\|Card40`), so the same `parse_generated_decks` reader works downstream. |
| `compute_basic_lands` (manabase heuristic) | `src/sealed/domain/manabase.py:20` | Computes `40 − N_chosen` basics from a chosen-nonland mana-pip histogram | **Reuse unmodified.** Per FR-007 the picker uses this exact helper. Lands in the chosen list silently contribute no pips. |
| `is_land_embedding` (deterministic-feature partition flag) | `src/sealed/domain/card_embedding_layout.py:65` | Reads the `IS_LAND` slot of the trailing feature block | **Reuse unmodified.** FR-008 mandates identical semantics. |
| `TrainScorerConfig` / `TrainScorerUseCase` (scorer training loop) | `src/sealed/application/train_scorer.py:40, 235` | Provides the resume/bootstrap/persist patterns the picker needs: `--resume` mutually exclusive with bootstrap, architecture-flag rejection on resume, best-vs-latest dual-file persistence, fail-fast width checks | **Mirror, do not share.** The training loops differ structurally (pairwise BCE vs. REINFORCE; matched-pair batch vs. pool batch with N_samples). Reuse the *patterns* — same CLI ergonomics, same dataclass-config shape, same `--random-seed 42` convention — but each trainer keeps its own file. |
| `ScorerConfig.d_model` derived from `.npz` cache width | `train_scorer.py:96`, `_resume_or_build_model:521` | Picker has the same constraint (FR-002, FR-034) | **Replicate the same `_check_*_width` fail-fast pattern** for both the scorer-checkpoint width and the picker-checkpoint width vs. embedding cache. |

**Outcome**: 11 vocabulary overlaps surveyed. Of these, 9 are reused as-is or
mirrored at the pattern level; 2 (`GreedyDeckBuilder` and `BuildDecksUseCase`)
are explicitly retained as parallel concepts because the picker is a different
*model*, not a different *implementation of the same model*. Zero entities are
silently duplicated.

### Adjacent prior art

| Sub-problem | Existing solution | Decision |
|---|---|---|
| Set-input transformer trunk | `SAB` in `scorer_model.py:26` | **Reuse.** Plain SAB stack matches the spec's § 1 architecture. |
| Pool-batch padding (variable N across pools) | `_concat_batches` in `greedy_deck_builder.py:322` zero-pads `(B, max_len)` indices and a separate boolean mask | **Reuse pattern.** Picker training batches `B = batch_size` pools with variable N; same `(B, max_N, d_model)` plus `(B, max_N)` real-card mask. nn.MultiheadAttention's `key_padding_mask` convention is already followed by `SetTransformerScorer.forward_prenormalized` — same convention here. |
| Per-card embedding lookup from a name list | `BuildDecksUseCase._build_one_deck` (`build_decks.py:166`) loads `pool_embeddings[name]` via `ConvertedCardLocator.load_embedding` | **Reuse.** Identical pool-loading code path. |
| Scoring sampled decks with a frozen scorer in a batched forward | `_score_batch` in `greedy_deck_builder.py:516` packs `(B, max_len)` index + mask, calls `model.forward_prenormalized`, runs under `torch.no_grad()` + autocast fp16 on CUDA | **Reuse pattern.** REINFORCE step 4 needs exactly this: stack `batch_size × N_samples` decks, pad to `max_len`, one frozen forward. The pool array is normalized once per batch (not per call) — same trick. |
| Frozen-encoder embedding cache (`.npz` per card, width = pooled text + features) | `EmbeddingStore` (`embedding_store.py`) saves; `ConvertedCardLocator.load_embedding` loads | **Reuse unmodified.** The picker consumes the cache the scorer already consumes. |
| `.pt` checkpoint serialization with a dataclass config | `torch_checkpoint.save_checkpoint`/`load_checkpoint` (`torch_checkpoint.py:14, 20`) — used by both encoder and scorer stores | **Reuse.** `PickerStore` is a thin wrapper, same as `ScorerStore`. |
| Resume-precedence resolution (CLI > stored `train_config` > dataclass default) | `_RESUMABLE_FLAG_NAMES` + `_dataclass_default` in `cli.py:882, 902` | **Reuse pattern.** Identical resume semantics expected of `train-picker`. |
| Per-parameter-group AdamW + per-group gradient-norm clip | `_build_optimizer` + `_clip_per_group` in `train_scorer.py:591, 817` | **Reuse pattern.** Picker uses a single "picker" parameter group; the clip helper works as-is. |
| Append-and-skip output-file resume (deck-builder-style) | `_count_complete_lines_and_truncate_partial` in `build_decks.py:26` + the `open_mode = "a"` / `pools[skip:]` flow at `build_decks.py:97` | **Reuse pattern.** `pick-decks --resume` has identical semantics. |
| Argparse-based CLI registration with subparsers and `set_defaults(func=...)` | `cli.py:60` and all `_build_*_parser` helpers | **Reuse pattern.** Two new `_build_*_parser` functions + two `run_*` dispatchers. |
| Plackett-Luce / sequential without-replacement sampler on GPU | None in the codebase | **New, inline in `train_picker.py`.** No prior art to reuse; the sampler is ~30 lines of `torch.multinomial` + mask updates per § 3.2 of the spec. Single use site, no abstraction. |
| Auxiliary scalar head (Linear + mean-pool over tokens) | None in the codebase exactly; sealed `train_encoder.py` has a `nn.ModuleDict` of regression heads off a pooled vector | **Implement inline** in `PickerModel`. The aux head is a single `Linear(d_model, 1)` over the mean-pooled token outputs, with a masked mean to ignore padding tokens. Too small to deserve a dedicated module. |
| Rank-correlation diagnostic for cross-scorer audit | None — no existing scorer-vs-scorer correlation site | **New, inline in `train_picker.py`.** Use `scipy.stats.spearmanr` (already pulled in transitively by `scikit-learn`; explicit import added). One call per validation epoch when the auditor checkpoint is set. |
| Distributional summaries (color count, CMC histogram, creature count, type balance) on decks | None exactly; `evaluate_scorer` has a `format_decks_for_display` text formatter | **New, inline in `train_picker.py`.** Compute from the same `pool_embeddings` block that produced the validation decks — color count uses the `COLOR_FLAGS` slot, creature count uses a per-card type flag (TODO: confirm the type slot is in the deterministic feature block — see "Open question" below). |

**Outcome**: 11 prior-art reuses, 3 new inline helpers (sampler, rank-correlation
diagnostic, distributional summaries). No reimplementation of solved problems.

### Convention alignment

The picker mirrors the scorer's conventions in the sealed package — they are
the closest sibling because both are end-stage models trained against an
upstream artifact (encoder for the scorer; scorer for the picker), both load
a frozen `.npz` cache, both have Phase A / Phase B distinction (the picker
only ships Phase A; Phase B is explicit OOS in the spec), and both expose
roughly the same CLI ergonomics.

Specific mirrored conventions:

| Convention | Scorer site | Picker plan |
|---|---|---|
| Folder layout `domain/<model>_model.py`, `application/train_<model>.py` + `<verb>_<model>s.py`, `infrastructure/<model>_store.py` | `scorer_model.py`, `train_scorer.py`, `evaluate_scorer.py`/`build_decks.py`, `scorer_store.py` | `picker_model.py`, `train_picker.py`, `pick_decks.py`, `picker_store.py` |
| Dataclass-backed CLI config | `TrainScorerConfig` (`train_scorer.py:40`) | `TrainPickerConfig` + `PickDecksConfig` |
| Hardcoded `random_seed = 42` | `TrainScorerConfig.random_seed = 42` (`train_scorer.py:61`); spec 017 §FR-018 mandates the same for the picker (no CLI flag, hardcoded) | Inline constant in `train_picker.py`. |
| Best-checkpoint dual-file persistence (`best_*.pt` + `latest.pt`) | `TrainScorerConfig.best_checkpoint_name()` / `latest_checkpoint_name()` (`train_scorer.py:77`) | Mirror: `latest.pt` overwritten each epoch; `best_{timestamp}.pt` overwritten on new val-reward best. Run-stamped best name (vs. the scorer's arch-stamped name) so runs don't clobber each other. No per-epoch snapshot files. (Spec 017 §FR-037.) |
| Fail-fast width check (scorer vs. cache) | `_check_scorer_width` (`train_scorer.py:430`) | `_check_picker_width` (cache vs. scorer; cache vs. picker checkpoint). |
| `--resume` mutually exclusive with bootstrap; architecture flags rejected on either; resume-precedence resolution | `cli.py:_RESUMABLE_FLAG_NAMES`, `_TRAIN_SCORER_ARCHITECTURE_FLAGS`, `run_train_scorer` validation block | Same pattern with `_TRAIN_PICKER_ARCHITECTURE_FLAGS` + `_RESUMABLE_PICKER_FLAG_NAMES`. |
| `--label` parser rejects `;` / `\|` / whitespace | `_parse_label` (`cli.py:35`) | Reuse the same `_parse_label` for `pick-decks --label`. |

**Outcome**: All conventions in the sealed package have a clear mirror site
for the picker. No deviations from established patterns.

### Third-instance check

| Pattern instance count |
|---|
| **Best-checkpoint helper**: `train_scorer.py` (val_acc), `train_encoder.py` (val_loss). Picker would be the third instance (val_reward). |
| **Training-loop scaffolding** (resume / bootstrap / persist / fail-fast width check / early-stop on patience): `train_scorer.py` and `train_encoder.py` both have it; picker would be the third. |
| **Pool-file-driven generated-decks writer**: only `build_decks.py` has it; `pick_decks.py` would be the second. Not a third-instance trigger. |

The third-instance trigger fires for **best-checkpoint helper** and
**training-loop scaffolding** patterns. Spec 016's plan (research.md §"Third-instance check")
already noted that the encoder and scorer's `_BestCheckpoint`-style logic
diverges enough (val_acc vs. val_loss; different metadata fields) that a
shared abstraction was deferred. The picker introduces yet another metric
(val_reward) and a different update semantic (REINFORCE per-pool vs.
pairwise BCE).

**Decision**: do **not** extract a shared abstraction in this feature. The
three trainers have diverged across:

- Loss type (pairwise BCE / weighted MSE + MLM / REINFORCE-with-baseline)
- Best-checkpoint metric direction (acc maximizes, loss minimizes, reward maximizes)
- Per-step dataset shape (matched pairs / token sequences / pool batches with sampled decks)
- Phase A/B vs. Phase A-only (scorer has both, encoder has none, picker has only A)

Any abstraction that accommodates all three would have to be parameterized on
loss type, metric direction, and dataset shape — i.e., a generic training
loop, which is the kind of thing the constitution's Simplicity First principle
flags as premature abstraction. Each trainer is ~700 lines; an abstraction
saving a few dozen lines per file while complicating control flow across three
files is a net negative.

**Follow-up**: add a TODO comment at the top of `train_picker.py` noting the
three trainers as candidate extraction sites if/when a fourth REINFORCE-style
trainer arrives (e.g., the contingency plan's Option A supervised pretraining,
or a future actor-critic variant).

## Phase 0 design decisions

The spec is prescriptive enough that no NEEDS CLARIFICATION items remain after
the `/speckit.clarify` session. The decisions below document the *how* of
applying the spec's *what* — sized to choices the implementer would otherwise
have to make blind.

### D1: Picker model layout

**Decision**: One `PickerModel(nn.Module)` class in
`src/sealed/domain/picker_model.py` containing:

- Optional `Linear(embedding_dim, d_model)` input projection (inserted only
  when `d_model != embedding_dim`, per FR-002).
- A `nn.ModuleList[SAB]` of `n_layers` SAB blocks, importing `SAB` directly
  from `sealed.domain.scorer_model`.
- A `nn.Linear(d_model, 1)` per-card head applied to each token output.
- A `nn.Linear(d_model, 1)` auxiliary head applied to the masked mean-pool
  of the token outputs (FR-005; § 1.2).
- A `PickerConfig` dataclass at the top of the same file (mirrors
  `ScorerConfig`).
- A `forward(pool_cards, pool_mask) -> (logits, aux_pred)` returning the
  N-logit tensor and the per-pool aux scalar in one pass. (No "training vs.
  inference" branching; the aux head is structurally always present, the
  caller decides whether to use it.)

**Rationale**: Single-file layout matches `scorer_model.py`. Co-locating the
config dataclass with the model class makes the architecture self-contained
and importable from both the use case and the store.

**Alternative considered**: split aux head into its own module. Rejected —
the aux head is two lines of code, splitting just inflates file count.

### D2: Sampler implementation

**Decision**: Sequential without-replacement sampling implemented as a
vectorized `torch.multinomial` loop over the `(B × N_samples, N)` probability
tensor. Per pick step:

1. Mask out already-picked positions by setting their logits to `-inf`.
2. Recompute probs (or rely on mask handling — see below).
3. `torch.multinomial(probs, 1)` to draw the next pick across all rows.
4. Update mask and bucket each row's pick into the per-row "chosen spells"
   / "chosen lands" tracker.
5. Halt when every row's `len(chosen_spells) == 23` (or when no row has any
   remaining mass).

Per § 3.2, the sampler walks ~25 iterations per sampled deck — bounded by
the spell quota plus the upper-bound land count. Vectorized across all
`B × N_samples` rows simultaneously.

**Implementation detail**: the spell-quota stopping rule is per-row (some rows
finish in 23 picks, some in 29). Use a per-row "active" flag derived from
`spells_picked < 23`; inactive rows still execute the multinomial call but
their pick is discarded (cheaper than dynamic-batching active rows in a
Python loop). Loop exits when no row is active OR when the universal
upper bound (23 + max nonbasic-lands-in-pool) is reached.

**Plackett-Luce log-prob**: computed in a *separate pass* over the recorded
pick sequence — the sampler returns `(pick_indices, picked_mask)` to the
caller; the caller computes log-prob via differentiable
`logit_picked − logsumexp(remaining_logits)` summed across pick steps. This
keeps the sampler under `torch.no_grad()` for the sampling itself, but the
log-prob path is differentiable.

**Rationale**: Matches the spec's § 3.2 prescription verbatim. Single-pass
on-GPU. The two-pass split (sample without grad, then recompute log-prob with
grad) is necessary because `torch.multinomial` itself is not differentiable.

**Alternative considered**: Gumbel-top-K. Rejected by spec § 3.2 because the
variable trajectory length defeats the Gumbel vectorization advantage.

### D3: Pool source streaming

**Decision**: Load the entire pools file into memory at startup (one
`parse_pools()` call), split the front `--val-fraction` off as the fixed
validation slice, shuffle the remainder per epoch with a `random.Random(42)`
generator seeded from `random_seed=42` (FR-018), then iterate `batch_size`
pools at a time. No streaming reader, no file-position tracking.

**Rationale**: A 100k-pool file is ~10–20 MB of text in memory — trivial.
The shuffling-per-epoch and validation-slice semantics match
`sklearn.train_test_split`-with-fixed-seed and `train_scorer.py:_load_dataset`
(`train_scorer.py:405`). Streaming would only matter at >>1M pools, which is
out of the spec's stated scale.

**Alternative considered**: lazy chunked reader. Rejected — premature
optimization. Spec § 3.1 "Pool source" makes the in-memory choice fine.

### D4: Per-step batching shape

**Decision**: The training step processes `B = batch_size` pools and
`S = n_samples` decks per pool. Tensor shapes:

- Pool cards: `(B, max_N, embedding_dim)` from the cache, with mask
  `(B, max_N)` where True = real pool card.
- Picker forward output: logits `(B, max_N)` and aux pred `(B,)`.
- Sampled decks: per-pool `(S, deck_max_len)` indices into the pool plus a
  mask, flattened to `(B*S, deck_max_len)` for the scorer forward.
- Sampler intermediate: probs `(B*S, max_N)`, picked-mask `(B*S, max_N)`.
- Scorer reward: `(B, S)` after un-flattening.
- Advantage: `(B, S)` after subtracting per-pool mean.
- Log-prob: `(B, S)` from the Plackett-Luce sum.
- Aux loss: `MSE(aux_pred, rewards.mean(dim=1).detach())` → scalar.

**Rationale**: Standard policy-gradient batching. `B*S` is the natural
"deck-batch" dimension for the frozen scorer forward — matches `_score_batch`
in `greedy_deck_builder.py` which the picker reuses through `forward_prenormalized`.

### D5: Frozen-scorer integration

**Decision**: Load the scorer via `ScorerStore().load_checkpoint(path)` at
training start. Move to GPU. Call `.eval()` on it. Wrap every scorer forward
in `torch.no_grad()` and `torch.autocast(device_type=..., dtype=torch.float16,
enabled=(device.type == "cuda"))`. The pool cards are normalized once per
batch via `scorer.normalize_features(...)`; sampled decks index into the
normalized array and are passed via `scorer.forward_prenormalized(...)` (same
trick `GreedyDeckBuilder._score_batch` uses to avoid per-call clones).

**Scoring input is chosen-only.** The deck handed to the scorer is the chosen
spells + nonbasic lands — basic lands are NOT scored (FR-012). This is
bit-for-bit the input `GreedyDeckBuilder` scores (`deck_spells + deck_lands`,
`greedy_deck_builder.py:516`), so there is no reward-comparability divergence
with prior SA results. Basic lands are added by `compute_basic_lands` only
when materializing the final 40-card deck for output (inference / `pick-decks`);
since that fill is deterministic and post-hoc, feeding basics into the scorer
would be off-distribution noise with zero upside.

Width check at startup: `scorer.config.d_model == embedding_dim` (where
`embedding_dim` is read from the first `.npz` in the cache via
`ConvertedCardLocator.load_embedding(...).shape[-1]`). On mismatch, raise
`ValueError` with the same message shape as `_check_scorer_width` in
`train_scorer.py:430` (FR-034, FR-035).

**Rationale**: Bit-for-bit reuse of the scorer integration pattern that
`GreedyDeckBuilder` and `BuildDecksUseCase` already use.

### D6: Auditor scorer integration

**Decision**: When `--auditor-scorer-checkpoint` is set, load a second
scorer the same way at startup, also frozen + eval-mode. At validation time
(once per epoch), after computing deterministic-inference decks and scoring
them with the training scorer, also score them with the auditor scorer in
one batched forward. Compute `scipy.stats.spearmanr(training_scores,
auditor_scores)` over the validation set and log the correlation alongside
mean validation reward.

**Width compatibility note**: the auditor scorer must have been trained on
the same `.npz` cache width as the training scorer (otherwise the same
sampled-deck embedding tensor can't feed both). Enforced by a width check
at startup; mismatch raises with the same error shape as the training-scorer
width check.

**Rationale**: One extra scorer forward at validation time is negligible
(spec § "Reward hacking" item 1 explicitly calls it negligible). The
width-compatibility constraint is real but is exactly the same one the
training scorer faces, so the same check covers it.

### D7: Distributional summaries

**Decision**: At validation epoch end, after computing the deterministic-inference
decks, compute four summaries over the decks and log them as a single
structured line:

- **Mean color count per deck**: per-deck, count of `COLOR_FLAGS` sets that
  the chosen-spells' embeddings flag any pip in; mean across validation decks.
- **Mean creature count per deck**: a per-card "is creature" flag — *open
  point*: the current deterministic-feature block (FEATURE_COUNT = 32, see
  `card_embedding_layout.py`) does **not** carry an explicit "is creature"
  bit. Power/toughness slots (`POWER` index 23, `TOUGHNESS` index 24) are
  nonzero for creatures and zero for non-creatures; use `POWER > 0 or
  TOUGHNESS > 0` as the creature heuristic. Validate this against
  `deterministic_features.py` during implementation; if it turns out P/T
  is also nonzero for some non-creature types (e.g., vehicles), fall back
  to reading the converted card text via `ConvertedCardLocator.load_text(...).type_line()`.
  (See "Implementation note" below.)
- **Mean CMC histogram**: condensed to 5 bins (CMC≤2, 3, 4, 5, 6+) via
  the `MANA_VALUE` slot (index 9) of the deterministic-feature block; log
  the across-decks mean count per bin.
- **Mean type-balance ratios**: creature share and noncreature share — two
  scalars, summing to 1. Future-extensible (could split noncreature into
  spells/permanents) but the minimum viable is creature share.

**Log format**: one line per epoch in addition to the validation reward line,
e.g.:

```
epoch 12  val_reward=4.213  audit_corr=0.81  | colors_mean=2.3 creatures_mean=14.8 type_creature_share=0.64  cmc_hist=[5.1, 6.7, 5.3, 3.9, 2.0]
```

**Rationale**: Five bins keep the line readable per Q4 of the clarify session.
All four metrics are computable from data already in memory at validation
time — no extra forward passes, no extra disk reads. Implementation note: if
the P/T heuristic for creature detection is wrong for some card types, the
fix is a one-line change to read the type line through `ConvertedCardLocator`;
this will be checked against `deterministic_features.py` during the early
training implementation task and corrected if needed.

### D8: Best-checkpoint persistence

**Decision**: Two checkpoint files in `models/sealed/picker/`:

- `latest.pt` — overwritten every epoch with the latest checkpoint; the
  resume point.
- `best_{timestamp}.pt` — one per run (`{timestamp}` fixed at run start),
  overwritten in place whenever a checkpoint sets a new validation-reward
  best. The run-stamped name (vs. the scorer's architecture-stamped
  `best_l6_h4_...pt`) prevents sequential or concurrent runs from clobbering
  each other's best.

No per-epoch snapshot files. This drops the original three-file design's
`{timestamp}.pt`-per-epoch snapshots, which the rest of the project does not
use (the scorer and encoder both persist only a best + a latest). The
downstream consequence: the spec's end-of-training Forge validation compares
the **best** (`best_{timestamp}.pt`) and the **final** (`latest.pt`)
checkpoints rather than an arbitrary top-K of per-epoch snapshots. Since
SC-003 expects validation reward to climb monotonically and then plateau,
best and final are normally the two strongest checkpoints anyway, so the
two-file scheme loses little practical top-K signal.

Both files carry the same payload (FR-038):

- `model_state_dict` — picker weights only.
- `optimizer_state_dict` — AdamW state.
- `config` — `PickerConfig` (architecture; includes input width).
- `epoch` — current epoch counter.
- `best_val_reward` — float; the best validation reward seen so far.
- `train_config` — JSON-friendly dict of `TrainPickerConfig` for resume
  precedence (mirrors `train_scorer.py:_build_train_config`).

**Rationale**: Matches `train-scorer`'s convention (`_persist_checkpoint`
in `train_scorer.py:371`). The `train_config` field is what enables
`--resume` to fall back to recorded values for resumable flags.

### D9: Test surface

**Decision**: Three pytest files:

- `tests/unit/sealed/domain/test_picker_model.py` — forward shape, projection
  inserted only on width mismatch, aux head present in `state_dict`,
  divisibility error raised by config validation.
- `tests/unit/sealed/application/test_train_picker.py` — sampler exits at
  spell quota; sampler respects mask (basic-land embeddings are not in the
  pool input anyway, but lands in the pool are sampled into the lands
  bucket; the spell quota terminates the walk); per-pool baseline formula;
  advantage detach; aux loss target detach; Plackett-Luce log-prob
  factorization on a hand-checked tiny example; resume precedence loads
  recorded train_config; architecture-flag rejection on resume; KL coef
  != 0 requires `--picker-checkpoint`; width-mismatch errors at startup.
- `tests/unit/sealed/application/test_pick_decks.py` — deterministic walk
  matches the § 1.1 pseudocode on a tiny pool; resume append-and-skip
  semantics; manabase fill to 40; label correctly written; pools-file
  parse errors propagate.
- `tests/unit/sealed/infrastructure/test_picker_store.py` — round-trip save
  / load; `latest.pt` and `best_{timestamp}.pt` written to the right paths.

No integration tests in this feature: the end-to-end Forge match validation
is documented in the spec as a manual procedure run a handful of times per
project lifetime; building it as a pytest integration target would violate
the documented "this isn't CLI infrastructure" decision.

**Rationale**: Unit tests cover every functional requirement that's
testable without a real GPU, real Forge JVM, or real `.npz` cache. Constitution
Principle I (Fast Automated Tests) is satisfied: every test in this list
runs in milliseconds on CPU.

## Open items handed to Phase 1 / implementation

- Confirm `POWER > 0 or TOUGHNESS > 0` is a sound creature heuristic against
  `src/sealed/domain/deterministic_features.py`. If not, fall back to
  reading the converted card text type line. (D7 above.)
- Decide where to place the `random.Random(42)` instance — module-level
  constant on `train_picker.py` or per-config field — at implementation
  time. (D3.)

These are sub-decisions inside Phase 1 implementation, not blocking design
questions.
