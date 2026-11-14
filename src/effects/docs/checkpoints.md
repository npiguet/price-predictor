# Checkpoints and resume state

Rules to respect when changing what a checkpoint or `resume.pt` holds, a loader, or `--resume`. The code is `infrastructure/effect_model_store.py`, `application/model_loading.py` and `resume_state`; what a checkpoint contains is specified in `specs/2026-10-30-ability-effect-model-gen3.md` § 16.

## Checkpoints

- Checkpoints live under `models/effects/effect-model/` (`{timestamp}.pt` plus a rolling `latest.pt`). A variant writes to `models/effects/effect-model/{variant}/`, so a baseline run can never overwrite the checkpoint the cache was built from.
- A checkpoint records the split it trained against, enumerated rather than derived: the held-out card list and both `game_id` sets. No command recomputes a split, because the corpus is append-only and a recomputed one would score the gates partly on games the model trained on.
- It also records the vocabulary and keyword-definition paths with their content hashes, the withheld keyword, the curated dataset's manifest path and digest, the sidecar roots, the holdout unit, the encoder size and the training settings.
- Every inference command re-hashes what it actually loaded and fails fast on a mismatch. Encoding a cache against a different vocabulary produces vectors that look fine and mean nothing.
- Every loader builds the encoder from the recorded size, so an arm at a non-default size loads without restating its flags. A checkpoint recording no size loads at 4 layers and width 256.
- The MLM, script-API and value heads are training-only and are filtered out at save time.
- A gen-3 checkpoint's weights are the averaged ones. It records `best_control_score`, the loss denominators, `--rule-mask`, `--deviance-floor`, `--ema-epochs`, `--stratum-weight`, `--relation-values`, the manifest's `rule_map_digest` and the `selection_set_digest`.
- `training_settings` holds `input_layout` with the relation types, subtypes, named counters and closed output lists, whose order fixes the model's tables. A loader refuses a checkpoint whose layout or lists differ.

## `resume.pt`

- It is not a checkpoint. It is rewritten at the end of every epoch, to a temporary name and renamed, and carries everything a checkpoint drops: the training-only heads, the optimizer, the step, the early stopper, the noise on `e`, the averaged weights, the loss denominators, the best control score and the generator states.
- `--resume` continues at the next epoch and trains bit for bit what the uninterrupted run would; `test_training_loop_workers.py` checks it across the curriculum switch. No other state needs saving because every draw is seeded by `(seed, epoch, position, step)`.
- A resume under a different seed, corpus, size or batching is refused. Every `TrainEffectModelConfig` field is compared except those in `resume_state.RESUME_MAY_CHANGE`, so a new flag is checked by default.
- `--resume` decides under this run's patience and epochs whether anything is left, so a run that stopped early can be continued.
- A new best epoch copies `resume.pt` to `resume-best.pt`, which another run directory can continue from.
- A resume compares the selection set's digest with the one the run recorded.
