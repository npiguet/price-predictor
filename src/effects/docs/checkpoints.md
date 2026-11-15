# Checkpoints and resume state

Rules to respect when changing what a checkpoint or `resume.pt` holds, a loader, or `--resume`. The code is `infrastructure/effect_model_store.py`, `application/model_loading.py` and `resume_state`; what a checkpoint contains is specified in `specs/2026-10-30-ability-effect-model-gen3.md` § 16.

## Checkpoints

- A training run writes its checkpoints (`{timestamp}.pt` plus a rolling `latest.pt`) to `--model-output`, its run directory under `models/effects/<gen>/effect-model/`. A baseline variant trains into a model directory of its own (`effect-model-identity`, ...; without `--model-output`, a `{variant}/` subdirectory of the default), so a baseline run can never overwrite the checkpoint the cache was built from.
- A checkpoint records the split it trained against, enumerated rather than derived: the held-out card list and both `game_id` sets. No command recomputes a split, because the corpus is append-only and a recomputed one would score the gates partly on games the model trained on.
- It also records the vocabulary and keyword-definition paths with their content hashes, the withheld keyword, the curated dataset's manifest path and digest, the sidecar roots, the holdout unit, the encoder size and the training settings.
- Every inference command re-hashes what it actually loaded and fails fast on a mismatch. Encoding a cache against a different vocabulary produces vectors that look fine and mean nothing.
- Every loader builds the encoder from the recorded size, so an arm at a non-default size loads without restating its flags. A checkpoint recording no size loads at 4 layers and width 256.
- The MLM, script-API and value heads are training-only and are filtered out at save time.
- A gen-3 checkpoint's weights are the averaged ones. It records `best_control_score`, the loss denominators, `--rule-mask`, `--deviance-floor`, `--ema-epochs`, `--stratum-weight`, `--relation-values`, the manifest's `rule_map_digest` and the `selection_set_digest`.
- `training_settings` holds `input_layout` with the relation types, subtypes, named counters and closed output lists, whose order fixes the model's tables. A loader refuses a checkpoint whose layout or lists differ.

## Loading

- `application/model_loading.py` rebuilds encoder, model and batcher from a saved checkpoint, the weights read by `effect_model_store.load_weights`.
- Loading is strict: a model state missing anything but the training-only heads, or carrying an unexpected key, is refused. Loaded leniently, the missing part would run at random initial weights. The encoder and model come back in eval mode.
- The `identity` baseline's `e` comes from a free embedding table and not from the encoder, so the table rides in the checkpoint's `extra` with its texts, and the runner rebuilds it from them. Rebuilt empty, the baseline is scored on random vectors and gate 1 gets a margin nobody earned.
- The table indexes texts exactly: one row per text the run's sidecars encode to (`ability_text.known_texts`), plus one shared row for any other text, such as a counterfactual copy's. A hashed table would let two texts share a row, which is the memorization the baseline exists to measure.

## `resume.pt`

- It is not a checkpoint. It is rewritten at the end of every epoch, to a temporary name and renamed, and carries everything a checkpoint drops: the training-only heads, the optimizer, the step, the early stopper, the noise on `e`, the averaged weights, the loss denominators, the best control score and the generator states.
- `--resume` continues at the next epoch and trains bit for bit what the uninterrupted run would; `test_training_loop_workers.py` checks it across the curriculum switch. No other state needs saving because every draw is seeded by `(seed, epoch, position, step)`.
- A resume under a different seed, corpus, size or batching is refused. Every `TrainEffectModelConfig` field is compared except those in `resume_state.RESUME_MAY_CHANGE`, so a new flag is checked by default.
- `--resume` decides under this run's patience and epochs whether anything is left, so a run that stopped early can be continued.
- A new best epoch copies `resume.pt` to `resume-best.pt`, which another run directory can continue from.
- A resume compares the selection set's digest with the one the run recorded.
