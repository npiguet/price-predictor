# Performance invariants of the hot paths

Six properties of the training and encoding hot paths are load-bearing, and each has a failure mode that looks like a working run. Read before changing the batcher, step preparation, the step workers, loss read-back, shard loading or `encode-abilities`' encoding loop.

## 1. `e` enters the head by row index, never by value

- A training surface's `Slot.e_row` is an `int` naming a row of the batch's encoded-ability matrix, and its `Slot.e_vector` is None; `scatter_e_rows` turns rows into vectors on the device in one gather.
- The value head reads `e` the same way: its targets carry the matrix row each line's text sits at, and the loss gathers `matrix[rows]` on the device, so a text two lines share is read once per line without a second encoder row.
- Reading the vectors back to the host per slot would stall once per slot and sever the gradient. The effect head's loss reaches the ability encoder through this path and no other, so the model would optimize the head against a frozen random encoder and still report a falling loss.

## 2. Every per-field number is read back once, not once per field

- Loss parts come back stacked on the device and are read with one `.tolist()`. Each read synchronizes the device; a read per field stalled every validation batch some forty-five times.
- The per-entity loss scores its fields by kind, not one by one: fields of one type and width are one call. `test_effect_losses.py` checks the loss, every part and the gradient against the field-by-field loop.

## 3. `encode-abilities` batches by unique text

- `encode_tree` deduplicates every line of the whole tree by its encoder input (text and role spans), encodes the distinct inputs once and copies the rows back into each sidecar's line order.
- It loads the checkpoint once and calls `eval()`. Without it dropout would make two runs of the same command write different caches.
- Noise on `e` is applied in the training batcher only, never here.

## 4. A training step encodes its texts in length-sorted groups

- `encode_by_length` (`text_staging.py`) sorts a step's texts by length and pads each group to its own longest, for both the main pass and the MLM pass.
- Padded to the step's longest instead, one 512-token chain makes every line in the step cost 512 positions, and the activations saved for backward overran the 8 GB card at 16 records per batch.
- Padding is masked out of attention, so grouping changes no output and no gradient; `test_encode_by_length.py` checks both.

## 5. The trainer holds one shard, never the corpus

- A parsed record costs about 45 KB resident, so the whole corpus would need hundreds of gigabytes. Loading it was an out-of-memory crash that printed nothing first, because the load ran before the first log line.
- Under `--prefetch-workers N` each worker process holds the one shard it is preparing; under `0` the training process holds one.
- Two consequences are deliberate:
  - Rarity weighting reads the manifest's corpus-wide table, never a count of the resident shard. Coverage and variant shards are built dense in scarce texts on purpose, so a per-shard count would read exactly those texts as common and down-weight them.
  - The split and the validation samples are read whole from what `build-corpus` wrote, never accumulated from what a run happens to read. A run that stops early records the same split as one that finishes, and a validation set redrawn each epoch would make `--patience` measure which records got drawn.

## 6. A step's host half is prepared in a worker process, and every draw is seeded by the step's place in the run

- A step has a host half (the plan, tokenizing, surfaces, targets, the MLM mask, all CPU tensors; `application/step_preparation.py`, with `shard_plan.py` (a shard's weights, labels and weighted batches), `lethality_augmenter.py` (the lethality copies) and `head_target_builders.py` / `record_head_targets.py` (the heads' targets)) and a device half (encoder, noise on `e`, model, losses, backward; `application/step_loss.py`). The host half runs in spawned processes (`application/step_workers.py`) because on threads it held the interpreter lock against the training step.
- No `EffectRecord` crosses the process boundary. A task names a shard path and what returns is finished tensors, each step packed into one flat buffer (`infrastructure/tensor_packing.py`): `torch.multiprocessing` pays a shared-memory mapping per tensor, and on Windows that cost is per tensor rather than per byte.
- No draw depends on where a step was prepared. The shuffle draws from `plan_rng(seed, epoch, position)` and each step from `step_rng(seed, epoch, position, step)`, so a run is the same run under any worker count, `0` included. The draws, and on one CPU thread the weights, are equal; torch's default thread count and CUDA bf16 change the last float bits. `test_training_loop_workers.py` checks it on one thread.
- No target depends on which shards a preparer read before, since each worker reads its own share of them. A memo on `ApiTargetBuilder` (`StepPreparer.head_builder.api`) holds what one sidecar line determines, keyed by that line's text, and is never read for a text the step's batch holds no line for (`ApiTargetBuilder.batch`: a granted keyword's text has a line in one batch and none in another).
- `e` is never prepared ahead. The encoder and the noise draw stay in the device half, or the gradient would not reach the encoder and the device generator would be read out of order.
- A worker never exits on its own while the run lasts, even after failing: on Windows a queued step's shared memory dies with the process that made it. Its exception reaches the training process at the shard it belongs to, so the CLI maps a worker's refusal to the same exit code as the in-process path. The pool terminates every worker on every way out, and a watchdog thread exits a worker whose training process was killed.
- A shard's rarity weights read every record's text, so the texts are resolved for the whole shard; a record's effect family, which only the epoch line counts, is resolved when a batch draws it (`ShardPlan.family`, once per record). `ShardClosed.unresolved` therefore tallies one lookup per record for its text and a second for each drawn record's family.
- Validation's host half is prepared once per field set, not every epoch: the `Validator` holds each sample's prepared steps (`validation_pass.HostSteps`), prepared on the sample's first pass and again only when the field set or the preparer change, which is the curriculum epoch alone. The steps are read, never edited, by the scoring pass and its collectors.
- Gate 1, gate 2, `model_loading` and the probes call `build`, the two halves back to back in process. Validation runs the same two halves as training.
