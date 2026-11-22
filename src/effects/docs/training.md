# Training

Rules to respect when running or changing `train-effect-model`. Each says what must hold and what breaks when it does not. This file does not describe the trainer:

- Flags and defaults: `python -m effects train-effect-model --help`, `TrainEffectModelConfig` in `application/train_effect_model.py`.
- What is trained and how it is scored: `specs/2026-10-30-ability-effect-model-gen3.md` § 11–13.
- Why the design is what it is: `experiments/2026-09-04-ability-effect-model-design.md`, `experiments/2026-09-18-effect-model-gen2-improvements-design.md`, `experiments/2026-10-30-effect-model-gen3-design.md`.

## Inputs and the split

- `--corpus` is the trainer's only input. The split, both validation strata and the rarity table come from its manifest. Nothing in the trainer computes a split or sweeps the raw corpus, and no flag may bring one back (`--split-from`, `--context-cache` and `--cache-refresh` are gone on purpose). Every `--variant` run inherits the same split.
- The holdout is keyed on ability text, never on card name. Functional reprints compile to the same script, so a name-keyed holdout trains on Lightning Strike while holding out Searing Spear. The rule itself is in `corpus-building.md`.
- The trainer never filters a shard. It refuses one holding a game the manifest places in a validation stratum (`ValidationGameInTrainingError`), because that shard and the manifest come from different builds. Do not turn the refusal into a filter.
- A manifest whose `subtypes` or `keywords` hold a name the checked-in closed output lists (`output_vocabularies.SUBTYPES_CHANGED`, `KEYWORDS_CHANGED`) lack stops the run at startup (`ClosedListMismatchError`, `application/closed_list_check.py`): no column would hold that name, so the change would train as "none gained". The refusal names the missing names and `scripts/scan_output_vocabularies.py --manifest`, whose regeneration changes the output widths and so needs a new run. A name the lists hold and the manifest lacks is no refusal.
- A corpus with no frozen selection set stops the run at startup (`SelectionSetError`). Freezing one: `evaluation.md`.

## Sampling

- An epoch's shards are drawn at random across the whole training-shard list, never taken as a contiguous run. The list is in path order, so a contiguous run is one collection run's consecutive worker lifetimes.
- Record weights count effective games, not records. One long game writing hundreds of records of one ability must not make that ability look well observed.
- The weights drive one weighted shuffle without replacement over the resident shard. The class mixture is fixed on disk by `build-corpus --class-mix`; the trainer does not re-mix per batch.
- `--grad-accum` counts micro-steps within the epoch and drops a partial accumulation at the epoch's end. Carrying it over would shift every later epoch's optimizer boundary.

## Validation and checkpoint selection

- Validation is scored in `--batch-size` batches over the shuffled sample, never one game per batch. `active_fields` keeps a field only while some class in the batch supervises it, so a whole-game batch measures a smaller objective than the training batch printed beside it.
- The gate-1 margins on the epoch line are read over the whole card-disjoint sample. They are a progress reading, not the evaluator's gate, which scores the unique-text stratum alone.
- Validation, gate 1, the breakdowns, the knowledge probes and the control score all read the model through one chunked forward (`application/record_forward.py`): evaluation mode, no gradient, bf16 autocast on CUDA. A second forward path would report a checkpoint at a different precision from the one it was selected at.
- Validation, early stopping and every saved checkpoint read the averaged weights (`application/weight_average.py`), never the trained ones. The best checkpoint is the epoch with the best control score on the averaged weights, and the curriculum switch resets the best and the patience.
- A field's validation loss is reported with its explained deviance against the constant-predictor floor. A loss in nats alone cannot tell a head that learned the field from one that learned how often it fires.

## Optimizer and memory

- The encoder and the head are separate parameter groups, clipped and reported apart. The encoder's own gradient norm is the only evidence that gradient reaches it through `e`; one shared norm would be dominated by the head.
- On CUDA the per-process memory reservation is capped and the allocator's cache is released once per shard. The corpus's variable batch shapes fragment the caching allocator, and a Windows driver pages the excess to host RAM instead of raising out-of-memory.
- The hot-path invariants are in `training-hot-paths.md`.

## Text and auxiliary heads

- Training, evaluation and `encode-abilities` all encode through `AbilityTokenizer.encoding_tokens`. Under `--withhold-keyword` it expands the withheld keyword to its reminder text and drops its token everywhere, inside another keyword's expansion too (flying's reminder text names flying). A second encoding path would leak the token or make the cache disagree with the model.
- Noise on `e` is applied in the training batcher only. It never reaches validation, evaluation or `encode-abilities`.
- The value head's and the script-API head's targets are read from the sidecar's `script_text` on both surfaces, because prose has no `$` parameter to read. A non-literal value masks its target.
- The value head's at-least-k bits (k = 1…40) are averaged so the block weighs one binary target. They exist to make `e` carry amounts as a thermometer the trunk can compare.
- The MLM pass zeroes a masked position's numeric value and number flag along with its token. The encoder adds the number embedding beside the token's, so leaving them would hand the masked digit back.

## One batching pipeline

- `application/surface_batching.py` turns records into the trunk's inputs (`TrunkInputs`), and the trainer and gate 1's evaluator both go through it. They must agree exactly.
- The encoding text has one definition, `SurfaceBatcher.text_of`. With two, one caller keys a batch's encodings by prose and the other looks them up by script: the effect head trains on an all-zero `e`, every loss still falls, and the four evaluation baselines come out identical to the shipping model.
- The variant masks have one definition, `SurfaceBatcher._masked`. They are the whole difference between a baseline and the model gate 1 compares it against.
  - `state-only` zeroes every `e`, `[ACT]`'s included.
  - `no-state` zeroes the context abilities' `e` but keeps `[ACT]`'s, and zeroes every card and player feature and every `[GLOBAL]` input except the trailing record-kind, moment and subkind one-hots (`KIND_FLAG_WIDTH`). The average-effect control sees which ability acts and what kind of record it is, and nothing of the board.
  - The masks zero inputs and never drop slots, so the per-entity targets keep their shape.

## Layout and magnitudes

- A checkpoint records its input layout and the ordered lists behind it (relation types, subtypes, named counters, the closed output lists). Every loader refuses another layout or a differing list (`CheckpointLayoutError`). Reordering any of those lists invalidates every existing checkpoint. See `model-layout.md` and `checkpoints.md`.
- Every count and signed-delta magnitude is capped at 40 (`MAGNITUDE_CAP`) in four places that must agree: the field losses, the created-objects counts, the constant-predictor floor and the evaluation's deviance. Above the cap an amount is the game compounding, not the text.
- The thermometer codes on card and player slots use the same 40 steps, so a stated amount compares with power, toughness and life in one attention score.

## Lethality copies

- `--lethality-copies K` appends counterfactual copies of damage and -X/-X resolutions with the toughness and the amount redrawn and the deaths relabelled by rule 704.5f-g (`domain/lethality_counterfactuals.py`). Forge's AI aims removal at creatures it can kill, so the corpus alone rarely shows damage falling short, and the model learns "small creatures die" instead of comparing amount with toughness.
- `K = 0` must train bit for bit as a run without the feature.
- Only natural records qualify for a copy: no fork, intervention, directed fork or synthetic record, no hit creature whose zone outcome is unknown, and no record whose hit creature a regeneration shield or a replacement of its move or destruction may save.

## Losses

- Every head reduces the same way (`effect_losses.per_row`): summed over a row's targets, averaged over the batch's rows.
- The affected gate trains on every entity with a slot (`model-layout.md` § Input slots) of a record that is not a playability record. A field trains only where the gate's target fires and the field is present. The legality fields are the exception: they train wherever present, gate or no gate.
- A field is present where the record wrote it, or where it has a silent value, the record's sampling class supervises it and the entity is in the field's scope. Silent values are "nothing happened": zone stayed, no damage, no counters, no change.
- `tap_state`, both durations, the legality bits and `min_blockers` have no silent value and train only where an event or a payload sets them. A creature no event mentions is neither tapped nor untapped; do not give these a default.
- A continuous contribution that displaces a line (a `=`, or a whole-class token: `model-layout.md` § Outputs) on an entity its board lacks marks the matching `*_lost` field unknown, and no silent value overrides it.
- A `trigger` record derives no per-entity targets. Its events are what caused it; what it does arrives as its resolution record. It supervises the verdict head's fired bit, and the gate trains on its entities as negatives.
- Per-entity counts are scored from 0 to the cap, never below zero (`damage_taken` nets healing).
- A `zone_change` to a zone no zone outcome names (`stack`, `merged`, `sideboard` and Forge's other rare zones) marks the moved card's zone outcome unknown, never "stayed".
- A `destroyed` or `sacrificed` event is overridden by the move that follows it, so a destruction an exile replacement turned into an exile reads `exiled`.
- A `life_change` or `counter_change` whose provenance `mode` names a loss (`event_schema.LOSS_MODES`, pinned to Forge's enums by a test) is a negative delta whatever sign its amount carries.
- Damage dealt as -1/-1 counters (wither or infect to a creature) adds nothing to `damage_taken`: the record's own `counter_change` adds the counters. A `deathtouch` hit matched by such a counter change is taken back out too, because Forge reports a deathtouch source with wither or infect as `deathtouch`.
- Three events name no output column and are counted per record in `RecordTargets.unsupervised`, their subjects affected and nothing else written: a hidden-keyword `keyword_change`, a `name_change` and a `mana_cost_change`. Supervising the last two needs a layout change. The same counter holds `subtype_outside_closed_list` and `keyword_outside_closed_list`: one count per changed subtype or keyword (event, created token or continuous line, `closed_list_misses.py`) that the closed output lists lack and no column therefore holds.
- A phased-out permanent and a last-known copy (role `last_known`, whatever zone it last had) are entities of the board but not `EntityState.on_battlefield`, so no attack, block, state-based or grid reading reaches them.
- A rewrite's set-aside ability's run (`buffered_events`) is supervised on the record that holds the run. A rewrite whose incoming event is unknown supervises no outgoing event and trains no affected gate.
- A mana record's own events beside `tapped` and `mana_produced` are the mana ability's outcomes and are supervised as any resolution's.
- The game-end marker names nobody and derives no targets.
- Every shipped term is divided by its loss denominator, its constant-prediction loss measured once before the first step. The denominators are stored in `resume.pt` and in every checkpoint, and a resume reuses them rather than recomputing. The loss descended is scaled, while the parts the log prints stay in nats so they compare with the floor.

## The rule mask

- Under `--rule-mask hard` (the default), a copy that rule budgets added trains only the effective outputs of the rules it was drawn for, and the gate only on its drawn rule's tag entities. Under `soft` it trains everything, the other outputs at `natural_share / copies`.
- A natural copy of a record with no added copy, a natural copy of an aimed record and a lethality copy train everything at full weight.
- The value, script-API and MLM heads read texts and no mask reaches them. Evaluation reads records unmasked.
- Training, validation and the denominator pass all build the mask through the same host half (`StepPreparer.rule_mask`), and the constant-predictor floor weighs each target as the run does.
