# `effects` — ability effect model

Entry point: `python -m effects <subcommand>` (see `src/effects/__main__.py` and `infrastructure/cli.py`). A pretrained model of **what each card ability does in play**, shipping two artifacts: a per-ability embedding cache (one fixed-width vector `e` per unique ability line, computed offline) and a state-conditional effect head that predicts an ability's effect on a given game state. `effects` **imports from** `sealed` and `price_predictor`; never the reverse, and never `sealed.application`. The boundary is asserted by `tests/unit/effects/test_import_boundaries.py`.

The record schema is fixed within a generation before that generation's first record is collected, so a later capability widens the corpus without invalidating what came before; a new generation may change a field's meaning, and every consumer but `rule-counts` reads gen-3 records only. Collection also runs on stock Forge, through the public event bus and a bracket around stack resolution, stamping every record `degraded`; what gen 3 collects beyond that needs the hooks of the `effect-record-hooks` branch (§ The engine hooks). The three generations are three features: 023 (`specs/023-ability-effect-model/`), 024 (§ Generation 2) and 025 (§ Generation 3).

Every subcommand that writes files keeps a `run.json` run manifest in its run directory (`.claude/skills/artifact-layout/SKILL.md`), declared in `RUN_MANIFESTS` in `infrastructure/cli.py` (a collector's is its shard directory, `evaluate-effect-model`'s the `<checkpoint stem>.evaluation/` it draws into; `validate-corpus`, `field-coverage` and `build-corpus --verify` write none); no shard, sidecar or cache reader takes it for data, since each finds its files by suffix.

## Topic files — read on demand

This file holds what applies to any work in the package. The detail is in `src/effects/docs/`, one file per topic, and none of it loads by itself. Read a topic file when its condition below is met, in full, before starting; do not read the others.

| Read | Before |
|---|---|
| `docs/record-format.md` | reading or writing effect-record shards or records in any code or script; changing a record member |
| `docs/text-encoding.md` | touching provenance sidecars, the tokenizer, the vocabulary, keyword definitions, the ability cache or the variant-script tree; running `extract-keyword-definitions`, `build-vocab` or `encode-abilities` |
| `docs/collection-internals.md` | changing a Java collector, a hook, a cap flag or the random seat; interpreting what a record's board, events, duration or absence of an event means |
| `docs/forks.md` | changing or diagnosing interventions, damage-step probes, protection forks or directed forks; registering a directed rule |
| `docs/collection-runs.md` | running, babysitting or changing `collect-coverage`, `collect-variants`, `validate-corpus` or `field-coverage` |
| `docs/rules-catalogue.md` | editing `rules/`, a detector family or rule tagging; running or reading `rule-counts` |
| `docs/corpus-building.md` | running or changing `build-corpus` or `holdout-cards`; anything that depends on the split, the strata, curation or the manifest |
| `docs/training.md` | running or changing `train-effect-model`, a loss, the rule mask or a training flag |
| `docs/training-hot-paths.md` | changing the batcher, step preparation, the step workers, loss read-back, shard loading or `encode-abilities`' encoding loop |
| `docs/checkpoints.md` | changing what a checkpoint or `resume.pt` holds, a loader, or `--resume` |
| `docs/evaluation.md` | running or changing `evaluate-effect-model`, a gate, the knowledge or embedding probes, the control score or the selection set |
| `docs/model-layout.md` | changing the input layout, the relation attention or an output head |

A topic file is the current state across all generations. The spec-kit contracts under `specs/023-ability-effect-model/`, `specs/024-ability-effect-model-gen2/` and `specs/025-ability-effect-model-gen3/` hold each generation's full contract and its changes. A change to a behaviour a topic file describes updates that file in the same change.

## Subcommands

- `extract-keyword-definitions` — writes Forge's keyword → reminder-text table (`docs/text-encoding.md`).
- `build-vocab` — builds the vocabulary from converted cards, token scripts and the keyword definitions; the script surface has a vocabulary path of its own (`docs/text-encoding.md`).
- `train-effect-model` — trains encoder and effect head jointly from a curated `--corpus`, its only input (`docs/training.md`).
- `build-corpus` — curates the raw shard corpus into a fixed dataset: the split, the three strata and `manifest.json` (`docs/corpus-building.md`).
- `encode-abilities` — computes the per-ability embedding cache once, offline (`docs/text-encoding.md`).
- `collect-coverage` — plays decks over the whole converted corpus, weighted toward the cards with the fewest records (`docs/collection-runs.md`).
- `collect-variants` — perturbs Forge source scripts and plays the variants, untargeted or steered at short rules (`docs/collection-runs.md`).
- `holdout-cards` — writes the depletion list `sealed generate-pools --exclude-cards` reads (`docs/corpus-building.md`).
- `validate-corpus` — measures the corpus invariants over a shard directory; meant for the first minutes of a collection pass (`docs/collection-runs.md`).
- `field-coverage` — names every record field a corpus never varied (`docs/collection-runs.md`).
- `rule-counts` — tags every record with the catalogue's detectors and writes per-rule counts (`docs/rules-catalogue.md`).
- `evaluate-effect-model` — the three gates and the reported checks; splits come from the checkpoint, never from a flag (`docs/evaluation.md`).

## Generation 2

Feature 024 (`specs/024-ability-effect-model-gen2/`) changed what is decided at collection time, what curation keeps, and what training and evaluation report. Run order: `specs/024-ability-effect-model-gen2/quickstart.md`.

## Generation 3

Feature 025 (`specs/025-ability-effect-model-gen3/`, root spec `specs/2026-10-30-ability-effect-model-gen3.md`) ties the corpus, the model's inputs and outputs, curation and checkpoint selection to the rules of the game. Run order: `specs/025-ability-effect-model-gen3/quickstart.md`.

The topic files describe both generations where they apply today: gen-3's curation, rule mask, loss denominators, averaged weights, selection set and control score are described beside gen-2's counterparts.

## Collection

Instrumentation is an opt-in on a command that already exists:

```bash
python -m sealed match-outcomes --effect-records output/effects/records/
```

`--effect-records` has **no default** there — collection is the opt-in, not the point, and a default would quietly turn every self-play run into a collection run. Absent the flag nothing changes: no shard is opened, and `match-outcomes.txt` and `cards-played.txt` keep their exact format and content. The instrumentation costs no extra simulation; it rides matches that were going to be played anyway.

Attribution mode is **detected at worker startup**, not compiled in: the patch lives in the sibling `../forge` checkout, which is rebuilt independently and re-patched by hand after every Forge upgrade, so a build-time switch would silently mislabel a corpus collected after the patch lapsed. Every record carries the detected `mode`.

`MatchWorkerMain` also has a **records-only mode**: with `-Deffect.records.dir` set and `-Doutput.file` absent, neither sealed writer is constructed at all. The coverage and variant collectors reuse this worker, and their decks are built for coverage rather than for a fair self-play sample — mixing them into the sealed corpus would corrupt the scorer's training data.

### Why the workers are short-lived, and what follows from it

Every collecting supervisor runs its Forge workers under `-Xmx1200m` and has `ForgeWorkerPool` kill the longest-running one every status interval (60 s), restarting it at once. **Neither is a tuning knob, and they are one measure rather than two.** Forge misbehaves late in a JVM's life — loops that do not terminate, allocation that grows without bound — and the pair is the containment: the small heap turns a runaway worker into an early, cheap death the pool restarts instead of one worker dragging the machine down hours into a run, and the hard recycle bounds the pathologies that do not allocate. Raising the heap or lengthening the interval hides the pathology rather than containing it. The price — a handful of GC crashes per overnight run, each costing the block of records in flight and nothing else — is paid deliberately. Do not propose removing either without saying what contains Forge instead; a corpus analysis nearly recommended undoing the heap setting before learning why it exists.

**What follows is a correctness constraint, not a throughput one.** Restarts are the normal operating condition: an eight-hour run is roughly 530 JVM lifetimes across six worker slots, many of them killed mid-game by `taskkill /F /T`, which runs no shutdown hook. So record identity, game boundaries, and any resume logic have to be correct across hundreds of restarts *and* across a JVM killed mid-game:

- Anything a worker counts from zero must be namespaced by the JVM **lifetime**, not by the worker slot. The first corpus was not: 71–75% of its `record_id` values were repeats and 804 `game_id` values covered 31,662 games, merging unrelated games under the corpus's own join key. That is why the lifetime segment is in the id and in the shard name.
- A file boundary is not a game boundary. A game can be cut in half by a kill and its remainder never written.
- A shard ends at its last complete 256-record block, so the last ≤255 records of each lifetime — about 0.5% of a run, always a game's tail — are absent. A resume that assumes a clean shutdown left a marker is wrong.
- Catch this in minutes, not hours: `python -m effects validate-corpus --effect-records <dir>` over the first few minutes of a pass. Every defect above was visible in the first minute of shards and none of them raised anything for eight hours.

### The engine hooks

The hooks live on the **`effect-record-hooks` branch of the sibling `../forge` checkout**, an ordinary branch: changes land as new commits on top, and a Forge upgrade is merged in rather than rebased under them. There is no exported patch series in this repository: the branch is the history, and a directory of `.patch` files beside it was a second copy of that history maintained by hand, which git already does better.

`PatchHooks.REQUIRED` is the inventory of what a patched Forge must have, and `PatchHooks.report()` prints what is missing at worker startup. A partly-applied patch still reads `patched` while a channel is quietly empty, so read that line and not just the mode (`docs/collection-internals.md` § The hook inventory).

```bash
cd ../forge && git checkout effect-record-hooks   # or: git merge master, after an upgrade
mvn -pl forge-core,forge-game,forge-ai -am install -DskipTests
cd ../price-predictor/forge-connector && mvn install -DskipTests
```

Rebuild the fat JAR afterwards, always: the worker links against the freshly installed Forge jars. Applied, a run reaches **all eight** sampling classes; unapplied, three, and every record is stamped `degraded`.

### Forks

An interventional resolution forks the game and forces an ability the AI never casts; a damage-step probe re-runs one combat with a keyword stripped; a protection fork and a directed fork are played on to the end of the turn. How each is taken, checked and budgeted: `docs/forks.md`.

The real-versus-fork difference is computed at **evaluation time** and never becomes a training target: a model trained on its own disagreements would learn to reproduce its errors rather than correct them.

## The gates

`evaluate-effect-model` reports three gates and exits 0 whatever their verdict, printing `GATES FAILED: …`. Gate 1, the identity baseline, asks whether the encoder is reading text (pass/fail). Gate 2, the damage-step keyword canary, only routes: its per-keyword verdict decides whether the probe machinery is worth building. Gate 3, the collapse canaries, checks the embedding geometry (pass/fail). Thresholds, populations and how each is read: `docs/evaluation.md` § The gates.

## File formats

Append-only data contracts spanning the Java writers and the Python readers — no single source file teaches a whole one, and a mismatch silently corrupts a corpus that cannot easily be rebuilt.

Where each format is described: effect-record shards, the record envelope and dropped-event counts in `docs/record-format.md`; provenance sidecars, keyword definitions, the ability cache and variant scripts in `docs/text-encoding.md`; the selection set in `docs/evaluation.md`; the curated dataset and its manifest in `docs/corpus-building.md`.

Three rules hold whatever the task:

- `mode`, `interventional`, `fork`, `synthetic`, `random_seat`, `what_if`, `generation`, `aimed_at`, `directed` and `curation` are collection and curation metadata and never reach the model.
- The record envelope is frozen within a generation: a later capability adds `kind` values and members, and never redefines a field. `tests/unit/effects/domain/test_schema_compatibility.py` enforces the compatibility rules.
- The ability cache lives under `output/effects/abilities/`, outside `output/cardsfolder/`, because the sealed pipeline's `encode-cards --clean` deletes every `.npz` under that tree.

## One pipeline, two callers

`application/surface_batching.py` turns records into the trunk's inputs (`TrunkInputs`), and both the trainer and gate 1's evaluator go through it. They have to agree exactly: when they did not — one keying the batch's encodings by prose, the other looking them up by script — the effect head trained on an all-zero `e`, every loss still fell, and the four evaluation baselines came out byte-identical to the shipping model. The encoding text therefore has **one** definition (`SurfaceBatcher.text_of`), and so do the variant masks (`SurfaceBatcher._masked`, FR-094), which are the whole difference between a baseline and the model gate 1 compares it against. `state-only` zeroes every `e`, `[ACT]`'s included. `no-state` zeroes the context abilities' `e` but keeps `[ACT]`'s, and zeroes every card and player feature and every `[GLOBAL]` input except the trailing record-kind, moment and subkind one-hots (`KIND_FLAG_WIDTH`): the average-effect control sees which ability acts and what kind of record it is, and nothing of the board. The masks zero inputs rather than drop slots, so the per-entity targets keep their shape.

`application/model_loading.py` is the other half: it rebuilds encoder, model and batcher from a saved checkpoint, the weights read by `effect_model_store.load_weights`. The `identity` baseline is the case that would break quietly — its `e` comes from a free embedding table rather than the encoder, so the table rides in the checkpoint's `extra` and a run that rebuilt it empty would score the baseline on random vectors and hand gate 1 a margin nobody earned. The table indexes texts exactly: one row per text the run's sidecars encode to (`ability_text.known_texts`), plus one shared row for any other text, such as a counterfactual copy's. Its texts are saved in the checkpoint beside the weights, and the runner rebuilds the table from them; a hashed table would let two texts share a row, which is the memorization the baseline exists to measure. Loading is strict: a model state missing anything but the training-only heads, or carrying an unexpected key, is refused rather than run at random initial weights, and the encoder and model come back in eval mode.

## Performance invariants

Six things on the hot paths are load-bearing rather than incidental, and each has a failure mode that looks like a working run. The six, with the reason and the failure mode of each in `docs/training-hot-paths.md`:

- `e` enters the head by row index, never by value. Reading the vectors back per slot severs the gradient to the encoder while the loss still falls.
- Every per-field number is read back once, not once per field.
- `encode-abilities` batches by unique text, in `eval()` mode.
- A training step encodes its texts in length-sorted groups.
- The trainer holds one shard, never the corpus; rarity weights and the split come from the manifest, never from what a run happens to read.
- Each training step's host half is prepared in a worker process, every draw is seeded by the step's place in the run, and `e` is never prepared ahead.

## Model artifacts

Checkpoints live under `models/effects/effect-model/` (`{timestamp}.pt` plus a rolling `latest.pt`); a variant writes to `models/effects/effect-model/{variant}/`, so a baseline run can never overwrite the checkpoint the cache was built from. A checkpoint records the split it trained against, the vocabulary and keyword-definition hashes, the corpus manifest's digest and its input layout. Every inference command re-hashes what it loaded and fails fast on a mismatch, and no command recomputes a split. `resume.pt` is not a checkpoint. What each holds and what a loader or a resume refuses: `docs/checkpoints.md`.
