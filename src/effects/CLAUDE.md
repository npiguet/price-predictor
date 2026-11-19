# `effects` — ability effect model

Entry point: `python -m effects <subcommand>` (see `src/effects/__main__.py` and `infrastructure/cli.py`). A pretrained model of **what each card ability does in play**, shipping two artifacts: a per-ability embedding cache (one fixed-width vector `e` per unique ability line, computed offline) and a state-conditional effect head that predicts an ability's effect on a given game state. `effects` **imports from** `sealed` and `price_predictor`; never the reverse, and never `sealed.application`. The boundary is asserted by `tests/unit/effects/test_import_boundaries.py`.

The three generations are three features, each with its run order in its `quickstart.md`: 023 (`specs/023-ability-effect-model/`), 024 (`specs/024-ability-effect-model-gen2/`) and 025 (`specs/025-ability-effect-model-gen3/`, root spec `specs/2026-10-30-ability-effect-model-gen3.md`). The spec-kit contracts under those directories hold each generation's full contract and its changes; a topic file below is the current state across all generations.

## Topic files — read on demand

This file holds what applies to any work in the package. The detail is in `src/effects/docs/`, one file per topic, and none of it loads by itself. Read a topic file when its condition below is met, in full, before starting; do not read the others.

| Read | Before |
|---|---|
| `docs/record-format.md` | reading or writing effect-record shards or records in any code or script; changing a record member |
| `docs/text-encoding.md` | touching provenance sidecars, the tokenizer, the vocabulary, keyword definitions, the ability cache or the variant-script tree; running `extract-keyword-definitions`, `build-vocab` or `encode-abilities` |
| `docs/collection-internals.md` | changing a Java collector, a hook, a cap flag or the random seat; interpreting what a record's board, events, duration or absence of an event means |
| `docs/forks.md` | changing or diagnosing interventions, damage-step probes, protection forks or directed forks; registering a directed rule |
| `docs/collection-runs.md` | running, babysitting or changing `collect-coverage`, `collect-variants`, `validate-corpus`, `field-coverage` or a `sealed match-outcomes --effect-records` run; changing how workers are recycled, stopped or killed; proposing a change to the workers' heap or lifetime |
| `docs/rules-catalogue.md` | editing `rules/`, a detector family or rule tagging; running or reading `rule-counts` |
| `docs/corpus-building.md` | running or changing `build-corpus` or `holdout-cards`; anything that depends on the split, the strata, curation or the manifest |
| `docs/training.md` | running or changing `train-effect-model`, a loss, the rule mask or a training flag; changing `application/surface_batching.py` or a baseline variant's masks |
| `docs/training-hot-paths.md` | changing the batcher, step preparation, the step workers, loss read-back, shard loading or `encode-abilities`' encoding loop |
| `docs/checkpoints.md` | changing what a checkpoint or `resume.pt` holds, a loader, or `--resume` |
| `docs/evaluation.md` | running or changing `evaluate-effect-model`, a gate, the knowledge or embedding probes, the control score or the selection set |
| `docs/model-layout.md` | changing the input layout, the relation attention or an output head |

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

Collection itself is an opt-in on a `sealed` command, `python -m sealed match-outcomes --effect-records DIR` (`docs/collection-runs.md`).

## Rules for any work in the package

- The record schema is fixed within a generation before that generation's first record is collected: a later capability adds `kind` values and members and never redefines a field, so it widens the corpus without invalidating what came before. A new generation may change a field's meaning. Every consumer reads gen-3 records only. `tests/unit/effects/domain/test_schema_compatibility.py` enforces the compatibility rules.
- `mode`, `interventional`, `fork`, `fork_kind`, `synthetic`, `random_seat`, `what_if`, `generation`, `aimed_at`, `directed`, `copy_dropped` and `curation` are collection and curation metadata and never reach the model.
- The raw corpora (`records` runs under `output/effects/<gen>/`) are append-only and cannot easily be rebuilt. A format spans the Java writers and the Python readers, no single source file teaches a whole one, and a mismatch corrupts a corpus without raising anything: read the format's topic file (§ File formats) before touching either side.
- No command recomputes a split. It comes from the curated corpus's manifest at training and from the checkpoint afterwards, because the corpus is append-only and a recomputed split would score the gates partly on games the model trained on.
- The real-versus-fork difference is computed at evaluation time and never becomes a training target: a model trained on its own disagreements would learn to reproduce its errors.
- The collecting workers' small heap (`-Xmx1200m`) and their recycling are the containment of Forge's late-life pathologies, not tuning knobs. Do not propose changing either without saying what contains Forge instead (`docs/collection-runs.md` § Short-lived workers).
- Worker restarts are the normal operating condition of a collection. Anything that counts, joins or resumes over shards must be correct across hundreds of JVM lifetimes and across a JVM ended mid-game (`docs/collection-runs.md` § Restarts are the normal condition).
- A collection needs the engine hooks below: what gen 3 collects beyond `degraded` records depends on them. Judge an engine by its gaps and not just the mode: a partly-applied patch or a stale installed Forge jar still reads `patched` while a channel is empty, and `PatchHooks.report()` names each missing hook, listener callback and argument (`docs/collection-internals.md` § The hook inventory). Every collecting supervisor stops a run on an engine that is not whole, unless `--allow-degraded` (`docs/collection-runs.md` § The workers' engine), and every collecting command refuses to start over a converted tree converted at another Forge commit than its connector JAR's (`docs/collection-runs.md` § The conversion the keys join against).
- Several failures here look like a working run, with every loss still falling: a second definition of the encoding text or of a baseline's masks (`docs/training.md` § One batching pipeline), `e` read back by value instead of entering the head by row index (`docs/training-hot-paths.md`), a baseline's embedding table rebuilt empty at load (`docs/checkpoints.md` § Loading). Read the named file before changing the code it covers.
- Every subcommand that writes files keeps a `run.json` run manifest in its run directory (`.claude/skills/artifact-layout/SKILL.md`), declared in `RUN_MANIFESTS` in `infrastructure/cli.py`. A collector's is its shard directory, `evaluate-effect-model`'s its `--output`, the directory it draws into; `validate-corpus`, `field-coverage` and `build-corpus --verify` write none. No shard, sidecar or cache reader takes it for data, since each finds its files by suffix.
- A change to a behaviour a topic file describes updates that file in the same change (the `maintaining-claude-md` skill).

## The engine hooks

The hooks live on the `effect-record-hooks` branch of the sibling `../forge` checkout, an ordinary branch: changes land as new commits on top, and a Forge upgrade is merged in, never rebased under them. This repository holds no exported patch series; the branch is the history.

```bash
cd ../forge && git checkout effect-record-hooks   # or: git merge master, after an upgrade
mvn -pl forge-core,forge-game,forge-ai -am install -DskipTests
cd ../price-predictor/forge-connector && mvn install -DskipTests
```

Rebuild the fat JAR afterwards, always: the worker links against the freshly installed Forge jars.

## File formats

Where each format is described:

- Effect-record shards, the record envelope and dropped-event counts: `docs/record-format.md`.
- Provenance sidecars, keyword definitions, the ability cache and variant scripts: `docs/text-encoding.md`.
- The selection set: `docs/evaluation.md`.
- The curated dataset and its manifest: `docs/corpus-building.md`.
- Checkpoints and `resume.pt`, in the run directory `--model-output` names: `docs/checkpoints.md`.

Paths follow the `artifact-layout` skill: effects work reads and writes only under `output/effects/<gen>/` and `models/effects/<gen>/`, every path passed explicitly. No flag defaults to a path under `output/` or `models/`: the trainer's `--model-output`, `encode-abilities --checkpoint`, the records, corpus, keyword, vocabulary, variant-script, cache, report and evaluated-checkpoint paths are required flags (`--effect-records` on a collector may be replaced by `--training-corpus`), and `domain/default_paths.py` holds the sources only (`DEFAULT_CARDS_FOLDERS`, the gen-3 converted tree, and Forge's card scripts and MTGJSON's dump). A config field with no default is a required flag (`cli.add_config_flags`). Each command's `RUN_MANIFESTS` entry lists every path it reads, those it takes from a corpus manifest or a checkpoint included (`infrastructure/recorded_inputs.py`), and a refusal's reason reaches `run.json`'s outcome (`cli.exits_with`). The ability cache (`encode-abilities --output-root`) is never put in a converted card tree: `sealed encode-cards --clean` deletes every `.npz` where it encodes, by default that tree.
