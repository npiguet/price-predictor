# Implementation Plan: Ability effect model

**Branch**: `023-ability-effect-model` | **Date**: 2026-09-06 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/023-ability-effect-model/spec.md`

## Summary

Build a pretrained model of what each card ability does in play, shipping two artifacts: a per-ability
embedding cache (one fixed-width vector `e` per unique ability line, computed offline) and a
state-conditional effect head that predicts an ability's effect on a given game state. Training data
is a corpus of game-effect records collected from instrumented Forge matches, written as append-only
JSONL shards under one fixed schema.

The work lands as a new `src/effects/` package plus Java collectors in `forge-connector`, staged in
four increments that each widen the corpus without invalidating earlier records. Stage one needs no
Forge patch at all — it collects through the public event bus and a bracket around stack resolution —
so the first embeddings and all three evaluation gates exist before any engine patch is written.

## Technical Context

**Language/Version**: Python 3.14.3 (`requires-python >=3.14`); Java 17 for the collectors
**Primary Dependencies**: torch ≥ 2.2 (CUDA 12.6 wheel index), numpy ≥ 1.26; Java side compiles
against forge-game / forge-core / forge-gui / forge-ai `2.0.15-SNAPSHOT` from the sibling `../forge`
checkout, plus the engine patch set under `forge-connector/patches/`
**Storage**: append-only JSONL record shards (`output/effects/records/`), per-card JSON provenance
sidecars beside the converted corpus, `.npz` ability caches (`output/effects/abilities/`), `.pt`
checkpoints (`models/effects/`)
**Testing**: pytest for the fast unit suite (`tests/unit/effects/`), JUnit 5 for `forge-connector`;
anything requiring a JVM carries the `integration` marker declared at `pyproject.toml:36`
**Target Platform**: local Windows 11 workstation with a single CUDA GPU; workers are headless JVMs
**Project Type**: CLI-driven ML pipeline — a third Python package alongside `price_predictor` and
`sealed`, plus Java worker mains in the existing `forge-connector` module
**Performance Goals**: observational record kinds cost no extra simulation — they ride matches that
were going to be played anyway; fork-based collection is budgeted per game (`--interventions-per-game`,
`--probes-per-game`, both defaulting to 2)
**Constraints**: 8 GB VRAM is the binding limit on batch composition; the engine patch must be
re-applied after every Forge upgrade, and workers must degrade rather than fail without it; collection
must never write to the sealed corpora
**Scale/Scope**: 33,680 converted cards / 66,080 ability lines / 37,787 unique line texts as the
encoding surface; the record corpus reaches the order of 10^8 records at full collection, of which
training sees a sampled fraction

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status |
|---|---|
| I. Fast Automated Tests | **Pass.** Domain layers (record schema, geometry of the effect-head input, split derivation, sampling mixture, gate arithmetic) are pure and unit-testable with no JVM and no GPU. JVM-dependent tests carry the existing `integration` marker so the fast suite stays fast. |
| II. Simplicity First | **Pass.** No new abstraction is introduced speculatively; the one extraction proposed (§ Codebase Survey, third-instance check) is justified by three existing implementations, meeting the "three concrete use cases" bar. Staging exists to avoid building fork machinery before a canary says it is needed. |
| III. Data Integrity | **Pass.** The record schema is fixed before collection and versioned by construction (later stages widen, never redefine). Splits and vocabulary/keyword-definition hashes are recorded in the checkpoint and re-checked at inference, so a run is reproducible against the exact corpus and vocabulary it trained on. |
| IV. DDD & Separation of Concerns | **Pass.** `effects/domain` holds the record schema, input geometry, and model definitions with no torch-free violations of layering; `application` holds the collectors, trainer, and evaluator; `infrastructure` holds JSONL IO, the checkpoint store, the CLI, and the Java connectors. Dependencies point inward, and `effects` imports from `sealed` and `price_predictor` but never the reverse. |
| V. Forge Interoperability | **N/A for this feature.** The Java stub library and its remote API are untouched. This feature adds CLI worker mains to `forge-connector`, which is the module's second, already-established role. |
| VI. Documentation | **Deferred to implementation, tracked.** New CLI subcommands and artifacts require README and package-`CLAUDE.md` updates in the same change that adds them; this is a task in `tasks.md`, not an assumption. |
| VII. Codebase-Aware Planning | **Pass with one required follow-up.** See below. |
| VIII. Performance-Conscious Implementation | **Applies.** See below. |

### Codebase Survey (Principle VII — required)

Full findings: [research.md § Codebase Survey](research.md#codebase-survey).

- **Overlapping vocabulary**: 2 concepts reused (`ConvertedCardText`, `compute_basic_lands`), 1
  extended (`ConvertedCardLocator`, gaining source-tree awareness), 2 parallel concepts introduced
  with justification (`ability_cache_layout`, the effect record dataclasses). 0 renames required.
- **Adjacent prior art**: 6 reused (`ForgeWorkerPool`, `torch_checkpoint`, `clip_per_group`,
  `build_vocabulary` + sealed's truncation, `probe_lib` ridge harness, `ConvertedCardLocator`),
  2 extended additively on the Java side (`RulesParser` for the sidecar, `MatchWorkerMain` for
  instrumentation), 1 reimplemented with a documented reason (the card-disjoint split, whose
  stratification key differs from sealed's).
- **Convention alignment**: mirrors `src/draft/` file-for-file; no deviation proposed.
- **Third-instance check**: **one extraction required.** Partial-line-tolerant append-only reading
  exists three times (`pool_file_reader`, `cards_played_reader`, `draft_record_io` — the last
  documenting that it mirrors the other two). The effects reader would be a fourth. Extract the
  completeness primitive to `src/price_predictor/infrastructure/append_only.py` and refactor the three
  call sites onto it.

**Follow-up tasks this surfaced, to be carried into `tasks.md`:**

1. Extract `append_only.py` and refactor the three existing readers onto it — a prerequisite for the
   effects record reader, not a later cleanup.
2. Extend `ConvertedCardLocator` with source-tree awareness rather than adding a parallel locator.
3. Reuse sealed's `_truncate_to_target_size` from `effects build-vocab` rather than copying it.

### Performance Review (Principle VIII — required when applicable)

This feature both moves data and runs model compute, so the checklist applies in full.

- **I/O batching & caching** — *Addressed.* Ability `e` vectors are computed once per unique ability
  text and persisted under `output/effects/abilities/`; consumers read the cache, never the encoder.
  Within a training batch each unique ability text is encoded once, not once per occurrence. Provenance
  sidecars are read once per card and held; record shards are read as a stream.
- **GPU placement** — *Addressed.* Both transformers (ability encoder and effect-head trunk) and their
  inputs are co-located on the GPU for training, encoding, and evaluation.
- **GPU batching** — *Addressed.* Batches mix several games and group each game's records so grouped
  records share a board, which is what makes one encode serve many records. No per-record host↔device
  transfer sits in the training loop; per-epoch metric reads are hoisted out. The documented fallback
  when live context re-encoding exceeds the 8 GB budget is `--context-cache` (a stop-gradient momentum
  cache refreshed every `--cache-refresh` batches), not a smaller batch.
- **Streaming & load-once** — *Addressed.* The record corpus is append-only JSONL read as a stream and
  never fully materialized; the tokenizer, vocabulary, keyword definitions, and model are constructed
  once per run and reused.

No optimization beyond this checklist is planned. Anything further must be backed by a profile
identifying the hot path (Principle II).

## Project Structure

### Documentation (this feature)

```text
specs/023-ability-effect-model/
├── plan.md              # This file
├── research.md          # Phase 0 output — codebase survey and decisions
├── data-model.md        # Phase 1 output — record schema and model entities
├── quickstart.md        # Phase 1 output — stage-one operator walkthrough
├── contracts/           # Phase 1 output — CLI and corpus contracts
│   ├── cli.md
│   ├── record-schema.md
│   └── provenance-sidecar.md
├── checklists/
│   └── requirements.md  # Spec quality checklist (from /speckit.specify)
└── tasks.md             # Phase 2 output (/speckit.tasks — NOT created here)
```

### Source Code (repository root)

```text
src/effects/                          # new package, mirrors src/draft/
├── domain/
│   ├── records.py                    # record envelope, per-kind payloads, event list
│   ├── event_schema.py               # canonical event vocabulary + per-type normalization
│   ├── state_snapshot.py             # snapshot blocks, entities, inclusion tiers
│   ├── effect_head_input.py          # snapshot -> [GLOBAL]/[ACT]/[PLAYER]/[CARD] token geometry
│   ├── ability_encoder.py            # encoder + [CLS] pooling to e
│   ├── effect_model.py               # trunk + per-entity / created-objects / verdict heads
│   ├── ability_cache_layout.py       # (n_lines, e_dim) cache layout and row alignment
│   ├── provenance.py                 # provenance key type and sidecar shape
│   ├── damage_step_keywords.py       # gate-2 table: predicate, direction, affected fields
│   ├── ward_twins.py                 # ward canary's functional-twin texts
│   └── script_variants.py            # perturbation whitelist (stage four)
├── application/
│   ├── collect_coverage.py
│   ├── collect_variants.py           # stage four
│   ├── build_vocab.py                # wraps price_predictor.build_vocabulary
│   ├── extract_keyword_definitions.py
│   ├── train_effect_model.py
│   ├── encode_abilities.py
│   └── evaluate_effect_model.py
└── infrastructure/
    ├── cli.py                        # python -m effects
    ├── record_io.py                  # JSONL shard reader/writer
    ├── sidecar_io.py                 # provenance sidecar reader
    ├── effect_model_store.py         # checkpoint save/load via torch_checkpoint
    ├── ability_cache_store.py        # .npz cache read/write
    └── collector_connector.py        # spawns the instrumented Java worker

src/price_predictor/infrastructure/
└── append_only.py                    # NEW, extracted (third-instance check)

forge-connector/
├── patches/                          # engine patch set, applied to ../forge
└── src/main/java/com/pricepredictor/connector/
    ├── effects/                      # collector classes
    ├── KeywordDefinitionMain.java    # new worker main
    ├── MatchWorkerMain.java          # extended: --effect-records
    └── RulesParser.java              # extended: emits the provenance sidecar

tests/unit/effects/                   # mirrors src/effects/, fast, no JVM/GPU
tests/integration/                    # JVM-dependent, `integration` marker
```

**Structure Decision**: a third top-level Python package, `src/effects/`, laid out in the same
hexagonal shape as `src/price_predictor/` and `src/sealed/` and mirroring `src/draft/` file-for-file
(see [research.md § Convention alignment](research.md#convention-alignment)). Java collectors go in the
existing `forge-connector` module rather than a new one, because that module's second documented role
is already "CLI workers invoked by the Python side" and the collectors share its Forge classpath and
`ForgeEnvironmentInitializer` bootstrap. The one file outside these two homes is the extracted
`append_only.py`, which belongs in `price_predictor` because that is the package both `sealed` and
`draft` already depend on.

### Post-design re-check

Re-evaluated after Phase 1 (`data-model.md`, `contracts/`, `quickstart.md`). No gate changed status.
Three things the design surfaced that the pre-design check did not:

- **Principle III gained a concrete mechanism.** The reproducibility requirement is met by the
  checkpoint recording its split *and* the content hashes of the vocabulary and keyword-definition
  files, with every inference command re-checking them ([contracts/cli.md](contracts/cli.md) §
  exit-code contract). Schema evolution is governed by the five compatibility rules in
  [contracts/record-schema.md](contracts/record-schema.md), which make widening the corpus safe and
  redefinition impossible.
- **Principle IV held under the Java boundary.** The collectors sit in `forge-connector` and reach
  Python only by writing shard files, so no Forge type crosses into `effects/domain`. The provenance
  key is the only shared vocabulary, and it is a tuple of strings and integers.
- **Principle I is satisfiable per stage, not only at the end.** Every domain module in the structure
  above is pure — record schema, event vocabulary, snapshot tiers, input geometry, split derivation,
  gate arithmetic — so the fast suite covers the contracts without a JVM or a GPU. The gate-2 table and
  the ward-twin list are checked-in data, which makes both gates unit-testable against fixtures.

## Complexity Tracking

No constitution violations require justification. The two parallel domain concepts introduced
(`ability_cache_layout`, the effect record dataclasses) are recorded in
[research.md § Overlapping domain vocabulary](research.md#overlapping-domain-vocabulary) with their
reasons and are permitted by Principle VII's clause (c); neither displaces an existing concept, so no
rename is owed.
