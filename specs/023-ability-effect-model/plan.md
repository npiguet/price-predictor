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
checkout, whose `effect-record-hooks` branch carries the engine hooks
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
re-applied after every Forge upgrade, and workers must degrade rather than fail without it; the
coverage and variant collectors must never write to the sealed corpora, and instrumenting
`match-outcomes` must leave that command's own two outputs untouched
**Scale/Scope**: 33,680 converted cards / 66,080 ability lines / 37,787 unique line texts as the
encoding surface; the record corpus reaches the order of 10^8 records at full collection, of which
training sees a sampled fraction

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status |
|---|---|
| I. Fast Automated Tests | **Pass.** The record schema, event vocabulary, snapshot tiers, and effect-head input geometry are pure `domain` modules; split derivation, the sampling mixture, and gate arithmetic sit in `application` but are pure functions of their inputs. All are unit-testable with no JVM and no GPU. JVM-dependent tests carry the existing `integration` marker so the fast suite stays fast. |
| II. Simplicity First | **Pass.** No new abstraction is introduced speculatively. Three extractions are proposed and each has a reason that is not "it might be useful later": `append_only.py` has four call sites and meets the "three concrete use cases" bar outright; the `_truncate_to_target_size` promotion has only two, but the alternative is a private cross-context import that Principle IV forbids; and the ridge-probe harness has to move regardless, because `scripts/` is not an importable package. Staging exists to avoid building fork machinery before a canary says it is needed. |
| III. Data Integrity | **Pass.** The record schema is fixed before collection, and the compatibility rules make later stages widen it rather than redefine it. There is no `schema_version` field — the root spec defines none — so the discipline is enforced by the schema contract and its tests, not by a version stamp; that is the honest scope of the claim. The split and the vocabulary/keyword-definition hashes are recorded in the checkpoint and re-checked at inference, so a run is reproducible against the exact games and vocabulary it trained on. The hashes cover those two files only — a reconversion that leaves the vocabulary unchanged passes, so corpus regeneration stays an operator-sequencing concern, stated as such in the spec's Assumptions. |
| IV. DDD & Separation of Concerns | **Pass, with one documented repo-wide deviation.** `effects/domain` holds the record schema, snapshot, input geometry, and model definitions; `application` holds the collectors, trainer, and evaluator; `infrastructure` holds JSONL IO, the checkpoint store, the CLI, and the Java connectors. Dependencies point inward, and `effects` imports from `sealed` and `price_predictor` but never the reverse. The deviation: `domain/ability_encoder.py` and `domain/effect_model.py` import torch, against the principle's "domain MUST be free of framework imports". This follows established repo convention — `sealed/domain/scorer_model.py`, `sealed/domain/encoder_model.py`, and `draft/domain/draft_agent_model.py` all do the same, because a model architecture *is* domain here and torch is its notation. Introducing a framework-free model abstraction for this one feature would diverge from three siblings. Raised rather than assumed; if the project wants it resolved, that is a constitution amendment, not a per-feature choice. |
| V. Forge Interoperability | **N/A for this feature.** The Java stub library and its remote API are untouched. This feature adds CLI worker mains to `forge-connector`, which is the module's second, already-established role. |
| VI. Documentation | **Deferred to implementation, tracked.** New CLI subcommands and artifacts require updates to the README, the new `src/effects/CLAUDE.md`, and the **root `CLAUDE.md`** (which says "Three Python packages live under `src/`" and carries the per-corpus file-format contracts, both of which this feature changes), in the same change that adds them; this is a task in `tasks.md`, not an assumption. |
| VII. Codebase-Aware Planning | **Pass with one required follow-up.** See below. |
| VIII. Performance-Conscious Implementation | **Applies.** See below. |

### Codebase Survey (Principle VII — required)

Full findings: [research.md § Codebase Survey](research.md#codebase-survey).

- **Overlapping vocabulary**: 2 concepts reused (`ConvertedCardText`, `compute_basic_lands`), 1
  extended (`ConvertedCardLocator`, gaining source-tree awareness), 2 parallel concepts introduced
  with justification (`ability_cache_layout`, the effect record dataclasses). 0 renames required.
- **Adjacent prior art**: 4 reused as they stand (`ForgeWorkerPool`, `torch_checkpoint`,
  `clip_per_group`, `build_vocabulary`), 2 reused only after being moved somewhere importable (the
  `--target-size` truncation, the `encoder_probes` ridge harness), 1 extended
  (`ConvertedCardLocator`, also counted under overlapping vocabulary), 2 extended additively on the
  Java side (`RulesParser` for the sidecar, `MatchWorkerMain` for instrumentation), 1 mirrored (the
  one-connector-per-Java-main convention), and 1 reimplemented with a documented reason (the
  card-disjoint split, whose stratification key and exclusion rule both differ from sealed's).
- **Convention alignment**: mirrors `src/draft/` file-for-file; no deviation proposed.
- **Third-instance check**: **one extraction required here, plus two forced elsewhere.** Partial-line-tolerant append-only reading
  exists three times (`pool_file_reader`, `cards_played_reader`, `draft_record_io` — the last
  documenting that it mirrors the other two). The effects reader would be a fourth. Extract the
  completeness primitive to `src/price_predictor/infrastructure/append_only.py` and refactor the three
  call sites onto it.

**The import surface FR-002 declares, made explicit.** FR-002 names `manabase.compute_basic_lands` as
the `sealed` surface, which is what `effects` needs for deck building — but the plan also reaches into
`sealed` three other ways, and an import-direction test cannot be written against an implicit list. The
allowed set is:

| From | Symbol | Why |
|---|---|---|
| `price_predictor` | tokenizer, `build_vocabulary` (+ the promoted truncation), `forge_jvm` | FR-002's declared surface, verbatim |
| `price_predictor` | `torch_checkpoint`, `torch_training.clip_per_group`, `append_only`, `ridge_probes` | widens FR-002; the last two are extractions this plan lands there |
| `sealed.domain` | `manabase.compute_basic_lands` | FR-002, verbatim |
| `sealed.infrastructure` | `ConvertedCardLocator` | the extension decided in the survey; a parallel locator would duplicate its prefix-fallback lookup and letter-index caching (it *imports* `sanitize_card_name` and `FILENAME_CORRECTIONS` rather than owning them) |
| `sealed.domain` / `sealed.infrastructure` | `card_embedding_layout`, `embedding_store` | FR-115's scorer smoke test writes the sealed `.npz` layout into a scratch tree |
| `sealed.application` | **nothing** | `train-scorer` Phase A is re-run as a subprocess, not imported, which keeps the application layers disjoint |

Anything outside this table is a boundary violation, and the import-direction test asserts exactly it.
Every row above is FR-002 as amended. The table originally widened the declared boundary in three
rows, and widening one is a spec change rather than a plan-level decision, so FR-002 was amended to
declare the full surface and FR-090 to include the withheld keyword. The import-direction test now
asserts a surface the spec describes.

**Follow-up tasks this surfaced, to be carried into `tasks.md`:**

1. Extract `append_only.py` and refactor the three existing readers onto it — a prerequisite for the
   effects record reader, not a later cleanup.
2. Extend `ConvertedCardLocator` with source-tree awareness rather than adding a parallel locator.
3. Promote `_truncate_to_target_size` from `sealed/application/build_vocab.py` into
   `price_predictor/application/build_vocabulary.py` and have both `sealed` and `effects` call it
   there. Importing it where it sits would mean reaching into another context's private symbol,
   which Principle IV forbids and which FR-002's declared `sealed` surface
   (`manabase.compute_basic_lands`) does not cover.
4. Extract the ridge-probe helpers (`_ridge_solve`, `fit_probes`, `HeadProbe`, `ProbeSet`) from
   `scripts/encoder_probes/probe_lib.py` into `price_predictor/application/ridge_probes.py`, leaving
   the script as a caller. FR-114 cannot import them where they sit: `scripts/` is not a package and
   the module runs side effects and hardcoded paths at import time.
5. Amend FR-002's declared import surface and FR-090's checkpoint contents to match the table above,
   before the import-direction test is written against them.

**Constitution follow-ups.** Two principles are deviated from below with reasons stated. The
constitution's Disputes rule says a conflict is resolved by amending it, not by bypassing it, so both
need an amendment task in `tasks.md` rather than standing as per-feature waivers: Principle IV's
"domain MUST be free of framework imports" (broken today by `sealed/domain` and `draft/domain`, and by
`effects/domain` once this lands — torch being the notation model architecture is written in) and Principle VII clause (c)'s unconditional rename requirement
(which has no target when the parallel concept's name does not collide).

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
  transfer sits in the training loop; per-epoch metric reads are hoisted out. The root spec's documented
  fallback when live context re-encoding exceeds the 8 GB budget is `--context-cache`, a stop-gradient
  momentum cache refreshed every `--cache-refresh` batches.
- **Vectorize hot loops** — *Addressed.* The two per-item Python loops this feature adds are the
  snapshot-to-tensor derivation in `domain/effect_head_input.py` and per-line JSON parsing in
  `infrastructure/record_io.py`. The first builds one batch's entity tensors with numpy/torch array
  operations over the whole batch rather than per entity; the second is I/O-bound line parsing, where
  the lever is the streaming pass above, not vectorization. Neither does element-wise arithmetic in a
  Python loop.
- **Streaming & load-once** — *Addressed.* The record corpus is append-only JSONL read as a stream and
  never fully materialized. The trainer is the one consumer that needs records indexed by game and by
  sampling class, and it builds those indexes one shard at a time: a shard is parsed, trained on for
  its share of the epoch's steps, and released before the next is read, so resident memory is one
  shard whatever the corpus size. The tokenizer, vocabulary, keyword definitions, and model are
  constructed once per run and reused.

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
│   ├── ability_tokenizer.py          # wraps MtgTokenizer: [MASK]/[CLS], role spans
│   │                                 #   by character offset, keyword expansion,
│   │                                 #   compositional script selectors (stage four)
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
    ├── collector_connector.py        # spawns the instrumented Java worker
    ├── keyword_definition_connector.py  # spawns KeywordDefinitionMain
    └── castability_connector.py      # spawns CastabilityMain

src/price_predictor/
├── infrastructure/
│   ├── append_only.py                # NEW, extracted (third-instance check)
│   └── cli.py                        # extended: convert also emits sidecars and
│                                     #   converts tokenscripts -> output/tokenscripts/
├── application/
│   ├── build_vocabulary.py           # gains the promoted target-size truncation
│   └── ridge_probes.py               # NEW, extracted from scripts/encoder_probes/
└── (domain unchanged)

src/sealed/                           # the stage-one collection opt-in lives here
├── infrastructure/
│   ├── converted_card_locator.py     # extended: source-tree awareness
│   ├── cli.py                        # extended: --effect-records + cap/budget flags
│   │                                 #   on match-outcomes
│   └── match_worker_connector.py     # extended: forwards them as -D properties
├── application/
│   ├── match_outcomes.py             # extended: threads the flags to the workers
│   └── build_vocab.py                # calls the promoted truncation helper
└── (domain unchanged)

forge-connector/
└── src/main/java/com/pricepredictor/connector/
    ├── effects/                      # collector classes
    ├── KeywordDefinitionMain.java    # new worker main
    ├── CastabilityMain.java          # new worker main: FR-047's consult
    ├── ConvertMain.java              # extended: token-script tree + sidecar output
    ├── MatchWorkerMain.java          # extended: --effect-records, plus a
    │                                 #   records-only mode (no output.file, no
    │                                 #   CardsPlayedWriter) for the effects collectors
    └── RulesParser.java              # extended: emits the provenance sidecar

tests/unit/effects/                   # mirrors src/effects/, fast, no JVM/GPU
tests/integration/                    # JVM-dependent, `integration` marker
```

**Structure Decision**: a third top-level Python package, `src/effects/`, laid out in the same
hexagonal shape as `src/price_predictor/` and `src/sealed/` and mirroring `src/draft/` file-for-file
(see [research.md § Convention alignment](research.md#convention-alignment)). Java collectors go in the
existing `forge-connector` module rather than a new one, because that module's second documented role
is already "CLI workers invoked by the Python side" and the collectors share its Forge classpath and
`ForgeEnvironmentInitializer` bootstrap.

Three files sit outside those two homes, each for a reason the survey established: the extracted
`append_only.py` belongs in `price_predictor` because that is the package both `sealed` and `draft`
already depend on; `price_predictor`'s `convert` (and its Java `ConvertMain` / `RulesParser`) is
extended because FR-011 and FR-007 put the sidecar and the token-script tree there, not in `effects`;
and `sealed`'s `ConvertedCardLocator` gains source-tree awareness rather than being duplicated.

### Post-design re-check

Re-evaluated after Phase 1 (`data-model.md`, `contracts/`, `quickstart.md`). No gate changed status.
Three things the design surfaced that the pre-design check did not:

- **Principle III gained a concrete mechanism.** The reproducibility requirement is met by the
  checkpoint recording its split *and* the content hashes of the vocabulary and keyword-definition
  files, with every inference command re-checking them ([contracts/cli.md](contracts/cli.md) §
  exit-code contract). Schema evolution is governed by the four compatibility rules in
  [contracts/record-schema.md](contracts/record-schema.md), which make widening the corpus safe, enforced by
  contract tests rather than by a version field.
- **Principle IV held under the Java boundary.** The collectors sit in `forge-connector` and reach
  Python only by writing shard files, so no Forge type crosses into `effects/domain`. The provenance
  key is the only shared vocabulary, and it is a tuple of strings and integers.
- **Principle I is satisfiable per stage, not only at the end.** The record schema, event vocabulary,
  snapshot tiers, and input geometry are pure `domain` modules testable with no JVM and no GPU. Split
  derivation, the sampling mixture, and gate arithmetic live in `application`
  (`train_effect_model.py`, `evaluate_effect_model.py`) but are pure functions of their inputs and
  testable the same way. The gate-2 table and the ward-twin list are checked-in data, which makes both
  gates unit-testable against fixtures.

## Requirement traceability

Every requirement in [spec.md](spec.md) has an owning module or file. The rows below are its twelve
`####` sections verbatim, with their real FR ranges, so the table cannot drift out of step with the
spec's numbering — sub-ranges invented here would. This is the map `tasks.md` should follow.

| Spec section (FR range) | Owner |
|---|---|
| Package layout and boundaries (FR-001…006) | the source tree above; FR-002's dependency direction enforced by an import-direction test over the surface named below, FR-005's degraded-mode detection by the Java collectors and the `mode` field |
| Ability identity (FR-007…012) | `RulesParser.java` + `ConvertMain.java`, surfaced by `price_predictor/infrastructure/cli.py` (write); `effects/domain/provenance.py` + `infrastructure/sidecar_io.py` (read); `contracts/provenance-sidecar.md` |
| Corpus (FR-013…029) — envelope, snapshot, events, payloads, caps and budgets | `effects/domain/records.py`, `state_snapshot.py`, `event_schema.py`; caps surfaced by `infrastructure/cli.py` and enforced in the Java collectors; `contracts/record-schema.md` |
| Collectors (FR-030…043) — channels, brackets, continuous, mana, playability, interventions, probes, fork guards | `forge-connector/.../effects/` + the `effect-record-hooks` branch of `../forge`; `effects/infrastructure/collector_connector.py` for the effects-owned supervisors. **FR-030/031's opt-in on `sealed match-outcomes`** is owned by `sealed/infrastructure/cli.py`, `sealed/application/match_outcomes.py`, and `sealed/infrastructure/match_worker_connector.py` — the User Story 1 acceptance path runs through sealed, not through `effects`. **FR-042** (the real-vs-fork diff is never a training target) is a constraint *on* the evaluator, honoured in `application/evaluate_effect_model.py`. **FR-052 and FR-059** (coverage and variant matches write effect records only) are enforced in `MatchWorkerMain`'s records-only mode, not in the Python supervisors: they reuse that worker, which today requires `-Doutput.file` and builds a `CardsPlayedWriter` unconditionally, so a Python-side guard would not bind |
| Coverage collector (FR-044…052) — rounds, split exclusion, deck weighting, castability-ranks-not-drops, the satisfaction predicate, retirement, and the two residues | `effects/application/collect_coverage.py`, except **FR-047's castability consult**, which needs a live Forge: `CastabilityMain.java` behind `effects/infrastructure/castability_connector.py`, keeping the repo's one-connector-per-main convention |
| Synthetic script variants (FR-053…059) — perturbation, output tree, script-surface-only, volume cap, and split inheritance | `effects/application/collect_variants.py`, `effects/domain/script_variants.py` |
| Keyword definitions (FR-060) | `KeywordDefinitionMain.java`, `effects/infrastructure/keyword_definition_connector.py`, `effects/application/extract_keyword_definitions.py` |
| Model (FR-061…082) — encoder surfaces and vocabulary, tokenization (`domain/ability_tokenizer.py`, because the shared `MtgTokenizer` returns bare strings with no character offsets and seeds neither `[MASK]` nor `[CLS]`), number and role embeddings, keyword-expansion dropout, effect-head input geometry, per-kind input variation, the three output heads, losses, curriculum | `effects/domain/ability_encoder.py`, `effect_head_input.py`, `effect_model.py`; `effects/application/build_vocab.py`; save-time head filtering in `infrastructure/effect_model_store.py` |
| Training (FR-083…095) — batching, mixture, rarity weighting, records with no acting text, splits and their whole-game exclusion, split/hash provenance, checkpoints, variants | `effects/application/train_effect_model.py`. **FR-092** sits in this section but binds the *inference* commands: the hash check belongs to `encode_abilities.py` and `evaluate_effect_model.py` |
| CLI surface (FR-096…097) | `effects/infrastructure/cli.py`; `contracts/cli.md` |
| Embedding cache (FR-098…105) — cache layout, idempotence, variant resolution, and the downstream card representation incl. the `[ALTERNATE]` layout tag | `effects/application/encode_abilities.py`, `effects/domain/ability_cache_layout.py`, `effects/infrastructure/ability_cache_store.py` |
| Evaluation (FR-106…124) — strata, the twelve reported checks, and **all three gates with their numeric thresholds** | `effects/application/evaluate_effect_model.py`; `effects/domain/damage_step_keywords.py` (gate 2's per-keyword table), `effects/domain/ward_twins.py` (the ward canary's twin list) |

The gate thresholds are contract and live in spec.md FR-118…FR-123, not restated here: gate 1's three
margins (≥ 0.05 affected-gate F1, ≥ 0.05 zone-outcome accuracy, ≥ 5% relative Poisson-deviance
reduction), gate 2's 200-record minimum and 70% direction threshold, and gate 3's ≤ 0.5 mean pairwise
cosine over 10,000 pairs and ≤ 30% top principal component. `tasks.md` must carry them into the
evaluator's tests.

Two requirements deserve a pointer because they are easy to lose between owners:

- **FR-088's whole-game exclusion** — every game holding a record that names a held-out card is
  excluded from training, not merely the record. It belongs to the split derivation in
  `train_effect_model.py` and is the guard gate 1 and SC-009 rest on; see
  [research.md § The split excludes whole games](research.md#the-split-excludes-whole-games-not-just-held-out-rows).
- **FR-117's keyword withholding** — the trainer must be able to withhold one implemented keyword's
  token so the zero-shot check has something to measure. Withholding is a trainer flag, but the
  *identity* of the withheld keyword is checkpoint state: the evaluator reads it from `--checkpoint`
  alongside the split, exactly as it reads every other datum that must describe the model as trained
  rather than as remembered.

## Complexity Tracking

| Item | Why needed | Simpler alternative rejected because |
|---|---|---|
| Two parallel domain concepts (`ability_cache_layout`, the effect record dataclasses) | The effects cache is an `(n_lines, e_dim)` matrix row-aligned to a sidecar, not the sealed `.npz`'s single pooled vector; the effect record is a different domain from `DraftRecord` | Reusing either would mean overloading a type whose shape, arity, and consumers all differ |
| Principle VII clause (c) asks for a rename of the older concept when a parallel one is introduced; no rename is proposed | Neither new concept displaces or shadows an existing one — the names do not collide (`card_embedding_layout` vs `ability_cache_layout`; `DraftRecord` vs `EffectRecord`), so there is nothing for a rename to disambiguate | Renaming `card_embedding_layout` or `DraftRecord` would churn two working modules to satisfy the letter of a clause aimed at name collisions. **Raised here rather than silently reinterpreted**, per the constitution's Disputes rule |
