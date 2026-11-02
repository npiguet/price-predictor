# Implementation Plan: Ability effect model — generation 3

**Branch**: `025-ability-effect-model-gen3` | **Date**: 2026-10-31 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/025-ability-effect-model-gen3/spec.md`

## Summary

Gen-3 indexes the effect-record corpus by the rules of the game and trains rule by rule. The work
falls into six layers, built in the spec's story order:

1. **Catalogue and detectors.** Scenarios gain detector declarations, operation classifications and
   departures. Detectors are parameterized families in `effects/domain/rule_detectors/`, and
   `rule-counts` runs them over gen-2's corpus before any gen-3 collection.
2. **Collection.** A gen-3 record schema with every line of a permanent, keyword lines, the new
   snapshot facts and relation sources, the combat snapshot before damage, a held effect half that
   spans the state-based loop, a collector-side playability enumerator for both seats, land plays included, with
   `timing` and `failed_checks`, `turn_action` records, per-attacker records, and the dropped-event
   count. Three engine hooks and a `GameCopier` patch land on `effect-record-hooks`.
3. **Directed forks.** A Java registry of directed rules (board precondition plus per-variant
   applications), ten fork operations, and a driver that plays a fork to the end of the turn with
   its own collectors and seats.
4. **Model.** Width 256 throughout, the gen-3 input layout, a relation-aware attention layer replacing
   `nn.TransformerEncoderLayer`, the 22 new outputs, gate-free legality outputs, and the negative
   binomial. Gen-1 and gen-2 checkpoints stop loading.
5. **Curation and training.** Per-rule budgets after the natural share, per-record rule copies with a
   `curation` annotation, global stratum placement steered by per-game rule inventories, the rule
   mask, loss denominators from a startup pre-pass, and averaged weights.
6. **Selection.** Probe, grid and rule-test scoring move from `scripts/` into the package. A frozen
   selection set with halves, balance and worst-case standard errors drives a per-epoch control score.

Most of it extends features 023 and 024 in place. Decisions are in [research.md](research.md); the
survey found five places where the spec stated a mechanism the code contradicts, and those are
corrected in `spec.md` and the root spec (research S1).

## Technical Context

**Language/Version**: Python 3.14 (`requires-python >=3.14`); Java 17 for `forge-connector` and the
Forge patch branch
**Primary Dependencies**: torch ≥ 2.2 (CUDA 12.6 wheel index), numpy, scikit-learn (probes); Forge
`2.0.15-SNAPSHOT` jars from `../forge` on `effect-record-hooks`, with new commits (three hooks, the
`GameCopier` copy)
**Storage**: append-only JSONL.gz record shards; provenance sidecars; `.npz` ability caches; `.pt`
checkpoints; curated corpus directories with a JSON manifest; `rules/` JSON and JSONL; selection-set
and scorecard JSON
**Testing**: pytest fast suite (`tests/unit/effects/`, `tests/unit/scripts/`); JUnit 5 for
`forge-connector` (`ForgeExtension`, `TestBoard`, `TestCollectors`); Forge-side
`EffectRecord*Test`; JVM-dependent Python tests carry the `integration` marker
**Target Platform**: local Windows 11 workstation, one CUDA GPU with 8 GB VRAM
**Project Type**: CLI-driven ML pipeline (`src/effects/`) with Java worker mains (`forge-connector/`)
and an engine patch branch (`../forge`)
**Performance Goals**: one training step at width 256 with relations fits 8 GB at batch 32 (SC-005);
in-epoch selection scoring costs at most a fifth of an epoch's training time (research E2);
`build-corpus` keeps two passes over the records (research R3)
**Constraints**: `build_corpus`, `corpus_plan`, detectors and the freeze stay torch-free; gen-3
records only past `rule-counts`; no Python call from the JVM during a game; collection metadata
(`generation`, `aimed_at`, `directed`, `curation` aside) never reaches the model
**Scale/Scope**: 410 units, 3,166 rules (1,422 mechanical, 226 legality), 1,763 scenarios, 2,488
non-base variants, 1,647 heads lines; corpus of the order of 10⁷ records per collection run; slots per
record p99 195 on gen-2 records, more in gen-3

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status |
|---|---|
| I. Fast Automated Tests | **Pass.** Detectors, rule budgets, placement, the selection-set freeze and its standard errors, the relation layer's four properties, the averaged weights, the negative binomial and the rule mask are each pure functions tested without a JVM or, for the layer, on CPU with a tiny config. Collector changes and the fork operations are JUnit-tested on scripted Forge games as feature 023's are. Directed forks played forward are an `integration` test. |
| II. Simplicity First | **Pass with notes.** Detectors are families, not one function per scenario (research R1). One enumerator serves decision records and the random seat. One scorer serves selection, reporting and the probe scripts. The clean break (M1) removes three compatibility paths instead of adding a fourth. |
| III. Data Integrity | **Pass.** The schema change is versioned by `generation` (FR-018), and every consumer but `rule-counts` refuses earlier generations. Gen-3 is collected from scratch, so no migration is needed (Assumptions). Placement, halves, subsampling and rule copies are seeded and deterministic for a given corpus. Placement is no longer stable across corpus growth (R3); the docs that promised it are corrected. Map snapshot, selection set and checkpoints each carry content digests. |
| IV. DDD & Separation of Concerns | **Pass.** Detectors, budgets, placement, the selection set, score scales and rule outcomes live in `effects/domain`. Orchestration lives in `application`, IO and the CLI in `infrastructure`. The architecture exception covers `relation_attention.py`. Probe cores move into the package, so `scripts/` imports `effects` and never the reverse. Board preconditions live in `forge-connector` (FR-078). |
| V. Forge Interoperability | **N/A.** The stub library and remote API are untouched. Collection stays in `forge-connector`'s worker role. Engine changes are new commits on `effect-record-hooks`. |
| VI. Documentation | **Tracked as tasks.** `CLAUDE.md`'s effect-record paragraph, the base record-schema contract's gen-3 section, `rules/README.md`'s new scenario fields, `src/effects/CLAUDE.md` (commands, flags, placement), and the README's effects workflow, each in the change that alters the behaviour. |
| VII. Codebase-Aware Planning | **Pass.** See below. |
| VIII. Performance-Conscious Implementation | **Applies.** See below. |

### Codebase Survey (Principle VII — required)

Full findings: [research.md § Codebase Survey](research.md#codebase-survey).

- **Overlapping vocabulary**: 50 existing concepts surveyed. Most are extended in place, 9 replaced,
  1 removed. 12 new concepts, each citing its nearest prior art. 3 renames: `rule_families` →
  `effect_families`, `_selection_half` → `_what_if_half`, `PROBE_TYPES` → `PROBE_MODELS`.
- **Adjacent prior art**: reuse of `Forks.copyAndVerify`, `mainLoopStep`, Forge's public card and
  player mutators, live trait lists, per-check restriction methods, `run_parallel`/`merge_into`, the
  water-filling allocator for the natural share, and lethality copies, whose qualify-then-copy shape
  protection copies follow in a module of their own (Decision R5). `FloatingEffects` extends the
  layer-table walk of `LayerContributions` (Decision C4). Patched in Forge: the
  delayed-trigger accessor and the `GameCopier` copy. One documented reimplementation: a
  collector-side `MustBlock` predicate, because Forge has none per (blocker, attacker).
- **Convention alignment**: features 023 and 024's `src/effects/` layout and `FE/` collectors. One
  deviation: probe scoring moves from `scripts/` into the package, because training scores it every
  epoch.
- **Third-instance check**: 13 extractions (`ForkRun`, `PriorityCandidates`, `TargetingClauses`,
  `KeywordEdit`, `seeded_draw`, one canonical digest, `score_scales`, `rule_outcomes`, detectors as
  the shared tagger, generalized `loss_floor`, one chunked forward, a derived head-output list, one
  catalogue schema module).

**Follow-up tasks this surfaced, to be carried into `tasks.md`:**

1. The three renames, each before the new concept that would collide with it lands.
2. Extract `ForkRun` from `InterventionFork` and `DamageStepProbe` before the directed-fork driver.
3. Extract `TargetingClauses` and `PriorityCandidates` before the enumerator (FR-030–033).
4. Move the probe cores (`READOUTS`, `ladder.score`, `probe_set`, grid rule functions) into `E`, with
   the scripts importing them, before the control score.
5. Move the scenario and heads enums into `E/domain/rule_catalogue.py`; derive `HEAD_OUTPUTS` from
   `HeadField`.
6. Spike: play a `GameCopier` copy forward with `mainLoopStep` to the end of the turn, before the
   directed-fork driver is built on it.
7. Remove `EARLIER_LAYOUT`, the `option_kind` gen-1 path and the `e_noise` shim with their tests (M1).
8. Correct the `GameRouting` docstring and `src/effects/CLAUDE.md`'s placement promise (R3).
9. Update `CLAUDE.md`, the base record-schema contract and `rules/README.md` with
   [contracts/record-schema-delta.md](contracts/record-schema-delta.md) and
   [contracts/catalogue-extensions.md](contracts/catalogue-extensions.md).

### Performance Review (Principle VIII — required when applicable)

The feature moves data (collection, `rule-counts`, curation, the freeze) and runs model compute
(training, selection, evaluation, probes).

- **I/O batching & caching**: *Addressed.* `rule-counts` and the survey read each shard once in
  parallel workers; sidecars come through the existing `SidecarCache`; the catalogue and map are read
  once per command. Rule copies reach the writers as a per-record map sharded by game hash, so each
  writer loads only its share. The selection set's host-side preparation is built once at training
  startup and cached.
- **GPU placement**: *Addressed.* The relation tables, averaged weights, dispersion scalars and the
  selection scorer live on the training device.
- **GPU batching**: *Addressed.* The relation key and value terms are batched gathers and scatters
  over a COO tensor, never per pair. The averaged-weight update is one `torch._foreach_lerp_` per
  step. Selection scores accumulate on the device and are read once per epoch.
- **Streaming & load-once**: *Addressed.* Shards stay streamed. The placement candidate pool keeps a
  per-game supply only for pool games, avoiding a third read. The kept-keyword-flag list is a
  checked-in constant, not a scan at startup. The loss-denominator pre-pass reads a fixed seeded
  sample of shards once and is stored in `resume.pt`.

Measurement planned, not optimization: the US4 step-time and peak-memory measurement, repeated on a
gen-3 curated shard once the gen-3 layout exists, and the in-epoch selection time, are recorded before
the first training run. The deviance sample cap is tuned from them. The stage-3 pilot records gzipped
bytes per game and games per hour against gen-2 with the same flags, because gen-3 enumerates both
seats' candidates at every priority stop with uncapped `legal_targets` and lists every line of every
permanent.

## Project Structure

### Documentation (this feature)

```text
specs/025-ability-effect-model-gen3/
├── plan.md                     # This file
├── research.md                 # Codebase survey, decisions, constants, spec corrections
├── data-model.md               # Entities added or changed
├── quickstart.md               # Gen-3 run plan, stage by stage
├── contracts/
│   ├── cli.md                  # CLI delta
│   ├── record-schema-delta.md  # gen-3 envelope, kinds, snapshot and payload delta
│   ├── catalogue-extensions.md # scenario detector/operation/departure, map snapshot, directed-rule registry
│   └── selection-set.md        # frozen selection set, freeze steps, control score
├── checklists/requirements.md
└── tasks.md                    # /speckit.tasks
```

### Source Code (repository root)

```text
../forge (effect-record-hooks, new commits)
├── forge-game/.../trigger/TriggerHandler.java     # delayed-trigger accessor
├── forge-game/.../phase/PhaseHandler.java         # state-based-loop begin/end; priority-settled
├── forge-ai/.../simulation/GameCopier.java        # copy delayed triggers, end-of-turn commands, this-turn state
└── forge-game/src/test/java/forge/game/**/EffectRecord*Test.java

forge-connector/src/main/java/com/pricepredictor/connector/
├── effects/
│   ├── EffectRecord.java, CollectorContext.java   # generation, aimed_at, directed
│   ├── RecordKind.java, EventType.java, BusEventReader.java   # turn_action; phasing
│   ├── PatchHooks.java, PatchedCollectors.java    # three hooks; dispatch by Game identity
│   ├── BusBracketCollector.java, HeldResolutionWindow.java, StackBrackets.java  # loop-spanning held half; dropped-event count
│   ├── CombatStepBracket.java                     # snapshot at the damage step's start
│   ├── SnapshotBuilder.java, EntityJson.java, PlayerJson.java  # lines, keyword lines, facts, links, base P/T
│   ├── FloatingEffects.java                       # NEW: timestamp → line bookkeeping (C4)
│   ├── PriorityCandidates.java                    # NEW: both seats' candidates and verdicts (C3)
│   ├── TargetingClauses.java                      # NEW: extracted clause walk
│   ├── MustBlock.java                             # NEW: per (blocker, attacker) requirement
│   ├── PlayabilityVerdict.java, DecisionCandidate.java, PlayabilityChannel.java  # timing, failed_checks; per-attacker records
│   ├── ClauseOutcomeRouter.java, RecordPayloads.java, RunParamNormalizer.java    # conditions_met; probe sources; zero prevention
│   ├── ProvenanceResolver.java, CharmModes.java, VariantRegistry.java            # FR-039 keying
│   ├── Forks.java (+ ForkRun), ForkBudget.java, ForkPlayForward.java (NEW)       # skeleton; directed budget; play to end of turn
│   ├── InterventionFork.java, InterventionScheduler.java  # FR-057 scheduling; FR-058 protection forks played on
│   ├── CollectionCaps.java                        # directed-fork and dropped-event properties
│   └── directed/                                  # NEW: DirectedRules, DirectedRule, BoardPrecondition, Binding,
│                                                  #      ForkOperation, KeywordEdit, the ten operations, DirectedForkScheduler
├── RandomSeatController.java, GamePlayer.java     # FR-040; seats inside a fork; JVM-lifetime directed counters
├── MatchWorkerMain.java                           # new properties
└── DirectedRulesMain.java                         # NEW: print registered rule ids

src/effects/
├── domain/
│   ├── rule_catalogue.py           # NEW: scenario/heads enums, DetectorDeclaration, Departure, ForkOperation
│   ├── rule_detectors/             # NEW: RuleTag, registry, one module per family
│   ├── rule_outcomes.py            # NEW: lethal, step_deaths, may_block (moved)
│   ├── rule_budgets.py             # NEW: per-rule shares, halves, per-record copies (R2)
│   ├── stratum_placement.py        # NEW: global placement and halves (R3)
│   ├── seeded_draw.py              # NEW: stable_order, smallest_k
│   ├── selection_set.py            # NEW: frozen set, balance, standard errors (E3, E4)
│   ├── score_scales.py             # NEW: AUC, R², rescaling, Hanley–McNeil
│   ├── probe_readouts.py           # NEW: READOUTS (moved)
│   ├── relation_types.py           # NEW: § 10.3 types, target-index cap
│   ├── relation_attention.py       # NEW: RelationEncoderLayer
│   ├── kept_keyword_flags.py       # NEW: generated, checked in
│   ├── effect_families.py          # RENAMED from rule_families.py
│   ├── records.py, shard_generation.py      # generation, aimed_at, directed; GEN_3
│   ├── state_snapshot.py           # new snapshot fields
│   ├── effect_head_input.py        # gen-3 layout, line flags, new slots and inputs, relations
│   ├── surface_collation.py        # relation tensor
│   ├── effect_model.py             # relation trunk, global head, line-flag projection, subtype embedding
│   ├── effect_head_layout.py, aux_heads.py, aux_head_targets.py   # outputs, verdict columns, global head
│   ├── effect_targets.py, entity_field_targets.py  # new targets; gate-free legality
│   ├── field_kinds.py, effect_losses.py, gate_metrics.py, loss_floor.py  # negative binomial; masks; denominators
│   ├── corpus_plan.py, corpus_curation.py, corpus_manifest.py, sampling.py  # signature condition bit; RuleRow; manifest
│   ├── protection_copies.py        # NEW: FR-058 rule-labelled copies (qualifier and copy, pure)
│   └── canonical_digest.py         # sha256 variant
├── application/
│   ├── rule_counts.py              # NEW
│   ├── corpus_survey.py, corpus_write.py, build_corpus.py, validation_samples.py  # inventories, pool, copies, curation, strata
│   ├── freeze_selection_set.py     # NEW
│   ├── control_score.py            # NEW
│   ├── record_forward.py           # NEW: one chunked forward
│   ├── weight_average.py           # NEW
│   ├── training_loop.py, training_schedule.py, resume_state.py, run_context.py, step_preparation.py,
│   │   surface_batching.py, validation_pass.py, run_reporter.py, train_effect_model.py  # mask, EMA, stopper, denominators, lines
│   ├── evaluate_effect_model.py, gate_one.py, breakdowns.py   # reporting half, NB deviance
│   └── validate_corpus.py, field_coverage.py                  # gen-3 fields
└── infrastructure/
    ├── cli.py                      # rule-counts; new flags; refusals
    ├── rules_catalogue_io.py       # NEW: read rules/scenarios, rules/heads
    ├── shard_io.py, record_codec.py  # require_generation; envelope
    └── effect_model_store.py       # checkpoint fields; gen-3 only

src/sealed/infrastructure/
├── cli.py, effect_collection_flags.py   # directed-fork and dropped-event flags; FR-014 startup check

scripts/
├── rules_catalogue.py              # detector/operation/departure checks; imports effects enums
├── scan_keyword_flags.py           # NEW: writes kept_keyword_flags.py
├── measure_trunk_step.py           # NEW: US4 step time and peak memory
└── effect_knowledge_probes/        # import the moved cores; freeze without a checkpoint; § 14.5 probes

rules/README.md, rules/scenarios/*.json    # new fields
tests/unit/effects/, tests/unit/scripts/    # mirror the modules above
forge-connector/src/test/java/...           # collector, fork, operation, registry-contract tests
```

**Structure Decision**: no new package or module root. Every change extends features 023 and 024's
`src/effects/` and `FE/` in place. Two new sub-packages hold the two families gen-3 adds:
`effects/domain/rule_detectors/` and `FE/directed/`. Probe scoring moves into the package because
training needs it; the probe CLI stays under `scripts/`.

### Post-design re-check

Re-evaluated after Phase 1. No gate changed status. The design added four points the pre-design check
did not cover:

- **Principle III**: the `curation` member exists only in curated shards. `validate-corpus` refuses it
  in a collected shard, and `test_schema_compatibility.py` checks it never reaches `SurfaceBuilder`
  as an input.
- **Principle III**: the selection set records r, the standard errors under both masks and every item
  left out, so a run's control score can be audited against the set it read.
- **Principle IV**: Python and Java each check their half of FR-014 (the detector and classification
  in Python, the registry by a JVM query), so neither side reads the other's internals.
- **Principle I**: the directed-fork play-forward is an `integration` test. Its pieces (operations,
  precondition binding, budget order) are fast JUnit tests on scripted boards.

## Requirement traceability

| Spec section (FR range) | Owner |
|---|---|
| Catalogue and review (FR-001…005) | `rules/`, `scripts/rules_catalogue.py`, `E/domain/rule_catalogue.py` |
| Detectors and counts (FR-006…009a) | `E/domain/rule_detectors/`, `E/application/rule_counts.py`, `cli.py`; `aimed_at` in `CollectorContext` |
| Directed forks (FR-010…014a) | `FE/directed/`, `ForkPlayForward`, `ForkRun`, `ForkBudget`, `GamePlayer`, `DirectedRulesMain`; flags and startup check in `src/sealed/infrastructure/cli.py`; `GameCopier` patch |
| Map (FR-015…017) | `scripts/rules_heads_workflow.js`, `rules_catalogue.py gen3-fields`, `rules/MODEL-FIELDS.md` |
| Schema and envelope (FR-018…020) | `EffectRecord.java`, `records.py`, `record_codec.py`, `shard_generation.py`, `shard_io.require_generation`; schema-compatibility tests; base contract |
| Lines of a permanent (FR-021…023) | `EntityJson`, `SnapshotBuilder`, `ProvenanceResolver` |
| Snapshot, links, records (FR-024…042) | `SnapshotBuilder`, `EntityJson`, `PlayerJson`, `FloatingEffects`, `CombatStepBracket`, `BusBracketCollector`, `HeldResolutionWindow`, `PriorityCandidates`, `PlayabilityVerdict`, `PlayabilityChannel`, `MustBlock`, `ClauseOutcomeRouter`, `RunParamNormalizer`, `StackBrackets`, `RandomSeatController`; three hooks; `collect_coverage.py`; `validate_corpus.py`, `field_coverage.py` |
| Model input (FR-043…045) | `effect_head_input.py`, `state_snapshot.py`, `effect_model.py`, `kept_keyword_flags.py`, `scan_keyword_flags.py`, `train_effect_model.py` (e-dim) |
| Relation trunk (FR-046…051) | `relation_attention.py`, `relation_types.py`, `effect_head_input.SurfaceBuilder`, `surface_collation.py`, `surface_batching._masked` |
| Outputs (FR-052…053b) | `effect_head_layout.py`, `effect_targets.py`, `entity_field_targets.py`, `aux_heads.py`, `aux_head_targets.py`, `effect_model.py` |
| Curation (FR-054…060a) | `rule_budgets.py`, `stratum_placement.py`, `corpus_survey.py`, `corpus_write.py`, `build_corpus.py`, `corpus_manifest.py`, `corpus_curation.py`, `protection_copies.py`, `rules_catalogue_io.py`; FR-057 in `InterventionScheduler` |
| Training (FR-061…067) | `surface_batching.py`, `effect_losses.py`, `field_kinds.py`, `loss_floor.py`, `weight_average.py`, `training_loop.py`, `resume_state.py`, `train_effect_model.py`, `run_reporter.py` |
| Selection and evaluation (FR-068…075) | `selection_set.py`, `freeze_selection_set.py`, `control_score.py`, `score_scales.py`, `probe_readouts.py`, `rule_outcomes.py`, `validation_pass.py`, `evaluate_effect_model.py`, `gate_one.py`, `scripts/effect_knowledge_probes/` |
| Operator surface, artifacts (FR-076…077) | `cli.py`, `src/sealed/infrastructure/cli.py`, `effect_model_store.py` |
| Boundaries (FR-078…080) | the import-direction test; nothing out of scope built |

Five requirements are easy to lose between owners:

- **FR-028 and FR-029** need the state-based-loop hook in Forge, the `beginBracket` change and the
  dropped-event count to land together, or a pilot files loop deaths in trigger records.
- **FR-033's enumerator** and **FR-040's random-seat filter** both read `PriorityCandidates`; the seat
  must switch to it in the same change, or its plays and its records disagree on `can_play`.
- **FR-061's mask** is built on the host from `curation`, but the gate's per-row mask is in
  `effect_losses._gate_term`. A test must check that a hard-masked added copy trains the gate only on
  its rule's entities.
- **FR-068a's placement** changes which games are in the strata, and the selection set is frozen
  against the corpus digest. Rebuilding the corpus invalidates the set, and training refuses to start
  without one (research E1).
- **FR-044a's flag scan** must keep every keyword gate 2 strips and the grids edit, or gate 2 and the
  grids silently stop editing anything.

## Complexity Tracking

| Item | Why needed | Simpler alternative rejected because |
|---|---|---|
| Three new engine hooks and a `GameCopier` patch on `effect-record-hooks` | FR-024 (pending delayed triggers), FR-028 (the state-based loop's extent), FR-033 (both seats at every stop), FR-010 (forks played to the end of the turn) | public API cannot observe the loop's end, the delayed-trigger lists or a settled priority, and an unpatched copy drops delayed triggers (user decision, research D3) |
| Global stratum placement with a per-game supply pool | FR-068a | gen-2's per-game hash threshold cannot put both polarities of every rule in both halves; an inventory-only pre-pass would add a third read of the corpus |
| Probe scoring moved from `scripts/` into the package | FR-068 scores probes every epoch | `src/effects` cannot import `scripts/` |
| Collector-side `MustBlock` predicate | FR-037's `must_block` | Forge has no per-(blocker, attacker) requirement predicate |
