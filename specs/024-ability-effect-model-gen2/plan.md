# Implementation Plan: Ability effect model — generation 2

**Branch**: `024-ability-effect-model-gen2` | **Date**: 2026-10-06 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/024-ability-effect-model-gen2/spec.md`

## Summary

Gen-2 changes what the ability effect model is trained on and how it is judged, in four layers:

1. **Encoding text.** `convert` writes each sidecar line's whole script chain with chain labels
   renamed by position. Charm modes become keyed `option` lines. The script surface gets its own
   tokenization rules and vocabulary seeding.
2. **Collection.** A random seat that sometimes plays, targets, attacks and blocks at random. Two
   envelope fields mark its records and the legality records that are real decisions. `actor_player`
   on playability records names the deciding player, and modal resolutions split into one effect
   half per mode.
3. **Curation and training.** A masked-template holdout, a hash-stable game-disjoint stratum, and
   selection balanced across rule families and outcome signatures with per-record copy counts.
   Training adds covariance-scaled noise on `e`, a value head, configurable encoder size, and wires
   the verdict, created-objects, MLM and script-API losses that feature 023 defined but never called.
4. **Measurement.** Per-text, per-family and per-policy evaluation reports, a scorer smoke-test
   command, checkpoint-aware embedding probes, and a new knowledge-probe suite.

Nearly all of it extends feature 023's code in place. The new pieces are four pure domain modules
(`rule_families`, `budget_allocation`, `value_targets`, `rarity`), one evaluation module
(`breakdowns`), the random seat (three Java classes plus one extracted helper) and the probe suite
under `scripts/`.

## Technical Context

**Language/Version**: Python 3.14 (`requires-python >=3.14`); Java 17 for `forge-connector`
**Primary Dependencies**: torch ≥ 2.2 (CUDA 12.6 wheel index), numpy, scikit-learn (already used by
`scripts/effect_embedding_probes/`); Forge `2.0.15-SNAPSHOT` jars from `../forge` on
`effect-record-hooks`, unchanged
**Storage**: append-only JSONL.gz record shards, provenance sidecars, `.npz` ability caches, `.pt`
checkpoints, curated corpus directories with a JSON manifest, probe-set and scorecard JSON
**Testing**: pytest fast suite (`tests/unit/effects/`, `tests/unit/scripts/`); JUnit 5 for
`forge-connector`; JVM-dependent Python tests carry the `integration` marker
**Target Platform**: local Windows 11 workstation, one CUDA GPU with 8 GB VRAM
**Project Type**: CLI-driven ML pipeline (`src/effects/`) with Java worker mains (`forge-connector/`)
**Performance Goals**: knowledge-probe run ≤ 2 GPU hours per checkpoint (SC-011), recorded in each
scorecard; `build-corpus` keeps its two-pass shape (one parallel survey, one parallel write)
**Constraints**: no engine hook added or changed (FR-024); `.txt` conversion byte-identical (FR-004);
gen-1 checkpoints, manifests and shards stay loadable for stage 0 (FR-033, FR-061); 8 GB VRAM bounds
the encoder sweep, with batch size reduced by hand where an arm does not fit
**Scale/Scope**: 33,680 converted cards, ~66,000 ability lines (plus ~2,000 `option` lines, of which
the charm modes gain keys); corpus of the order of 10⁸ records; one curated dataset shared by every
sweep arm

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status |
|---|---|
| I. Fast Automated Tests | **Pass.** Every new rule is a pure function testable without a JVM or GPU: chain-label renaming and masked templates (string → string), script tokenization, family assignment, the allocator, copy counts, stratum placement, value targets, breakdowns, the share and its minimum-gap rule, fold assignment. The random seat's draw rules are unit-tested in JUnit against a scripted Forge game, as feature 023's collector tests are. The knowledge probes are tested on fixtures cut from real records (FR-089). |
| II. Simplicity First | **Pass.** One allocator serves three levels of FR-049. One random-choices class serves the random seat and `ForkCollector`. No configuration point is added that the spec does not name. The copy-count model replaces the threshold model rather than sitting beside it. |
| III. Data Integrity | **Pass.** Both envelope fields are additive. `actor_player` on playability records is redefined for gen-2 shards (FR-030a), and the redefinition is versioned by shard generation (FR-033): the presence of `random_seat` marks a gen-2 shard, and every reader refuses a records set mixing the two. It is also a named exception in the schema-compatibility tests and both contracts. Gen-1 shards keep their meaning and stay readable. Manifest and checkpoint additions are omitted at their gen-1 value, so existing digests and `check_corpus` keep passing. Stratum placement and selection are deterministic functions of record and game hashes. |
| IV. DDD & Separation of Concerns | **Pass.** New rules live in `effects/domain` (families, allocation, value targets, rarity, masked templates, stratum placement). Orchestration stays in `application`, and IO and the CLI in `infrastructure`. The model-architecture exception covers only `effect_model.py` and `ability_encoder.py`, as before. `effects` still imports nothing from `sealed.application`: the smoke test runs Phase A as a subprocess. |
| V. Forge Interoperability | **N/A.** The stub library and remote API are untouched. The random seat is a worker-side controller in `forge-connector`'s established second role. |
| VI. Documentation | **Tracked as tasks.** `CLAUDE.md`'s shard and sidecar paragraphs, both base contracts (FR-034), `src/effects/CLAUDE.md` (new commands and flags), and the README's effects workflow, each in the change that alters the behaviour. |
| VII. Codebase-Aware Planning | **Pass.** See below. |
| VIII. Performance-Conscious Implementation | **Applies.** See below. |

### Codebase Survey (Principle VII — required)

Full findings: [research.md § Codebase Survey](research.md#codebase-survey).

- **Overlapping vocabulary**: 9 concepts extended (`ProvenanceKey` in both languages, `SidecarLine`,
  `TraitScript`, `EffectRecord`, `CollectionCaps`, the holdout entry point, `CorpusManifest`/
  `SplitProvenance`, the rarity weights), 2 reused, 1 moved (`KeywordResolver` into the torch-free
  domain module), 1 replaced (threshold admission → copy counts), 4 heads wired, 2 removed
  (`pairing_*`, `e_noise`). 6 new concepts, none colliding with an existing name. 0 renames.
- **Adjacent prior art**: reuse of Forge's `CombatUtil`/`TargetRestrictions`/`ComputerUtilMana`,
  `numeric_params`, `ManaCost.parse`, `write_scorer_smoke_cache` and the probe idioms. Extension of
  `PatchedCollectors`' emitters, `MatchWorkerMain`'s records-only gate, the sealed CLI flag plumbing,
  `collect_coverage` and `KeywordDefinitionMain`. One documented reimplementation: a connector copy of
  `ComputerUtil.handlePlayingSpellAbility`, because its `AiCostDecision` is not injectable.
- **Convention alignment**: feature 023's `src/effects/` layout; the probe suite mirrors
  `scripts/effect_embedding_probes/`. No deviation.
- **Third-instance check**: the FR-049 allocator (three uses) and `rarity_bucket` (three uses) are
  extracted. Random play choices are extracted at their second instance, because the two callers must
  agree and the existing one is wrong on Pawprint, repeats and minimum modes.

**Follow-up tasks this surfaced, to be carried into `tasks.md`:**

1. Extract `RandomChoices` from `ForkCollector` and widen it (MinCharmNum, repeats, Pawprint,
   `setChosenList`) before the random seat is built. `ForkCollector`'s per-mode output (FR-029f)
   depends on it.
2. Move `KeywordResolver` and `qualifying_observations` from `application/gate_two.py` to
   `domain/damage_step_keywords.py` before build-corpus uses them.
3. Make `PatchedCollectors`' three emitters public, taking the deciding player and `whatIf`, and
   route the hook handlers through them, before either the random seat or FR-030a lands.
4. Remove `tokenize_script` with its tests and move `keyword_expansion.py`'s call to `tokenize`.
5. Invert `PatchedCollectorTest`'s legality de-dup pin (`:1310-1317`) and `test_validate_corpus.py`'s
   three-halves test (`:389`).
6. Update `CLAUDE.md`, the two base contracts and `src/effects/CLAUDE.md` with the deltas in
   [contracts/record-schema-delta.md](contracts/record-schema-delta.md).

### Performance Review (Principle VIII — required when applicable)

The feature moves data (collection, curation, probes) and runs model compute (training, encoding,
evaluation, probes).

- **I/O batching & caching**: *Addressed.* Value targets and family lookups are computed once per
  text and cached beside `e` in the batcher. The build-corpus survey gathers family, signature and
  keyword-combat flags in its existing single pass over each shard, and sidecars are read through the
  existing `SidecarCache`. The probe suite extracts rung features once per stratum and fits every probe
  on the cached arrays.
- **GPU placement**: *Addressed.* The noise covariance, its Cholesky factor and the sampled noise live
  on the training device. The value, verdict, created-objects, MLM and API heads run with the model.
  The MLP probes train on the GPU.
- **GPU batching**: *Addressed.* Noise is one batched matmul per step. Covariance updates use
  `torch.no_grad()` with no host reads. The new epoch-line counters (rarity bucket and family shares)
  accumulate on the host from the batch plan, which is already host-side, so they add no
  device→host transfer.
- **Streaming & load-once**: *Addressed.* Shards stay streamed. The tokenizer's per-text cache keeps
  working with the surface fixed at construction. The keyword-definition table and its formatter map
  load once. Probe sets are read once per run. On the collection side, a what-if legality record
  builds its snapshot only after the rate draw, so the de-dup key change (FR-028) does not multiply
  snapshot builds.

No optimization beyond this checklist is planned.

## Project Structure

### Documentation (this feature)

```text
specs/024-ability-effect-model-gen2/
├── plan.md                  # This file
├── research.md              # Codebase survey, decisions, constants
├── data-model.md            # Entities added or changed
├── quickstart.md            # Gen-2 run plan, stage by stage
├── contracts/
│   ├── cli.md               # CLI delta
│   ├── record-schema-delta.md   # envelope, key, sidecar and modal-record delta
│   └── knowledge-probes.md  # probe suite layout, rungs, share, sweeps, scorecard
├── checklists/requirements.md
└── tasks.md                 # /speckit.tasks
```

### Source Code (repository root)

```text
forge-connector/src/main/java/com/pricepredictor/connector/
├── effects/
│   ├── TraitScript.java              # chain render, label renaming, missing-SVar report
│   ├── ProvenanceRecorder.java       # attribute charm option lines with mode keys
│   ├── ProvenanceKey.java            # option component; clone → Choices$ index
│   ├── ProvenanceSidecar.java        # option lines' keys, chains, API types
│   ├── EffectRecord.java             # random_seat, what_if
│   ├── PatchedCollectors.java        # public emitters; deciding-player actor; FR-027/028/029; per-mode halves via clause hook
│   ├── BusBracketCollector.java      # per-mode effect halves sharing link_id
│   ├── ForkCollector.java            # modes via RandomChoices; per-mode halves
│   └── RandomChoices.java            # NEW, extracted from ForkCollector and widened
├── RandomSeatLobbyPlayer.java        # NEW: installs the controller on one seat
├── RandomSeatController.java         # NEW: PlayerControllerAi subclass (FR-021–023)
├── RandomSpellPlayer.java            # NEW: copy of handlePlayingSpellAbility + RandomCostDecision
├── GamePlayer.java                   # seat choice per match, collector injection
├── MatchWorkerMain.java              # per-match records-only; random-seat properties
├── RulesParser.java                  # option TextAbility attribution
└── KeywordDefinitionMain.java        # formatter field

src/effects/
├── domain/
│   ├── provenance.py                 # ProvenanceKey.option; option-row adjacency
│   ├── records.py                    # random_seat, what_if; defaults; actor exception
│   ├── ability_tokenizer.py          # script-surface rules; display-name expansion; formatter fill; tokenize_script removed
│   ├── ability_encoder.py            # size from config; e_noise removed; MAX tokens report hook
│   ├── effect_model.py               # option-kind embedding; mlm_head width; pairing removed; value head
│   ├── effect_head_input.py          # option rows; decision [ACT] candidate e
│   ├── effect_targets.py             # verdict targets at [ACT]
│   ├── text_holdout.py               # masked_template; unit parameter
│   ├── corpus_curation.py            # copy counts; stratum placement
│   ├── corpus_manifest.py            # new fields, omitted at gen-1 values
│   ├── damage_step_keywords.py       # + KeywordResolver, qualifying_observations (moved)
│   ├── collection_caps.py            # random-seat share and probability
│   ├── keyword_formatting.py         # NEW: value formatting per Keyword.type
│   ├── rule_families.py              # NEW
│   ├── budget_allocation.py          # NEW: equal split with redistribution
│   ├── value_targets.py              # NEW
│   └── rarity.py                     # NEW: rarity_bucket
├── application/
│   ├── build_vocab.py                # staging order, seeding, reports, no generated_script
│   ├── extract_keyword_definitions.py  # formatter
│   ├── holdout_cards.py              # template report
│   ├── build_corpus.py               # survey gathers family/signature/combat flag; decide → copies
│   ├── validation_samples.py         # round-robin per held-out text
│   ├── collect_coverage.py           # --only-cards text mode
│   ├── surface_batching.py           # noise hook; candidate_index; value targets cache
│   ├── training_loop.py              # NoiseState; all loss terms; epoch shares
│   ├── train_effect_model.py         # rarity ceiling p99, within class; flags into config
│   ├── encode_abilities.py           # shared expansion constant; truncation report
│   ├── evaluate_effect_model.py      # breakdowns, slices, zero-shot, --win-rates
│   ├── breakdowns.py                 # NEW: per-text/bucket/family/policy grouping
│   ├── gate_one.py                   # measure returns per-record results
│   ├── geometry_checks.py            # smoke cache key aligned with sealed
│   ├── scorer_smoke_test.py          # NEW: write vectors, run Phase A subprocess
│   └── validate_corpus.py            # widened link_id; field-presence check
└── infrastructure/
    ├── cli.py                        # new and changed flags, refusals, scorer-smoke-test
    ├── record_io.py                  # new envelope keys and defaults
    ├── sidecar_io.py                 # option key component
    ├── effect_model_store.py         # e_noise shim; new checkpoint fields; head filter
    ├── ability_encoder_runner.py     # expansion constant; size from checkpoint
    └── model_runner.py               # size from checkpoint

src/sealed/infrastructure/
├── cli.py                            # --random-seat-share/-probability, FR-026 refusal
└── match_worker_connector.py         # (unchanged code path; new properties ride the caps dict)

src/price_predictor/infrastructure/cli.py   # convert: report missing SVars

scripts/effect_embedding_probes/      # --checkpoint/--abilities-root; taxonomy optional; PR
scripts/effect_knowledge_probes/      # NEW suite (common, labels, ladder, sweeps, ablation, compare, run)
scripts/make_fixture_records.py       # NEW: real-record test fixture

tests/unit/effects/                   # mirrors the modules above
tests/unit/scripts/                   # knowledge-probe tests, loaded by path
tests/fixtures/effects/               # gen-1 records and their sidecars, cut from real shards
forge-connector/src/test/java/...     # RandomChoices, RandomSeatController, emitters, modal halves, TraitScript chains
```

**Structure Decision**: no new package or module root. Every change extends feature 023's
`src/effects/` and `forge-connector/.../effects/` in place, with new pure domain modules beside their
siblings. The random seat sits beside `GamePlayer` because it is a seat-construction concern. The
knowledge-probe suite sits beside the embedding probes under `scripts/`, which keeps analysis tooling
out of the installed package (FR-090).

### Post-design re-check

Re-evaluated after Phase 1. No gate changed status. The design added three things the pre-design
check did not cover:

- **Principle III**: the `actor_player` redefinition is visible at every layer that could misread
  it. The contracts show the gen-1 and gen-2 meanings side by side. The compatibility test names the
  exception. `validate-corpus` refuses a shard that mixes records with and without the new fields.
- **Principle IV**: the random seat reaches the collectors through public `PatchedCollectors`
  methods, never through the hook proxies, so the Java-side boundary between collection and engine
  hooks stays one-directional.
- **Principle I**: the knowledge probes are testable without a GPU. The MLP probe runs on CPU in
  tests with a tiny hidden size passed by the test, while the fixed constants stay the defaults.

## Requirement traceability

Every spec section and its FR range, with its owner. `tasks.md` follows this map.

| Spec section (FR range) | Owner |
|---|---|
| Encoding text (FR-001…006) | `TraitScript`, `ProvenanceRecorder`, `RulesParser`, `ProvenanceSidecar` (write); `provenance.py`, `sidecar_io.py` (read); `ability_encoder.py` + `encode_abilities.py` (truncation report); `price_predictor` `convert` (missing-SVar report) |
| Script tokenization (FR-007…011) | `ability_tokenizer.py`, `application/build_vocab.py` |
| Keyword expansion (FR-012…020) | `ability_tokenizer.py`, `KeywordDefinitionMain` + `extract_keyword_definitions.py`, `build_vocab.py`, `build_corpus.py` (FR-015) |
| Random seat (FR-021…026) | `RandomSeatLobbyPlayer`, `RandomSeatController`, `RandomSpellPlayer`, `RandomChoices`, `GamePlayer`, `MatchWorkerMain`; flags and FR-026 in `sealed/infrastructure/cli.py` |
| Legality records (FR-027…029) | `PatchedCollectors` emitters |
| Modal resolutions (FR-029a…029f) | `PatchedCollectors.clauseHandler`, `BusBracketCollector`, `ProvenanceKey.resolve`, `ForkCollector`; `validate_corpus.py` (FR-029d readers) |
| Envelope fields (FR-030…034) | `EffectRecord.java`, `records.py`, `record_io.py` (incl. `shard_generation`), `validate_corpus.py`; schema-compatibility tests; `CLAUDE.md` and both base contracts |
| Held-out coverage round (FR-035…038) | `collect_coverage.py`, `effects/infrastructure/cli.py` |
| Holdout (FR-039…043) | `text_holdout.py`, `train_effect_model.text_keyed_holdout`, `holdout_cards.py`, `corpus_manifest.py`, `effect_model_store.SplitProvenance` |
| Game-disjoint stratum (FR-044…046) | `corpus_curation.in_game_disjoint_stratum`, `build_corpus.py`, `damage_step_keywords.py` (moved predicate) |
| Rule families and selection (FR-047…052) | `rule_families.py`, `budget_allocation.py`, `corpus_curation.py`, `build_corpus.py`, `validation_samples.py` |
| Manifest (FR-053) | `corpus_manifest.py`, `build_corpus.py` |
| Training (FR-054…063b) | `train_effect_model.py` (rarity), `training_loop.py` (epoch line, noise state, loss terms), `surface_batching.py` (noise, candidate `e`, value targets), `value_targets.py`, `effect_model.py`, `ability_encoder.py`, `effect_head_input.py`, `effect_model_store.py` |
| Evaluation (FR-064…071) | `evaluate_effect_model.py`, `breakdowns.py`, `gate_one.py`, `scorer_smoke_test.py`, `geometry_checks.py`, `cli.py` |
| Embedding probes (FR-072…073) | `scripts/effect_embedding_probes/common.py` and each script |
| Knowledge probes (FR-074…089) | `scripts/effect_knowledge_probes/*`; tests in `tests/unit/scripts/` |
| Boundaries (FR-090…092) | the import-direction test (unchanged surface); gate 2/3 code untouched |

Three requirements are easy to lose between owners:

- **FR-022d's land-first rule** sits in `RandomSeatController.chooseSpellAbilityToPlay`, but its test
  needs a board where the AI would play a land and a spell. It belongs in the JUnit scripted-game
  fixture, not a unit test of the draw.
- **FR-030a** changes the `actor` argument at three emit sites and the `random_seat` stamp at every
  emit site. Both must land in one change, or a pilot would write records whose `random_seat`
  follows the old actor.
- **FR-060a's `candidate_index`** is in the batcher, but the verdict targets come from
  `effect_targets.py`. A test must check that a `decision` record yields both a non-zero `[ACT]` and
  verdict targets, or the head trains on zeros as it did in gen-1.

## Complexity Tracking

| Item | Why needed | Simpler alternative rejected because |
|---|---|---|
| A connector copy of `ComputerUtil.handlePlayingSpellAbility` (~50 lines) | FR-022c randomises additional-cost choices; Forge hard-codes `new AiCostDecision(...)` inside it | overriding a controller method cannot reach the cost decision (`chooseCardsForCost` is an unused placeholder), and FR-024 forbids an engine hook |
| `actor_player` redefined on gen-2 playability records | FR-030a (clarified 2026-10-06) | a separate `deciding_player` field was the additive alternative; the spec chose the redefinition, which the contracts and tests carry as a named exception |
