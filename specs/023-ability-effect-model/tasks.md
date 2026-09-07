---

description: "Task list for 023-ability-effect-model"
---

# Tasks: Ability effect model

**Input**: Design documents from `/specs/023-ability-effect-model/`
**Prerequisites**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/](contracts/)

**Tests**: Mandatory. Constitution Principle I is non-negotiable and the template requires test tasks in
every list. Tests are written before or alongside the code they cover, never deferred.

**Organization**: Grouped by user story. The four stories are the root spec's four stages, so a story
boundary is also a corpus-capability boundary — each one widens what can be collected without
invalidating what came before.

**Editing design documents**: T007 and the Polish-phase task that fills the design record's Outcome
section write under `specs/` and `experiments/`. Load the **feature-workflow** skill before each such
edit — `experiments/*.md` carries a binding prose style that applies per edit.

**Prior art**: Per Principle VII, every task that creates a new module names the prior art it reuses,
extends, or deliberately diverges from. The references are to
[research.md § Codebase Survey](research.md#codebase-survey).

## Path Conventions

Three Python packages under `src/` (`price_predictor`, `sealed`, `effects`), the Java connector at
`forge-connector/src/main/java/com/pricepredictor/connector/`, fast tests under `tests/unit/` mirroring
the source tree, and JVM-dependent tests under `tests/integration/` carrying the `integration` marker.
Java paths written `forge-connector/.../effects/X.java` elide the package directory pinned in T006.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: parallelizable — different file, no dependency on an incomplete task
- **[Story]**: US1–US4; absent on Setup, Foundational, and Polish tasks

---

## Phase 1: Setup

**Purpose**: package skeleton and test scaffolding. Nothing here has behaviour.

- [X] T001 Create the `src/effects/` package with `domain/`, `application/`, `infrastructure/` subpackages and `__init__.py` in each, mirroring the layout of `src/draft/` (prior art: research.md § Convention alignment)
- [X] T002 Create `src/effects/infrastructure/cli.py` with an argparse `main()` and an empty subcommand table, mirroring `src/draft/infrastructure/cli.py`. Later tasks register subcommands into it; it must exist first because `__main__.py` imports it
- [X] T003 Create `src/effects/__main__.py` delegating to `effects.infrastructure.cli:main`, mirroring `src/draft/__main__.py`
- [X] T004 [P] Create the test tree `tests/unit/effects/{domain,application,infrastructure}/` mirroring the source layout, with `__init__.py` where the sibling suites have them
- [X] T005 [P] Add an `effects` entry to `pyproject.toml`'s package discovery if the existing `where = ["src"]` config does not pick it up automatically; add `umap-learn` to the project dependencies (FR-110's UMAP check is the only third-party library this feature adds); verify with `pip install -e .` that `python -m effects --help` resolves
- [X] T006 Create `forge-connector/src/main/java/com/pricepredictor/connector/effects/` and `forge-connector/patches/` with a `README.md` in the latter stating that patches are applied to the sibling `../forge` checkout and must be re-applied after every Forge upgrade

**Checkpoint**: `python -m effects --help` runs and prints an empty subcommand list.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: the record schema, the provenance join, and the three extractions the codebase survey
forced. **No user story can start until this phase is complete** — the schema is frozen before any
collection, and every story reads the sidecar.

### Extractions and spec amendments (survey follow-ups)

- [x] T007 Amend `spec.md` FR-002 to declare the full `price_predictor` and `sealed` import surface, and FR-090 to include the withheld keyword in the checkpoint contract, per the table in [plan.md § The import surface FR-002 declares](plan.md). Do this first: the import-direction test below asserts against it
- [x] T008 Amend `.specify/memory/constitution.md` Principle IV to permit framework imports in `domain` for model-architecture modules, or record an explicit exception. The current wording is broken today by six modules — five in `sealed/domain` (`card_encoder`, `encoder_model`, `greedy_deck_builder`, `picker_model`, `scorer_model`) and one in `draft/domain` (`draft_agent_model`) — and by `effects/domain` once T064 and T067 land; plan.md requires this be resolved by amendment, not by a per-feature waiver. Follow the file's own Governance section: bump the version and update the Sync Impact Report header
- [x] T009 Amend `.specify/memory/constitution.md` Principle VII clause (c) so the rename requirement applies only when the parallel concept's name collides with the existing one. `ability_cache_layout` vs `card_embedding_layout` and `EffectRecord` vs `DraftRecord` do not collide, so the clause as written has no target
- [X] T010 Extract the partial-line-tolerance primitive into `src/price_predictor/infrastructure/append_only.py` — iterate complete (newline-terminated) lines, count them, optionally truncate a trailing partial (prior art: this is the fourth instance; `pool_file_reader.py:88`, `cards_played_reader.py:61`, `draft_record_io.py` all implement it today)
- [X] T011 [P] Unit-test `append_only.py` in `tests/unit/infrastructure/test_append_only.py`: complete lines, a trailing partial, an empty file, a file ending exactly on a newline
- [X] T012 Refactor `pool_file_reader.py`, `cards_played_reader.py`, and `draft_record_io.py` onto `append_only.py`, keeping each reader's own record parsing. Their existing tests must pass unchanged — that is the regression check
- [X] T013 [P] Add an import-direction test in `tests/unit/effects/test_import_boundaries.py` asserting that `price_predictor` and `sealed` never import `effects`, and that `effects` imports only the symbols the amended FR-002 declares
- [X] T014 Promote `_truncate_to_target_size` from `src/sealed/application/build_vocab.py` into `src/price_predictor/application/build_vocabulary.py` as a public function; have `sealed`'s wrapper call it there. Sealed's existing vocab tests must pass unchanged (prior art: research.md § Adjacent prior art — the alternative is a private cross-context import Principle IV forbids)
- [X] T015 Extract the ridge machinery from `scripts/encoder_probes/probe_lib.py` into `src/price_predictor/application/ridge_probes.py`, free of module-scope side effects and hardcoded paths; leave the script importing them. The set is the closure of `fit_probes`, not just its name: `fit_probes`, `_ridge_solve`, `HeadProbe`, `ProbeSet`, `to_logit`, `_choose_alpha`, `_r2`, `_pearson`, and the `HEADS`, `ALPHA_GRID` and `LOGIT_CLIP` constants (`LOGIT_CLIP` is `to_logit`'s default argument and binds at definition time, so omitting it reproduces the very import error this task warns about). Extracting only the first four yields a module that raises `NameError` on import. `scripts/` is not an importable package, so FR-114 cannot use them where they sit
- [X] T016 Move the probe label/feature join out of `scripts/` too: `load_labels` (`probe_lib.py:69`) and `build_join` (`:370`) build the DataFrame `fit_probes` consumes, and `load_labels` defaults to a NAS path resolved at module scope. Reimplement both in `ridge_probes.py` taking the win-rate table path as a parameter, so FR-114's battery runs without a NAS mount and without importing `scripts/`
- [X] T017 [P] Unit-test `ridge_probes.py` in `tests/unit/application/test_ridge_probes.py` against a small synthetic matrix with a known closed-form solution

### Record schema (frozen before any collection)

- [X] T018 [P] Implement the record envelope dataclasses in `src/effects/domain/records.py` — `EffectRecord` with every field in [contracts/record-schema.md](contracts/record-schema.md) § Envelope, plus the `kind`/`moment`/`subkind` enums (prior art: `draft/domain/draft_geometry.py` for the pure-dataclass-plus-infrastructure-serializer split)
- [X] T019 Implement the per-kind payload dataclasses in `src/effects/domain/records.py` per § Payloads: resolution activation/effect, rewrite, continuous, combat, trigger, and the three playability subkinds
- [X] T020 [P] Implement `src/effects/domain/state_snapshot.py`: the `global`/`players[]`/`entities[]`/`refs`/`pending_event` blocks, with `granted_attached` and `granted_temporary` as **separate** entity fields, and the four inclusion tiers as an enum with a "tier absent means uncollected" reader rule
- [X] T021 [P] Implement `src/effects/domain/event_schema.py`: the canonical event-type vocabulary (union of Forge trigger types, bus events, bracket diffs) and the per-type field normalization
- [X] T022 Unit-test the schema in `tests/unit/effects/domain/test_records.py`: every flag signature from § Flag signatures (ordinary / interventional / probe), `link_id` absent on each partnerless outcome, `ability` absent for `combat` and `playability`, and that `mode`/`interventional`/`fork`/`synthetic` are excluded from any model-input projection
- [X] T023 [P] Test the four schema compatibility rules in `tests/unit/effects/domain/test_schema_compatibility.py` — a field may be added but not repurposed; a new `kind`/`subkind` may be introduced but existing values not reused; tiers are additive; collection metadata never becomes a model input. plan.md's Principle III claim rests on these being enforced by tests rather than a version field
- [X] T024 [P] Unit-test the snapshot in `tests/unit/effects/domain/test_state_snapshot.py`: tier ordering, absent-tier semantics, the two grant channels staying separate, and that no perspective is stored (controllers absolute)
- [X] T025 Write the effect-API completeness test in `tests/unit/effects/domain/test_event_schema_completeness.py`: every Forge effect API class maps to a covered event type or an explicit, commented exclusion. Use a checked-in class list at `src/effects/domain/forge_effect_apis.py`, generated once from the sibling checkout by a documented one-liner recorded in that file's docstring, so the fast suite needs no JVM and no `../forge` checkout

### Corpus IO

- [X] T026 Implement `src/effects/infrastructure/record_io.py` — JSONL shard writer and directory reader over `append_only.py`, tolerating a trailing partial line (prior art: `draft/infrastructure/draft_record_io.py`, same shape)
- [X] T027 [P] Unit-test `record_io.py` in `tests/unit/effects/infrastructure/test_record_io.py`: round-trip every record kind, multi-shard directory load, trailing partial line skipped, unknown field preserved on read

### Provenance sidecar (the join everything rests on)

- [X] T028 Implement provenance-key computation in `forge-connector/.../effects/ProvenanceKey.java`: (script file incl. tree, face, trait kind, index within kind) from `getCardState()`, `isIntrinsic()`, `getKeyword()`, `isCopiedTrait()`. **Do not use `getOriginalHost()` to key granted traits** — it returns the recipient, not the donor
- [X] T029 Implement the granted/copied/copy-spell fallback chain in the same class: granted abilities resolve through the grantor accessors (`getGrantorStatic()`, `SpellAbility` only), copied abilities through the original-ability back-reference, copy-spell effects through the stack object's source card
- [X] T030 Extend `forge-connector/.../RulesParser.java` to record, per rendered line, its provenance keys, sub-ability links, script API type, parameter keys, script text, and prose role spans — and to collect keys for traits that map to no line into `dropped_keys`. The rendered text must not change (prior art: research.md — additive extension; the parser already holds the runtime trait)
- [X] T031 Extend `forge-connector/.../ConvertMain.java` to write `<name>.provenance.json` beside each converted `.txt`, and to convert `../forge/forge-gui/res/tokenscripts/` into `output/tokenscripts/` with sidecars of their own
- [X] T032 Extend `run_convert` in `src/price_predictor/infrastructure/cli.py` to surface the token-script output path, keeping the existing `--cards-path` / `--output-path` contract intact
- [X] T033 [P] JUnit-test the sidecar in `forge-connector/src/test/java/.../ProvenanceSidecarTest.java`: a line merged from several traits carries several keys; a deduplicated trait lands in `dropped_keys`; a granted ability keys to the donor; the converted text is byte-identical to the pre-change output for a sample of cards
- [X] T034 [P] Implement `src/effects/domain/provenance.py`: the `ProvenanceKey` dataclass (script file incl. tree, face, trait kind, index within kind) and the per-line sidecar record, including `dropped_keys` (prior art: `draft/domain/draft_geometry.py`, the pure-dataclass half of the same split)
- [X] T035 Implement `src/effects/infrastructure/sidecar_io.py` reading the sidecar into `src/effects/domain/provenance.py` dataclasses, with the join rule: a key in `lines` resolves; a key in `dropped_keys` resolves to no line and is kept; a key in neither fails loudly
- [X] T036 [P] Unit-test the join in `tests/unit/effects/infrastructure/test_sidecar_io.py` covering those three cases plus the cache-row alignment rule (row *i* ↔ `lines[i]`)

### Shared locator

- [X] T037 Extend `src/sealed/infrastructure/converted_card_locator.py` with source-tree awareness (`cardsfolder`, `tokenscripts`, `variant-scripts`) rather than adding a parallel locator; existing sealed callers keep their current behaviour by default (prior art: research.md § Overlapping domain vocabulary)
- [X] T038 [P] Unit-test the extension in `tests/unit/sealed/infrastructure/test_converted_card_locator.py`: same filename resolving differently per tree, and every existing sealed test still passing

**Checkpoint**: the schema is frozen, the sidecar joins runtime traits to converted lines, and shards
can be written and read. User stories can begin.

---

## Phase 3: User Story 1 — First embeddings without patching Forge (Priority: P1) 🎯 MVP

**Goal**: produce the first `e` vectors and all three gate verdicts against a **stock, unpatched**
Forge, using only the public event bus and a bracket around stack resolution.

**Independent Test**: run `match-outcomes --effect-records` briefly, train, `encode-abilities`,
`evaluate-effect-model`; gates 1 and 3 return pass/fail and gate 2 returns a per-keyword routing
verdict. Nothing in `../forge` has been modified.

### Tests for User Story 1 (write first, expect failure)

- [X] T039 [P] [US1] Contract test in `tests/unit/effects/test_cli_contract_us1.py`: every flag and default in [contracts/cli.md](contracts/cli.md) for `build-vocab`, `extract-keyword-definitions`, `train-effect-model`, `encode-abilities`, `evaluate-effect-model` — parsed values only, no execution. Exclude the rows cli.md marks stage four (`--variant-scripts`, `--surface script`), which US4's own contract test covers
- [X] T040 [P] [US1] Test in `tests/unit/effects/application/test_split.py` that the card-disjoint split holds out cards by newest first printing until they cover ≥ 8% of the cards under `output/cardsfolder/` (token scripts and variant scripts are not in the printing order and stay out of the denominator), **excludes every game holding a record naming a held-out card**, and takes game-disjoint validation from 10% of what remains
- [X] T041 [P] [US1] Test in `tests/unit/effects/application/test_sampling.py` that `--kind-mix` renormalizes over the classes present (a stage-one corpus has only three of eight), that rarity weighting is ∝ effective_games^(−0.5) capped at 20×, and that `combat` and `playability-legality` sample uniformly
- [X] T042 [P] [US1] Test in `tests/unit/effects/application/test_gates.py` that gate 1 requires **all three** of >= 0.05 absolute affected-gate F1 improvement, >= 0.05 absolute zone-outcome accuracy improvement on affected entities, and >= 5% relative reduction in mean Poisson deviance over count-valued fields, on the card-disjoint split's unique-text stratum over resolution records; that gate 2 requires ≥ 200 qualifying records and ≥ 70% direction agreement per keyword and blocks nothing, and gate 3 requires ≤ 0.5 mean pairwise cosine **over 10,000 random pairs** and ≤ 30% top principal component
- [X] T043 [P] [US1] Test in `tests/unit/effects/domain/test_effect_head_input.py` that position ids reset at each `[CARD]`, `[ACT]` is empty for `combat`, controller tags derive from `actor_player`, and numeric scalars enter as raw plus log1p
- [X] T044 [P] [US1] Contract test in `tests/unit/effects/test_cli_contract_match_outcomes.py`: `--effect-records` has **no default** on `sealed match-outcomes` (FR-031) and the five cap/budget flags carry the defaults in [contracts/cli.md](contracts/cli.md). US1 ships this surface, so its fast-suite coverage belongs here rather than in US2
- [X] T045 [P] [US1] Integration test in `tests/integration/test_effects_collection.py` (marked `integration`): a short instrumented `match-outcomes` run writes shards of `resolution` and `combat` records, every record carries `mode = degraded`, and `match-outcomes.txt` / `cards-played.txt` keep their format and row semantics

### Collection — stage one, no patch

- [X] T046 [US1] Implement `forge-connector/.../effects/SnapshotBuilder.java`: build the `global` / `players[]` / `entities[]` / `refs` / `pending_event` blocks of every record's `state` per [contracts/record-schema.md](contracts/record-schema.md), with computed (post-layer) characteristics, `granted_attached` and `granted_temporary` written as separate fields, absolute controller ids and no stored perspective, and **inclusion tiers 1 and 2** — the two FR-022 requires from stage one
- [X] T047 [P] [US1] JUnit-test the snapshot in `forge-connector/src/test/java/.../SnapshotBuilderTest.java`: tiers 1-2 present, a referenced object carried as an entity in whatever zone it sits with that zone recorded, the two grant channels distinct, and characteristics computed rather than printed
- [X] T048 [US1] Implement the JSONL shard writer in `forge-connector/.../effects/RecordShardWriter.java`, writing `{run_id}.{worker}.jsonl` with the `record_id` / `game_id` construction from the schema contract
- [X] T049 [US1] Implement the event-bus subscriber and stack-resolution bracket in `forge-connector/.../effects/BusBracketCollector.java`: subscribe via `Game.subscribeToEvents`, read the resolving ability from the stack, attribute events by bracket. The `mode` value comes from T050's detection rather than being hardcoded here
- [X] T050 [US1] Implement hook detection at worker startup in `forge-connector/.../effects/AttributionMode.java`: probe for the patch hooks, fall back to bracket-only, and stamp every record's `mode` accordingly
- [X] T051 [US1] Implement bracket attribution rules in `BusBracketCollector`: state-based-action deaths attribute to the bracket they follow; combat damage attributes to its damage-step bracket; stat changes with no trigger are captured by diffing computed stats inside the bracket when the stats-changed bus event fires, coalesced per bracket, with trigger-channel events taking precedence where both report
- [X] T052 [US1] Implement resolution-record emission (cost half and effect half, linked by `link_id`, with the five outcomes) in `forge-connector/.../effects/ResolutionCollector.java`. Attribution granularity is the sub-ability; an event attributed to a link the sidecar does not map falls back to the **root line** rather than being dropped (edge case 3). On a modal resolution, `ability` is the chosen `option` line's key, not the parent `spell` line's (FR-017)
- [X] T053 [US1] Implement combat-record emission in `forge-connector/.../effects/CombatCollector.java` — one record per damage step, so a first-strike combat yields two
- [X] T054 [US1] Extend `forge-connector/.../MatchWorkerMain.java` with `--effect-records` and the cap/budget system properties, **and** a records-only mode in which `output.file` is absent and no `CardsPlayedWriter` is constructed (the effects collectors reuse this worker; without the mode they would write the sealed corpora)
- [X] T055 [US1] Thread `--effect-records` and the cap/budget flags through `src/sealed/infrastructure/cli.py`, `src/sealed/application/match_outcomes.py`, and `src/sealed/infrastructure/match_worker_connector.py`. Absent the flag, all three behave exactly as today
- [X] T056 [P] [US1] JUnit-test the collectors in `forge-connector/src/test/java/.../CollectorTest.java`: bracket attribution of an SBA death, two records for a first-strike combat, and a countered spell producing a cost record with no partner

### Keyword definitions and vocabulary

- [X] T057 [P] [US1] Implement `forge-connector/.../KeywordDefinitionMain.java` emitting keyword → reminder-text template for every keyword as JSON
- [X] T058 [P] [US1] Implement `src/effects/infrastructure/keyword_definition_connector.py` spawning it, mirroring `match_worker_connector.py`'s shape (prior art: research.md § Adjacent prior art — one connector per Java main)
- [X] T059 [US1] Implement `src/effects/application/extract_keyword_definitions.py`
- [X] T060 [US1] Implement `src/effects/application/build_vocab.py` wrapping `price_predictor.application.build_vocabulary` and the promoted truncation (T014), scanning converted cards, token scripts, and the keyword-definition file, seeding `[PAD]`, `[UNK]`, `cardname`, `[MASK]`, `[CLS]`
- [X] T061 [P] [US1] Unit-test in `tests/unit/effects/application/test_build_vocab.py`: seeded specials present, `--target-size` respected, and the prose/script vocab paths kept separate

### Model

- [X] T062 [US1] Implement `src/effects/domain/ability_tokenizer.py` wrapping `MtgTokenizer`: `[MASK]`/`[CLS]`, character offsets so role spans can be applied per token, and keyword-expansion hooks. The shared tokenizer returns bare strings with no offsets and seeds neither special
- [X] T063 [P] [US1] Unit-test the tokenizer in `tests/unit/effects/domain/test_ability_tokenizer.py`: offsets map to the sidecar's role spans, unknown words become `[UNK]` with no subword fallback, unknown keywords always expand
- [X] T064 [US1] Implement `src/effects/domain/ability_encoder.py`: token + position + role embeddings, the monotone number embedding (learned base plus log1p(n) × learned direction), N transformer layers, `[CLS]` pooling to `e` with train-time additive noise (prior art: `sealed/domain/encoder_model.py`)
- [X] T065 [US1] Implement keyword-expansion dropout in the encoder: expand at `--keyword-expand-p`, always expand unknown keywords, instantiate parameterized templates with the instance's own values (a keyword referenced without an instance, inside another definition, expands with the template's generic wording), never expand host-card-bodied keywords (saga chapters, class levels), leave nested keywords as tokens
- [X] T066 [US1] Implement `src/effects/domain/effect_head_input.py` (prior art: `draft/domain/draft_state.py`, the typed model-input module in the sibling package): snapshot → `[GLOBAL] [ACT] [PLAYER]… [CARD] e e …` with position ids resetting at each `[CARD]`, controller tags relative to `actor_player`, context-ability dropout, entity ability tokens limited to printed and attachment-granted lines with temporary grants riding the overlay (FR-073), and the per-kind input variations from the root spec's record-kind table
- [X] T067 [US1] Implement `src/effects/domain/effect_model.py` (prior art: `draft/domain/draft_agent_model.py`, the multi-headed model in the sibling package): the trunk plus the per-entity head mapped over **every `[CARD]` and `[PLAYER]`** output (gate, then permanent fields, player fields, legality bits), the created-objects head at `[GLOBAL]` (K = 4 canonically ordered slots plus overflow flag), and the verdict head at `[ACT]`
- [X] T068 [US1] Implement the losses in `effect_model.py`: Poisson-family count regression for magnitudes, direction-plus-magnitude decomposition for signed deltas, categorical for closed vocabularies, binary for gates and bits, per-record normalization over entity count, the sparse-field curriculum at `--curriculum-step`, and — the rule that makes a three-of-eight-kind stage-one corpus trainable — no loss contribution from fields whose record kinds are absent from the corpus (FR-085)
- [X] T069 [P] [US1] Unit-test the heads in `tests/unit/effects/domain/test_effect_model.py`: the per-entity head produces player-field outputs at `[PLAYER]` positions, created-object slots follow canonical order, overflow sets its flag, and conditional fields contribute no loss where the gate target is off

### Training

- [X] T070 [US1] Implement `src/effects/application/train_effect_model.py`: batch assembly (mix games, group each game's records, encode each unique ability text once), the sampling mixture, rarity weighting, and the no-acting-text rule
- [X] T071 [US1] Implement split derivation in the same module per T040's contract, and record the split — held-out cards plus the `game_id` set across both strata — into the checkpoint
- [X] T072 [US1] Implement the two training-only auxiliary heads in `src/effects/domain/effect_model.py` (FR-063): an MLM head over masked tokens (`--mlm-weight`, `--mlm-mask-prob`) and a script-API classification head from `e` predicting the trait's API type plus its parameter-key set (`--api-weight`). The script-API head is not stage-four work — it stands in for script structure through stage three, so the MVP needs it
- [X] T073 [US1] Implement `src/effects/infrastructure/effect_model_store.py` over `price_predictor.infrastructure.torch_checkpoint`, recording vocabulary and keyword-definition paths, their content hashes, and the withheld keyword; filtering the MLM, script-API, and pairing heads at save time (prior art: `draft/infrastructure/draft_agent_store.py`)
- [X] T074 [US1] Implement `--split-from` inheritance in `train_effect_model.py` (split, vocabulary and keyword-definition paths; no in-repo prior art — no sibling trainer shares a split across runs) and fail-fast when a variant run omits it
- [X] T075 [US1] Implement the four `--variant` baselines exactly as defined in FR-094: `identity`, `state-only`, `no-state` (entity ability `e` tokens zeroed too), `taxonomy`
- [X] T076 [US1] Implement `--withhold-keyword`, holding one keyword's token out of training and always expanding its occurrences
- [X] T077 [US1] Implement the training loop in `src/effects/application/train_effect_model.py`: an epoch is `--steps-per-epoch` optimizer steps, validation runs between epochs on both strata, `--epochs` bounds the run, and training early-stops after `--patience` epochs with no new card-disjoint validation best
- [X] T078 [US1] Select the best checkpoint by **card-disjoint validation loss** (FR-089) and write it plus a rolling `latest.pt` under `--model-output`, which defaults to `models/effects/effect-model/` for `--variant full` and `models/effects/effect-model/{variant}/` otherwise (prior art: the timestamp-plus-latest convention in `sealed/infrastructure/scorer_store.py`)
- [X] T079 [US1] Wire the optimizer, schedule, and hardcoded constants: AdamW, lr 1e-4 constant after **linear** warmup over the first 5% of scheduled steps, per-parameter-group clip 1.0 via `torch_training.clip_per_group`, seed 42, and the architecture constants
- [X] T080 [US1] Implement `--context-cache` / `--cache-refresh` as the documented 8 GB fallback (stop-gradient momentum cache), verifying against the budget per Principle VIII
- [X] T081 [P] [US1] Test the evaluator's and trainer's fail-fast rows from cli.md's exit-code contract in `tests/unit/effects/application/test_failfast.py`: a `--variant-checkpoint` whose split or hashes disagree with `--checkpoint` fails naming both, and a variant training run without `--split-from` fails
- [X] T082 [P] [US1] Unit-test the checkpoint contract in `tests/unit/effects/infrastructure/test_effect_model_store.py`: split round-trip, hash mismatch fails fast, training-only heads absent from the saved artifact, withheld keyword recorded

### Cache and evaluation

- [X] T083 [US1] Implement `src/effects/domain/ability_cache_layout.py` — the `(n_lines, e_dim)` layout and its row-alignment rule (prior art: parallel to `sealed/domain/card_embedding_layout.py`, which describes a different artifact: one pooled vector plus deterministic features)
- [X] T084 [US1] Implement `src/effects/application/encode_abilities.py` and `src/effects/infrastructure/ability_cache_store.py`: one file per source under `output/effects/abilities/<tree>/…`, `--variant` resolving checkpoint and suffix together, the vocabulary and keyword-definition hash check against the checkpoint, `taxonomy` emitting its lookup into the same row layout, cache-time keyword handling matching inference (known keywords stay tokens, unknown always expand — FR-103), idempotent, `--clean` removing only what it wrote
- [X] T085 [P] [US1] Unit-test the cache in `tests/unit/effects/application/test_encode_abilities.py`: row alignment against a sidecar, per-tree subtrees, variant suffixes written beside rather than over the shipping cache, idempotence
- [X] T086 [US1] Implement the consumer-side layout contract in `src/effects/domain/ability_cache_layout.py`: a card is a `[CARD]` token plus its ability `e` rows, position ids reset to 0 at each `[CARD]`, and multi-face cards separate faces with an `[ALTERNATE]` token tagged from the converted `layout:` line, positions continuing across faces
- [X] T087 [P] [US1] Unit-test that contract in `tests/unit/effects/domain/test_ability_cache_layout.py`: position reset at each `[CARD]`, continuation across `[ALTERNATE]`, and the layout tag read from the `layout:` line
- [X] T088 [US1] Implement `src/effects/application/evaluate_effect_model.py`: load `--checkpoint` and every `--variant-checkpoint`, read the split and hashes from the checkpoint, fail fast on disagreement, score only recorded `game_id`s, and report per record kind and per stratum
- [X] T089 [US1] Implement FR-097 in both inference commands: `--vocab-path` and `--keyword-definitions` default to the paths the loaded checkpoint recorded, and an explicit value overrides while still being hash-checked. Test it by execution, not by argparse inspection — a default resolved from a checkpoint is invisible to a parser-only contract test
- [X] T090 [US1] Implement the four held-out strata (unique-text, shared-text, novel combinations, numeric extrapolation) and class-balanced per-field metrics conditioned on affected entities, with `state-only` reported as every kind's floor
- [X] T091 [P] [US1] Implement `src/effects/domain/damage_step_keywords.py`: one row per keyword — qualifying predicate, rules direction, affected fields — for the eight damage-step keywords
- [X] T092 [US1] Implement gate 2 in the evaluator: a **model-side** input perturbation removing the keyword from the ability token or the temporarily-granted-keywords channel, over qualifying game-disjoint combat records; report a per-keyword routing verdict; block nothing
- [X] T093 [US1] Implement gates 1 and 3 with their pinned thresholds; both block shipping the model or the cache
- [X] T094 [US1] Implement `src/effects/domain/ward_twins.py` (the checked-in functional-twin texts) and the ward canary — ward's `e` closer in cosine distance to each checked-in twin than the **median** of ward's cosine distances to all bare single-keyword `e` vectors (FR-112, the criterion SC-004 rests on)
- [X] T095 [US1] Implement the remaining stage-one checks: nearest-neighbour and UMAP, zero-shot keyword (reading the withheld keyword from the checkpoint), scaling calibration, the `identity` variant on the game-disjoint split, the decodability battery via `ridge_probes.py` (T015) over pooled per-card `e` (mean and max concatenated over the card's ability rows), reported side by side with the sealed encoder at `models/sealed/encoder/latest.pt`, the pooled-`e` scorer smoke test writing only into a scratch cards folder, the `taxonomy` comparison, and the `no-state` average-effect control
- [X] T096 [US1] Make checks whose records do not exist yet skip rather than fail: matched real-vs-fork and probe-diff wait for US3, the role-polarity probe for US2
- [X] T097 [US1] Wire every US1 subcommand into `src/effects/infrastructure/cli.py`
- [X] T098 [US1] Write `src/effects/CLAUDE.md` covering the US1 subcommands and artifacts, and update the root `CLAUDE.md` (it says "Three Python packages live under `src/`" and carries the per-corpus file-format contracts) with the effect-record, sidecar, and ability-cache formats. Constitution Quality Gate: documentation ships in the same change as the workflow, so the MVP is not documentation-free
- [ ] T099 [US1] Run [quickstart.md](quickstart.md) end to end and confirm each stated check, including that the sealed corpora are unaffected by the flag

**Checkpoint**: US1 delivers a populated embedding cache and all three gate verdicts against stock
Forge. **Gate 2's per-keyword output decides whether US3 builds probe machinery at all.**

---

## Phase 4: User Story 2 — Complete attribution and the corpus match play cannot reach (Priority: P2)

**Goal**: the three-hook patch set, the four record kinds it unlocks, snapshot tier 3, and the coverage
collector.

**Independent Test**: apply the patches, re-run collection, confirm `mode = patched` and the four new
kinds; run `collect-coverage` to completion on a small `--target-records` and read its two residues.

### Tests for User Story 2

- [X] T100 [P] [US2] Test in `tests/unit/effects/domain/test_records_us2.py` that `continuous` records coalesce per (game, static, board hash), that `trigger` negatives are drawn same-event-type at ~1:1, and that no policy verdict can be represented in a `playability` payload
- [X] T101 [P] [US2] Contract test in `tests/unit/effects/test_cli_contract_collectors.py`: every flag and default in [contracts/cli.md](contracts/cli.md) for `sealed match-outcomes --effect-records` (no default), `collect-coverage`, and the shared cap/budget table — `--mana-cap` 2000, `--playability-rate` 0.1, `--interventions-per-game` 2, `--probes-per-game` 2, `--probe-keywords` empty
- [X] T102 [P] [US2] Test in `tests/unit/effects/application/test_coverage.py`: the castability consult ranks but never drops; satisfaction counts acting-host / event-subject / referenced-ref records and **not** mere snapshot presence; a card with no new qualifying record for `--no-progress-rounds` retires; the run terminates
- [X] T103 [P] [US2] Integration test in `tests/integration/test_effects_coverage_isolation.py` (marked `integration`): after a `collect-coverage` run, `output/sealed/match-outcomes.txt` and `cards-played.txt` are byte-identical to their pre-run state (US2 acceptance 8, FR-052, half of SC-007)
- [X] T104 [P] [US2] Integration test in `tests/integration/test_effects_patched.py` (marked `integration`): against a patched checkout, records carry `mode = patched` and rewrite/continuous/trigger/playability kinds appear

### The patch set

- [X] T105 [US2] Write `forge-connector/patches/01-trigger-cause.patch`, the trigger-handler cause-channel hook, plus the one-line addition of `AbilityKey.Cause` at the `Destroyed` firing site where the ability is in scope but currently dropped
- [X] T106 [US2] Write `forge-connector/patches/02-replacement-hook.patch`, the replacement-execution-point hook, deep-copying the parameter map before the call (the handler's own copy is shallow and replacements mutate nested structures in place)
- [X] T107 [US2] Write `forge-connector/patches/03-subability-pointer.patch`, the threaded currently-resolving-sub-ability pointer
- [X] T108 [US2] Write `forge-connector/patches/04-logging-points.patch`, the trigger-fire and playability logging points: the trigger handler's condition evaluation, and the AI's candidate computation, combat-setup legality, and legality/cost-adjustment checks
- [X] T109 [P] [US2] Document the patch set in `forge-connector/patches/README.md`: what each hook is for, how to apply and re-apply, and how to verify the worker reports `mode = patched`

### Record kinds the patch unlocks

- [X] T110 [US2] Implement cause-attributed trigger collection and sub-ability attribution in `forge-connector/.../effects/TriggerCollector.java`
- [X] T111 [P] [US2] Implement mana-record collection in `forge-connector/.../effects/ManaCollector.java` riding the inline cast/resolution triggers, honouring `--mana-cap` (per unique mana-ability text, per worker process)
- [X] T112 [P] [US2] Implement rewrite-record collection in `forge-connector/.../effects/RewriteCollector.java` at the shared replacement execution point
- [X] T113 [P] [US2] Implement continuous-effect collection in `forge-connector/.../effects/ContinuousCollector.java` from the per-card layer tables after a recompute, coalesced per stable board, with static id 0 (temporary pumps) staying on the resolution bracket, and the acting static's own contributions removed from the record's entity inputs
- [X] T114 [P] [US2] Implement trigger-fire collection in `forge-connector/.../effects/TriggerFireCollector.java` at the condition-evaluation hook, with same-event-type negatives
- [X] T115 [US2] Implement playability collection in `forge-connector/.../effects/PlayabilityCollector.java` for all three subkinds, honouring `--playability-rate` (which samples `decision`-subkind logging points only; `attackers` and `blockers` are always logged), snapshotting defensively — a verdict may be abandoned mid-evaluation and the legality check mutates the checked ability's targets, so never reuse a checked ability object
- [X] T116 [US2] Add snapshot tier 3 (unreferenced stack contents) to the snapshot builder in `forge-connector/.../effects/SnapshotBuilder.java`

### Coverage collector

- [X] T117 [US2] Implement `forge-connector/.../CastabilityMain.java` — FR-047's consult, which needs a live Forge
- [X] T118 [P] [US2] Implement `src/effects/infrastructure/castability_connector.py` spawning it (one connector per Java main)
- [X] T119 [US2] Implement `src/effects/infrastructure/collector_connector.py` spawning the records-only worker (T054) over coverage decks, reusing `ForgeWorkerPool` (prior art: its third supervisor; already extracted at the second instance)
- [X] T120 [US2] Implement `src/effects/application/collect_coverage.py`: rounds, held-out exclusion via `--split-from`, corpus-wide 40-card deck building (23 nonlands plus basics from `compute_basic_lands`) weighted by record scarcity, with deck candidates and the coverage unit drawn from the `output/cardsfolder/` entry of `--cards-folder` alone, the consult as a ranking input only, satisfaction and retirement, and the two residues
- [X] T121 [P] [US2] JUnit-test the US2 collectors in `forge-connector/src/test/java/.../PatchedCollectorTest.java`: cause-attributed trigger attribution, a stacked-replacement pair each carrying the event it received, a coalesced continuous record per stable board, a same-event-type trigger negative, and snapshot tier 3 present
- [X] T122 [US2] Wire `collect-coverage` into the CLI with its documented defaults
- [X] T123 [US2] Implement the role-polarity probe in the evaluator, now that mana records give it its effect-position half: compare the model's predicted mana-pool sign for `{R}` in cost position against effect position

**Checkpoint**: attribution is exact, the corpus's largest line kind has records, and the cards sealed
pools cannot contain have reached play.

---

## Phase 5: User Story 3 — Counterfactual records for what observation cannot reach (Priority: P3)

**Goal**: interventional resolutions, snapshot tier 4, and — only for keywords gate 2 failed — the
damage-step probe.

**Independent Test**: interventional records appear with the fork's own state, paired to their real
counterparts by `mirror_of`; the evaluator reports matched real-vs-fork agreement.

### Tests for User Story 3

- [X] T124 [P] [US3] Test in `tests/unit/effects/domain/test_fork_records.py`: an interventional record carries `interventional = true` + `fork = true`, no `link_id`, and no activation partner; a probe carries `fork = true` + `interventional = false`; at most two forks name one real resolution
- [X] T125 [P] [US3] Integration test in `tests/integration/test_effects_forks.py` (marked `integration`): a fork failing its copy-score check writes no record but still counts against the budget

### Interventional resolutions

- [X] T126 [US3] Implement interventional resolution in `forge-connector/.../effects/InterventionCollector.java`: run the game simulator on a fork with chosen targets and modes, route unaffordable candidates through the play-without-paying-mana path, verify the located ability through its provenance key, store the fork's own state, write the effect half only
- [X] T127 [US3] Implement `--probe-keywords` (comma-separated, default empty) as the runtime switch: with it unset no probe fork is taken at all, whatever the build state. This is US3 acceptance 3, and it is separate from the build decision gate 2 drives
- [X] T128 [US3] Enforce the budgets: `--interventions-per-game` and `--probes-per-game`, plus the fixed constant of at most two forks per real resolution, independent of either flag. Every fork counts against its budget even when discarded by the copy-score guard
- [X] T129 [US3] Score-check every fork against the live game at creation before any perturbation; on mismatch log a warning, discard the fork, and still count it against the budget
- [X] T130 [US3] Add snapshot tier 4 (unreferenced hand and graveyard) to `forge-connector/.../effects/SnapshotBuilder.java`

### Damage-step probes (contingent on gate 2)

- [X] T131 [P] [US3] JUnit-test the US3 fork path in `forge-connector/src/test/java/.../ForkCollectorTest.java`: an interventional record storing the fork's own state with no activation partner, a discarded fork still counting against its budget, and snapshot tier 4 present
- [X] T132 [US3] Implement the probe in `forge-connector/.../effects/DamageStepProbe.java` — **only if gate 2 failed for at least one keyword**: fork at declare-blockers after blocks lock, strip one keyword below the layer system with a keyword-cache refresh, resolve the damage step
- [X] T133 [US3] Install a seeded random source for both branches and restore it in a `finally` (Forge's own restore is not in a `finally`), and forbid two concurrent games per JVM while probes are on
- [X] T134 [US3] Record the fork branch as an ordinary `combat` record with `mirror_of` set; compute the real-vs-fork diff only at evaluation time, never as a training target
- [X] T135 [US3] Implement matched real-vs-fork agreement (prediction agreement over pairs joined by `mirror_of`) and the probe-diff re-check (gate 2's canary re-run over the real-and-fork combat pairs, probed keywords only) in the evaluator

**Checkpoint**: the corpus reaches abilities Forge never chose to use, and gate 2's failures have their
isolated counterfactual.

---

## Phase 6: User Story 4 — Encoding the mechanism instead of its description (Priority: P4)

**Goal**: the script surface as the primary encoding, with prose as the paired secondary, plus
synthetic script variants.

**Independent Test**: `build-vocab --surface script` writes its own vocabulary and leaves the prose one
untouched; stage-one-to-three checkpoints still load and encode against their recorded path.

### Tests for User Story 4

- [ ] T136 [P] [US4] Contract test in `tests/unit/effects/test_cli_contract_us4.py`: the stage-four CLI surface — `collect-variants`' flags and defaults, `build-vocab --surface script`, and `--variant-scripts` on the trainer, `encode-abilities` and `evaluate-effect-model`
- [ ] T137 [P] [US4] Test in `tests/unit/effects/domain/test_script_tokenizer.py` that `Creature.nonDragon+OppCtrl` splits into `Creature`, `nonDragon`, `OppCtrl`
- [ ] T138 [P] [US4] Test in `tests/unit/effects/application/test_variants.py`: variant records carry `synthetic = true` and `variant_of`, a variant of a held-out card is held out with it, variants contribute no pairing loss, and no variant is ever converted to prose

- [ ] T139 [P] [US4] Integration test in `tests/integration/test_effects_variant_isolation.py` (marked `integration`): after a `collect-variants` run, the two sealed corpora are byte-identical to their pre-run state (US4 acceptance 5, FR-059, the other half of SC-007)

### Script surface

- [ ] T140 [US4] Add `--surface script` to `build-vocab`, scanning the sidecars' script lines and writing `models/effects/vocab-script.txt` — a path of its own, so the rebuild never overwrites the prose vocabulary earlier checkpoints recorded
- [ ] T141 [US4] Implement the compositional script tokenizer in `ability_tokenizer.py`
- [ ] T142 [US4] Make the script line the primary encoding surface with prose paired, in `src/effects/domain/ability_encoder.py` (surface selection and the paired encode) and `src/effects/domain/effect_model.py` (the pairing loss),, under an asymmetric loss with stop-gradient on the script side. The surface follows the loaded `--vocab-path`, so `--surface` keeps its `prose` default and stage-one-to-three checkpoints continue to encode against the vocabulary they recorded
- [ ] T143 [US4] Extend `KeywordDefinitionMain` to capture the generated implementation script as text at the keyword factory for the script-generated majority; the engine-coded minority keeps its reminder template
- [ ] T144 [US4] Switch keyword-expansion dropout in `src/effects/domain/ability_encoder.py` to the captured script on the script surface, falling back to the template where no script exists

### Synthetic variants

- [ ] T145 [US4] Implement `src/effects/domain/script_variants.py`: the perturbation rules (a numeric parameter shifted by up to ±3 or doubled, floored at zero in either case; a selector swapped from the checked-in whitelist)
- [ ] T146 [US4] Make `output/effects/variant-scripts/` loadable by the worker JVM: `ForgeEnvironmentInitializer` resolves only `forge-gui/res/cardsfolder` (line 37) and Forge's own custom-card path is the fixed `ForgeConstants.USER_CUSTOM_CARDS_DIR` (`FModel.java:203`), so extend the initializer and `MatchWorkerMain` to accept an extra card-source directory. FR-055's "loaded from there as custom cards" is a Java-side capability; without it `collect-variants` cannot put a perturbed card into a game
- [ ] T147 [US4] Implement `src/effects/application/collect_variants.py`: read source scripts from `--forge-cards-path`, write perturbed scripts and their sidecars to `output/effects/variant-scripts/`, hold a variant out whenever its source card is held out (FR-057), load them as custom cards, deck and schedule them exactly as coverage decks are through `collector_connector.py` and the records-only worker (so no sealed corpus is written), and honour `--variant-volume`
- [ ] T148 [US4] Add `--variant-scripts` to `train-effect-model` (`src/effects/application/train_effect_model.py`), defaulting to `output/effects/variant-scripts/` at stage four, and make the trainer read that tree and its sidecars so variant records are trainable. Without this the corpus `collect-variants` produces cannot be trained on
- [ ] T149 [US4] Add `--variant-scripts` to `evaluate-effect-model` (`src/effects/application/evaluate_effect_model.py`) with the trainer's default, so held-out strata and the reported checks can score variant records
- [ ] T150 [US4] Extend `encode-abilities` to the variant tree, aligning rows to script lines rather than rendered lines
- [ ] T151 [US4] Wire `collect-variants` into the CLI with its documented defaults

**Checkpoint**: the encoder reads mechanism rather than description, and the corpus contains texts that
never existed on a real card.

---

## Phase 7: Polish & Cross-Cutting Concerns

- [ ] T152 Extend `src/effects/CLAUDE.md` and the root `CLAUDE.md` with the US2-US4 subcommands and artifacts, on top of the US1 documentation already written
- [ ] T153 [P] Update `README.md` with the effects workflows: collection, coverage, training, encoding, evaluation
- [ ] T154 Performance review per Principle VIII across all stories — I/O batching and caching, GPU placement, GPU batching with no per-item host↔device transfers in hot loops, vectorized hot loops (the snapshot-to-tensor derivation especially), streaming for the shard corpus, load-once reuse — verified against the 8 GB budget
- [ ] T155 Fill the Outcome section of [`../../experiments/2026-09-04-ability-effect-model-design.md`](../../experiments/2026-09-04-ability-effect-model-design.md) with the first run's gate results — that file is the only home for run numbers

---

## Dependencies & Execution Order

### Phase dependencies

- **Setup (Phase 1)**: no dependencies
- **Foundational (Phase 2)**: depends on Setup; **blocks every user story**. The schema is frozen here and every story reads the sidecar
- **US1 (Phase 3)**: depends on Foundational only
- **US2 (Phase 4)**: depends on Foundational; independent of US1 in principle, but shares the collectors US1 builds, so in practice follows it
- **US3 (Phase 5)**: depends on US2's collectors for the interventional path, and on **US1's gate-2 output** for whether the probe half is built at all
- **US4 (Phase 6)**: depends on Foundational for the sidecar's `script_text`, and on US2's coverage collector for variant scheduling. Otherwise independent of US2 and US3
- **Polish (Phase 7)**: after the stories being delivered

### Critical ordering

T002 (create `cli.py`) → T003 (`__main__.py` imports it) → T005 (which verifies `python -m effects
--help`). T007 (spec amendment) → T013 (the import test asserts against it). T010 → T012 → T026 (the
record reader needs the extraction). T028/T029 → T030 → T031 → T035 (keys before the writer before the
reader). T071 (split derivation) → T073 (the checkpoint store it records into). T048 (shard writer)
before T049/T052/T053, which emit through it.

### Parallel opportunities

`[P]` marks a task that shares no file with another task in its group. It does not mean "start now" —
a test still runs after the thing it tests exists.

- T004 and T005 in Setup, once T001–T003 have created the package
- T011, T013, T017 once their subjects exist; T018 then T019 (same file), with T020 and T021
  independent of both
- Within US1: the six test tasks T039–T045 together; then T057/T058 alongside the model tasks
- Within US2: T111–T114 are four independent collector classes once the patch set lands
- US4's script-surface work can proceed alongside US3, since neither touches the other's files

---

## Implementation Strategy

### MVP: User Story 1 only

Phase 1 → Phase 2 → Phase 3, then **stop and validate**. That yields the first `e` vectors, gates 1
and 3, and the gate-2 routing verdicts — against a stock Forge, with nothing in `../forge` modified.
If gate 1 or gate 3 fails, the design is wrong and no patch has been written yet. That is the whole
point of the staging.

### Incremental delivery

1. Setup + Foundational → schema frozen, sidecar joining
2. + US1 → embeddings and gates (MVP, no patch)
3. + US2 → full attribution, statics covered, coverage hole closed
4. + US3 → counterfactuals, and probes **only for the keywords gate 2 failed**
5. + US4 → mechanism-level encoding and synthetic variants

### Notes

- Every task above that creates a module names its prior art or its justification for divergence, per Principle VII
- Tests are written before or alongside their implementation, never deferred (Principle I)
- Commit per task or per logical group; each checkpoint is a validation point
- Run results go in the design record's Outcome section, never in the root spec and never here
