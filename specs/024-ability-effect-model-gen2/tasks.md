---

description: "Task list for 024-ability-effect-model-gen2"
---

# Tasks: Ability effect model — generation 2

**Input**: Design documents from `/specs/024-ability-effect-model-gen2/`
**Prerequisites**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/](contracts/), [quickstart.md](quickstart.md)

**Tests**: Mandatory. Constitution Principle I is non-negotiable, and FR-089 names the knowledge-probe
tests explicitly. Each test task sits beside the code it covers and is written before or with it.
Fixtures come from real records and real converted output, never hand-built imitations of them.

**Organization**: Grouped by the spec's seven user stories, in priority order. Stories 1–3 change what
is decided at collection time and must all be done before the first gen-2 game. Stories 5 and 7 can
also be exercised on gen-1 artifacts (run plan stage 0) while 1–3 are in progress.

**Editing design documents**: any task that writes under `specs/` or `experiments/` (T068, T131, T132)
loads the **feature-workflow** skill first, for every edit.

**Prior art**: per Principle VII, a task that creates a new module names the prior art it reuses or
diverges from, citing [research.md § Codebase Survey](research.md#codebase-survey).

## Path Conventions

- Python: `src/effects/`, `src/sealed/`, `src/price_predictor/`; fast tests under `tests/unit/`
  mirroring the source tree; JVM-dependent tests carry the `integration` marker.
- Java: `FC/` stands for `forge-connector/src/main/java/com/pricepredictor/connector/`, and `FCT/` for
  `forge-connector/src/test/java/com/pricepredictor/connector/`.
- Forge sources cited as `forge/…` are in the sibling `../forge` checkout on `effect-record-hooks`,
  which this feature does not modify (FR-024).
- Build the connector with `mvn install -DskipTests`, never `package`.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: parallelizable (different file, no dependency on an incomplete task)
- **[Story]**: US1–US7; absent on Setup, Foundational and Polish tasks

---

## Phase 1: Setup

**Purpose**: capture the baselines that later tasks must compare against, before any code changes.

- [X] T001 Capture golden converter output before any converter change: pick about 30 Forge scripts covering a trigger with `Execute$` → `SubAbility$` chain, a `RepeatSubAbility` loop, a `ReplaceWith$` replacement whose ability has a sub-ability, a spell charm, a triggered charm, a chapter charm, a Pawprint charm, Doomsday Confluence (descriptionless charm), a die-roll card with `option` lines (e.g. Aberrant Mind Sorcerer), keyword lines (`Ward:2`, `Flying`, `Enchant:Creature`), a token script, and a synthetic script `forge-connector/src/test/resources/golden-024/undefined_svar.txt` whose trigger names `Execute$ TrigMissing` with no such SVar (FR-003). Run the current converter on them and store each `.txt` and `.provenance.json` under `forge-connector/src/test/resources/golden-024/` (list the chosen scripts in a `README.md` there)
- [X] T002 [P] Build a real-record fixture for this feature's tests: write `scripts/make_fixture_records.py` that streams gen-1 shards from `output/effects/records/` and writes a small, kind-balanced sample (resolution cost and effect halves including at least one modal resolution, trigger fired and unfired, continuous, rewrite, combat with a damage-step keyword, playability `decision`/`attackers`/`blockers`) to `tests/fixtures/effects/gen1-records.jsonl.gz`, plus the sidecars those records name under `tests/fixtures/effects/gen1-sidecars/`. Check the output against real shards before using it (memory: fixtures must mirror real output)
- [X] T003 [P] Record the baseline: run `pytest -m "not integration"` and `cd forge-connector && mvn install -DskipTests && mvn test` and note the counts in the PR description draft; every failure found is fixed before Phase 2, not carried

**Checkpoint**: goldens and fixtures exist; both suites green.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: the identity, envelope and bucketing changes several stories read. No user story starts
before this phase is complete.

- [X] T004 [P] Add `option: int | None = None` to `ProvenanceKey` in `src/effects/domain/provenance.py:54-100`: included in equality, hashing and ordering; serialized as an `"option"` member only when set; `ProvenanceSidecar._row_by_key` (`:198-208`) keys on the full key so a mode key never collapses into its root (data-model.md § ProvenanceKey)
- [X] T005 [P] Read and write the `option` member in `src/effects/infrastructure/sidecar_io.py` (`sidecar_from_dict:101`, `line_for:313`) and `src/effects/infrastructure/record_io.py` (ability-key parsing); add `ProvenanceSidecar.option_rows_after(row) -> list[int]` returning the indices of keyed `option` lines that immediately follow `row`
- [X] T006 [P] Unit-test T004–T005 in `tests/unit/effects/domain/test_provenance_resolution.py` and `tests/unit/effects/infrastructure/test_sidecar_io.py`: round-trip with and without `option`; a mode key and its root map to different rows; `option_rows_after` on a charm root, a non-charm `option` line (empty provenance, not returned), and a line with no options
- [X] T007 Add the `option` component to `FC/effects/ProvenanceKey.java` (field, JSON member written only when set, equality) without yet changing `resolve`; extend `FCT/effects/ProvenanceKeyTest.java` for serialization and inequality with the root key
- [X] T008 Add `random_seat: bool` and `what_if: bool | None` to `EffectRecord` in `src/effects/domain/records.py:419-449`, add both to `COLLECTION_METADATA_FIELDS` (`:141-143`), and extend `_check_flags` (`:509-522`) so `what_if` is rejected on any record that is not `playability`/`attackers` or `blockers`
- [X] T009 Read and write both fields in `src/effects/infrastructure/record_io.py`: `_KNOWN_ENVELOPE_KEYS` (`:94-98`), `record_to_dict` (`:543`), `record_from_dict` (`:574-603`) with an absent `random_seat` read as `False` and an absent `what_if` as `None` (FR-033)
- [X] T010 Update `tests/unit/effects/domain/test_schema_compatibility.py`: add both fields to `_FROZEN_ENVELOPE`; add a named exception for `actor_player` on `playability` records (FR-030a) so the rule-one test documents it rather than skips it; extend `tests/unit/effects/infrastructure/test_record_io.py` and `tests/unit/effects/domain/test_records.py` to load every record in `tests/fixtures/effects/gen1-records.jsonl.gz` with the defaults and to round-trip gen-2 records with both fields
- [X] T011 Add `randomSeat(boolean)` and `whatIf(Boolean)` to `FC/effects/EffectRecord.java` (fields `:59-77`, setters `:90-185`); `toJson` (`:207-227`) writes `random_seat` after `synthetic` on every record and `what_if` after it only when non-null; update `FCT/effects/CollectorTest.java:151-160` (`anEnvelopeCarriesEveryFrozenField`)
- [X] T012 [P] Create `src/effects/domain/rarity.py` with `RARITY_BUCKETS = ("1", "2-4", "5-19", "20+")` and `rarity_bucket(games: int) -> str` (prior art: `price_predictor/domain/price_buckets.py` for the `index_for` pattern; research.md § Third-instance check); test in `tests/unit/effects/domain/test_rarity.py` at every boundary
- [X] T013 [P] Move `KeywordResolver` and `qualifying_observations` from `src/effects/application/gate_two.py:71-147` into `src/effects/domain/damage_step_keywords.py`, keeping them torch-free; `gate_two.py` imports them from there. `tests/unit/effects/application/test_gate_two.py` and `tests/unit/effects/domain/test_damage_step_predicates.py` pass unchanged, and a new test asserts `import effects.domain.damage_step_keywords` does not import torch

**Checkpoint**: gen-1 fixtures load; gen-2 envelope round-trips in both languages; keys carry modes.

---

## Phase 3: User Story 1 — The encoder reads the whole ability (Priority: P1) 🎯 MVP

**Goal**: sidecar `script_text` is the whole chained, label-renamed ability; charm modes are keyed
`option` lines; the script surface has its own tokenization, seeding and keyword expansion.

**Independent Test**: spec Story 1 — run `extract-keyword-definitions`, `convert`, `build-vocab --surface script`; inspect the golden cards' sidecars; read the build log; expand every definition with no `%` or `[UNK]`.

### Converter (Java)

- [X] T014 [US1] Rewrite `TraitScript.of` in `FC/effects/TraitScript.java:57-141` to render the chain (FR-001): root segment from the trait's own `getMapParams()` in `TreeMap` order; then, for a trigger, `Trigger.getOverridingAbility()`; each `SpellAbility.getSubAbility()` in chain order; `getAdditionalAbility("RepeatSubAbility")`; for a replacement, `ReplacementEffect.getOverridingAbility()` (forge/`ReplacementEffect.java:325-327`) and its chain. `effectOf` (`:81-89`) stops returning null for replacements. Segments are joined by ` [SEG] `; each non-root segment opens with its referencing label and a colon. A visited set emits each SVar once at its first reach (FR-003)
- [X] T015 [US1] Implement chain-label renaming in `FC/effects/TraitScript.java` (FR-002a): the values of `Execute$`, `SubAbility$`, `RepeatSubAbility$`, `ReplaceWith$`, each comma item of `Choices$`, and every segment opener become `SV1…` by first appearance reading the rendered `script_text` from its start; one label keeps one number throughout the line; numbering restarts per line; amount SVars and `Count$` definitions are untouched; an undefined referenced label is still numbered
- [X] T016 [US1] Detect and report undefined SVars in `FC/effects/TraitScript.java`: a chain parameter naming a label the accessor cannot resolve is collected as `(card, line, label)` and the segment is omitted; surface the collected list from `FC/BatchConverter.java:109-112` / `FC/ConvertMain.java:52-61` as one report block on stderr, and have `src/price_predictor/infrastructure/cli.py`'s `run_convert` pass it through to the operator
- [X] T017 [US1] Charm root lines in `FC/effects/TraitScript.java`: a `Charm` root's `script_text` is its own segment only (its `Choices$` renamed `SV1,SV2,…`, `CharmNum$`, `MinCharmNum$`, repeat and Pawprint parameters), with no mode inlined (FR-001)
- [X] T018 [US1] Attribute charm modes in `FC/RulesParser.java:335-338` and `FC/effects/ProvenanceRecorder.java:38,58-76`: each `TextAbility(OPTION)` built by `FC/ability/CharmAbility.java:56-95` (also via `CharmAbility.optionsFrom:178-188`, `FC/ability/TriggeredAbilityEntry.java:45-72`, `FC/ability/ChapterAbility.java:45-53`) is recorded with source = the mode's SA from `getAdditionalAbilityList("Choices")` and key = root key + `option` index; its `script_text` is the mode's own chain opened `SV1:`; its `script_api_type` is the mode's API type (FR-005a). Non-charm `option` lines (die-roll outcomes, `TriggeredAbilityEntry.java:137-158`) keep empty provenance and `script_text`
- [X] T019 [US1] Descriptionless charms in `FC/effects/ProvenanceSidecar.java:30-112`: when a charm renders no root line (`CharmAbility.java:92-95`, Doomsday Confluence), its root key goes into `dropped_keys` and its option lines carry their mode keys
- [X] T020 [US1] JUnit tests in a new `FCT/effects/TraitScriptChainTest.java` against the T001 scripts: spec Story 1 scenarios 1, 2, 3, 11, 12; a `RepeatSubAbility` loop terminates; an undefined SVar is reported and skipped; a charm with two modes naming the same sub-ability carries it in each mode's chain; Doomsday Confluence's root key is in `dropped_keys`; the die-roll card's `option` lines are unkeyed
- [X] T021 [US1] Byte-identity test in `FCT/BatchConverterTest.java`: converting the T001 scripts reproduces every golden `.txt` byte for byte (FR-004, SC-001)
- [X] T022 [US1] Record each keyword's value formatting in `FC/KeywordDefinitionMain.java:83-98`: add `"formatter"`, the simple name of the keyword's `Keyword.type` (forge/`Keyword.java:218`); extend `FCT/KeywordDefinitionMainTest.java`
- [X] T023 [US1] Read `formatter` in `src/effects/application/extract_keyword_definitions.py:26-55` (`KeywordDefinition.formatter`, default `None` for old files) and replace `’` with `'` in `reminder_template` at load (FR-017)

### Python: sidecars, tokenization, expansion

- [X] T024 [US1] Make `[SEG]` a special token: add it to `SEEDED_SPECIALS` in `src/effects/application/build_vocab.py:46` and teach `AbilityTokenizer._split_with_offsets` (`src/effects/domain/ability_tokenizer.py:198-212`) to emit `[SEG]` (and the other bracketed specials) as one token rather than `[`, `seg`, `]`
- [X] T025 [US1] Give `AbilityTokenizer` a surface fixed at construction from `surface_of(vocab_path)` (`src/effects/domain/ability_encoder.py:191-205`); on the script surface, before lowercasing and splitting, apply in order: the whole-token exceptions of FR-009 (values of `Execute$`, `SubAbility$`, `RepeatSubAbility$`, `ReplaceWith$`, each `Choices$` item, segment-opening labels, `CounterType$` values) matched by parameter name on the raw text; the `TokenScript$` value split at `_` with `_` dropped (FR-009a); the lowercase→uppercase word break over keys and values (FR-008). `$` stays its own token. The prose surface is unchanged. The per-text tokenize cache (`:181-189`) keeps working
- [X] T026 [US1] Remove `AbilityTokenizer.tokenize_script`, `_split_selector`, `_SCRIPT_TOKEN_RE` and `_SELECTOR_SPLIT_RE` (`src/effects/domain/ability_tokenizer.py:368-424`) and delete `tests/unit/effects/domain/test_script_tokenizer.py`; move `scripts/effect_embedding_probes/keyword_expansion.py:361-362` onto `tokenize` (FR-007)
- [X] T027 [US1] Tokenizer tests in `tests/unit/effects/domain/test_ability_tokenizer.py`: spec Story 1 scenarios 4 and 5; `ConditionCompare$ GE3` → `condition compare $ ge 3`; `Count$Valid Creature.YouCtrl` → `count $ valid creature . you ctrl`; `TokenScript$ w_1_1_soldier` → `token script $ w 1 1 soldier`; `[SEG]` is one token; `sv2` and `p1p1` stay whole; `price_predictor`'s `MtgTokenizer.tokenize` returns the same tokens on a script line as before this feature, pinned from the pre-change output (FR-010)
- [X] T028 [US1] Expansion probability constant: add `INFERENCE_KEYWORD_EXPAND_P = 0.0` in `src/effects/domain/ability_tokenizer.py`; read it in `src/effects/infrastructure/model_runner.py:145`, `src/effects/infrastructure/ability_encoder_runner.py:103-106` (replacing the hardcoded 1.0), and `src/effects/application/training_loop.py:463`; unknown keywords keep expanding at any probability (FR-012)
- [X] T029 [US1] Display-name expansion in `src/effects/domain/ability_tokenizer.py`: `display_name_of(line_text)` = the text before the first colon of a keyword line; matched case-insensitively against the definitions; the whole span of tokens it splits into is replaced; the text after the colon supplies instance values; keyword words inside non-keyword lines keep the token path (FR-013, FR-014); `HOST_BODIED_KEYWORDS` (`:45-47`) is compared by display name, fixing `level up` and `read ahead` (FR-016). Callers that expand keyword lines (`surface_batching.py:208-210`, `ability_encoder_runner.py`) pass the line's instance values
- [X] T030 [US1] Template filling in `_instantiate` (`src/effects/domain/ability_tokenizer.py:339-359`): fill placeholders in order, `%1$s` repeats its value, values formatted by the definition's `formatter` through a table in a new `src/effects/domain/keyword_formatting.py` (one entry per `Keyword.type` in the definitions file; unknown formatter → raw value); remove every unfilled `%s`, `%d` and positional specifier (FR-017, FR-018) (prior art: Forge's per-class `formatReminderText`, `KeywordWithCost.java:30-35`, `KeywordWithCostAndAmount.java:55-60`, `KeywordWithAmount`; mirrored rather than called, because expansion runs in Python)
- [X] T031 [US1] Expansion tests in `tests/unit/effects/domain/test_ability_tokenizer.py`: spec Story 1 scenarios 6, 7, 10; `%1$s` used twice; more specifiers than values; a keyword in `HOST_BODIED_KEYWORDS` never expands; display name with no definition stays tokens; and a sweep over every definition in the real `output/effects/keyword-definitions.json` (skipped when absent) asserting no `%` token and no `[UNK]` (SC-002)
- [X] T032 [US1] `build-vocab --surface script` in `src/effects/application/build_vocab.py`: stage script lines through `AbilityTokenizer`'s script rules before the prose files (`:116-167`); stop scanning `generated_script` (`:111`, FR-020); after `_seed_specials` (`:209-213`) seed `sv1…svN` (N = longest chain's label count), every template word after specifier removal (FR-019), and every part of every parameter key and `$`-prefix in the staged lines (FR-009b), raising `domain_token_count` (`:173-187`) so `truncate_to_target_size` keeps them; fail when any definition expands to text containing `[UNK]`
- [X] T033 [US1] Build-vocab reports in `src/effects/application/build_vocab.py:202-205`: script-line length distribution in tokens (min, p50, p90, p99, max, count over 512); unknown-token rate over script parameters with every `*Description$` value removed; number of distinct camel-case parts; seeded-token counts by kind (FR-006, FR-011)
- [X] T034 [US1] Build-vocab tests in `tests/unit/effects/application/test_build_vocab.py`: spec Story 1 scenario 8; a key part used once survives a small `--target-size`; `generated_script` text is absent from the vocabulary; the report lines are printed
- [X] T035 [US1] Truncation reporting (FR-006): `prepare_line` (`src/effects/domain/ability_encoder.py:242-262`) returns whether it truncated; `encode_abilities.py`, `training_loop.py` (once per text per run) and `evaluate_effect_model.py` report each truncated line by provenance key; test in `tests/unit/effects/application/test_encode_abilities.py` (spec Story 1 scenario 9), and assert `encoding_text` (`ability_encoder.py:208-219`) returns prose for a keyword-derived line and a synthetic land mana line and `script_text` for every scripted line (FR-005)
- [X] T036 [US1] `build-corpus` fails when a display-name expansion lands on a sidecar line whose `script_api_type` is not `Keyword` (FR-015): check in `src/effects/application/build_corpus.py` where texts are resolved (`_held_out_text_of_key` `:590-613` neighbourhood); test in `tests/unit/effects/application/test_build_corpus.py` (spec Story 4 scenario 8)

**Checkpoint**: Story 1's independent test passes on the real Forge checkout; goldens unchanged.

---

## Phase 4: User Story 2 — Hold out whole templates, cover every held-out text (Priority: P2)

**Goal**: masked-template holdout shared by three commands; a text-keyed coverage round.

**Independent Test**: spec Story 2 — `holdout-cards --holdout-unit template` report; same-side check for number-only variants; a short `collect-coverage --only-cards` run.

- [X] T037 [P] [US2] Add `masked_template(text) -> str` to `src/effects/domain/text_holdout.py`: whitespace-normalize, replace every `*Description$` value and `CARDNAME` with `#`, then every digit run except the digits of a chain label at the FR-002a positions (a value of `Execute$`, `SubAbility$`, `RepeatSubAbility$`, `ReplaceWith$`, a `Choices$` item, a segment opener). Letters and mana colours stay (FR-039)
- [X] T038 [US2] Add `unit: Literal["template", "text"]` to `select_holdout` (`src/effects/domain/text_holdout.py:52-76`) and to `text_keyed_holdout` (`src/effects/application/train_effect_model.py:279-310`): under `template`, carriers are counted per template, eligibility and the crc32 permille test apply to the template, and every text with a held-out template is held out; `text` is today's rule unchanged (FR-040, FR-041)
- [X] T039 [US2] Holdout tests in `tests/unit/effects/domain/test_text_holdout.py`: spec Story 2 scenarios 1, 2, 4; `GE3`/`GE4` share a template; `SV1` vs `SV2` references differ; `P1P1` vs `M1M1` differ; colour variants differ; SC-003 over a sample of real sidecar texts
- [X] T040 [US2] `holdout-cards --holdout-unit` (default `template`) in `src/effects/infrastructure/cli.py:660-759` and `src/effects/application/holdout_cards.py:23-57`; the report adds held-out templates, texts, cards and the depleted share (FR-043); test in `tests/unit/effects/application/test_holdout_cards.py`
- [X] T041 [US2] `build-corpus --holdout-unit` (default `template`) in `src/effects/infrastructure/cli.py:507-520` → `src/effects/application/build_corpus.py:1352-1358`; `_held_out_text_of_key` (`:590-613`) resolves held-out texts under the chosen unit
- [X] T042 [US2] Record the unit: `CorpusManifest.holdout_unit` in `src/effects/domain/corpus_manifest.py:46-232` and `SplitProvenance.holdout_unit` in `src/effects/infrastructure/effect_model_store.py:87-138`, both omitted from serialization when `"text"` and read with default `"text"` (FR-042); a test loads a real gen-1 manifest and checkpoint payload and asserts the digest and `check_corpus` still pass
- [X] T043 [US2] `train-effect-model --holdout-unit` in `src/effects/infrastructure/cli.py:1175-1272`, with no default: absent, the trainer uses the manifest's recorded unit; given, it must equal the manifest's unit or training is refused before any step. The trainer recomputes the holdout with `text_keyed_holdout` under that unit over its sidecar roots and refuses when its held-out texts differ from the manifest's `held_out_texts`; the unit is copied into the checkpoint (FR-041, spec Story 2 scenario 3). Test both paths with a real gen-1 manifest (unit `text`) in `tests/unit/effects/application/test_train_startup.py`
- [X] T044 [US2] Text-keyed coverage in `src/effects/application/collect_coverage.py`: `--only-cards PATH` and `--min-text-games` (default 5) in `src/effects/infrastructure/cli.py:307-436`; a `TextCoverage(text, games: set[str], retired: bool)` unit; a picklable shard counter beside `count_shard` (`:155`) mapping each record's `ability` keys to held-out texts through `SidecarCache` and collecting distinct `game_id`s; decks from carriers of unsatisfied texts plus basics via `build_coverage_decks` (`:311-345`), carriers ranked by `rank_by_consult` (`:273`) through `CastabilityMain`, and texts whose carriers are all uncastable sent to an uncastable residue in the end report (FR-037); full-strength caps (`CollectionCaps` at 1.0); retirement reuses `retire_stalled` (`:348-388`) per text; the run ends per FR-037 and prints the texts under the floor (prior art: the per-card `CardCoverage` loop it mirrors)
- [X] T045 [US2] Refuse `--only-cards` beside `--exclude-cards`, `--training-corpus` or `--split-from` in `coverage_config_from` (`src/effects/infrastructure/cli.py:422-436`) before any worker starts (FR-038); tests for spec Story 2 scenarios 5–7 in `tests/unit/effects/application/test_coverage.py` and `tests/unit/effects/test_cli_contract_collectors.py`

**Checkpoint**: holdout computed identically by all three commands; coverage round runs on a text unit.

---

## Phase 5: User Story 3 — Off-policy outcomes and real legality decisions (Priority: P3)

**Goal**: the random seat, real/what-if legality, deciding-player actors, per-mode modal records.

**Independent Test**: spec Story 3 — a pilot of a few hundred games with the random seat, then `validate-corpus` and `field-coverage`.

### Collectors (Java)

- [X] T046 [US3] Make the three emitters public in `FC/effects/PatchedCollectors.java`: `recordAttackers(Player decider, GameEntity defender, List<Card> candidates, List<Card> legal, Boolean whatIf)`, `recordBlockers(Player decider, Card attacker, List<Card> candidates, List<Card> legal, int minBlockers, Boolean whatIf)`, `recordCandidate(Player decider, SpellAbility sa, boolean canPlay, boolean affordable, boolean hasTarget)`; route `combatLegalityHandler` (`:1072-1097`) and `playabilityHandler` (`:922-1008`) through them; `actor_player` = `decider` (FR-030a: candidates' controller, blockers' controller, candidate ability's activator)
- [X] T047 [US3] Classify and sample legality records in `FC/effects/PatchedCollectors.java`: a `whatIf == null` caller (the hook path) is classed by FR-027 on the redefined actor; real decisions skip the `--legality-rate` draw (`:1118-1119`, `:1153-1154`); for what-ifs the rate draw runs first and the snapshot is built only for survivors; the de-dup key becomes `(subkind, payload, snapshot hash)` via `allowDistinctRecord(subkind, payload, state)` (`:1177-1215`, FR-028); invert `FCT/effects/PatchedCollectorTest.java:1310-1317`
- [X] T048 [US3] Stamp `random_seat` at the three record-building sites (`FC/effects/BusBracketCollector.java:1228`, `FC/effects/ForkCollector.java:846`, `FC/effects/PatchedCollectors.java:3155`) from a per-game `randomSeatPlayerId` (null when the match has none) compared with the record's `actor_player` (FR-030); `what_if` on attackers/blockers only (FR-031)
- [X] T049 [US3] Collector tests in `FCT/effects/PatchedCollectorTest.java`: spec Story 3 scenarios 6–10; a `blockers` record names the blocking player; an `attackers` record built for the non-active player during declare-attackers is a what-if; a `decision` record evaluated on the opponent's turn names the candidate's controller
- [X] T050 [US3] Extract `FC/effects/RandomChoices.java` from `ForkCollector.chooseTargets` (`:463-485`), `chooseModes` (`:418-442`) and `announceX` (`:495`), widened: target count uniform in [min, max] then distinct uniform targets; mode count uniform in [`MinCharmNum`, `CharmNum`] with repeats only where the charm permits; Pawprint drawn one mode at a time among those whose pawprint cost fits; X uniform in [min legal, `ComputerUtilMana.determineLeftoverMana`]; always `setChosenList`; `MAX_DRAWS = 20`. `ForkCollector` calls it. Test in a new `FCT/effects/RandomChoicesTest.java` with a seeded `Random`
- [X] T051 [US3] Per-mode modal resolution in `FC/effects/PatchedCollectors.java` `clauseHandler` (`:1705-1735`) and `FC/effects/BusBracketCollector.java` (`beginBracket:241-254`, `endBracket:287-327`): on `onClauseResolving` of each mode's first clause, close the previous mode's effect half and open the next with a fresh snapshot; map the k-th mode to the k-th entry of the sorted chosen list and that to its `Choices$` index by identity against `getAdditionalAbilityList("Choices")`; every half shares the cost half's `link_id`; repeated and Pawprint modes give one half per resolution; a fizzled mode follows feature 023's outcome rules on its own half (FR-029a–d)
- [X] T052 [US3] Resolve a cloned mode to its key in `FC/effects/ProvenanceKey.java` `resolve` (`:213-233`): after climbing to the root, a clone belonging to a resolving charm yields root key + `option` from the mapping in T051
- [X] T053 [US3] Per-mode halves for forks in `FC/effects/ForkCollector.java` `intervene` (`:345-361`) by the same rule (FR-029f); degraded mode keeps one root effect half (FR-029e) — assert it with a test that disables the hooks
- [X] T054 [US3] Modal tests in a new `FCT/effects/ModalResolutionTest.java`: spec Story 3 scenarios 12–14; a mode that fizzles; each half's snapshot shows the earlier modes' effects; `link_id` shared by one cost half and N effect halves

### Random seat (Java)

- [X] T055 [US3] Create `FC/RandomSeatLobbyPlayer.java` extending `LobbyPlayerAi` and overriding `createIngamePlayer` (forge/`LobbyPlayerAi.java:49`) to install a `RandomSeatController` via `Player.setFirstController` (prior art: research.md § The random seat is a lobby-player subclass)
- [X] T056 [US3] Create `FC/RandomSeatController.java` extending `PlayerControllerAi`, holding `P`, a seeded `Random`, the per-game collector, and a live-game check (`belongsToLiveGame`, `PatchedCollectors.java:1608`) that makes every override defer to `super` inside a fork (`GameCopier.clonePlayer` keeps the subclass) (prior art: `ForkCollector`, which acts only inside forks through the same test, used here inverted; research.md § The random seat is a lobby-player subclass)
- [X] T057 [US3] Play decision in `RandomSeatController.chooseSpellAbilityToPlay` (forge/`PlayerControllerAi.java:877`): land-first check (`player.canPlayLand` and a playable land in `ComputerUtilAbility.getAvailableCards`) → `super` with no draw; else the `P` draw; on success, candidates = the AI's filter (`canPlay`, `canCastTiming`, `ComputerUtilCost.canPayCost`, not a mana ability, not a land ability), `recordCandidate` per evaluated candidate, uniform pick, mark the pick as randomly drawn; nothing playable → pass (FR-022a, FR-022d)
- [X] T058 [US3] Create `FC/RandomSpellPlayer.java`: a copy of forge/`ComputerUtil.handlePlayingSpellAbility` (`:80-124`) that takes a `CostDecisionMakerBase` and a targets consumer, plus `RandomCostDecision extends AiCostDecision` overriding the `CostDiscard`, `CostExile`, `CostTapType` and `CostSacrifice` visits (`AiCostDecision.java:91,188,461,492`) with uniform distinct legal objects and leaving `CostPartMana` to the AI. Document the copy's reason in its class comment (research.md § Adjacent prior art)
- [X] T059 [US3] `RandomSeatController.playChosenSpellAbility` (`PlayerControllerAi.java:882`): a randomly drawn play uses `RandomChoices` for modes, targets and X and `RandomSpellPlayer` for costs, with no `P` gate; rejection redraws up to `MAX_DRAWS`, then abandons and redraws over the remaining candidates, passing when none remains (FR-022c); an AI-chosen play goes to `super` with its targets under their own `P` draw (FR-022)
- [X] T060 [US3] Attacks and blocks in `RandomSeatController`: `declareAttackers` (`:867`) per defender, each legal attacker (`CombatUtil.canAttack`) with probability ½, validated by `CombatUtil.validateAttackers`, redraw then AI fallback; `declareBlockers` (`:872`) per blocker uniform over no block plus legal attackers (`CombatUtil.canBlock`), validated by `CombatUtil.validateBlocks` (the engine never calls it), redraw then AI fallback; on success `recordAttackers`/`recordBlockers` with `whatIf = false` (FR-022b, FR-022d); `chooseTargetsFor` (`:1414`) draws targets by FR-022b
- [X] T061 [US3] Seat installation in `FC/GamePlayer.java:105-177`: per match, with probability `effect.random.seat.share` from the match-seeded `Random`, build one seat (A or B, ½ each) as `RandomSeatLobbyPlayer`; after `match.createGame()` give the controller the game's collector and set the collectors' `randomSeatPlayerId`; `PlayedMatch` carries `randomSeat`
- [X] T062 [US3] Per-match records-only in `FC/MatchWorkerMain.java:102,255-258,389-408`: `MatchGenerationResult` carries `randomSeat` (its row-count validation accepts zero rows for it); `recordMatch` writes the progress line and skips `MatchResultWriter` and `CardsPlayedWriter` for a random-seat match (FR-025); read `-Deffect.random.seat.share` and `-Deffect.random.seat.probability`
- [X] T063 [US3] Random-seat tests in a new `FCT/RandomSeatControllerTest.java` on scripted Forge games (`ForgeExtension`): spec Story 3 scenarios 1–5, 15–18; a seat with only mana abilities playable passes; a single legal option is taken; a fork of the game does not act at random; and `FCT/MatchWorkerMainTest.java` asserts a random-seat match adds no outcome or cards-played row

### Python plumbing

- [X] T064 [P] [US3] Add `random_seat_share` and `random_seat_probability` to `src/effects/domain/collection_caps.py` and `as_system_properties` (`:125`); test in `tests/unit/effects/domain/test_collection_caps.py`
- [X] T065 [US3] `--random-seat-share` (default 0) and `--random-seat-probability` in `src/sealed/infrastructure/cli.py` (`_add_effect_record_flags` `:1053+`, `_effect_collection_caps` `:1023-1050`); FR-026 refusals in `run_match_outcomes` (`:1836-1890`) beside the side-b check; tests in `tests/unit/effects/test_cli_contract_match_outcomes.py` and `tests/unit/sealed/infrastructure/test_match_worker_connector.py` (spec Story 3 scenario 11)
- [X] T066 [US3] Widen halves pairing in `src/effects/application/validate_corpus.py:582-584,770-790`: a `link_id` joins one cost half and one or more effect halves, several only when each acts through an `option` key of the same root; add the per-shard field-presence check of FR-033; invert `tests/unit/effects/application/test_validate_corpus.py:389` and add cases for both new checks
- [X] T067 [US3] Pilot integration test in `tests/integration/effects/test_random_seat_pilot.py` (`integration` marker): a short `match-outcomes --effect-records --random-seat-share 1 --random-seat-probability 0.5` run on one worker; assert random-seat matches wrote no row to either outcome file, both envelope fields are present and vary, every `what_if = false` record is present regardless of `--legality-rate`, a resolved charm has one effect half per chosen mode, and no modal effect half acts through a key without `option` (SC-001a)
- [X] T068 [US3] Document the delta (FR-034): apply [contracts/record-schema-delta.md](contracts/record-schema-delta.md) to `specs/023-ability-effect-model/contracts/record-schema.md` and `specs/023-ability-effect-model/contracts/provenance-sidecar.md`, and update `CLAUDE.md`'s effect-record shard and provenance sidecar paragraphs. Load the feature-workflow skill before editing the two contracts

**Checkpoint**: pilot collection runs; `validate-corpus` passes; stories 1–3 together gate the first gen-2 game.

---

## Phase 6: User Story 4 — Curated corpus balanced across families and outcomes (Priority: P4)

**Goal**: family and signature balancing with copy counts, hash-stable stratum, round-robin card-disjoint sample, manifest shortfalls.

**Independent Test**: spec Story 4 — `build-corpus` over pilot shards into scratch; read the manifest; rebuild over grown shards and check stratum stability.

- [X] T069 [US4] Shard-generation check (FR-033): `shard_generation(path)` in `src/effects/infrastructure/record_io.py` reads a shard's first complete record and returns `"gen-1"` or `"gen-2"` by the presence of `random_seat`; `src/effects/application/build_corpus.py`, `train_effect_model.py`, `evaluate_effect_model.py` and `scripts/effect_knowledge_probes/common.py` refuse a records set holding both, before reading records. Test in `tests/unit/effects/infrastructure/test_record_io.py` with one gen-1 fixture shard (T002) and one gen-2 shard
- [X] T070 [P] [US4] Create `src/effects/domain/rule_families.py`: `rule_family(record, sidecars) -> str` per research.md § Rule families — `resolution` → acting line's `script_api_type`; `trigger` → `Mode$` parsed from the root segment (`(no-mode)` when absent); `continuous` → acting static line's `script_api_type`; `rewrite` → `Event$` from the root segment; `combat` → sorted damage-step keyword set via `KeywordResolver` or `none`; `playability` per FR-047a; any keyword acting line → that keyword's display name (FR-048). A `decision` record's family is per candidate (prior art: none; `sampling_class` in `train_effect_model.py:108-131` is the sibling classification)
- [X] T071 [P] [US4] Test `rule_families.py` in `tests/unit/effects/domain/test_rule_families.py` on the T002 fixtures: every kind; spec Story 4 scenarios 7 and 11; a trigger with no `Mode$`; a multi-mode static; a `blockers` record whose `forbidden` names two statics with different modes
- [X] T072 [P] [US4] Create `src/effects/domain/budget_allocation.py`: `allocate(budget: int, capacities: Mapping[K, int]) -> dict[K, int]` — equal shares, capacity-capped, leftover redistributed equally over members with remaining capacity until all are full or the budget is spent; deterministic tie-breaking by sorted key (research.md § Third-instance check)
- [X] T073 [P] [US4] Test `budget_allocation.py` in `tests/unit/effects/domain/test_budget_allocation.py`: spec Story 4 scenario 1; empty members; budget above total capacity (shortfall); integer remainders sum exactly to the budget
- [X] T074 [US4] Outcome signature in `src/effects/domain/corpus_curation.py`: `outcome_signature(record) -> frozenset[tuple[str, bool]]` from `derive_targets` (`src/effects/domain/effect_targets.py:95-129`), `zone_outcome` or `"stayed"` paired with `affected`; defined for the `resolution-effect`, `rewrite`, `combat`, `continuous` classes; test that a wipe of three and of seven creatures share a signature
- [X] T075 [US4] Copy counts in `src/effects/domain/corpus_curation.py`: `copies_for_text(record_hashes, quota, text_cap, reuse_cap) -> dict[hash, int]` — over the cap, the `cap` smallest hashes once each; otherwise the quota spread evenly, extra copies to the smallest hashes; capacity `min(reuse_cap × n, text_cap)`; replace `CapHeap` admission (`:25-96`); test spec Story 4 scenario 4 in `tests/unit/effects/domain/test_corpus_curation.py`
- [X] T076 [US4] Game-disjoint placement in `src/effects/domain/corpus_curation.py`: `in_game_disjoint_stratum(game_id, names_held_out, has_keyword_combat, share, keyword_share)` with `crc32(game_id) % 10**6 / 10**6`; tests for spec Story 4 scenarios 5 and 6
- [X] T077 [US4] Extend the survey pass in `src/effects/application/build_corpus.py` (`_worker_sidecars:238-245`, survey `:322-325`): per record emit `(class, half, family, signature, text, record_hash, random_seat)`, where `half` is `real` for `what_if = false` legality records and `what-if` otherwise; per game emit `has_keyword_combat` from `qualifying_observations` over `--game-disjoint-keywords`
- [X] T078 [US4] Replace selection in `decide` (`src/effects/application/build_corpus.py:755-789`): the class budgets from `class_targets` (`corpus_curation.py:99-139`, unchanged); `allocate` over families; within `resolution-effect`, `rewrite`, `combat`, `continuous`, `allocate` over signatures; within `playability-legality`, real decisions take up to half the budget first, what-ifs the rest, families within each half; `copies_for_text` per text per cell; `random_seat` is never a key (FR-050). Replace the seeded game sample (`:787-789`) with `in_game_disjoint_stratum`
- [X] T079 [US4] Write copies in `write_shard_pass` (`src/effects/application/build_corpus.py:961-1070`): each record written `copies` times, determined by `record_hash` and `--seed` (FR-052); keep the class-admit hash where it still applies
- [X] T080 [US4] Flags in `src/effects/infrastructure/cli.py:473-656`: add `--game-disjoint-share` (0.01), `--game-disjoint-keyword-share` (0.15), `--game-disjoint-keywords` (FR-045 list), `--reuse-cap` (4); remove `--game-disjoint-games` (`:542-548`) and `game_disjoint_target` (`build_corpus.py:494,533,632`) so argparse rejects it (spec Story 4 scenario 10)
- [X] T081 [US4] Card-disjoint validation sample in `src/effects/application/validation_samples.py:88-125`: resolution slots filled round-robin over held-out texts, 4 records per text per round by smallest record hash within the text, until the class quota is met (FR-051); test spec Story 4 scenario 9 in `tests/unit/effects/application/test_validation_samples.py`
- [X] T082 [US4] Manifest fields in `src/effects/domain/corpus_manifest.py` and the writer `src/effects/application/build_corpus.py:1524-1615` per data-model.md § CorpusManifest: per class and family available/share/written/repeats/shortfall; per family written per signature; per class on/off-policy counts; per legality subkind real/what-if/unknown; keyword-threshold games; held-out texts with no gate-one resolution record and those under five games; the flags as run. Each omitted when at its gen-1 value; the summary log prints shortfalls and real/what-if counts (FR-053)
- [X] T083 [US4] Build-corpus tests in `tests/unit/effects/application/test_build_corpus_decide.py` and `test_build_corpus.py`: spec Story 4 scenarios 2 and 3; a family with no records takes no share; a legality class with fewer real decisions than half its budget; two builds with the same seed are identical; a rebuild over a superset of shards keeps every game-disjoint game (SC-007); `random_seat` changes no selection
- [X] T084 [US4] Manifest digest regression test in `tests/unit/effects/domain/test_corpus_manifest.py`: a real gen-1 manifest re-serializes to the same digest

**Checkpoint**: a pilot build writes a manifest with family, signature, policy and real/what-if tables.

---

## Phase 7: User Story 5 — Training that reaches the tail and keeps amounts in `e` (Priority: P5)

**Goal**: rarity ceiling, epoch shares, covariance noise, value head, every specified head wired, encoder size flags, option rows.

**Independent Test**: spec Story 5 — three noise-pilot runs on gen-1's corpus; one arm at a non-default size loads in `encode-abilities` and `evaluate-effect-model` without restating flags.

- [X] T085 [US5] Rarity weights in `src/effects/application/train_effect_model.py:155-223` and their use in `training_loop.py:549-564,882-885`: weights ∝ effective_games^(−0.5), normalized within each sampling class, capped at 20× the weight of the text at the 99th percentile of effective games (FR-054); test spec Story 5 scenario 1 in `tests/unit/effects/application/test_sampling.py`
- [X] T086 [US5] Epoch-line shares in `src/effects/application/training_loop.py` (`_train_on_shard:895-932`, epoch line `:797-805`): count trained records by `rarity_bucket` of their text and by `rule_family`, from the batch plan on the host, and print both share tables (FR-055); test in `tests/unit/effects/application/test_training_loop_execute.py` (spec Story 5 scenario 2)
- [X] T087 [US5] Remove `AbilityEncoderConfig.e_noise` and its application (`src/effects/domain/ability_encoder.py:64,82,177-181`); drop an `e_noise` key from `payload["encoder_config"]` before `AbilityEncoderConfig(**…)` in `src/effects/infrastructure/effect_model_store.py:257` (FR-057, FR-061)
- [X] T088 [US5] Noise on `e`: a `NoiseState` (Σ on device, step) owned by `TrainingLoop` in `src/effects/application/training_loop.py` and passed to each per-batch `SurfaceBatcher` (`:442-474`); `SurfaceBatcher.build` (`src/effects/application/surface_batching.py:308-341`) adds `L z` to every scattered `e` row of every variant, `L` = Cholesky of `r²Σ + 1e-6·I`; Σ initialized from the first batch, updated with decay 0.99 under `no_grad`; `r` ramps linearly to `--e-noise` (default 0.1) over `--steps-per-epoch`; nothing in validation, evaluation or encoding (FR-056)
- [X] T089 [US5] Noise tests in `tests/unit/effects/application/test_surface_batching.py`: spec Story 5 scenarios 3 and 4; covariance of added noise ≈ r²Σ on a synthetic batch; no gradient through Σ; the ramp at step 0, mid-epoch and after
- [X] T090 [P] [US5] Create `src/effects/domain/value_targets.py` per data-model.md § ValueTargets: split on `[SEG]`, sum integer literals of the amount keys per segment via `numeric_params` (`src/effects/domain/script_variants.py:44-79`), mask a target any segment states with a non-literal; cost targets from the root segment's `Cost$` via `ManaCost.parse` (`src/price_predictor/domain/value_objects.py:98-170`) plus `T` and `Sac<` tests; mana targets masked on spell lines (FR-058, FR-058a) (prior art: `numeric_params`, reused)
- [X] T091 [P] [US5] Test `value_targets.py` in `tests/unit/effects/domain/test_value_targets.py`: spec Story 5 scenario 5; Bone Splinters (`Cost$ B Sac<1/Creature>`) has masked mana and a set sacrifice target; a charm root has all amounts masked; a gen-1 one-segment text works
- [X] T092 [US5] Value head in `src/effects/domain/effect_model.py`: reads `e` only, outputs the count and binary targets of T090; loss under `--value-weight` (default 0.05) in `training_loop._loss_for` (`:478-547`); targets cached per text in `SurfaceBatcher` like `e`; added to `TRAINING_ONLY_HEADS` (`:332-334`) so `filter_training_only` drops it (FR-058, FR-059)
- [X] T093 [US5] Wire the verdict and created-objects losses (FR-060a): `SurfaceBatcher` passes `candidate_index=0` for `decision` records (`surface_batching.py:287-297`); `effect_targets.py` emits verdict targets at `[ACT]` (the three bits on `decision`, cost paid on cost halves, trigger fired on `trigger`) beside `_apply_playability` (`:243-262`); `_loss_for` adds `verdict_loss` (`effect_model.py:731`) and `created_objects_loss` (`:703`) at weight 1.0, normalized per record
- [X] T094 [US5] Wire the MLM and script-API losses (FR-060b): `mlm_loss` (`effect_model.py:751`) with masking at `--mlm-mask-prob` and weight `--mlm-weight`; `api_loss` (`:765`) at `--api-weight`, its targets the line's `script_api_type` and the parameter keys of every `[SEG]` segment; set `n_api_types`/`n_param_keys` from the sidecars' vocabularies in `training_loop.py:621-628` and record them in the checkpoint; size `mlm_head` (`:38,317`) from the configured `d_model`
- [X] T095 [US5] Loss-wiring tests in `tests/unit/effects/domain/test_effect_model.py` and `tests/unit/effects/application/test_training_loop_execute.py`: spec Story 5 scenario 9; a `decision` record's `[ACT]` is non-zero and it yields verdict targets; with `--mlm-weight 0` and `--api-weight 0` those terms are absent
- [X] T096 [US5] Remove `pairing_proj` and `pairing_loss` (`src/effects/domain/effect_model.py:329,923`) and the test at `tests/unit/effects/domain/test_effect_model.py:477` (FR-060)
- [X] T097 [US5] Encoder size: `--encoder-layers` (4) and `--encoder-d-model` (256) in `src/effects/infrastructure/cli.py:1175-1272` into `AbilityEncoderConfig` (`training_loop.py:616-620`); `ff_dim` follows `d_model × 4` instead of the constant (`ability_encoder.py:68`); refuse a width not divisible by `N_HEADS` before training (FR-061, FR-062)
- [X] T098 [US5] Build the encoder from the checkpoint everywhere: `src/effects/infrastructure/model_runner.py:106-108`, `src/effects/infrastructure/ability_encoder_runner.py:61,72-86`, `scripts/effect_embedding_probes/keyword_expansion.py:197`; a checkpoint recording no size loads at 4/256 (FR-061)
- [X] T099 [US5] `--cards-folder` (repeatable; default `output/cardsfolder/`, `output/tokenscripts/`) on `train-effect-model`, `encode-abilities` and `evaluate-effect-model` in `src/effects/infrastructure/cli.py`, replacing the roots hardcoded at `src/effects/application/train_effect_model.py:736` and feeding `SidecarCache` in `training_loop.py:378`, `model_runner.py` and `ability_encoder_runner.py`; the checkpoint records the roots (FR-063b). Test in `tests/unit/effects/application/test_train_startup.py` that a run pointed at a second tree resolves keys there
- [X] T100 [US5] Checkpoint fields (FR-063) in `src/effects/infrastructure/effect_model_store.py:220-243`: holdout unit, `--encoder-layers`, `--encoder-d-model`, `--e-noise`, `--value-weight`, `--mlm-weight`, `--mlm-mask-prob`, `--api-weight`, the script-API vocabularies, and the sidecar roots (FR-063b)
- [X] T101 [US5] Store tests in `tests/unit/effects/infrastructure/test_effect_model_store.py`: spec Story 5 scenarios 6–8 (scenario 8 against a real gen-1 checkpoint payload with `e_noise`); every training-only head is absent after save
- [X] T102 [US5] Option rows (FR-063a): `Slot.option` in `src/effects/domain/effect_head_input.py:193-212`; after each `ABILITY` slot whose sidecar row has `option_rows_after` rows, one `ABILITY` slot per row with `option = True`, positions continuing (`:539-561`); `collate_surfaces` (`src/effects/domain/effect_model.py:790-853`) emits `option_kinds`; `EffectModel.forward` (`:350-358`) adds a two-row `option_kind_embedding` initialized to zeros
- [X] T103 [US5] Option-row tests in `tests/unit/effects/domain/test_effect_head_input.py`: a charm entity gets root then mode rows in `Choices$` order; a gen-1 checkpoint state dict loaded with `strict=False` gives identical outputs with and without option rows' flag (zero embedding)

**Checkpoint**: the stage-0 noise pilot runs on the gen-1 corpus; a non-default-size arm reloads without flags.

---

## Phase 8: User Story 6 — Evaluation per text, per family and per policy (Priority: P6)

**Goal**: breakdown reports, slices, zero-shot, `--win-rates`, scorer smoke test, checkpoint-aware embedding probes.

**Independent Test**: spec Story 6 — evaluate a gen-2 arm with no identity variant; run each embedding probe with `--checkpoint`/`--abilities-root`; run `scorer-smoke-test` and check `output/cardsfolder/` unchanged.

- [X] T104 [US6] Make `gate_one.measure` (`src/effects/application/gate_one.py:79-163`) return per-record results (record id, text, family inputs, per-field loss and prediction) beside the pooled `GateOneMetrics`
- [X] T105 [P] [US6] Create `src/effects/application/breakdowns.py`: pure grouping of per-record results by text (per-text mean), `rarity_bucket`, `rule_family` (and the mean over families), `random_seat`, `what_if`; memorization gap per family = game-disjoint − card-disjoint on the same fields; withheld-keyword slice (keyword on the acting line or carried by an entity) beside trained keywords (FR-064–068) (prior art: none; `EvaluationReport` consumes its output)
- [X] T106 [P] [US6] Test `breakdowns.py` in `tests/unit/effects/application/test_breakdowns.py` on per-record fixtures: spec Story 6 scenarios 1–5
- [X] T107 [US6] Wire breakdowns into `src/effects/application/evaluate_effect_model.py` as `REPORTED` `CheckResult`s after the gate-2 block (`:837-842`); replace the zero-shot placeholder (`:863-873`); keep "gate 1 skipped" without an identity variant (`:825-830`, FR-069)
- [X] T108 [US6] `--win-rates PATH` (default `output/sealed/cards-win-rates.txt`) in `src/effects/infrastructure/cli.py:1439-1499`, threaded into `check_decodability` (`evaluate_effect_model.py:848-851`) in place of `DEFAULT_WIN_RATES` (`:894`) (FR-070); test spec Story 6 scenario 7
- [X] T109 [US6] Create `src/effects/application/scorer_smoke_test.py` and the `scorer-smoke-test` subcommand: load pooled `e` per card from `--checkpoint`'s cache and the sealed vectors of `--sealed-encoder-checkpoint` via `sealed.infrastructure.embedding_store`; write them with `geometry_checks.write_scorer_smoke_cache` (`src/effects/application/geometry_checks.py:411-440`, key checked against `src/sealed/infrastructure/embedding_store.py:18`) under `--scratch-dir`; run `python -m sealed train-scorer` Phase A as a subprocess with `--cards-path` and `--checkpoint-dir` under `--scratch-dir` and `--embedding-lr 0` (FR-071) (prior art: `write_scorer_smoke_cache`, reused)
- [X] T110 [US6] Test `scorer_smoke_test.py` in `tests/unit/effects/application/test_scorer_smoke_test.py` with the subprocess call mocked: argument vector, nothing written outside `--scratch-dir`, no import of `sealed.application` (SC-012, spec Story 6 scenario 8)
- [X] T111 [US6] Embedding probes take `--checkpoint` and `--abilities-root` (FR-072): move the path constants of `scripts/effect_embedding_probes/common.py:21-24` into a `resolve_paths(args)` that reads the checkpoint's vocabulary and the cache's width of `e`; add the two flags to every script; output under `output/effects/reports/embedding-probes-<checkpoint stem>-<date>/`; `keyword_expansion.py` drops its hardcoded `CHECKPOINT` (`:80-83`)
- [X] T112 [US6] `build_texts.py` (`:74-82`) keeps a sidecar whose `taxonomy` cache is absent and leaves `e_tax` empty; `linear_probes.py:99`, `type_merging.py:114`, `neighbours.py:103`, `pca_directions.py:75` omit their taxonomy columns when it is empty (FR-072)
- [X] T113 [US6] `pca_directions.py` reports the participation ratio (Σλ)²/Σλ² in its variance table (`:~130`) from `common.pca` (`:259`) (FR-073)
- [X] T114 [US6] Script tests in `tests/unit/scripts/test_effect_embedding_probes.py`, loading modules by path as `tests/unit/scripts/test_regenerate_forge_api_list.py` does: `resolve_paths` reads width from a cache; the participation ratio on a known spectrum; `build_texts` with no taxonomy cache yields rows (spec Story 6 scenarios 9 and 10)

**Checkpoint**: an arm's evaluation report carries every new section.

---

## Phase 9: User Story 7 — Knowledge-probe suite (Priority: P7)

**Goal**: a frozen probe set per curated corpus; the read-out ladder over ten families; sweeps; ablation; scorecards and comparison.

**Independent Test**: spec Story 7 — freeze and probe gen-1; read the scorecard; compare two scorecards.

- [X] T115 [P] [US7] Create `scripts/effect_knowledge_probes/common.py`: load any checkpoint (gen-1 included) with its vocabulary, cache and width of `e` through `effects.infrastructure`; load sidecars from `--cards-folder` trees; stream records of named games from `--records-dir`; build `EffectHeadInput` batches through `SurfaceBatcher`; no import-time work (contracts/knowledge-probes.md § Layout; prior art: `scripts/effect_embedding_probes/common.py`)
- [X] T116 [US7] Create `scripts/effect_knowledge_probes/labels.py`: one extractor per family of FR-079 following the probes record's family table and its section on assembled labels (`experiments/2026-09-19-effect-knowledge-probes-design.md`): magnitudes from parsed params and observed outcomes; mana production from mana params and `mana_produced` events; mana usage from `Cost$` (no spell lines) and decision records; timing and non-mana costs from the script; interactions by joining fired trigger records' pending events to resolution records of the same game, plus script-mined pairs, negatives from evaluated unfired triggers; side from affected entities and restrictions; evasion and blocking from `blockers` records; target legality from decision records' legal-target sets; duration and repeatability from event durations and line kind; state dependence as the spread of a text's effect profile, weighted n/(n+5)
- [X] T117 [US7] Create `scripts/effect_knowledge_probes/ladder.py`: feature extraction for rungs 0, 1, 1w (fixed random vector per text, seed 42), 1o, 2 (trunk output at `[ACT]` or the target's `[CARD]`) in one batched forward per stratum; rung 3 from the model's head; linear (logistic C=1.0 / ridge α=1.0, standardized) and MLP (2×256, ReLU, AdamW 1e-3, wd 1e-4, batch 512, 30 epochs, GPU) probes; 5 folds grouped by text (by card for pooled items); AUC or R²; share per probe type with 1,000-resample bootstrap CI over texts; withheld when rung 3 − rung 0 < 0.05 (FR-080–083) (prior art: `pca_directions.cv_r2`, `linear_probes.probe_binary`)
- [X] T118 [US7] Create `scripts/effect_knowledge_probes/sweeps.py`: the four sweeps of FR-084 as edits of real records' snapshots (toughness 1–8 of one target; untapped production 0 to cost+2; opposing creature count 0–8 by copying an entity; `NumDmg$` 1–8 re-encoded by the checkpoint's encoder on a spell and on a triggered ability), each reading the full model's prediction
- [X] T119 [US7] Create `scripts/effect_knowledge_probes/ablation.py`: per output field, the loss increase under matched-noise, API-type-mean and nearest-other-text replacements of `e`, each on every slot, `[ACT]` only, and card slots only (FR-085)
- [X] T120 [US7] Create `scripts/effect_knowledge_probes/run.py`: flags per contracts/cli.md; `--freeze-probe-set` enumerates items, labels, both strata's probe games, sweep records and the interaction join rate, and writes them with their sha256 digest (FR-076, FR-078); a probing run refuses without a matching frozen set, reports board-dependent results per stratum and line-level results for held-out vs trained lines (FR-077), writes tables and `scorecard.json` to `output/effects/reports/knowledge-probes-<checkpoint stem>-<date>/` (FR-088), the scorecard recording wall time and `torch.cuda.max_memory_allocated()` for the run; `--per-layer` via forward hooks; `--method-c` trains a 0- or 1-layer trunk on frozen `e` (FR-086)
- [X] T121 [US7] Create `scripts/effect_knowledge_probes/compare.py`: side-by-side per family and rung; ranking on rung 1 − rung 1w per probe type; warning on mismatched digests (FR-087)
- [X] T122 [US7] Tests in `tests/unit/scripts/test_effect_knowledge_probes.py` (modules loaded by path, fixtures from T002): the share and its minimum-gap rule; each label extractor on real records; the sweep editor changes exactly one input; fold assignment never puts one text on both sides and groups pooled items by card; the probe-set digest is stable across runs; `compare` ranking (FR-089). The MLP probe runs on CPU in tests with a test-supplied hidden size

**Checkpoint**: gen-1 scorecard produced within two GPU hours on 8 GB (SC-010, SC-011).

---

## Phase 10: Polish & Cross-Cutting Concerns

- [X] T123 [P] Update `src/effects/CLAUDE.md`: the encoding text, script tokenization and seeding, the random seat, real/what-if, `actor_player` redefinition, per-mode modal records, masked-template holdout, family/signature selection and copy counts, the stratum rule, the new training terms and flags, the evaluation sections, `scorer-smoke-test`, and both probe suites
- [X] T124 [P] Update `README.md`'s effects workflow with the gen-2 commands and flags (contracts/cli.md) and the stage order of quickstart.md
- [X] T125 [P] Confirm the import-direction test (`tests/unit/effects/test_import_boundaries.py`) passes unchanged, `scorer_smoke_test.py` included (FR-090)
- [X] T126 [P] Regression: gate 2, gate 3 and the baseline variants are unchanged — `tests/unit/effects/application/test_gate_two.py`, `test_gates.py`, `test_geometry_checks.py` pass, and `evaluate-effect-model` on the gen-1 checkpoint gives the same gate-2 and gate-3 verdicts as before this feature (FR-091)
- [X] T127 Performance review against plan.md § Performance Review: confirm value targets and families are cached per text, noise adds no host transfer, what-if snapshots are built after the rate draw, and probe features are extracted once per stratum; profile one training epoch before and after T088–T094 and record the step time
- [X] T128 Full verification: `pytest -m "not integration"`, `pytest -m integration` (needs the JAR), `cd forge-connector && mvn install -DskipTests && mvn test`, `ruff check`, and IDE type diagnostics on every touched file; fix every failure and warning found, whoever's code it is
- [ ] T129 Walk quickstart.md stages 1 and 2 end to end on the real Forge checkout (conversion, vocabulary, holdout, pilot collection, validation, scratch build) and fix whatever breaks
- [X] T130 Run stage 0 on gen-1 (quickstart.md): the knowledge probes and the three noise-pilot runs; check the scorecard's recorded wall time and peak GPU memory against SC-011 (≤ 2 GPU hours, within 8 GB); hand the epoch lines and scorecard to the user for the noise-ratio decision
- [ ] T131 After the sweep runs, fill the Outcome sections of `experiments/2026-09-18-effect-model-gen2-improvements-design.md` and `experiments/2026-09-19-effect-knowledge-probes-design.md` with the user — load the feature-workflow skill before each edit
- [X] T132 If implementation changes any contract in `specs/024-ability-effect-model-gen2/` or the root spec, update it in the same change — load the feature-workflow skill before each edit

---

## Dependencies & Execution Order

### Phase dependencies

- **Setup (Phase 1)** → **Foundational (Phase 2)** → user stories.
- **US1** needs T004–T007 (keys). **US2** needs US1's `script_text` for real data, but its code and
  tests run on synthetic strings, so it can be built in parallel with US1.
- **US3** needs T004–T011 (keys and envelope). It is independent of US1/US2 in code; its modal tests
  need T018's option-line keys to resolve mode keys against a sidecar.
- **US4** needs T008–T013 and US3's envelope semantics (real/what-if, `random_seat`); its pilot data
  needs US1–US3.
- **US5** needs T004–T005 and T012; T102 needs T005's `option_rows_after`. Its stage-0 noise pilot
  needs only gen-1 artifacts plus T009's defaults.
- **US6** needs US4's family function (T070) and T012; T111–T114 are independent of everything but
  T098.
- **US7** needs T009 (gen-1 reads), T098 (checkpoint loading) and T070 (families); it can start on
  gen-1 artifacts as soon as those land.
- **Polish** after the stories it documents.

### Story order for the run plan

US1, US2 and US3 must all be complete before the first gen-2 game (stage 2). US5 and US7 are needed
for stage 0, which can run while US1–US3 are in progress. US4 is needed for stage 2's scratch build.
US6 is needed for stage 5.

### Within each story

Tests are written with or before the code they cover. Java tasks build with `mvn install
-DskipTests` before Python integration tests run. T036 (US1) implements and tests spec Story 4
scenario 8, because FR-015 sits in the keyword-expansion section.

## Parallel Examples

```text
# Foundational: independent files
T004 provenance.py   |  T008 records.py   |  T012 rarity.py   |  T013 damage_step_keywords.py

# US1: Java converter and Python tokenizer proceed side by side
T014–T021 (TraitScript, RulesParser, goldens)   ||   T024–T031 (tokenizer, expansion)

# US3: collectors and random seat are separate classes
T046–T049 (PatchedCollectors)   ||   T050 (RandomChoices)   ||   T064–T065 (Python flags)

# US4: pure domain modules first
T070 rule_families.py   ||   T072 budget_allocation.py   ||   T074–T076 corpus_curation.py

# US7: label extractors and the ladder are separate modules
T116 labels.py   ||   T117 ladder.py   ||   T118 sweeps.py   ||   T119 ablation.py
```

## Implementation Strategy

### MVP

US1 alone: the chained, label-renamed encoding text with its tokenization and vocabulary. It is the
first thing the run plan does, everything downstream keys on it, and it is verifiable on the Forge
checkout without collecting a game.

### Incremental delivery

1. Setup + Foundational.
2. US1 → verify conversion and vocabulary (stage 1, first half).
3. US7 + US5's noise pilot on gen-1 (stage 0), in parallel with:
4. US2 → holdout (stage 1, second half); US3 → random seat and records.
5. Pilot collection (stage 2) once US1–US3 are in; US4 → scratch build of the pilot.
6. Full collection (stage 3), final build (stage 4), then US5 sweep arms and US6 evaluation (stage 5).

### Notes

- Every new module cites its prior art; every [P] task touches a file no concurrent task touches.
- Principle VIII items are checked in T127, not assumed.
- Commit per task or per small group, referencing the task ID.
