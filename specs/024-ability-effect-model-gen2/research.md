# Research: Ability effect model — generation 2

**Feature**: `024-ability-effect-model-gen2` | **Date**: 2026-10-06
**Spec**: [spec.md](spec.md) | **Root spec**: [`../2026-10-06-ability-effect-model-gen2.md`](../2026-10-06-ability-effect-model-gen2.md)
| **Base feature**: [`../023-ability-effect-model/`](../023-ability-effect-model/plan.md)

The root spec and the clarification sessions pin the behaviour. What remains for planning is where
each change attaches to the code feature 023 left, and the handful of constants the spec leaves to
the plan. The survey below is what grounds both. It surfaced several places where the code did not
match an assumption the spec made; each was resolved in the spec before this plan was written
(spec Clarifications, session 2026-10-06, last six bullets, and the edits to FR-005a, FR-022d,
FR-030a, FR-033, FR-060a, FR-060b, FR-061, FR-063a, FR-092).

Paths are relative to the repository root unless prefixed `forge/` (the sibling checkout on its
`effect-record-hooks` branch). `FC` abbreviates `forge-connector/src/main/java/com/pricepredictor/connector`
and `E` abbreviates `src/effects`.

## Codebase Survey

Constitution Principle VII. Every bullet cites a concrete file or symbol.

### Overlapping domain vocabulary

| Existing concept | Where | Decision |
|---|---|---|
| `ProvenanceKey` (Java) | `FC/effects/ProvenanceKey.java:51-55` (kinds), `resolve` `:213`, root climb `:230-233` | **Extend.** Gains an optional `option` component (FR-005a). `resolve` keeps climbing a sub-ability to its root, then, for a cloned charm mode, maps the clone to its `Choices$` position (see Decisions). No new trait kind: a mode is still a `spell`/`trigger`/… trait of the charm, distinguished by the component. |
| `ProvenanceKey` (Python) | `E/domain/provenance.py:54-100` | **Extend** with `option: int \| None = None`. `_row_by_key` (`:198-208`) uses `setdefault`, so two lines sharing a key would collapse; with distinct mode keys they no longer share one. |
| `SidecarLine`, `ProvenanceSidecar` | `E/domain/provenance.py:120-268`; reader `E/infrastructure/sidecar_io.py:101-351` | **Extend.** `option` lines now carry keys, chains and `script_api_type`. Adds `option_rows_after(row)` for FR-063a's expansion. |
| `TraitScript` | `FC/effects/TraitScript.java:57-141` | **Extend.** Renders the chain (FR-001), renames labels (FR-002a), keeps the `TreeMap` parameter order per segment (`:62`), which is the order FR-002a's "first appearance" reads. |
| `EffectRecord` (Java) / `EffectRecord` (Python) | `FC/effects/EffectRecord.java:59-227`; `E/domain/records.py:419-449`, `COLLECTION_METADATA_FIELDS` `:141-143` | **Extend** with `random_seat` and `what_if`, both collection metadata. |
| `CollectionCaps` | `E/domain/collection_caps.py:125` (`as_system_properties`); Java `CollectionCaps.legalityRate` (`PatchedCollectors.java:111-143`) | **Extend** with the random-seat share and probability, so both supervisors' `-D` spellings stay pinned by `tests/unit/effects/domain/test_collection_caps.py`. |
| `HeldOutCards`, `text_keyed_holdout`, `select_holdout` | `E/application/train_effect_model.py:229-310`; `E/domain/text_holdout.py:29-76` | **Extend.** `select_holdout` gains a `unit` parameter and a `masked_template` key function; `text_keyed_holdout` stays the single entry point for all three commands. `train-effect-model`, which today only reads the holdout from the manifest (`train_effect_model.py:1089-1097`), recomputes it with the same function under the manifest's unit and refuses a disagreement (FR-041). |
| `CorpusManifest`, `SplitProvenance` | `E/domain/corpus_manifest.py:46-232`; `E/infrastructure/effect_model_store.py:87-138` | **Extend.** New fields serialize only when they differ from the gen-1 value, so gen-1 manifests keep their digest (`corpus_manifest.py:219-232`) and gen-1 checkpoints keep passing `check_corpus` (`evaluate_effect_model.py:684-699`). |
| `sampling_class`, `DEFAULT_KIND_MIX`, `class_targets` | `E/application/train_effect_model.py:57-131`; `E/domain/corpus_curation.py:99-139` | **Reuse** the eight classes and the class mixture unchanged (FR-092). Selection *within* a class is replaced (FR-049). |
| `CapHeap` and threshold admission | `E/domain/corpus_curation.py:25-96`; `E/application/build_corpus.py:322-325, 755-774, 1033-1041` | **Replace.** A threshold admits each record at most once; FR-049 needs per-record copy counts. The survey pass already gathers per-key heaps; it now also gathers family and outcome signature per record, and `decide` returns a copy count per record hash. |
| `DamageStepKeyword.qualifies`, `KeywordResolver`, `qualifying_observations` | `E/domain/damage_step_keywords.py:415-612`; `E/application/gate_two.py:71-147` | **Move, then reuse.** `KeywordResolver` and `qualifying_observations` move into the torch-free domain module, because `gate_two.py` imports torch at module top (`:31`) and build-corpus survey workers must not load it for FR-045's predicate. |
| `derive_targets`, `EntityTargets` | `E/domain/effect_targets.py:67-129` | **Reuse** as the source of the outcome signature (FR-049 step 2): `fields["zone_outcome"]` (absent = stayed) and `affected` (= changed). |
| `RarityWeights` (`effective_games`, `rarity_weights`) | `E/application/train_effect_model.py:155-223` | **Extend.** The ceiling becomes 20× the p99 text's weight and weights normalize within a class (FR-054); today the ceiling is 20× the most-observed text and weights mix classes across a shard. |
| Verdict head, created-objects head | `E/domain/effect_model.py:241-243, 311, 375, 703-750` | **Wire** (FR-060a). Both heads and both losses exist; nothing calls the losses. |
| MLM head, script-API heads | `E/domain/effect_model.py:256-257, 316-334, 751-765` | **Wire** (FR-060b). `mlm_head` is built from the `ENCODER_D_MODEL` constant (`:38, :317`) and must follow `--encoder-d-model`. |
| `pairing_proj`, `pairing_loss` | `E/domain/effect_model.py:329, 923`; test `tests/unit/effects/domain/test_effect_model.py:477` | **Remove** (FR-060). |
| `AbilityEncoderConfig.e_noise` | `E/domain/ability_encoder.py:64, 82, 177-181` | **Remove** (FR-057), with a load shim that drops the key (`effect_model_store.py:257`). |
| `RandomSeatController` | — | **New, no prior concept.** The nearest relatives are `ForkCollector`'s random choosers (see adjacent prior art) and its live-game test (`belongsToLiveGame`), used here inverted. |
| `keyword_formatting` | — | **New.** Mirrors Forge's per-keyword-class reminder formatting (`KeywordWithCost.java:30-35`, `KeywordWithCostAndAmount.java:55-60`, `KeywordWithAmount`); no Python equivalent exists, and `_instantiate` (`ability_tokenizer.py:339-359`) fills `%s` with raw text today. |
| `RuleFamily`, `OutcomeSignature`, `MaskedTemplate`, `ProbeSet`, `Scorecard` | — | **New.** No existing concept shares the name or the responsibility. `price_predictor/domain/price_buckets.py` is the nearest pattern for rarity buckets but buckets prices, not game counts; the names do not collide. |

Nine concepts extended, two reused, one moved, one replaced, two wired, two removed, and six new
(the random seat controller, `keyword_formatting`, and four curation/training concepts); zero
renames required.

### Adjacent prior art

| Sub-problem | Prior art | Decision |
|---|---|---|
| Random targets, modes and X for a play Forge did not choose | `ForkCollector.chooseTargets` `:463-485`, `chooseModes` `:418-442`, `announceX` `:495` | **Extract and widen** into `FC/effects/RandomChoices.java`, used by `ForkCollector` and the random seat. `chooseModes` ignores `MinCharmNum`, repeats and Pawprint today; FR-022c and FR-029f need all three, so one widened implementation fixes both callers. |
| Installing a custom controller on one seat | `LobbyPlayerAi.createIngamePlayer` (forge/`forge-ai/.../LobbyPlayerAi.java:49`, public, non-final); `Player.setFirstController` (forge/`Player.java:2590`) | **Subclass**, no engine change (FR-024): `RandomSeatLobbyPlayer` overrides `createIngamePlayer`. `GamePlayer.registeredPlayers` (`FC/GamePlayer.java:105-110`) builds it for the chosen seat. |
| Legal option lists and validation | `CombatUtil.canAttack` `:192`, `validateAttackers` `:83`, `canBlock` `:907`, `validateBlocks` `:638`, `getMinNumBlockersForAttacker`; `AttackConstraints.getLegalAttackers` `:77`; `TargetRestrictions.getAllCandidates`; `ComputerUtilMana.determineLeftoverMana` `:1621` | **Reuse.** The engine never calls `validateBlocks`, so the random seat calls it itself before returning a block declaration. |
| Playing a chosen spell with a cost-choice override | `ComputerUtil.handlePlayingSpellAbility` (forge/`ComputerUtil.java:80-124`), which hard-codes `new AiCostDecision(ai, sa, false)` | **Reimplement a copy in the connector, documented reason.** The cost-decision instance is not injectable, so randomising sacrifice/discard/exile/tap choices (FR-022c) needs a ~50-line copy that passes a `RandomCostDecision extends AiCostDecision`. Mana payment stays the AI's `CostPartMana` visit. |
| Writing legality and decision records | `PatchedCollectors.emitAttackers` `:1099`, `emitBlockers` `:1132` (private), `playabilityHandler` `:922-1008` (package-private) | **Extend.** Expose `recordAttackers`, `recordBlockers`, `recordCandidate` as public methods taking the deciding player and a `whatIf`; the hook handlers call them too. The random seat receives the per-game instance from `GamePlayer` after `createGame` (`:140, 157-177`). |
| Records-only output | `MatchWorkerMain` `WorkerConfig.recordsOnly()` `:102, 255-258, 389-408` | **Extend per match.** Today records-only is per worker; FR-025 needs it per match, so `MatchGenerationResult` carries a `randomSeat` flag and `recordMatch` skips both writers for it. |
| Threading collection flags to workers | `src/sealed/infrastructure/cli.py:988-1050, 1836-1890`; `src/sealed/infrastructure/match_worker_connector.py:137-140` | **Extend.** Two flags join `_effect_collection_caps`; FR-026's refusal sits in `run_match_outcomes` beside the side-b check. |
| Per-card coverage loop | `E/application/collect_coverage.py:69-625`; refusals in `load_exclusions` `:427-457` and `coverage_config_from` (`E/infrastructure/cli.py:422-436`) | **Extend** with a text-keyed mode (`TextCoverage`, a shard counter keyed by held-out text and distinct `game_id`). Rounds, retirement and workers are reused. |
| Keyword definition extraction | `FC/KeywordDefinitionMain.java:83-206` | **Extend** with `formatter`, the keyword's `Keyword.type` simple name (forge/`Keyword.java:218`), which selects the value formatting (FR-018). |
| Script parameter parsing for value-head targets | `E/domain/script_variants.py:44-79` (`_PARAM_RE`, `NUMERIC_PARAM_KEYS`, `numeric_params`); `ManaCost.parse` (`src/price_predictor/domain/value_objects.py:98-170`) | **Reuse.** `numeric_params` already reads integer literals only, which is the masking rule; `ManaCost.parse` skips non-mana shards (`T`, `Sac<…>`). |
| Scorer smoke cache | `geometry_checks.write_scorer_smoke_cache` (`E/application/geometry_checks.py:411-440`), only called by tests | **Reuse** and add the missing half: load the sealed vectors, write under `--scratch-dir`, run `python -m sealed train-scorer` Phase A as a subprocess. The written key is checked against `src/sealed/infrastructure/embedding_store.py:18`. |
| Grouped folds, AUC, ridge R² in probes | `scripts/effect_embedding_probes/pca_directions.py` (`cv_r2`, `GroupKFold`), `linear_probes.py` (`probe_binary`) | **Reuse the idioms** in the knowledge-probe suite. No MLP probe exists anywhere; it is new. `price_predictor/application/ridge_probes.py` is sealed-head specific and not reused. |
| Embedding-probe loaders | `scripts/effect_embedding_probes/common.py:21-356` (module-constant paths) | **Extend.** Paths become arguments resolved from `--checkpoint`/`--abilities-root`; `build_texts.py` stops skipping a sidecar whose `taxonomy` cache is absent (`:80-82`). |

### Convention alignment

Mirror **feature 023's own `src/effects/` layout**, which mirrors `src/draft/`. Every Python change
lands in an existing module of the right layer, and the four new domain modules follow the
`domain/*.py` pure-dataclass style of `records.py` and `corpus_curation.py`. Java changes stay in
`FC/effects/` beside the collectors, and the random seat sits beside `GamePlayer` in `FC/`. The
knowledge-probe suite mirrors `scripts/effect_embedding_probes/`: a `common`-style loader module,
one module per measurement, argparse entry points, no import-time work. Its tests load modules by
path as `tests/unit/scripts/test_regenerate_forge_api_list.py` does. No deviation.

### Third-instance check

- **Random play choices** exist once (`ForkCollector`) and the random seat would be the second.
  They are extracted anyway, because the two must agree on the rules FR-022c lists and
  `ForkCollector`'s copy is wrong on three of them.
- **Equal split with redistribution** appears three times inside FR-049 alone (families within a
  class, signatures within a family, the two legality halves). One allocator,
  `E/domain/budget_allocation.py`, serves all three.
- **Rarity bucketing** is used by the epoch line (FR-055), the evaluator (FR-064) and the manifest's
  under-five-games list (FR-053). One function, `rarity_bucket(games)`, in a new
  `E/domain/rarity.py`.
- **crc32 threshold selection** appears twice (holdout permille, game-disjoint share). Two
  instances, different moduli and inputs: no extraction.
- **Halves pairing** (`validate_corpus.py:582-584, 770-790`) assumes exactly one effect half per
  `link_id`. The widened meaning (FR-029d) is applied at both sites; there is no third.

## Decisions

### A mode's key is the root key plus its `Choices$` position

**Decision**: `ProvenanceKey` gains `option: int | None`, the mode's 0-based position in the root's
`Choices$`. In JSON it is an extra `"option"` member of the key object, written only when set.

**Rationale**: Forge gives a mode no identity of its own. `CharmEffect.chainAbilities`
(forge/`CharmEffect.java:287-311`) copies each chosen mode, overwrites its `CharmOrder` with its chain
position, and sets no back-reference; `ProvenanceKey.resolve` climbs every copy to the root
(`ProvenanceKey.java:230-233`). The only stable identity is the position in `Choices$`, which both the
converter (`CharmAbility.java:57`, `getAdditionalAbilityList("Choices")`) and the collector (the spell's
chosen list, which `chainAbilities` sorts in place by original `CharmOrder`) can compute. An additive
member keeps compatibility rule 1.

**Alternatives considered**: a new `option` trait kind (rejected: a mode is not a separate trait in
the script's trait lists, so `index_within_kind` would have nothing to index); keying by the mode's
SVar name (rejected: FR-002a renames labels away, and names repeat across cards).

### Modes are mapped to clones at the clause hook, at the first clause of each mode

**Decision**: the collector cuts a modal resolution into per-mode effect halves on
`onClauseResolving` of each mode's first clause. It maps the k-th mode-opening clause in chain
order to the k-th entry of the sorted chosen list, and that entry to its `Choices$` index by
identity against `getAdditionalAbilityList("Choices")`. A repeated mode appears twice in the
chosen list and opens two halves.

**Rationale**: `resolveApiAbility` (forge/`AbilityUtils.java:1583-1608`) nests the hooks: mode 1's
"resolved" fires after mode 2 finishes, so "resolved" cannot delimit modes. "Resolving" of a mode's
first clause is exactly the moment FR-029b's snapshot must be taken. The chosen list is null unless
the AI or a fork set it, so the random seat and `RandomChoices` always call `setChosenList`.

### The random seat is a lobby-player subclass, and its collector arrives after game creation

**Decision**: `RandomSeatLobbyPlayer extends LobbyPlayerAi` overrides `createIngamePlayer` to install
`RandomSeatController extends PlayerControllerAi`. `GamePlayer` gives the controller the per-game
`PatchedCollectors` (or the degraded bracket collector) after `match.createGame()`, and the seat's
`Random` seeded from the match seed.

**Rationale**: `createControllerFor` is private but `createIngamePlayer` is public and not final;
`Player.setFirstController` is public. No engine change (FR-024). Forks clone players through
`GameCopier.clonePlayer` (forge/`GameCopier.java:211-219`), which keeps any `LobbyPlayerAi` subclass,
so the controller checks that its game is the live game (`belongsToLiveGame`, `PatchedCollectors.java:1608`)
and defers to `super` everywhere in a fork.

### The random seat's decision points and how each draw is made

**Decision**:

| Override | Behaviour on `P` success | On `P` failure |
|---|---|---|
| `chooseSpellAbilityToPlay` | land-first rule (FR-022d) checked before the draw; candidates = the AI's own filter (`canPlay`, `canCastTiming`, `ComputerUtilCost.canPayCost`, not mana, not land); uniform pick; `recordCandidate` per candidate | `super` |
| `playChosenSpellAbility` | for a randomly drawn play: modes, targets, X via `RandomChoices`, costs via `RandomCostDecision`, through the connector copy of `handlePlayingSpellAbility`; abandon and redraw after 20 failed draws | `super` (targets under their own `P` draw, FR-022) |
| `declareAttackers` | per defender, each legal attacker with probability ½; `validateAttackers`, redraw up to 20 times, then `AttackConstraints.getLegalAttackers` fallback to the AI; `recordAttackers(whatIf=false)` per defender | `super` |
| `declareBlockers` | per blocker, uniform over no block plus legal attackers; `validateBlocks`, redraw up to 20 times, then AI; `recordBlockers(whatIf=false)` per attacker | `super` |
| `chooseTargetsFor` (triggers, deferred targeting) | FR-022b draw | `super` |

The land-first test is `player.canPlayLand(null)` together with a playable land among
`ComputerUtilAbility.getAvailableCards`; when it holds the seat calls `super` without drawing.

**Rationale**: every method above is public and non-final on `PlayerControllerAi`
(`:867, 872, 877, 882, 1414`). Land choice lives in private helpers inside the AI's play choice
(`AiController.java:465, 528, 1448-1462`), which is why FR-022d hands that priority to the AI.

### The retry limit is 20

**Decision**: `RandomChoices.MAX_DRAWS = 20` for every redraw in FR-022b and FR-022c.

**Rationale**: a uniform draw over legal elements fails only on cross-element constraints
(attack requirements, block minimums, multi-slot target rules). Twenty draws make a failure that
persists rare without stalling a priority loop that runs thousands of times per game.

### Legality de-duplication keys on the snapshot hash, built lazily for what-ifs

**Decision**: the de-dup key becomes `(subkind, payload, snapshot_hash)` (FR-028). For a what-if the
`--legality-rate` draw runs first and the snapshot is built only for survivors; a real decision always
builds it. `PatchedCollectorTest.java:1310-1317`, which pins the old key, is inverted.

**Rationale**: the snapshot is built only after de-dup today (`PatchedCollectors.java:1128, 1165`).
Moving it into the key would otherwise build a snapshot for every AI re-ask, most of which the rate
discards.

### `actor_player` on playability records is the deciding player

**Decision**: `recordAttackers` takes the attacking player (controller of the candidates),
`recordBlockers` the blocking player (controller of the candidate blockers), `recordCandidate` the
candidate ability's activating player. `random_seat` is `actor_player == randomSeatPlayerId`, set
where each record is built. The Python schema-compatibility test gains a named exception for this
one redefinition.

**Rationale**: FR-030a. Today `decision` and `attackers` use `activePlayerId()` (`:1003, :3146`) and
`blockers` uses the attacker's controller (`:1164`).

### `random_seat` matches skip the outcome writers per match

**Decision**: `GamePlayer.PlayedMatch` and `MatchGenerationResult` carry `randomSeat`. `recordMatch`
writes the progress line for every match and skips the match-outcome and cards-played writers for a
random-seat match. Which matches get a random seat is drawn per match from the worker's seeded
`Random` with probability `--random-seat-share`, and the seat (A or B) with probability ½.

### Readers default the two fields for gen-1 shards, and never mix generations

**Decision**: `record_from_dict` reads an absent `random_seat` as `False` and an absent `what_if` as
`None` (unknown). `validate-corpus` adds a per-shard presence check: a shard in which any record
carries a field must carry it on every record of the kinds that must (FR-033). `shard_generation(path)`
in `record_io.py` reads a shard's first complete record and names it gen-1 or gen-2 by the presence
of `random_seat`; `build-corpus`, `train-effect-model`, `evaluate-effect-model` and the knowledge
probes refuse a mixed set before reading records.

**Rationale**: Principle III asks for a versioned schema change. The redefined `actor_player` has no
version field, so the generation marker is what tells a reader which meaning a shard carries.

### Sidecar roots are a flag on every command that reads them

**Decision**: `train-effect-model`, `encode-abilities` and `evaluate-effect-model` take `--cards-folder`
(FR-063b), replacing the roots hardcoded at `train_effect_model.py:736`. The checkpoint records them.

**Rationale**: stage 0 trains and evaluates on gen-1 while stage 1 reconverts `output/cardsfolder/`.
Without the flag the noise pilot would resolve gen-1 records against gen-2 chained sidecars.

### Script chain rendering lives in Java; the converted text is untouched

**Decision**: `TraitScript.of` builds the chain from runtime accessors —
`Trigger.getOverridingAbility()`, `SpellAbility.getSubAbility()`,
`getAdditionalAbility("RepeatSubAbility")`, `ReplacementEffect.getOverridingAbility()` — rendering each
segment's own `getMapParams()` in `TreeMap` order, joined by ` [SEG] `. Labels are renamed in the same
pass. A label referenced but not defined (FR-003) is detected because the accessor returns null for
a parameter the map names; `ConvertMain` reports it and omits the segment. A visited set enforces
FR-003's once-per-chain rule. `RulesParser.java:335-338` additionally attributes each charm
`TextAbility(OPTION)` (`CharmAbility.java:56-95`) to its mode's SA with the `option` component.

**Rationale**: the runtime accessors already resolve every SVar reference the chain needs, and
`ProvenanceRecorder` (`ProvenanceRecorder.java:58-76`) already holds the trait when it calls
`TraitScript.of`. Rendering in Python would re-parse SVar text Forge has already parsed. The rendered
prose path is not touched, so `.txt` files stay byte-identical (FR-004, SC-001), which a test over
the whole converted tree checks.

### Tokenization: one grammar, two surfaces, staged identically in `build-vocab`

**Decision**: `AbilityTokenizer.tokenize` takes the surface from `surface_of` (`ability_encoder.py:191-205`)
at construction. On the script surface it applies, before lowercasing: the `TokenScript$` value
split at `_` (FR-009a); the camel-case word break (FR-008); the whole-token exceptions of FR-009,
matched by parameter name on the raw text. `build-vocab --surface script` stages script lines through
the same function, because the shared builder tokenizes with `MtgTokenizer`, which lowercases before
splitting (`src/price_predictor/domain/tokenizer.py:115`) and would erase camel case.
`tokenize_script` (`ability_tokenizer.py:368-424`) and its tests
(`tests/unit/effects/domain/test_script_tokenizer.py`) are removed; its one remaining caller,
`scripts/effect_embedding_probes/keyword_expansion.py:361-362`, moves to `tokenize`.

**Seeding order**: specials, then `sv1…svN`, then template words (FR-019), then key parts
(FR-009b), all before `domain_token_count` (`build_vocab.py:173-187`). On the script surface the
script lines are staged before the prose `.txt` files, so frequency truncation drops prose tokens
first, not script tokens.

**Also fixed in passing**, because each is the same code the FRs touch: `generated_script` is no
longer scanned (FR-020; `build_vocab.py:111`); `HOST_BODIED_KEYWORDS` compares display names, which
fixes `level up` and `read ahead` never matching (FR-016); `encode-abilities` stops expanding at 1.0
(`ability_encoder_runner.py:103-106`) and reads the shared constant (FR-012).

### Tokenizer rules are versioned by checkpoint

**Decision**: every training run records `tokenizer_rules = gen-2` in the checkpoint's
`training_settings`, and every loader picks the tokenizer's grammar with
`tokenizer_surface(surface_of(vocab_path), checkpoint.training_settings)`: the vocabulary's surface
for a gen-2 checkpoint, the prose grammar for one that records no rules.

**Rationale**: gen-1 tokenized its script vocabulary with the prose grammar, so `surface_of` alone
hands it the gen-2 script rules it never trained on. Read that way, its `[UNK]` rate over script
text rises from 0.4% to 3.2% and 73% of texts change their token sequence, which would distort gate 2,
the knowledge probes and the noise pilots of stage 0. The prose grammar reproduces gen-1's tokens
exactly. A noise pilot trains from scratch under the gen-2 rules, so it reads a script vocabulary
rebuilt with them over gen-1's sidecars (`models/effects/vocab-script-gen1-sidecars.txt`).

### Keyword expansion by display name

**Decision**: `_keyword_token` keeps its token-path role (FR-014). A new `display_name_of(line)`
returns the text before the first colon of a keyword line's `script_text`; expansion of a keyword
line replaces every token of that span with the filled template. Formatting is a small table keyed by
the recorded `formatter` (FR-018): `KeywordWithCost` → mana cost rendered as Forge's
`costReminderText`, `KeywordWithAmount` → the integer, `KeywordWithType` → the type name, and so on
for each `Keyword.type` value present in the definitions file; an unknown formatter fills the raw
value. `_instantiate` (`ability_tokenizer.py:339-359`) gains positional specifiers and removal of
unfilled ones; apostrophes are normalized at load. Callers start passing `instance_values`, which
none does today.

### The value head's targets are computed in Python from `script_text`

**Decision**: `E/domain/value_targets.py` splits `script_text` on `[SEG]`, reads integer literals of
the amount keys per segment with `numeric_params`, sums them per target, and masks a target any
segment states with a non-literal (`X`, an SVar name). Cost targets read the root segment's `Cost$`
through `ManaCost.parse` plus `T` and `Sac<` tests, with mana targets masked on spell lines (FR-058a).
Targets are cached per text in the batcher like `e`.

**Alternatives considered**: computing targets in Java with Forge's `Cost` at convert time (rejected:
adds sidecar fields for a training-only head, and every target would still need a Python path for
gen-1 one-segment texts in the noise pilot).

### Rule families are a pure function of the record and the sidecars

**Decision**: `E/domain/rule_families.py`:

| Kind | Family |
|---|---|
| `resolution` | acting line's `script_api_type` (a mode's own, FR-005a) |
| `trigger` | `Mode$` parsed from the trigger line's root segment; `(no-mode)` when absent (244 of 3,081 triggered lines) |
| `continuous` | acting static line's `script_api_type`, which equals its `Mode$` (1,218 of 1,234); a comma list is kept whole |
| `rewrite` | `Event$` parsed from the replacement line's root segment (its `script_api_type` is `None`) |
| `combat` | sorted set of damage-step keywords on participants, through `KeywordResolver`; `none` when empty |
| `playability` | FR-047a, with a static's `Mode$` read through the same sidecar lookup |
| any kind, keyword acting line | that keyword's display name (FR-048) |

### Selection is per-record copy counts from a water-filling allocator

**Decision**: the survey pass emits, per record, `(class, family, signature, text, record_hash,
is_real)`. `decide` builds cells, computes each text's capacity
`min(reuse_cap × n, text_cap)`, and runs `allocate(budget, capacities)` — equal shares,
redistributing what a capped member leaves, until all are full or exhausted — at three levels.
Within a text it assigns copy counts: a text over the cap takes the `cap` smallest `record_hash`
values once each; otherwise the cap spreads evenly over its records, the extra copies going to the
smallest hashes. The write pass writes each record `copies` times. This preserves determinism under
`--seed` (FR-052) because every choice is a function of `record_hash`.

### Game-disjoint placement is a per-game crc32 test

**Decision**: `in_game_disjoint_stratum(game_id, has_keyword_combat, share, keyword_share)` in
`E/domain/corpus_curation.py`: `crc32(game_id) % 10**6 / 10**6` below the applicable share, never
for a game naming a held-out card. `has_keyword_combat` comes from the survey pass through the moved
`qualifying_observations`. `--game-disjoint-games` is removed from the parser so argparse rejects it
(spec Story 4 scenario 10).

### Noise on `e` lives in the training loop, applied in the batcher

**Decision**: a `NoiseState` (running Σ as a `d×d` tensor on the device, a step counter) is owned by
`TrainingLoop` and passed to each per-batch `SurfaceBatcher` (built per batch at
`training_loop.py:442-474`). `SurfaceBatcher.build` adds `L z` to every scattered `e` row, `L` the
Cholesky factor of `r²Σ + εI`, with `ε = 1e-6`. Σ is the batch covariance about the batch mean under
`torch.no_grad()`, decayed at 0.99.

### Verdict and created-objects losses join the per-entity loss at unit weight

**Decision**: `_loss_for` (`training_loop.py:478-547`) adds `verdict_loss` and `created_objects_loss`
at weight 1.0 each, normalized per record like the per-entity loss, on the record kinds the base
spec's table assigns them. `SurfaceBatcher` passes `candidate_index=0` for `decision` records
(`surface_batching.py:287-297`), and `effect_targets.py` emits the verdict bits as `[ACT]` targets.

**Rationale**: the base spec assigns these heads targets but no weights, and says every head's
outputs outside the curriculum group train from step zero. Unit weight is the reading that adds no
new setting.

### MLM and script-API losses use their existing flags

**Decision**: `_loss_for` adds `mlm_loss` (masking at `--mlm-mask-prob`) and `api_loss` at
`--mlm-weight` and `--api-weight`. `n_api_types` and `n_param_keys` are set from the vocabulary of
`script_api_type` values and parameter keys over the sidecars, recorded in the checkpoint so the
training-only heads can be rebuilt for a resumed run. `mlm_head` takes the configured `d_model`.

### The option-kind embedding starts at zero

**Decision**: `EffectModel` adds a two-row `option_kind_embedding` whose rows are initialized to
zero, added to ability rows like `slot_kind_embedding` (`effect_model.py:288, 356`).

**Rationale**: `model_runner` loads with `strict=False` (`model_runner.py:114`), so a gen-1 checkpoint
gets the module's initial value. Zero rows make that value a no-op, so gen-1 inference is unchanged
even on a board whose sidecars carry option lines.

### New manifest and checkpoint fields stay out of gen-1 digests

**Decision**: every field this feature adds to `CorpusManifest` and `SplitProvenance` is written only
when it differs from the value a gen-1 artifact implies (`holdout_unit == "text"`, empty breakdowns).
Readers use `.get(field, default)`.

### Constants fixed in code

| Constant | Value | Where |
|---|---|---|
| Random-draw retry limit | 20 | `FC/effects/RandomChoices.MAX_DRAWS` |
| Card-disjoint sample, records per held-out text per round | 4 | `E/application/validation_samples.py` |
| Knowledge-probe minimum gap (rung 3 − rung 0) | 0.05 AUC or R² | `scripts/effect_knowledge_probes/ladder.py` |
| Linear probe | logistic regression, L2, C = 1.0 (yes/no); ridge, α = 1.0 (amounts); features standardized per fold | `ladder.py` |
| MLP probe | two layers, hidden 256, ReLU, AdamW lr 1e-3, weight decay 1e-4, batch 512, 30 epochs, on GPU | `ladder.py` |
| Folds | 5, grouped by text (by card for pooled items), seed 42 | `ladder.py` |
| Share confidence interval | 1,000 bootstrap resamples over texts, 95% percentile | `ladder.py` |
| State-dependence weight | n / (n + 5), n = the text's resolutions | `labels.py` |
| Noise jitter ε | 1e-6 | `E/application/training_loop.py` |
| Verdict and created-objects loss weights | 1.0 | `training_loop.py` |

## Technology notes

No new runtime dependency. The knowledge probes use torch (already a dependency) for the MLP probe
and scikit-learn (already used by `scripts/effect_embedding_probes/`) for the linear probes and
`GroupKFold`. Java stays on Java 17 and the Forge `2.0.15-SNAPSHOT` jars from `../forge`.
