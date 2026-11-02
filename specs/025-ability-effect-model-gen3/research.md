# Research: Ability effect model — generation 3

**Feature**: `025-ability-effect-model-gen3` | **Date**: 2026-10-31
**Spec**: [spec.md](spec.md) | **Root spec**: [`../2026-10-30-ability-effect-model-gen3.md`](../2026-10-30-ability-effect-model-gen3.md)
| **Base features**: [`../023-ability-effect-model/`](../023-ability-effect-model/plan.md), [`../024-ability-effect-model-gen2/`](../024-ability-effect-model-gen2/plan.md)

The root spec and the clarification sessions pin the behaviour. What remains for planning is where
each change attaches to the code features 023 and 024 left, the constants the spec leaves to the plan
(the balance ratio r, the worst-case standard-error estimators, the target-index cap, the
remembered-list rule), and the mechanisms the survey found the spec had stated wrongly. Three of
those were put to the user on 2026-10-31 and decided:

- Forge's `GameCopier` is patched on `effect-record-hooks` to carry delayed triggers and
  end-of-turn commands into a fork (Decision D3).
- Gen-1 and gen-2 checkpoints stop loading: a clean break (Decision M1).
- The five spec statements the code contradicts are corrected in `spec.md` and the root spec in
  this pass (Decision S1).

Paths are relative to the repository root unless prefixed `forge/` (the sibling checkout on its
`effect-record-hooks` branch). `FC` abbreviates `forge-connector/src/main/java/com/pricepredictor/connector`,
`FE` is `FC/effects`, `FG` is `forge/forge-game/src/main/java/forge/game`, `FA` is
`forge/forge-ai/src/main/java/forge/ai`, and `E` is `src/effects`.

## Codebase Survey

Constitution Principle VII. Every row cites a concrete file or symbol.

### Overlapping domain vocabulary

Collection (Java and the record envelope):

| Existing concept | Where | Decision |
|---|---|---|
| `EffectRecord` (Java) | `FE/EffectRecord.java:238-270` (`toJson`, fixed field order); stamping `FE/CollectorContext.java:48-105` | **Extend** with `generation`, `aimed_at`, `directed`. `aimed_at` is stamped in `CollectorContext` as `random_seat` is (`:104`). |
| `EffectRecord` (Python), `ShardGeneration` | `E/domain/records.py` (`random_seat` `:530`), `E/infrastructure/record_codec.py`, `E/domain/shard_generation.py` (`ShardGeneration.of` keys on `random_seat`) | **Extend** with `GEN_3`, keyed on `generation`. |
| `refuse_mixed_generations` | `E/infrastructure/shard_io.py:423-434`; callers `build_corpus.py:522`, `train_effect_model.py:444`, `evaluation_context.py:181`, exit map `cli.py:738` | **Replace** with `require_generation(paths, GEN_3)`. `rule-counts` accepts gen-2 and gen-3. |
| `PatchedCollectors`, `PatchHooks.REQUIRED` | `FE/PatchedCollectors.java`; `FE/PatchHooks.java:319-325` (18 hooks) | **Extend.** Three hooks are added (Decision C1). The handlers dispatch by `Game` identity so a fork can be recorded (Decision D4). |
| `BusBracketCollector`, `HeldResolutionWindow`, `BusEventReader.isStateBasedLeave` | `FE/BusBracketCollector.java` (`beginBracket :131`, `fileMove :345`, `route :529`, `onPriority :579`); `FE/HeldResolutionWindow.java`; `FE/BusEventReader.java:99` | **Extend.** The held half's window is the state-based loop, not the frozen tracker (Decision C2). |
| `CombatStepBracket` | `FE/CombatStepBracket.java:98-104` (snapshot at the first filed damage event) | **Extend** (FR-027). The snapshot moves to `onPhase` of the damage step. |
| Fork machinery | `FE/Forks.java` (`copyAndVerify :162-179`, `scoreCheck`, `findInFork`, `asFork`); `FE/InterventionFork.java`; `FE/DamageStepProbe.java`; `FE/ForkBudget.java`; schedulers `FE/InterventionScheduler.java`, `FE/ProbeScheduler.java` | **Extend and extract.** A directed fork is the third fork driver, so the run skeleton is extracted (Third-instance check). `ForkBudget` gains the directed budget. |
| Damage-step probe keyword removal | `DamageStepProbe.stripKeyword` → `Card.addChangedCardKeywords(null, List.of(kw), false, ts, null)` | **Reuse** as the "add or remove a keyword" operation, lifted into `KeywordEdit`. |
| `RandomChoices` | `FE/RandomChoices.java` | **Reuse** unchanged. |
| `SnapshotBuilder`, `EntityJson`, `PlayerJson` | `FE/SnapshotBuilder.java:94-142`; `FE/EntityJson.java:41-69` (`ptJson :85-104`, `printedKeys :236`, `grantedAttached :250-256`, `addKeys :262`); `FE/PlayerJson.java:27-42` | **Extend** for FR-021–026. **Replace** `ptJson`: it treats printed power as base, counts only ±1/±1 counters and writes power twice. **Replace** `grantedAttached`: it lists the attachment's own spell abilities, not what it grants. `addKeys` widens from `SpellAbility` to `CardTraitBase`. |
| `ProvenanceResolver` | `FE/ProvenanceResolver.java:57-90`; Secondary handling `FC/FaceParser.java:445-461` | **Reuse.** It already resolves all five trait kinds. |
| `PlayabilityVerdict`, `DecisionCandidate`, `PlayabilityChannel` | `FE/PlayabilityVerdict.java:38-105`; `FE/DecisionCandidate.java` (`legalTargetsOf :72-89`, cap `:21`); `FE/PlayabilityChannel.java:61-80, 180-237` | **Extend.** Add `timing`, `failed_checks` and the static-prohibition check. Lift the 32-target cap and the top-clause-only walk. Decision records come from a collector-side enumerator (Decision C3). Attackers records become one per attacker. |
| `CollectionCaps` (Java and Python), flag plumbing | `FE/CollectionCaps.java`; `FC/MatchWorkerMain.java:223`; `src/sealed/infrastructure/effect_collection_flags.py:28` `add_cap_flags`; `src/sealed/infrastructure/cli.py:1017`; `E/domain/collection_caps.py` | **Extend** with the directed-fork properties and `effect.count.dropped.events`. `test_collection_caps.py` pins both spellings. |
| Random seat | `FC/RandomSeatController.java:248-277` `playableCandidates`; `FC/RandomSeatLobbyPlayer.java:46-49`; `FC/GamePlayer.java:224-239` | **Extend** for FR-040 and for seats inside a fork (Decision D4). |

Model, curation and evaluation (Python):

| Existing concept | Where | Decision |
|---|---|---|
| Trunk | `E/domain/effect_model.py:93-101`: `nn.TransformerEncoder(nn.TransformerEncoderLayer(d=256, nhead=4, ff=1024, dropout=0.1, batch_first, norm_first=True), 6 layers)` | **Replace** with `RelationEncoderLayer` in its own module, keeping the stock parameter names so a stock `state_dict` loads in for the equivalence test (Decision M2). |
| Widths | trunk and encoder already 256; `e_dim` defaults to 64 (`train_effect_model.py:112`, `ability_encoder.py:48`) | **Extend.** FR-043 is a change of the `e_dim` default only. |
| `InputLayout`, `INPUT_LAYOUT`, `EARLIER_LAYOUT` | `E/domain/effect_head_input.py:106-139`; `run_context.training_settings` `:128-137` | **Replace** with one layout, `gen-3`. The earlier layouts are removed (Decision M1). |
| Slot kinds, projections | `SlotKind` (`effect_head_input.py:46`); one `Linear` per kind (`effect_model.py:80-84`); ability slots carry `e` plus `option_kind_embedding` only (`:88`) | **Extend.** An `ability_proj` reads line flags. One subtype embedding table is shared by subtypes and chosen creature types. |
| Gen-2 link features, `_push_overlay` | `effect_head_input.py:401-429`; player is-a-target `:360`; stack extras `:449`; `Slot.option` `:253` | **Keep** every one except "card is attached", which is removed (FR-051). Relations sit beside them. |
| Face-down flag, granted keyword multi-hot, granted-abilities count | `_push_overlay` (`~:409`, `~:420-426`, `OVERLAY_KEYWORDS_GEN2`) | **Replace** with face-state one-hot and granted lines plus the kept-flag list (FR-044a, Decision M5). |
| Acting-ability resolution-outcome input | `act_features` (`effect_head_input.py:322-339`, `_RESOLUTION_OUTCOMES`) | **Remove**; it becomes a verdict-head output (FR-052). |
| `SurfaceBuilder` | `effect_head_input.py:542-632` (`build`, `_entity_block`, `_keeps :615-623`) | **Extend.** Relations are emitted here, after context dropout and gate-2 stripping have fixed slot indices. |
| `TrunkInputs`, `collate_surfaces` | `E/domain/surface_collation.py:20-140` | **Extend** with a COO relation tensor `(N, 4) = [batch, from, to, type]`. `tensor_packing` already carries tensors generically. |
| Baseline masks | `SurfaceBatcher._masked` (`E/application/surface_batching.py:481-498`) | **Extend.** `no-state` drops the relations too, or they would carry board structure. |
| `derive_targets`, `_apply_playability`, `EntityTargetBatch.scored` | `E/domain/effect_targets.py:139-162, 321-340`; `E/domain/entity_field_targets.py:82-85` | **Replace** the playability path: legality fields score on a gate-free path, write 0 outside the legal set, and no playability record supervises the gate (FR-053a). |
| Gate loss | `_gate_term` (`E/domain/effect_losses.py:150`) | **Extend** with a per-row mask (playability off; the hard rule mask limits it to the rule's entities). |
| Output layout | `PER_ENTITY_FIELDS`, `FIELD_SLICES`, widths (`E/domain/effect_head_layout.py:230-343`) | **Extend** for the 22 new outputs and the § 11.3 extensions. No layout versioning (Decision M1). |
| Heads, `AUX_HEADS` | `effect_model.py:105-124`; `E/domain/aux_heads.py:102-120` | **Extend.** The global head is one more `AUX_HEADS` row. The verdict head gains columns. |
| Poisson | `E/domain/field_kinds.py:55, 141-180`; `effect_losses.py:206, 239, 249-277`; `E/domain/gate_metrics.py:64-183` | **Replace** by the negative binomial inside `FieldKind` (Decision T2). |
| Constant-predictor floor | `E/domain/loss_floor.py:81-130` (`constant_predictor_floor`, `PartsMean`); `FloorCache` (`validation_pass.py:182-200`), card-disjoint only | **Extend** to every head, both strata and the negative binomial. One function serves FR-062's denominators and FR-068's deviance component. |
| `TrainingLoop`, `EarlyStopper`, `Curriculum` | `E/application/training_loop.py` (`_close_epoch :237-251`, `_checkpoint :253-269`, `_optimizer_step :119-125`, `_train_step :381-382`); `E/application/training_schedule.py:18-100` | **Extend** with the averaged weights and the table norms. **Replace** the stopper by a higher-is-better control-score stopper. |
| Parameter groups | `run_context.py:53-65`; AdamW lr 1e-4, wd 0.01, clip 1.0 per group | **Reuse.** The relation tables fall into `head`. |
| `resume.pt` | `E/application/resume_state.py:40-145` (`RESUME_MAY_CHANGE :40-44`) | **Extend** with the averaged weights and their start step. FR-065's refusals come from the existing compare-every-flag rule. |
| `Validator` | `E/application/validation_pass.py:241-263`; bf16 autocast `step_loss.py:32,123-142`; samples of 2048 records per stratum (`build_corpus.py:165`) | **Extend** to the four components per stratum and half (Decision E2). |
| `EffectCheckpoint` | `E/infrastructure/effect_model_store.py:90-140` (`best_val_loss`, `training_settings`, `load_weights` refuses size mismatches) | **Extend** (FR-077). `best_val_loss` is replaced by `best_control_score`. |
| Gate 1 | `gate_metrics.poisson_deviance`, `MarginAccumulator`; `E/application/gate_one.py:134-164` | **Replace** the deviance by the negative binomial's. |
| Scenario files and their validator | `rules/scenarios/<unit>.json` (`rules/README.md:153-222`); `scripts/rules_catalogue.py:362 validate_scenarios`, enums `:27-46` | **Extend** with `detector`, `variant.operation` and `departure` (contracts/catalogue-extensions.md). The enums move to `E/domain/rule_catalogue.py`, which the script imports. |
| Heads map | `rules/heads/<unit>.jsonl` (`id, output, outputs, input, inputs_missing, note, gen3_*`); `rules_catalogue.py:291 validate_heads`; produced by `scripts/rules_heads_workflow.js` | **Reuse.** Verdicts are already per rule id. `build-corpus` filters `outputs` to the heads the model has (Decision R4). |
| `HEAD_OUTPUTS` | `rules_catalogue.py:35-43`, a hand copy of `HeadField` | **Replace** with a list derived from `HeadField` and the verdict-head names. |
| `RuleFamily`, `rule_family` | `E/domain/rule_families.py:143`, used through `Cell.of` (`E/domain/corpus_plan.py:84`) | **Reuse the pattern** (a pure record + sidecars → label function) for detectors. **Rename** the gen-2 concept: see below. |
| Sampling classes, outcome signature | `E/domain/sampling.py:20,41`; `E/domain/corpus_curation.py:167-173` (`SIGNATURE_CLASSES`) | **Reuse** the classes. **Extend** the signature with the condition bit, and give decision and legality records a signature level (FR-056). |
| Water-filling allocator | `E/domain/budget_allocation.py:15 allocate`; `corpus_plan.py:264 AllocationNode`, `_split_in_rounds :332` | **Reuse** for the natural share. Rule copies are a new step after it (Decision R2). |
| `text_capacity`, `CopyPlan` | `corpus_curation.py:197-206` | **Reuse** for the natural share. Rule copies are per record (Decision R2). |
| Lethality copies | `E/domain/lethality_counterfactuals.py` (`qualify :301`, `lethal :197`); `StepPreparer._counterfactuals` (`step_preparation.py:286-346`) | **Reuse.** `lethal` moves into `E/domain/rule_outcomes.py` (Third-instance check). Protection copies follow its qualify-then-copy shape in a module of their own (Decision R5). |
| `GameRouting`, `admit_card_disjoint`, `_split_games` | `corpus_plan.py:484-558`; `corpus_curation.game_disjoint_draw :252`; `build_corpus.py:394` | **Replace** with a global placement step after the survey (Decision R3). |
| Survey and write passes | `E/application/corpus_survey.py:279 _ShardSurveyor` (`_route_games :339`, supply fold `:353`); `E/application/corpus_write.py:145 _ShardWriter`, `_hold :176` | **Restructure.** The survey returns per-game rule inventories, and placement runs before the training supply is final. The validation strata are written under the rule budgets. |
| `CorpusManifest`, `FamilyRow` | `E/domain/corpus_manifest.py:186-321` | **Extend** with a `RuleRow` modelled on `FamilyRow`, split by polarity and source, and the map snapshot. |
| `validation_samples` | `E/application/validation_samples.py:28,134` | **Extend.** Samples are drawn per half under the rule budgets, sized by the precision target (Decision E2). |
| Knowledge-probe suite | `scripts/effect_knowledge_probes/*` (`readout.READOUTS :79`, `ladder.score :89-110`, `probe_set.py`, `run.py --freeze-probe-set :323`, the three grid scripts) | **Split.** Rung-3 readouts, item and grid scoring, the frozen set and its freeze move into `E` (Decision E1). The scripts keep their CLI, rungs 0–2, tables and charts. |
| Parallel shard readers | `E/application/parallel_map.py:40 run_parallel`; `WorkerContext.records` (`corpus_survey.py:100`); `E/domain/tally_merge.py:17 merge_into` | **Reuse** for `rule-counts`. |

Concepts gen-3 adds, none colliding with an existing name once the renames below are made:

| New concept | Where | Nearest prior art |
|---|---|---|
| `RuleTag`, rule detectors | `E/domain/rule_detectors/` | `rule_families.rule_family`, `damage_step_keywords.qualifying_observations`, `lethality_counterfactuals.qualify` |
| `ScenarioCatalogue` (read-only view of scenarios and heads) | `E/domain/rule_catalogue.py`, `E/infrastructure/rules_catalogue_io.py` | `rules_catalogue.py` validators |
| `RelationEncoderLayer`, `RelationType` | `E/domain/relation_attention.py`, `E/domain/relation_types.py` | `nn.TransformerEncoderLayer` |
| `WeightAverage` | `E/application/weight_average.py` | the `e`-noise running covariance `NoiseState` (`E/application/surface_batching.py:46-111`; decayed update `:92-105`, `state_dict`/`load_state_dict :72-87`, kept in `resume.pt` by `resume_state.py:91, 123`) |
| `SelectionSet` (frozen probes, grids, rule tests, halves) | `E/domain/selection_set.py`, `E/application/freeze_selection_set.py` | `scripts/effect_knowledge_probes/probe_set.py` |
| `ControlScore`, `StratumScore` | `E/application/control_score.py` | `Validator.epoch` (`E/application/validation_pass.py:241-263`) and its compute-once `FloorCache :181-199`; the probe runner scoring a frozen set against one checkpoint, `scripts/effect_knowledge_probes/run.py:678 probe` with `ladder.score :89-110` |
| `DirectedRule`, `BoardPrecondition`, fork operations | `FE/directed/` | `InterventionFork`, `DamageStepProbe`, `VariantRegistry` |
| `DirectedForkScheduler` | `FE/directed/DirectedForkScheduler.java` | `InterventionScheduler.collect` (`FE/InterventionScheduler.java:43-60`, once per line per game through `interveneSeen :34`); `ProbeScheduler.collect :45-81` and its rotation over a flag's list, `probeFocus :89-96`; both called from `GameRecording.PhaseBridge.onPhase` (`FE/GameRecording.java:141-149`); per-game seeded `Random` as `InterventionFork :44` builds it; spend-when-taken in `ForkBudget` (`:52`, `:81`) |
| `PriorityCandidates` | `FE/PriorityCandidates.java` | `RandomSeatController.playableCandidates`, the AI playability hook |
| `ForkPlayForward` | `FE/ForkPlayForward.java` | `InterventionFork.forceResolution`; Forge's `AITest` driving `mainLoopStep` |
| `FloatingEffects` (floating effects as command-zone objects) | `FE/FloatingEffects.java` | `LayerContributions.addCard` (`FE/LayerContributions.java:43-69`) over the same timestamp-keyed layer tables; `ContinuousChannel.record` skipping static id 0 (`FE/ContinuousChannel.java:79-83`); `ClauseOutcomeRouter.clauseResolving` (`FE/ClauseOutcomeRouter.java:87-103`); `EventAttribution.declaredDuration :203-215`; `SnapshotBuilder.emblems :209-217`; `ProvenanceResolver.effectSource :331-334` (Decision C4) |
| Protection copies | `E/domain/protection_copies.py` | `lethality_counterfactuals.qualify :301` and `counterfactual :448`; `evasion_grid.edited :106` granting protection through `BoardEdit.grant_keyword` (`scripts/effect_knowledge_probes/board_edits.py:81`) and deciding the block by `may_block :63`; copies written by `_ShardWriter._train` (`E/application/corpus_write.py:186-205`) (Decision R5) |

Renames, required by Principle VII(c) because the names would read as the same thing:

| Older name | Collides with | Rename |
|---|---|---|
| `E/domain/rule_families.py`, `RuleFamily`, manifest key `families` | gen-3's "rule" (a numbered Comprehensive Rules rule) | `effect_families.py`, `EffectFamily`; the manifest key stays `families` (a gen-3 manifest is new, so its digest is free to change) |
| `corpus_plan._selection_half` (real versus what-if legality) | gen-3's selection half | `_what_if_half` |
| `ladder.PROBE_TYPES = ("linear", "mlp")` | the spec's "probe type" (a probe target) | `PROBE_MODELS` |
| `EffectCheckpoint.best_val_loss` | — | removed; `best_control_score` replaces it (Decision M1) |

Summary: 50 existing concepts surveyed. Most are extended or reused in place, 9 are replaced (the
generation check, the trunk, the input layout, the face-down flag and keyword multi-hot, the
playability targets, the Poisson, gate 1's deviance, `HEAD_OUTPUTS`, game routing) and 1 is removed
(the resolution-outcome input). 12 concepts are new, and 3 renames are required.

### Adjacent prior art

| Sub-problem | Prior art | Decision |
|---|---|---|
| Copying a game and checking the copy | `Forks.copyAndVerify`, `GameStateEvaluator` score check | **Reuse.** The directed fork spends its budget before verifying, as `InterventionFork.intervene :47-57` does. |
| Playing a copied game forward | Forge's `PhaseHandler.mainLoopStep()` (`FG/phase/PhaseHandler.java:1045`, public); `forge-gui-desktop/src/test/java/forge/ai/AITest.java:92` drives a game by it | **Wrap.** `ForkPlayForward` loops `mainLoopStep` until the turn number changes or the game ends. It is a spike task first: nothing here has run it on a `GameCopier` copy. |
| What a fork copy loses | `forge/.../GameCopier.java`: delayed triggers, `addUntil` closures, `ExiledWith :352`, attacked this turn `:115`, this-turn casts `:181`, counters added this turn `:233`, controller timestamps `:357` | **Patch Forge** (Decision D3). |
| Fork operations | `Card.addNewPT :4509`, `addPTBoost :4684`, `addChangedCardKeywords :5140`, `addColor :4291`, `addChangedCardTypes :4247`, `GameEntity.setCounters :326`, `Card.setTapped :4767`, `Player.setLife :414`, `CardFactory.getCard :169`, `Zone.remove :162` / `Zone.add(card, index) :91` (`FG/zone/Zone.java`), `Combat.removeFromCombat :608`, `Card.unattachFromEntity :4015`, `Card.setSickness :3686`, `Card.setGameTimestamp :6908`, `GameAction.checkStaticAbilities` | **Reuse**, silent edits where Forge has them (Decision D2). |
| Static condition, live trait lists | `Card.getTriggers :7119`, `getStaticAbilities :7081`, `getReplacementEffects :7143` (live via `CardState.getTriggers :688-697`); `StaticAbility.checkConditions :362` | **Reuse**, public and read-only. |
| Keyword instances | `Card.getKeywords :5091`, `KeywordInterface.getOriginal()` | **Reuse.** Protection lines read the instance's fragment. |
| Player and global facts | `Player.getMaxLandPlays :1696`, `getMaxLandPlaysInfinite :1732`, `getLandsPlayedThisTurn :2230`, `StaticEffect.getAffectedPlayers :130` (`FG/StaticEffect.java`), `Player.getSpeed :1882`, `hasBlessing :3657`, `hasInitiative :3551`, `Game.getMonarch :171`, `Game.getDayTime :1404`, `Card.getCurrentRoom :2332`, `StaticEffects.getStaticEffect(stAb).getAffectedCards()` (`StaticEffects.java:59`, `StaticEffect.java:109`), `getXManaCostPaid :1636`, chosen values `:2158-2302` | **Reuse.** |
| Pending delayed triggers | `TriggerHandler.delayedTriggers`, `thisTurnDelayedTriggers`, `playerDefinedDelayedTriggers` (`FG/trigger/TriggerHandler.java:49-51`, private, no getter) | **Patch Forge** with an accessor, shared with Decision D3. |
| Relation sources | `getExiledWith :2172`, `getMergedCards :1447`, `getMustBlockCards :1685`, `getGoaded :7798`, `getEncodingCard :1440`, `getHaunting :1398`, `getPairedWith :1405`, `getPromisedGift :2137`, `getProtectingPlayer :2146`, `getCopiedPermanent :3592`, `getCloneOrigin :1705`, `getCrewedByThisTurn :6821`, `getSaddledByThisTurn :6632`, `getBlockedThisTurn :1663`, `getDamageHistory :6103`, `getEffectSource :7248` | **Reuse**; `pending on` reads the new accessor. |
| Per-check playability | `Player.canPlayLand :1659` (its turn, sorcery-timing, zone, prohibition and limit parts, for land plays; `LandAbility.canPlay :51`), `StaticAbilityCantBeCast.cantPlayLandAbility :73`, `SpellAbilityRestriction.checkActivatorRestrictions :330`, `checkTimingRestrictions :282`, `checkZoneRestrictions :193`, `checkOtherRestrictions :349`, `getLimitToCheck`/`getGameLimitToCheck :580-598`; `SpellAbility.canCastTiming :2574`; `MagicStack.isSplitSecondOnStack`; `StaticAbilityCantBeCast.applyCantBeCastAbility :99`, `applyCantBeActivatedAbility :156` | **Reuse** (Decision C3). |
| Conditions at resolution | the clause hook fires right before `sa.metConditions()` (`FG/ability/AbilityUtils.java:1446-1473`) | **Reuse.** The collector calls `metConditions()` on identical state. The GIFT trigger runs between the two calls (`:1469`); a gift segment is read after it. |
| Attack constraints and taps | `combat.getAttackConstraints().getRequirements().get(c).hasRequirement()` (`Combat.java:197`, `AttackRequirement.java:84`); `GameEventCardTapped` at `PhaseHandler.java:607`, then `GameEventAttackersDeclared` at `:629` | **Reuse.** The second event flushes the held attackers records. |
| "Must block" | `CombatUtil.java:640-660, 843-955` (`getMustBlockCards`, `MustBeBlockedBy`, `MustBeBlockedByAll`, "All creatures able to block CARDNAME do so.", `StaticAbilityMustBlock`) | **Reimplement** as a collector-side `MustBlock` predicate, documented reason: Forge has no per-(blocker, attacker) predicate. One JUnit case per source pins it. |
| Turn-based actions | draw `PhaseHandler.java:270`, `Untap.executeAt :74`, `doDayTime :251`, lore counters `PhaseHandler.java:283-288`, cleanup discard `:374-395`, `GameEventCardPhased` (`Card.java:5768`) | **Reuse**; phasing gains an `EventType` and a `BusEventReader` case. |
| Rule outcome functions | `lethality_counterfactuals.lethal`; `scripts/.../combat_grid.step_deaths`, `evasion_grid.may_block` | **Move** into `E/domain/rule_outcomes.py`. |
| Rung-3 readouts, AUC and R² | `scripts/.../readout.READOUTS`; `ladder.score`; `scripts/_lib/stats.py:49 auc` | **Move** into `E/domain/probe_readouts.py` and `E/domain/score_scales.py`. |
| Frozen probe set | `scripts/.../probe_set.py` (sha256 digest, file `knowledge-probes-set-<corpus digest>.json`) | **Move** into `E/domain/selection_set.py` and extend. |
| Threading new collection flags to workers | `add_cap_flags` (`effect_collection_flags.py:28`), `_effect_collection_caps` | **Extend.** |
| CLI subcommands with parallel shard reads | `field-coverage`, `validate-corpus` (`E/infrastructure/cli.py:612-693`, sequential) | `rule-counts` follows `build-corpus`'s parallel pattern instead. |

### Convention alignment

Mirror features 023 and 024's `src/effects/` layout. Pure rules go in `E/domain`, orchestration in
`E/application`, IO and the CLI in `E/infrastructure`. The model-architecture exception of Principle
IV covers `relation_attention.py` beside `effect_model.py` and `ability_encoder.py`. `build_corpus`,
`corpus_plan`, the detectors and the selection-set freeze stay torch-free, as the import-boundary
test's torch-free list requires (`tests/unit/effects/test_import_boundaries.py:~175-185`).

Java stays in `FE/`, with the directed-fork registry in a new `FE/directed/` package beside the other
fork drivers. Java tests use `ForgeExtension`, `TestBoard` and `TestCollectors`, as
`StateBasedDeathTest`, `ForkRecordsTest` and `PlayabilityChannelTest` do. Forge-side patch tests live
in `forge/forge-game/src/test/java/forge/game/**/EffectRecord*Test.java`.

The knowledge-probe scripts keep their layout and import the moved cores from `effects`, as
`scripts/rules_catalogue.py` will. One deviation: the probe cores move from `scripts/` into the
package, because `src/effects` cannot import `scripts/` and training scores them every epoch.

### Third-instance check

| Sub-problem | Instances | Extraction |
|---|---|---|
| Fork run skeleton (asFork, spend budget, copyAndVerify, map objects, snapshot builder, run, catch `RuntimeException \| StackOverflowError`, emit with flags) | `InterventionFork.intervene`, `DamageStepProbe.probe`; directed forks third | `Forks.ForkRun` template |
| Candidate enumeration with verdict | `RandomSeatController.playableCandidates`, the AI playability hook path; FR-033 third | `FE/PriorityCandidates.java` |
| Targeting-clause walk | `PlayabilityVerdict.hasLegalTargets`, `DecisionCandidate.legalTargetsOf`, `RandomSeatController.targetingClausesOf :397-430` | `FE/TargetingClauses.java` |
| Keyword normalize and strip | `ProbeScheduler.normalizeKeyword`, `DamageStepProbe.stripKeyword`; the keyword operation third | `FE/directed/KeywordEdit.java` |
| Stable seeded ordering | `corpus_curation.game_hash :246`, `record_hash`, `validation_samples._sample_hash`, `admit_card_disjoint`'s shuffle, the probes' `common.stable_hash :45`, `sweeps.SweepSelector :297` | `E/domain/seeded_draw.py` (`stable_order(ids, seed)`, `smallest_k`) for placement, halves and the freeze subsampling |
| Canonical JSON digest | `E/domain/canonical_digest.py:14` (blake2b-16), `probe_set.canonical_digest` (sha256) | one module with both variants; the selection set and the map snapshot use it |
| AUC, R², rescaling | `scripts/_lib/stats.py:49`, `ladder.score`, `linear_probes.py:122`, `colour_pips.ranking_auc :179`, `ladder_check.within_record_auc :56` | `E/domain/score_scales.py` |
| Record → label tagging | `rule_families`, `qualifying_observations`, `lethality qualify`, probe `labels.py`, grid qualifiers | detectors become the shared tagger; the keyword-combat and lethality qualifiers are ported onto it |
| Rule outcome functions | `lethal`, `step_deaths`, `may_block` | `E/domain/rule_outcomes.py` |
| Per-field constant-prediction loss | `loss_floor.constant_predictor_floor`, `Validator` field parts, `ablation.field_losses :166-195` | generalized `loss_floor` |
| Chunked forward over records | `model_loading.forward_chunks :98`, `common.predict_all :259`, `Validator.score :265` | one helper in `E/application/record_forward.py` |
| Head output name list | `HeadField`, `rules_catalogue.HEAD_OUTPUTS`, the manifest map snapshot | derived from `HeadField` |
| Scenario and heads schema | the validator script and the new Python loader | `E/domain/rule_catalogue.py`, imported by the script |

Held-until-a-later-event records (`ActivationLedger`, `HeldResolutionWindow`, `ProbeQueue`, now the
attackers hold) differ enough that no generic class is extracted.

## Decisions

### Catalogue and detectors

#### R1. Detectors are parameterized families, registered by the scenario's declaration

**Decision**: each scenario gains a `detector` object naming a detector family and its parameters
(contracts/catalogue-extensions.md), or `{"none": reason}`. Families live in
`E/domain/rule_detectors/`, one module per family: zone move inside the state-based loop, trigger
fired by `Mode$`, playability verdict bit, keyword in combat, continuous contribution, rewrite by
`Event$`, legality list membership, and a few others as the catalogue needs them. A family takes one
record plus its sidecar lines and returns `RuleTag(rule, polarity, entities, scenario, observed)`.

- Polarity comes from the board: which variant's situation the record matches.
- `observed` is whether the scenario's outcome happened in the record.
- A record whose polarity and observed outcome disagree goes on the review list (FR-004) and is not a
  rule-test record until a scenario edit or an accepted departure makes them agree.
- A detector lists in `needs` the gen-3 fields it reads; on a gen-2 record such a detector reports
  "not countable on gen-2" (FR-007).
- Several scenarios of one rule may tag one record. The rule's tag takes `fires` if any scenario says
  so, else `does_not_fire`, else `other`. Entities are the union.
- Tags are per rule id (702.2b), not per unit (702.2). `rule-counts` prints a unit's rules together.

**Rationale**: 1763 scenarios over 1561 rules cannot each get a bespoke function, and most share a
handful of shapes (the survey counted `record_kind` resolution 636, playability 414, trigger 312,
continuous 216, rewrite 93, combat 92). The existing qualifiers already have these shapes.

**Alternatives considered**: one function per scenario (rejected: unmaintainable at this size);
detectors as data-only predicates in the scenario JSON (rejected: event derivations such as "a death
inside the state-based loop after this resolution" need code).

#### R2. Rule copies are per record, after the natural share, bounded by a per-record reuse cap

**Decision**: `plan_selection` (`corpus_plan.py:376`) stays the natural share. A new pure step,
`E/domain/rule_budgets.py`, runs per class after it:

1. share = class target ÷ number of budgeted rules tagged in the class; each polarity half = share ÷ 2;
2. natural written per (rule, polarity, source) = Σ natural copies over its tagged records;
3. each half is filled separately, never redistributing between halves;
4. added copies of a record ≤ `reuse cap` − its natural copies, so a record's total copies never exceed
   the reuse cap; `--text-cap` still bounds a text's total;
5. each added copy goes to the rule of the record furthest below its half, greedily, in seeded record
   order.

The survey keeps, per tagged training record, (cell, keys, record hash, tags, natural copies). The
write pass receives a per-record map `record hash → (added copies, drawn rule per copy)`, sharded by
the same game hash `_repack_outputs` uses so each writer loads only its part.

**Rationale**: FR-054's example (2,000 records × reuse cap 4 = 8,000) reads as a per-record cap.
`allocate` water-fills between members, which FR-054a forbids between polarity halves, and needs
disjoint members, which multi-tagged records are not. A greedy pass is simple and deterministic.

**Alternatives considered**: per-text copy plans like gen-2's (rejected: a rule's records span many
texts, and "furthest below its share" is per record); reusing `allocate` for the halves (rejected: it
redistributes a short half).

#### R3. Placement runs once, globally, after the survey, steered by per-game rule inventories

**Decision**: the survey returns, for every candidate game (Forge-AI only, any natural record), a
small inventory `game → {(rule, polarity): n}` over budgeted rules, rule-test-eligible records only.
A pure `E/domain/stratum_placement.py` runs after `merge_surveys`:

- Game-disjoint: start from gen-2's crc32 draw. For each (rule, polarity, half) below the precision
  target's record count, promote unplaced games carrying it, smallest game hash first.
- Card-disjoint: games are fixed by the text holdout; placement orders admission so under-filled
  (rule, polarity) cells go first, under the per-text cap and the held-out-resolution requirement.
- Halves are assigned per game within each stratum, greedily balancing every (rule, polarity) count,
  with at least 300 games in each half (Decision E5); ties go to the half with fewer games, in
  `stable_order(game_id, seed)` order. The manifest records the halves, and the freeze reads them.
- A rule left out of a half is named in the manifest.

The training supply is kept per game for a candidate pool (games whose game-disjoint draw is below
ten times the share, plus every card-disjoint candidate), and folded after placement. Games outside
the pool fold in the shard worker as today.

A game counts as a random-seat game when any of its records carries `random_seat = true`; games never span
shards, so the survey decides this per shard.

**Rationale**: FR-068a's goal needs a global view. A per-game inventory is small. A pool avoids a
third read of a corpus of the order of 10⁷ records.

**Alternatives considered**: an inventory-only pre-pass (rejected: a third full read); keeping the
hash threshold and accepting holes (rejected: FR-068a requires both polarities in both halves as far
as records allow).

Consequence: placement is no longer a pure function of the game, so a rebuild over a grown corpus can
move a game. The `GameRouting` docstring (`corpus_plan.py:486-490`) and `src/effects/CLAUDE.md` are
corrected.

#### R4. The map snapshot filters outputs to the heads the model has

**Decision**: `build-corpus` reads `rules/heads/` once through `E/infrastructure/rules_catalogue_io.py`.
For each tagged rule it records `{verdict, outputs}` in the manifest, `outputs` keeping only names in
the derived head-output list (untagged names; `new:`, `record:` and `planned:` entries are dropped
until the map's rerun lists them as existing outputs). The hash is the canonical digest of the parsed
lines.

**Rationale**: 182 trained lines mix existing outputs with tagged future ones. The mask and the rule
tests can only read outputs that exist.

#### R5. Protection copies are written by `build-corpus`, in the lethality copies' shape but not through their code

**Decision**: `E/domain/protection_copies.py` (torch-free, beside `lethality_counterfactuals` in the
domain layer, since both functions are pure rules) follows `lethality_counterfactuals`' two steps: a
pure qualifier (`qualify :301`) deciding whether a real record can carry the copy, then a copy function
(`counterfactual :448`) editing the record from a seeded generator. Only the write pass that calls it
is application code. What it reuses and extends:

- the outcomes come from `rule_outcomes`, extended with rule 702.16's: damage prevented, a block or
  target made illegal, an Aura put into the graveyard. The block outcome extends `may_block`'s
  protection clause (`scripts/effect_knowledge_probes/evasion_grid.py:63`, moved by the
  third-instance extraction), which knows only protection from a colour, to a creature type, a card
  type, a condition and everything;
- the grant is the edit `evasion_grid.edited :106` makes through `BoardEdit.grant_keyword`
  (`board_edits.py:81`), written as a granted keyword line plus its kept flag (Decision M5);
- the copies are written by the write pass beside the record's natural copies
  (`_ShardWriter._train`, `E/application/corpus_write.py:186-205`), as many as `rule_budgets` draws for
  the protection rules, each with `synthetic = true`, its `drawn_rule` and its `curation` (Decision T1).

**Rationale**: two behaviour gaps keep `counterfactual` and `StepPreparer._counterfactuals` from
serving.

- Lethality copies exist only in memory. `StepPreparer._counterfactuals`
  (`E/application/step_preparation.py:321-346`) draws them afresh at each training step and relabels
  the derived `EntityTargets`, which never reach a shard. A protection copy is written into the
  corpus, counted by the rule budgets and the manifest's `RuleRow`, and enters the validation strata
  (root spec § 12.4). Its labels must come out of `derive_targets` (`E/domain/effect_targets.py:139`),
  which reads the payload's events, so the copy rewrites the payload rather than the targets.
- The edits differ. A lethality copy changes an amount in the acting text, under a `counterfactual/`
  source-tree key the batcher memo carries for one step (`_edited_text :402-423`), and one toughness
  (`_with_toughness :426`). A protection copy leaves the acting line's text alone and grants a keyword
  line to an entity, so neither edit applies.

The two names do not collide, so Principle VII(c) asks for no rename.

**Alternatives considered**: drawing protection copies per training step inside
`StepPreparer._counterfactuals`, beside lethality copies (rejected: the per-rule amounts, the
`drawn_rule` the hard mask reads and the validation strata's copies are all fixed by `build-corpus`,
Decision T1).

### Collection

#### C1. Three engine hooks are added on `effect-record-hooks`

**Decision**: new commits on the branch add:

1. a pending-delayed-trigger accessor on `TriggerHandler` (FR-024, the `pending on` relation, and
   Decision D3);
2. a state-based-loop begin/end signal around the loop in `PhaseHandler.checkStateBasedEffects`
   (`:1168-1177`) (Decision C2);
3. a priority-settled signal in `mainLoopStep`, after the state-based loop and before the priority
   holder chooses (Decision C3).

Each joins `PatchHooks.REQUIRED`; an unpatched Forge still runs and stamps records `degraded`.

**Rationale**: none of the three can be observed through public API or existing hooks (survey §2 of
the collection side). The user's standing rule allows new commits on the branch.

#### C2. The held effect half spans the whole state-based loop

**Decision**: the held half opens at a resolution's end as today and stays open until the end signal
of the state-based loop that follows. Every death between the begin and end signals is filed in it,
whether the tracker is frozen or not. A trigger put on the stack between passes
(`addAllTriggeredAbilitiesToStack`, tracker unfrozen) opens its own bracket without closing the held
half; `beginBracket` no longer calls `onNextAction()` inside the loop. The held half is written at the
end signal. A death inside the loop with no held half and no open damage step is dropped and counted
under `--count-dropped-events`, instead of going to the open bracket as `brackets.filing()` routes it
today.

**Rationale**: the survey found that triggers never reach the stack while the tracker is frozen; they
arrive between passes, and `checkStaticAbilities` also freezes the tracker (`GameAction.java:1084`).
The loop boundary is the mechanism the spec's intent needs. Spec correction S1(1).

#### C3. Decision candidates come from a collector-side enumerator at the priority-settled signal

**Decision**: at the priority-settled signal, `PriorityCandidates` enumerates both players' candidates
(spell abilities of every available card, both seats) and computes each verdict from the rules:
`canPlay`, `canCastTiming`, the per-check methods, the static prohibitions with their responsible
static, the existing affordability path, and every targeting clause through `TargetingClauses`.
`can_play` is `canPlay` and `canCastTiming` and no static prohibition. A check Forge evaluates alone
is reported under its own identifier; `checkOtherRestrictions` reports `script_restriction`.
`legal_targets` is the union over clauses with no cap. The random seat reads the same enumerator.
The AI playability hook no longer produces decision records.

The verdict must not write to the game. `cantBeCastAbility` calls `card.setCastSA(spell)` and
`setActivatingPlayer` writes too, so the enumerator saves and restores both around each check.

Land plays are candidates too (FR-033a, user decision 2026-11-02): each land ability of an available
card (`SpellAbility.isLandAbility()`), marked `land_play`, its verdict `Player.canPlayLand` split
into `whose_turn`, `casting_speed`, `zone`, `static_prohibition` (`cantPlayLandAbility`) and a new
`land_play_limit` (`getLandsPlayedThisTurn() >= getMaxLandPlays()`, never under
`getMaxLandPlaysInfinite()`). The limit gets its own identifier rather than `activation_limit`,
which is `getLimitToCheck`/`getGameLimitToCheck` on one ability. Mana abilities stay out. The random
seat filters land-play candidates out of its draw, keeping gen-2's rule that the AI makes land drops
(`RandomSeatController.landDropPending`).

**Rationale**: in the patched `canPlayAndPayFor` (`FA/AiController.java:840-868`), a spell returns
`CantPlaySa` before the hook when `canPlayFromHost()` is null. The hook's list is pre-filtered (lands,
`RemAIDeck`, storm), and it fires for the priority holder only. FR-030 and FR-033 cannot be met
through it. Spec correction S1(3).

#### C4. Floating effects are reified by collector bookkeeping at clause resolution

**Decision**: effect cards in the command zone already carry a source and `Duration$`. For "until end
of turn" changes that live only as timestamped rows on a card, the clause handler records, at
`ON_CLAUSE_RESOLVING`, the timestamp Forge assigns, the line resolving and the clause's `Duration$`.
A snapshot builds one command-zone object per (line, timestamp) still present on some card. A
counting-down shield's amount reads the per-API SVar the effect card carries.

`FE/FloatingEffects.java` extends three existing pieces rather than walking the board a second way:

- The layer tables are the ones `LayerContributions.addCard` (`FE/LayerContributions.java:43-69`)
  already reads, Forge's `(timestamp, static id)` tables (`FG/card/Card.java`: `getPTBoostTable :4692`,
  `getChangedCardKeywords :4141`, `getChangedCardTraits :5066`) plus the patched type, colour and name
  accessors. `LayerContributions` files a row by its static id, and `ContinuousChannel.record`
  (`FE/ContinuousChannel.java:79-83`) drops static id 0, the rows a resolving ability writes.
  `FloatingEffects` files exactly those rows by their timestamp. Granted lines come from the same
  `getChangedCardTraits` walk `EntityJson.grantedTemporaryAbilities :166` makes.
- The timestamp is noted in `ClauseOutcomeRouter.clauseResolving` (`FE/ClauseOutcomeRouter.java:87-103`),
  the `ON_CLAUSE_RESOLVING` handler that already pushes one observation per clause. Forge draws it from
  `Game.getNextTimestamp` (`FG/Game.java:539`) as the effect runs (`PumpEffect.resolve`,
  `FG/ability/effects/PumpEffect.java:278`; `SpellAbilityEffect.createEffect :630` for an effect card).
  The duration reads the clause's `Duration$` and falls back to the root line's, as
  `EventAttribution.declaredDuration` (`FE/EventAttribution.java:203-215`) does. It is bucketed into a
  vocabulary of its own, because `EventDuration` has `instant` and no `until_leaves`.
- An effect card already in the command zone is an entity of the snapshot
  (`SnapshotBuilder.collectEntities :173-184`) and resolves to its line through
  `ProvenanceResolver.effectSource` (`FE/ProvenanceResolver.java:331-334`). `SnapshotBuilder.emblems`
  (`:209-217`) is the command-zone list the new one sits beside.

**Rationale**: the `GameCommand` closures in `game.getEndOfTurn()` are opaque, and `staticId = 0`
names no line. The timestamp is the only join.

#### C5. Lines of a permanent, keyword lines and base power

**Decision**: `printedKeys` reads the live trigger, static and replacement lists beside spell and
activated abilities, keyed through `ProvenanceResolver.of(CardTraitBase)`. Each static carries
`active = checkConditions()`. Keyword lines come from `getKeywords()`, each from its instance's
fragment. `grantedAttached` is removed: granted lines are already in the host's live lists, flagged
granted. `ptJson` is rewritten: base is `getCurrentPower/Toughness()` (set base, layer 7b), boosts are
the temporary boost plus every counter's contribution.

#### C6. Attackers records pivot to one per attacker; `must_block` is re-derived

**Decision**: the per-defender legality loop (`FG/combat/CombatLegalityObserver.java`, called at
`PhaseHandler.java:538`) is pivoted in the collector: one record per attacker, with
`legal_defenders` and `must_attack` from the attack constraints initialized just before the hook. A
declared anchor's record is held until `GameEventAttackersDeclared` and carries `tapped_by_attack`
from the `GameEventCardTapped` events before it. Since combat legality is reported only from the
turn-based declaration, no what-if attackers record exists; FR-037's what-if clause has nothing to
apply to (spec correction S1(4)). `must_block` is the collector-side `MustBlock` predicate.

#### C7. Turn-based actions open their own window

**Decision**: a `turn_action` window opens at `onPhase` for the draw, untap, cleanup and
beginning-of-main phases and closes at the next `onPriority` or `onPhase`. Today these events fall
into `StackBrackets` orphans (`FE/StackBrackets.java:83-104`). Phasing gains an `EventType` and a
`BusEventReader` case for `GameEventCardPhased`.

#### C8. The remembered-list rule

**Decision**: an entry of a card's `getRemembered()` is a relation only when the snapshot is taken
while no ability of that card is on the stack or resolving, and the entry is a card or a player.
Entries read at that moment have survived their resolution's `ClearRemembered$` cleanup, so they are
the lasting ones.

**Alternatives considered**: a whitelist of scripts whose remembered list persists (rejected: it
would need upkeep for every new card).

#### C9. The random seat skips a free activation that changed nothing, until the turn ends

**Decision**: `HeldResolutionWindow.close` reports an empty effect half for an activation whose cost
half paid no mana and moved no card. The seat keeps a per-turn set of (card, ability key) and
filters them from `playableCandidates`. The set clears at the next turn.

### Directed forks

#### D1. A directed rule is a Java class with a precondition and per-variant applications

**Decision**: `FE/directed/DirectedRule` registers by rule id. `precondition(Game)` returns an
optional binding (the subject objects the variants act on). Each classified variant has
`apply(Game fork, Binding)` using one `ForkOperation`. A JUnit contract test reads
`rules/scenarios/*.json` and checks that every registered variant exists with the same operation
classification.

At `match-outcomes` startup the supervisor checks the Python side (detector, classified variant) and
runs one short JVM query, `DirectedRulesMain`, which prints the registered rule ids. A rule missing
from either check exits 1 (FR-014).

**Rationale**: a boolean predicate is not enough. The operation needs the bound subject and value,
and scenario deltas are prose (`rules/README.md:62,200`). The registry lives in Java, so the
supervisor has to ask the JVM.

#### D2. Operations edit the board silently where Forge allows

**Decision**:

| operation | edit |
|---|---|
| power or toughness | `addNewPT` (set) with a fresh timestamp |
| keyword | `KeywordEdit` through `addChangedCardKeywords` |
| colour or type | `addColor`, `addChangedCardTypes` |
| counters | `setCounters` (silent) |
| tap or untap | `setTapped` (silent) |
| life total | `setLife` (fires life-gain and life-loss triggers; Forge has no silent path) |
| replace a card | `CardFactory.getCard`, raw `Zone.remove/add`, carrying controller, tapped state, summoning sickness, damage, counters, attachments and combat status to the new card |
| remove a permanent | `Combat.removeFromCombat`, unattach what is attached to it and what it is attached to (`unattachFromEntity`), raw `Zone.remove` from the battlefield; the card goes to no zone |
| add a permanent | `CardFactory.getCard` from the card database under the variant's owner, `setGameTimestamp` fresh, raw `Zone.add` to the controller's battlefield, then `setTapped`, `setSickness` and `setCounters` as the variant states |
| move a card | raw `Zone.remove` from one of hand, library and graveyard and `Zone.add(card, index)` to another (a library position from the top), or `CardFactory.getCard` for a new card put at the old one's position, the old one going to no zone |

`replace_card` is `remove_permanent` then `add_permanent`, carrying the old card's state, and shares
their code. Every operation ends with `fork.getAction().checkStaticAbilities()`. None goes through
`GameAction.moveTo`, which would fire zone-change triggers, replacements and the per-turn counts
(cards drawn, permanents to the graveyard, lands played) a variant does not change.

**Rationale**: a scenario variant describes a different board, not an event on the board. A silent
edit gives the minimal pair the scenario names; firing triggers would add changes no variant states.
The three board-membership operations exist because the first classification pass found most
variants add or remove a permanent or change a card in a hand, library or graveyard (user decision,
2026-11-02).

**Alternatives considered**: forcing a choice, a target or a timing on a seat inside the fork
(rejected by the user for gen-3: each kind of choice is a separate AI callback to override, and a
timing variant changes when to fork rather than the board; such variants stay unclassifiable).

#### D3. `GameCopier` carries delayed triggers and end-of-turn commands (user decision)

**Decision**: a commit on `effect-record-hooks` makes `GameCopier` copy pending delayed triggers (the
accessor of Decision C1), the end-of-turn and until-leaves commands, `ExiledWith`, attacked this turn,
this-turn casts, counters added this turn and controller timestamps, remapping card references to the
copy. Forge-side tests check each on a copy.

**Rationale**: a fork played to the end of the turn would otherwise drop end-step delayed triggers
such as Sneak Attack's sacrifice. Chosen by the user over accepting the loss or stopping forks early.

#### D4. A fork is played forward with its own collectors and its own seats

**Decision**: `ForkPlayForward` loops `fork.getPhaseHandler().mainLoopStep()` until the turn number
changes or the game ends, under the existing fork exception guard. During that span, hook handlers
dispatch by `Game` identity to a fork-bound `PatchedCollectors` and `BusBracketCollector`; the fork's
recorder schedules no intervention, probe or directed fork of its own. A random seat's controller in
the copy is attached to the fork with a `Random` seeded from the per-game fork seed, not drawn from the
match source.

The same driver plays on the interventional forks of FR-058 that grant protection, which keep their
own flags. FR-057's forks resolve the stack and stop, under the existing fork rules.

**Rationale**: hooks are JVM-global `volatile` slots (`FG/ability/EffectRecordHook.java`), and the AI
evaluates candidates on a separate thread (`FA/AiController.java:1642-1731`), so a thread-local
dispatch would miss fork callbacks. `GameCopier.clonePlayer` keeps the lobby-player subclass but never
attaches the new controller, so the seat would otherwise play as the plain AI. Drawing the seed from
the match source would shift later games' seats. Spec correction S1(2): interventional forks resolve
the stack and stop; they do not play the turn on.

Forge's AI decisions in a fork still draw from the shared `MyRandom`, as interventions already do.

#### D5. `mirror_of` on a directed-fork record

**Decision**: a directed-fork record carries `mirror_of` only when a live record was open at the fork
point. A fork taken at a phase boundary usually has none, because the live turn's records are written
after the fork ends. Directed-fork records are paired with the live turn by `game_id`, turn and
`directed`.

**Rationale**: FR-009a says "where one exists". Holding the fork's records to the live turn's end for
a join would add a held-record queue for a field no consumer reads.

### Model

#### M1. Clean break with gen-1 and gen-2 checkpoints (user decision)

**Decision**: gen-3 code loads gen-3 checkpoints only. `EARLIER_LAYOUT`, the gen-1 `option_kind`
compatibility path and the `e_noise` shim are removed. A checkpoint without
`training_settings["input_layout"] = "gen-3"` is refused with exit 1 by every loader.

**Rationale**: head widths are module constants and `load_weights` refuses size mismatches
(`effect_model_store.py:130-140`). The spec already refuses gen-2 records everywhere but `rule-counts`,
and gen-2's numbers live in the experiment records.

**Alternatives considered**: versioned output layouts (rejected by the user: every layout change would
be written twice).

#### M2. `RelationEncoderLayer`

**Decision**: `E/domain/relation_attention.py` holds a pre-norm layer with the stock parameter names
(`self_attn.in_proj_weight`, `self_attn.out_proj`, `linear1`, `linear2`, `norm1`, `norm2`) so a stock
`TransformerEncoderLayer` state dict loads into it. Per layer it holds `rel_k` and `rel_v`, each of
shape `(heads, types, head_width)`, zero-initialized; under `--relation-values off` `rel_v` is a
buffer.

Forward, per layer and head, with relations as COO `(b, i, j, r)`:

1. `scores = q kᵀ / √d`, explicit (no fused kernel).
2. Key term: `qK = einsum(q, rel_k)` gives `(B, H, S, T)`; `scores[b, h, i, j] += qK[b, h, i, r] / √d`
   by `index_put_` with accumulation over the COO rows.
3. Padding mask, softmax, attention dropout → `a`.
4. `out = a v`; value term: `W[b, h, i, r] = Σ a[b, h, i, j]` over the COO rows (`index_add_`), then
   `out += einsum(W, rel_v)`.

Relations are emitted in `SurfaceBuilder` after context dropout and gate-2 stripping, and pass through
`collate_surfaces` as a `(N, 4)` long tensor with batch offsets.

**Rationale**: the scatter form never builds a per-pair vector (FR-048). The explicit score matrix at
the survey's measured slot counts (p99 195, max 232; gen-3 likely about 350) costs under 1 GB over 6
layers at batch 32, beside a measured gen-2 peak of 2.9 GB. `line of the same card` and
`has the same name` grow quadratically in triples; their count is part of the US4 measurement.

The equivalence test runs the stock layer in train mode with dropout 0, or with
`torch.backends.mha.set_fastpath_enabled(False)`, because the eval fast path uses a different kernel.

#### M3. Target-index cap

**Decision**: `targets` has 4 index types per direction. Target index 3 and above share the last type.
The cap is recorded in the checkpoint as `relation_target_indices = 4`.

#### M4. Relation type table

**Decision**: `E/domain/relation_types.py` enumerates the § 10.3 families, each as two types (forward
and reverse), plus `targets` × 4 per direction. The enum order is frozen in the checkpoint as a list
of names, and a loader refuses a checkpoint whose list differs.

#### M5. Keyword flags follow a checked-in scan

**Decision**: `scripts/scan_keyword_flags.py` scans Forge's card scripts for selectors naming a
keyword and its engine code for `hasKeyword(...)` checks, and writes `E/domain/kept_keyword_flags.py`,
checked in. Gate 2's strip, the combat grid's grants and the evasion grid's flags edit both the flag
(where kept) and the granted keyword line. The eight damage-step keywords and the evasion keywords are
engine-checked by name, so they keep their flags. The scan's test fails if one of them would lose its
flag.

**Rationale**: removing the multi-hot would otherwise break gate 2's strip and both grids, which edit
keywords through that channel.

### Training

#### T1. Each copy carries its curation, written by `build-corpus`

**Decision**: each written record gains a `curation` member, curated-only, never in collected shards:
`{natural_share: n, copies: d, drawn_rule: rule | null, drawn_rules: [...], rule_entities: [...]}`.
`drawn_rule` is set on added copies only; `drawn_rules` lists every rule any added copy of the record
was drawn for. The surface batcher turns it into per-entity, per-field loss weights in
`entity_target_tensors`, and into the gate's per-row mask.

**Rationale**: the hard mask needs the copy's rule and entities; the soft mask needs n, d and every
drawn rule. Copies are planned per (cell, text), so none of this can be derived at training time.

#### T2. Negative binomial everywhere a Poisson is used today

**Decision**: every count and amount term moves to the negative binomial: per-entity counts and
signed-delta magnitudes, created-objects counts, the verdict head's cost per colour, and the value
head (which shares `FIELD_KINDS`). Each field has one learned log-dispersion scalar in the `head`
group. The constant prediction fits mean and dispersion by moments, then Newton steps on the
dispersion. Gate 1 reads the negative binomial deviance, each run with its own dispersion.

**Rationale**: one `FieldKind` serves all of them. Keeping a Poisson on some would give them a
different loss scale under FR-062.

#### T3. Loss denominators come from a startup pre-pass

**Decision**: at startup `train-effect-model` computes each field's constant-prediction loss on a
seeded sample of training shards, weighted as the run's `--rule-mask` weights the field, clamps it at
`--deviance-floor`, and stores the result in `resume.pt` and the checkpoint. A resume reuses it.

**Rationale**: `build_corpus` must stay torch-free, and the floor exists today on the validation
sample only.

#### T4. The averaged weights step per optimizer step

**Decision**: `WeightAverage` covers the encoder, the model and the identity table. It updates after
each optimizer step with rate `grad_accum / (steps_per_epoch × ema_epochs)`, because
`--steps-per-epoch` counts batches. It starts from the weights at the end of warmup. An epoch whose
last step is at or after warmup's end is validated on the average and is a best candidate.

#### T5. The curriculum switch resets the best score

**Decision**: as in gen-2, the stopper resets at the curriculum switch. The control score's best and
patience restart there, because the deviance component's field set changes at the switch.

### Selection and evaluation

#### E1. Probe, grid and rule-test scoring move into the package

**Decision**: the rung-3 readouts, item and grid scoring, rule-test scoring, the frozen selection set
and its freeze move into `E`: `domain/probe_readouts.py`, `domain/score_scales.py`,
`domain/rule_outcomes.py`, `domain/selection_set.py`, `application/freeze_selection_set.py`,
`application/control_score.py`. The scripts import them. The freeze needs no checkpoint and is run by
`scripts/effect_knowledge_probes/run.py --freeze-probe-set --corpus DIR`. The combat and evasion grids
read frozen records instead of scanning 200 shards.

`train-effect-model` finds the frozen set by the corpus digest under `output/effects/reports/` and
exits 1 at startup when it is absent. This startup check is added to FR-076 (spec correction S1).

**Rationale**: `src/effects` cannot import `scripts/`, and training scores these every epoch.

#### E2. In-training scoring cost and precision

**Decision**: at startup, training prepares the host side of every selection-half item once (probe
records, grid edits, rule-test records, deviance samples) and caches it; each epoch runs only the
device forward on the averaged weights. Selection and reporting both run under the same bf16 autocast
as validation, through the same scorer. The reporting half is scored once, on the selected
checkpoint, by `evaluate-effect-model` and the probes.

The deviance component reads a seeded sample per half of each stratum's written records, copies
included, sized to meet the precision target and capped at 16,384 records per half. The cap and the
in-epoch scoring time are checked against the US4 measurement; the target is selection costing at
most a fifth of an epoch's training time.

**Rationale**: the survey estimates about 27,000 rule-test records per stratum at gen-2's rate of
roughly 9 minutes, against 32-minute epochs, before caching. Scoring selection and reporting under
different precisions would make the halves incomparable.

#### E3. The balance ratio r is 4

**Decision**: r = 4, recorded in the frozen set.

**Rationale**: with the smaller side fixed at n, an AUC's variance relative to an unbounded larger
side is (1 + r) / r: 1.5 at r = 2 and 1.25 at r = 4. At r = 4, the 3,000-record probe types are cut to
about 900 to 1,000 records against the smallest (about 220), which keeps their standard error near
0.04. Balance changes no expected score, since AUC does not depend on prevalence and components
average items equally, so the looser ratio costs nothing but emphasis.

#### E4. Worst-case standard errors

**Decision**: all estimators are computed at the freeze from labels alone, and every one clusters by
game.

| item | estimator, on the rescaled scale |
|---|---|
| rule test, AUC probe | Hanley–McNeil at AUC 0.5, × 2: `2 √((n₊ + n₋ + 1) / (12 n₊ n₋))` (the spec's) |
| grid | `(1/C) Σ_c 1/(2√N_c) / B`, with `B = (1/C) Σ_c |ȳ − y_c|`, `y_c` the cell's rule outcome and `ȳ` its mean over cells; B = 0 makes the standard error infinite, so the item is left out |
| output field, R² probe | loss ratio: `√(Σ_g S_g²) / Σ_g S_g`, where `S_g` sums `w_i ℓ_i` over the game's records, `ℓ_i` the constant prediction's loss and `w_i` the record's weight in the component |

For an output field, the freeze computes the standard error under both mask modes and records both;
training reads the one for its `--rule-mask`.

**Rationale**:

- Grid: each cell mean of predictions in [0, 1] has a standard deviation of at most `1/(2√N_c)`. The
  mean absolute error is 1-Lipschitz in each cell mean. Averaging the bound over cells covers fully
  correlated cells, which edit grids are, since every cell edits the same records. The two combat-grid
  variants whose rule is constant (deathtouch, indestructible) have B = 0 and drop out by the same
  rule.
- Loss ratio: explained deviance is `1 − L_model / L_const`. Its worst case is a model whose
  per-record loss differs from the constant's by up to the constant's own loss. The root mean square
  form does not collapse at a balanced binary field, where the constant loss is the same on every
  record, as a coefficient-of-variation form would. Clustering by game covers the correlation of
  records within a game.

#### E5. Halves and gen-2's set size

**Decision**: each half of a stratum's probe set holds at least as many games as gen-2's whole set
(300 games per stratum), so each stratum holds at least 600 games. Halves are assigned by placement
(Decision R3) and recorded in the manifest; the freeze reads them and assigns none, and exits 1 naming
the stratum and half when a half holds fewer than 300 games. Line probes (`items`, `pairs`) read training records, are reported
only, and never feed a stratum score.

The thresholds (300 games per half, 0.1 per item, 0.05 per component, r = 4) are keyword arguments of
`stratum_placement`, `selection_set` and `freeze_selection_set`, with these values as defaults, so unit
tests pass small values. No operator flag overrides them. `build-corpus` writes the corpus and its
manifest before any precision exit, so a pilot corpus, which cannot meet the thresholds, exits 1 and
is still read (user decision).

#### E6. Rule-test entity and output readout

**Decision**: a rule test reads, for each output the manifest's map snapshot names, the readout of
`probe_readouts` at the tag's entities (the mean over entities when the outcome names several), or at
`[ACT]` for a verdict-head output. A categorical output reads the probability of the scenario's
category, a count or delta its expected value under the negative binomial.

### Spec corrections

#### S1. Corrections carried into `spec.md` and the root spec (user decision)

Made in this pass, after the plan artifacts:

1. FR-028 / root § 8.7: the held half spans the state-based loop between its begin and end signals,
   every pass included. Triggers reach the stack between passes and open their own bracket without
   closing it (Decision C2).
2. FR-010 and its clarification / root § 6: interventional forks resolve the stack and stop.
   Directed forks, and the interventional forks of FR-058, are played to the end of the turn by a new
   driver, on a copy that carries the live game's delayed triggers and end-of-turn effects (Decisions
   D3, D4).
3. FR-030, FR-033 / root § 8.8: decision candidates come from a collector-side enumerator at a
   priority-settled hook, for both players; the AI's playability hook no longer produces them
   (Decision C3).
4. FR-037 / root § 8.11: combat legality is recorded only from the turn-based declaration, so no
   what-if attackers record exists; the what-if clause goes.
5. FR-068a / root § 14.2: the cell-balance rule applies to grids whose cells hold different records.
   An edit grid edits the same records in every cell. A grid whose rule outcome is constant over its
   cells is left out by the precision target.

Also carried: FR-076 / root § 15 gains `train-effect-model`'s startup check for the frozen selection
set (Decision E1), and `rule-counts` gains `--cards-folder`, `--variant-scripts` and `--rules-dir`
with defaults, because detectors read sidecars and the catalogue.

## Constants

| constant | value | where recorded |
|---|---|---|
| balance ratio r | 4 | frozen selection set |
| target-index relation types | 4 per direction, index ≥ 3 sharing the last | checkpoint `relation_target_indices` |
| deviance sample cap | 16,384 records per half per stratum | frozen selection set |
| probe-set games | ≥ 300 per half per stratum | frozen selection set |
| item standard-error ceiling | 0.1 | spec (FR-068b) |
| component standard-error ceiling | 0.05 | spec (FR-068b) |
| placement candidate pool | game-disjoint draw below 10 × the share | `stratum_placement.py` |
| loss-denominator pre-pass | 8 training shards, seeded | checkpoint and `resume.pt` |
