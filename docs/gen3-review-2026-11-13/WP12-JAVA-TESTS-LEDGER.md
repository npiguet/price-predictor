# WP12 Java-test ledger (2026-11-14)

Paths relative to `forge-connector/src/test/java/com/pricepredictor/connector/` unless stated.
Run Maven through `WP12J-mvn.sh "<tests>"` (lock + reap built in).

## Bugs

- N1 (converter, saga chapter keys): FIXED. Cause: `ProvenanceRecorder.attributeToKeyword` claimed every keyword-derived trait on EVERY line the keyword rendered (`byKeyword`). Chapter (230 sagas) and Haunt (10 cards) render one line per generated trait. Fix: renderers tell the recorder which line each trait printed (`ProvenanceRecorder.rendered`, `ChapterAbility.fromKeyword(ki, rendered)`, `HauntKeyword.of(ki, card, rendered)`); a noted trait is claimed by its own line alone. Tests (seen red): `effects/TraitScriptChainTest` eachChapterLineCarriesItsOwnChaptersTriggersAlone, aSagaOfThreeAbilitiesKeysOneTriggerToEachChapterLine, aHauntingCreaturesTwoLinesEachCarryTheirOwnTriggers, aHauntingSpellsSpellLineCarriesTheSpellAlone. Fixture sidecar regenerated: konda only (only saga/haunt among the 680; it now also carries `prose_offset`, as the current converter writes). Python xfail removed and strengthened. Other constructs: Class levels, die-roll tables, additional-cost lines carry ONE trait over several lines (legit); no other multi-trait defect.
- N2 (probe keywords default): FIXED on the Java side (spec `specs/2026-09-05-ability-effect-model.md` and Python say eight). `CollectionCaps.DEFAULT_PROBE_KEYWORDS`; property read by presence (unset = eight, blank = none). `TestCaps.caps()` / `TestCaps.unprobed()` take no probe unless a test names one (ShardRun, TestCollectors, RandomSeatControllerTest recordings switched). Python xfail removed, test compares the lists.

## TC1

DELETE
- ApiEventsTest.anEmitterThatNeverThrowsReportsNothing — deleted (sibling: anExtraTurnCarriesItsCount observes the same Alchemist's Gambit clause and asserts its event; a throwing emitter yields none). b411deb8
- ApiEventsTest.anOwnershipChangeWithNoNamedOwnerIsStillAnOwnershipChange — KEPT and strengthened: the only test of the no-DefinedPlayer fallback; now asserts owner == activator. b411deb8
- UnannouncedOutcomesTest.theNewRulesAreKeyedByRealApis — deleted, replaced by behaviour: aCreatureMadeAttackingEnteredCombat parameterised over Token, CopyPermanent, ChangeZoneAll; new anAttackerAClauseBlockedNamesItsBlocker (Block); BecomesBlocked, ClassLevelUp, UnlockDoor, Bond, Mutate, ChangeZone already had behaviour tests. b411deb8

(sibling map from a Sonnet lookup, decisions mine; commit 312d8dd4)
- BusBracketCollectorTest.aCastWritesItsCostHalfAtOnceWithNoOutcome — deleted (sibling RecordMomentsTest.aSpellIsCastFromItsHandAndResolvesOffTheStack: no outcome, link; anActivatedAbilitysCostHalf…: events []; link format pinned by every golden)
- .aResolutionWritesBothHalvesUnderOneLink — deleted (same sibling: resolved, same link)
- .aSpellGoneFromTheStackWritesNoEffectHalf — deleted (StackItemBracketTest.aCounteredSpellIsWrittenOff…, RecordMomentsTest.aCounteredSpellsFate…)
- .aFizzleWritesAnEmptyEffectHalfSayingSo — deleted (FizzledSpellTest.aFizzledAbilitysHalfHoldsNothing; for a SPELL the engine writes its move off the stack, RecordMomentsTest.aFizzledSpellsEffectHalfHoldsItsMoveOffTheStack — the hand test pinned a spell shape no engine writes)
- .aDrawnGameOutcomeNamesNoWinner — deleted (BusOutcomeEventsTest.aSimultaneousLossNamesBothLosersAndNoWinner engine; BusEventsTest.aDrawNamesNoWinner)
- .aCastGoneFromTheStackByTheEndWritesNothingMore — KEPT: only test of finishGame with the cast no longer on the stack
- .aGameOutcomeAfterTheKillingBracketAlreadyClosedIsStillWritten — deleted after moving the winner ref into BusOutcomeEventsTest.theGamesOutcomeNamesItsLoser (engine, same sequence)
- .aCreatureHitInTheFirstStrikeStepOpensACombatRecord — deleted after moving cause (and amounts) into CombatStepBracketTest.eachDamageStepsSnapshotPrecedesItsDamage (engine, first_strike + regular)
- .anOfferTheControllerRefusedIsADeclinedEffectHalf — deleted after moving events []/link into aRefusalDeliveredThroughTheHookReachesTheRecord
- StackItemBracketTest.aSpellCastAgainWritesOffItsEarlierCast — KEPT: only test of the double-cast write-off (defensive path, no engine reach)
- MidResolutionCastTest.aCastSpellThatShowsNothingWritesAnEmptyLinkedHalf — KEPT: only test of an empty linked half for a mid-resolution cast
- ModalResolutionTest.aPatchedWorkerWritesOneEffectHalfPerChosenMode — KEPT: only readable assertion of per-mode snapshot timing (life 20 then 16), one link over cost + 3 halves, cost half without option (engine aCharmsFirstModeKeepsItsOwnEvents has no cast; golden holds bytes only)
- ModalResolutionTest.aRepeatedModeActsInOneHalfPerResolution — deleted after adding optionOf [0,0,1] to StateBasedDeathTest.aDeathTouchedByTwoModesJoinsTheLastOfThemOnce (engine)
- ClauseContractTest.theHandlerReadsEachArgumentOffItsOwnPosition / aClauseThatThrewWritesNoEvent — deleted, replaced by onlyAClauseThatRanWritesItsEvent over EVERY ClauseRun (RAN positive with count 1, THREW added, moment asserted); real argument order: engine clause events (EffectEndingTheTurnTest combat_ended, RecordMomentsTest charms) + PatchHooksTest declaration check
- ClauseContractTest.aSwappedCallWritesNoEvent — deleted (input no Forge build sends)
- OutcomeVocabularyContractTest.outcomeSubjectsReachTheShardAsEntityAndPlayerRefs — deleted (OutcomeHandlerTest damagePreventedNamesItsRecipient/aCoinFlipNamesItsFlipper/restrictionChange… pin card and player refs; conversion is per element)
- PatchHooksTest.theSubAbilityHookReadsNullOutsideAResolution — see STRENGTHEN
- CostPayloadTest.anAbilityThatPaidNoLifeSaysZero — deleted (lifePaidIsTheAmountPaidNotTheAmountPrinted; anUnreadableCostRenders… pins life 0)
- EventAttributionTest.attributionAlwaysNamesOneOfTheThreeStates — deleted after moving the non-ability pointer line into theRootActingAndNoPointerAreDifferentAnswers
- StateBasedDeathTest.theLegendRulesLoserJoinsTheSecondLegendsRecord — KEPT and strengthened (only legend-rule test; asserts the Isamaru in the graveyard is the one whose death is in the half, once, and not the other)

STRENGTHEN
- ApiEventsTest.everyChoiceMadeApiHasAReadableChoice / aSecretlyChosenNumberIsStillAChoice — done-before (WP3: fresh host holding earlier choices; reading follows the API).
- anExtraTurnCarriesItsCount / askippedTurnCarriesItsCount — strengthened: new anExtraTurnCountIsTheClausesNumTurns (3), aSkippedTurnCountIsTheClausesNumTurns (2). b411deb8
- BusBracketCollectorTest.theEffectHalfCarriesRefsReadAtResolution (hand-added target) — not done; engine refs are pinned by ChainTargetsRecordTest and record-resolution golden (WP4)
- BusBracketCollectorTest cause tests (CONSEQUENCE branch) — not done (both branches stamp through EventAttribution.stamp identically); wither: done-before (DamageTypeTest, engine combat)
- aRefusalDoesNotOutliveItsBracket — not done (partly; hand-driven)
- OutcomeOfTest.losingSomeTargetsButNotAllIsAPartialFizzle — done-before (ChainTargetsRecordTest.eachModesHalfNamesItsOwnTargets engine partially_fizzled; record-resolution golden Arc Trail)
- ModalResolutionTest.aModeThatDidNothingKeepsItsOwnEmptyHalf — strengthened (outcome resolved, link of cost and both halves). 909187a7
- RecordMomentsTest.aFizzledCharmWritesAnEmptyHalfPerChosenMode (refs.targets) — not done (ChainTargetsRecordTest asserts targets/targets_at_cast for the partial case)
- RecordMomentsTest.aSacrificedHostIsItsGraveyardCardInItsEffectHalf — strengthened (costs.sacrificed and costs.tapped = the Wilds). 909187a7; StackItemBracketTest.aPaymentInTheDamageStepIsNotTheCombats — not done
- aManaAbilityAtPriorityIsNotTheWaitingSpells / unmorphingInResponse… — done-before (partly: WP3 pinned the Shock half's exact events)
- OutcomeHandlerTest.cardRevealedNamesTheLibrariesItCameFrom — done-before (subjects and params exact)
- FirstStatChangeTest.aChangeTheSnapshotAlreadyShows… — done-before (positive, exact)
- MidResolutionCastTest cascade tests (bus-only) / aSpellCastMidResolutionStartsWithNothingOfTheCharms — not done (budget)
- StateBasedDeathTest.aStaticThatKillsOnEntryJoinsTheCreatureSpellsRecord (attributed root) — not done
- EventBufferTest.aRepeatedStateTransitionIsFiledOnce — done-before (gone with WP3's dedup fix)
- EventAttributionTest duration tests / FloatingEffectsTest.aDurationIsOneOfThree — done-before (WP3: one bucketing, FloatingEffects.bucket)
- PatchHooksTest.eachStandInIsForgesOwnDeclaration (same-typed swap) — not done (needs parameter names on both sides)
- OutcomeVocabularyContractTest.everyEngineOutcomeTypeReachesTheLiveShard (Map.of()) — not done
- ClauseContractTest.nestedClausesDoNotCrossTheirMemos (documented order) — not done
- card-in-hand fixtures (≈41 call sites in StateBasedDeathTest, ConditionsMetTest, FirstStatChangeTest, RecordMomentsTest; 0 use TestBoard.cast) — not done (mechanical, large; each resolves with the card in hand — hand-size-sensitive assertions would be wrong, none of these asserts hand size)
- unconditional patched-checkout test — written (b349ae74)
- endingTheCombatPhaseNamesTheAttackingPlayer (simulated) — the engine twin EffectEndingTheTurnTest.endingTheCombatPhaseKeepsTheSpellsEffectHalf now asserts combat_ended + attacking player: it FAILED — BUG: the clause's combat_ended was routed as the skipped-to step's and dropped. FIXED (BusBracketCollector Filing.CLAUSE). bf7014a0. Contract record-schema-delta.md ~388 to say "its turn_ended or combat_ended" (spec edit pending).

## TC2

DELETE (5da483ce)
- EffectRecordTest: theDetectedModeMatchesTheHooksPresence (AttributionMode.probe is that check; theSiblingForgeCheckoutIsThePatchedOne asserts PATCHED + no hook missing), anOrdinaryObservationCarriesNeitherFlag, aDamageStepProbeIsAForkThatIntervenedInNothing, anInterventionalResolutionIsBoth, theModeIsStampedOnEveryRecord, aCombatRecordNamesNoActingLine, aRewriteRecordNamesItsReplacement, anEnvelopeCarriesEveryFrozenField — deleted (siblings: record-resolution/combat/combat-probe/intervention/degraded/rewrite goldens, which no longer skip silently; Python TestGen3EnvelopeFields)
- ProvenanceResolverTest.aGrantedTraitStillKnowsWhichCardGrantedIt — KEPT (only test of the donor's script file for a granted trait)
- .aResolvedKeyCarriesNoReason — KEPT (only test of a null reason on a resolved key)
- .aStackCopyOfATokensAbilityKeysIntoTheTokenTree, .aStackCopyOfABackFaceAbilityKeepsItsFace — deleted (aStackCopyOfAnActivatedAbilityKeysToTheLineItCameFrom for the copy branch; aTokensAbilityKeysIntoTheTokenTree, aTraitOnTheBackFaceKeysToThatFacesOrdinal for the keys)
- ProvenanceSidecarTest.aKeywordLineKeysToItsKeyword — deleted (aKeywordDerivedTriggerIsClaimedByTheKeywordLine, TraitScriptChainTest keyword tests)
- .anAbilityLineCarriesItsScriptApiTypeAndParameterKeys — deleted (TraitScriptChainTest.theRootSegmentKeepsItsParameterKeysAndApiType exact)
- .everyKeyNamesTheCardsOwnScriptFile — deleted (TraitScriptChainTest exact ProvenanceKey equalities incl. script file)
- .aSidecarIsWrittenBesideEveryConvertedCard — deleted (aWrittenSidecarIsWellFormedJsonNamingItsCard, theTokenTreePrefixesItsKeysDifferently read the written files)
- RoleSpansTest.everySpanStaysInsideTheProse — deleted (ProvenanceSidecarTest.roleSpansCoverTheProseOfEveryIndexedLine, same bounds through the converter)
- ScriptLocatorTest.theKeyedPathIsTheOneTheConverterWouldHaveWritten — deleted (CardFilenamesTest:84 and ProvenanceResolverTest:459 pin +2 Mace)
- WireVocabularyTest.unresolvedReasons — deleted (tests/unit/effects/domain/test_records.py TestTheUnresolvedVocabularyIsOneSet reads UnresolvedReason.java both ways; order is not wire)

STRENGTHEN
1. assumeTrue skips everywhere — one unconditional patched-checkout test written (b349ae74); the skips stay (C1 #14, user's decision)
2. blind -Dcharacterization.update — written: per-line JSON-path report of what a rewrite changed; CharacterizationDiffTest (254e1401). "require naming the file" skipped (changes the documented regeneration command)
3. rewriteRecords/triggerRecords/aTraitsHostWithThePendingEvent hand maps — not done (engine-driven RunParamNormalizerTest exists, WP4)
4. resolutionRecords without hook — done-before (recordedWithClauseHook)
5. aClonesLinesKeyToWhatItCopied — done-before (real Clone, Serra Angel keyword case)
6. SnapshotLinksTest.aCloneIsACopyOfWhatItCopied copy_source — done-before elsewhere (SnapshotBuilderTest.aClonesCopySourceIsWhatItCopied, real resolve); links test left hand-applied
7/8. StaticSuppressionTest Frogify three members / field-by-field 8 Aura cases — not done (field-by-field already compares all members over 8 statics incl. Frogify, Control Magic)
9. StatDifferTest keyword/controller — done-before; forget and rebaseline — written (67bada54, forget seen red)
10. RecordReservoirTest hold/held/drain — done-before (heldRecordsAreDrainedOnce)
11. FloatingEffectsTest delayed end-step duration — strengthened to the contract's exact "permanent" (not a this-turn trigger). 909187a7
12. SnapshotBuilderTest empty-board ReferencedTier / hand-built mutate / designations — not done
13. WireVocabularyTest / CardFilenamesTest sanitizer cross-language — not done
14. roleSpans bounds / aWrittenSidecarIsWellFormedJson — the latter strengthened (parsed: members, script_file, lines [], dropped_keys = the implicit spell). 909187a7
15. RecordShardWriterTest.mintedTokensStayDistinctUnderABurstOfRestarts tolerance — KEPT by design (base-36 millis + 4 random chars; slots separate by worker index, which Python always passes; a same-ms burst in one JVM is not a real launch)
TOP MISSING: RunParamNormalizerTest engine-driven + mode names real Forge names — done-before (WP4: everyTriggerRowIsAForgeTriggerMode, everyReplacementRowIsAForgeReplacementMode, reverse direction); goldens for turn_action, non-front face_state, delayed trigger — written (item 3); patched-checkout — written

## TC3

DELETE
- InterventionSchedulerTest.theHandIsStillForcedFirst — KEPT (only pin of the zone order FORCED_ZONES holds lines in)
- ForkRecordsTest.aFreshGameMayIntervene — deleted (anExhaustedInterventionBudgetStopsFurtherForks; RecordCharacterizationTest.anInterventionalResolution intervenes on a fresh budget). 5da483ce
- ForkPlayForwardTest.forkLinesAreHeldUntilTheForkIsKept — deleted (aForkThatThrowsPartWayWritesNothing; fork/interventional flags of directed-fork records: record-directed-fork golden + Python TestTheForkShapesPythonReads). Its writer leak goes with it. 5da483ce
- ForkFidelityTest.aLiveAbilityMapsOntoItsCopysCard — KEPT (only test of the connector's Forks.mappedAbility wrapper; Forge's sibling calls counterpartAbility)
- DirectedRulesContractTest aVariantClassifiedAsRegisteredAgrees / aMissingVariantLabelFails / aScenarioNotListingTheRuleFails / anUnknownScenarioFails — KEPT, `disagreements` NOT moved to production. Why: the runtime readiness check already lives in Python's startup check (sealed rule_readiness / directed_fork_error) that every collecting command runs; DirectedRulesMain is the printer Python reads the registrations from, and a Java twin would be a second implementation to keep in step. The four are what make everyRegisteredRuleAgreesWithItsScenario falsifiable (a helper returning [] would pass it).

STRENGTHEN
- RegisteredRulesTest 72 twin forks — done-before (WP6); ~145 Binds negatives — done-before (WP6)
- toughnessZeroForksATougherCreature / theRegisteredToughnessZeroRule… — done-before (WP6 played-forward families)
- ForkOperationsTest.replacingABlockerCarriesItsStateAndChangesTheCardAlone — strengthened (the card's name changed: the loop could pass on an empty set). 909187a7; addingUnderAnotherController… entered_this_turn — done-before
- ForkEventSinkTest.aForkHearsEveryOutcomeAnObservedRecordHears — not done
- ForkRecordsTest forced==observed — done-before (ForcedAndLiveHalvesTest, hooks on both sides, WP5); mirror_of test — gone (f981c033); aProtectionForkIsTakenOnceAGame… — not done (budget-only)
- InterventionSchedulerTest stand-in greedy spend — done-before (WP5)
- ProbeSchedulerTest collect() — done-before (three tests over real combats)
- ForkPlayForwardTest live unmoved — strengthened (BoardStates before/after, 8c70161d); ForkPlayForwardIntegrationTest — not done
- ForksScoreCheckTest stream test discards the result — strengthened (asserts the check passed). 909187a7
- ForkFidelityTest differences blind spots — not done
SUPPORT
- InterventionSchedulerTest.variantBoard VariantRegistry leak — FIXED (VariantRegistry.forget, @AfterEach; also VariantRegistryTest's Echo Tracer). e4e2e6bb
- assertOnlyChanged no "something changed" — FIXED; it caught four library-swap rules (cascade, discover, partner, transmute) whose swap no visible member shows: those now assert no visible change, their library list already asserts the swap. 909187a7
- five tests re-inlining assertOnlyRemoved/Added/Replaced with drifting allow-lists, duplicated Board/Forker fixtures, ForkRecordsTest's six subjects — skipped (tidying, churn)
- a test leaking a writer — forkLinesAreHeldUntilTheForkIsKept deleted; aDirectedForkMirrorsALiveRecord… gone

TOP MISSING (verified, not redone)
- played-forward fork with Blood Artist + SBA death — done-before (ForkOperationsTest.anAddedPermanentTheFirstCheckKillsIsSeenDying; DirectedForkPlayedForwardTest per family, WP6)
- live game unmoved after forks — done-before (ForkIsolationTest: live records byte-equal forks off/on, all four kinds; Forks.LiveMarks)
- score check + SBA in twin for all rules — done-before (WP6 twin)
- rule negatives (land in hand, second permanent, non-active holder, off-phase) — done-before (WP6)
- no fork with a resolving stack — done-before (ScheduledInterventionsTest.noForkIsTakenWhileTheStackResolves)
- forced divided-as-you-choose — done-before (ForkHookEventsTest.aForcedDividedXSpellDealsTheXItAnnounced)
- KeywordEdit.add can't-have — done-before (ForkOperationsTest.aKeywordTheCardCannotHaveIsRefused)
- mirror_of through GameRecording — n/a (D3c: dead pairing removed; null is the contract)
- greedy-spend scheduler tests, forced==observed comparisons, 72 twin forks — done-before (WP5, WP6; see agent report items in STRENGTHEN)

## TC4

DELETE (5da483ce)
1. ContinuousChannelTest.aContinuousRecordIsKeyedByItsStaticAndBoard — KEPT (only test that two statics on one board each get a record at the per-board gate; engine twins cover identical rendering only)
2. .twoLinesThatRenderIdenticallyAreEachWritten — deleted (RecordDedupTest.theActingLineIsPartOfWhatARecordSays, same RecordDedup property)
3. PlayabilityChannelTest.anAttackDeclarationIsRealAndNamesTheDeciderTheEngineReports — deleted after moving random_seat false onto AttackersRecordTest.eachAttackerGetsARecord… per record
4. .aDeclarationIsRealWhateverTheStep — KEPT (only test that the actor is the hook's player, the non-active one)
5-8. PlayabilityVerdictTest aSecondLandFailsTheLandPlayLimit (timing true moved onto PriorityCandidatesTest.aSecondLandIsALandPlayPastTheLimit), aLandOffTurnFailsWhoseTurnAndCastingSpeed (PriorityCandidatesTest.aLandOfThePlayerWithoutPriority…), aLandOnTheBattlefieldFailsZone (DecisionSampleTest.aLandOnTheBattlefieldIsAZoneCandidate), aLandPlaysCandidateCarriesLandPlayAndNoCost (aSecondLandIsALandPlayPastTheLimit: land_play, legal_targets [], cost {}) — deleted
9. PriorityCandidatesTest.explorationMakesTheSecondLandPlayable — deleted (PlayabilityVerdictTest.explorationRaisesTheLandPlayLimit)
10. .aSpellCandidateIsNoLandPlay — KEPT (only land_play false on a record)
11. RewriteContractTest.everyResultIsWrittenInItsWireSpelling — deleted (WireVocabularyTest.replacementResults pins the spellings; goldens hold the record field)
12. TriggerRecordTest.preventedDamageIsNotRecordedAsDamageDealt — deleted (RunParamNormalizerTest.preventedDamageIsAPreventionOfItsAmount, engine)
13-16. MatchWorkerMainTest: progressFilePropertyIsRead deleted (progressFileIsReadableInRecordsOnlyMode); progressFilePropertyIsEmptyWhenAbsent KEPT (only unset case: csv(null) path); noProgressFileIsASilentNoOp KEPT (only Optional.empty() both); cardsPlayedLiveBesideTheMatchOutcomes deleted (progressLineIsWrittenAlongsideTheSealedCorpus writes both beside each other)
17. MatchGeneratorTest.generateMatchPopulatesMetadata — KEPT (only runId/timestamp of a generated header)
18. MatchGeneratorRoutingTest.withoutDecksOnlyTheRollStillMixes — deleted (rollIsFileSampleApproximatesWeightFraction)
19. MatchResultTest.theSequencesAreOneCharacterPerGameInGameOrder — deleted (aLineIsTenFieldsInTheDocumentedOrder; length is no code path)
20. GeneratedDecksIndexTest.randomDeckPicksFromAllDecks — deleted (loadParsesLabelSetCodeAndCardNames draws from every deck)
21. MaterializeDeckTest.afullyResolvableDeckIsUnchangedByTheCheck — KEPT (only FAIL-mode pass-through)
22. ToolMainArgsTest.theExitCodesAreTheDocumentedValues — KEPT (only numeric pin; IO_FAILURE == BAD_CONFIG is documented on purpose in ExitCode, not a bug)

STRENGTHEN
- RandomSeatControllerTest cast stops at the stack — strengthened (mana paid, resolves); attack/block "no record" tests never opened the recording — FIXED (recording.open()). cfb9579d; charm test without an artifact — not done
- RandomChoicesTest "uniform" reachability / chooseModes chain — not done
- TriggerRecordTest pins (CounterMap, unsigned life, face_up) — done-before (WP4: AddCounter replacement shape, signed -4, "front"); aCounterEventNamesTheCardThatPutTheCounterThere — strengthened (real CounterAdded params, type and delta). (TriggerRecordTest commit after 5da483ce)
- RewriteContractTest.aPreventedAmountIsARewriteWithBothHalves — not done; anUpdatedRewriteCountsTowardTheTally… — done-before (WP7: deltas between printed summaries, shard-counted)
- GameRecordingTest close tests assert 2 hooks — strengthened (every static EffectRecordHook of the ten owners is null after close). 909187a7; boundary-step test — not done
- CollectionCapsTest.theDefaultCapsAreTheContracts — fixed with N2
- PlayabilityChannelTest.aForkPriorityStop… vacuous — strengthened (premise asserted). 909187a7
- DecisionSampleTest test-side bucket rule — KEPT (production needs the candidate ability; the test's rule is stated for its three-card board; production read.sample() tests exist)
- TriggerNegativeSamplerTest 1:1 — done-before (downstream-refusal test drops 3 in 4)
- GeneratedDecksIndexTest.loadParsesLabelSetCodeAndCardNames — strengthened (card names per deck)
- VariantStagingTest integration tag — done-before (WP0)
- LineAppenderTest threads vs processes — strengthened (an appender per writer, no shared monitor). 909187a7
TOP MISSING
- real match through GamePlayer.playMatch — done-before (WP7 GamePlayerRecordingTest); a game that throws mid-match — done-before (same class)
- worker exit paths testable — done-before (ToolMain.Exit, ToolMainExitTest)
- parseConfig shard path — written (MatchWorkerMainTest.theShardIsNamedForTheRunTheSlotAndTheLifetime, aNegativeWorkerIndexIsRefused; openEffectRecords package-private). 909187a7
- HookFailureReport — done-before (HookFailureReportTest)
- every EVENT_BY_MODE key a real Forge mode — done-before (RunParamNormalizerTest)
- Channels.close when playability.close throws — not done (no seam: a closed writer makes every step throw)
- redrawTargets / RandomCostDecision — not done (untested)
- ContinuousChannel.collect at two boundaries — not done

## Items 3-5

- Item 3 goldens: written. aa4c139e (goldens alone) + 12a04016 (tests). record-turn-actions.jsonl (cleanup_discard, phasing, day_night, untap, draw, lore_counter, real loop); record-back-face-and-delayed-trigger.jsonl (Insectile Aberration face_state back, Graceful Reprieve's D<id> delayed trigger with source line + remembered); shard-two-games.txt + .jsonl.gz (one writer, two GameRecordings, one gzip member a game, marker last; Characterization.gzipMembers / assertShardMatches). No existing golden changed (all 4 characterization classes run). Python: test_shard_io TestJavaWrittenShards (+2: reads the Java gzip game by game, first member ends at game 0's marker; cut mid-second-member keeps game 0 ended), test_record_codec_wp5_wire TestTheBoardExtrasPythonReads (+2), TestTheTurnActionsPythonReads (+2); test_field_coverage: turn-action events now vary (asserted).
- Item 4 HookShapes/PatchHooksTest: the test already failed on an unrestated Forge listener method; the empty NOT_CONSUMED_YET allow-list removed (no escape hatch). b349ae74
- Item 4 unconditional patched-checkout assertion: written PatchHooksTest.theSiblingForgeCheckoutIsThePatchedOne (fails, never skips). b349ae74. forge-connector/CLAUDE.md updated.
- Item 4 real match through GamePlayer.playMatch with records: done-before (WP7: GamePlayerRecordingTest.eachGameThatEndedWritesItsMarkerLast decodes game by game, one marker last per game, records_withheld 0; aFinishedGameIsOnDiskWholeBeforeTheNextGameStarts).
- Item 4 Board.twin state-based check: done-before (WP6: RegisteredRulesTest.Board.twin score-checks, runs checkStateEffects(true), asserts no card moved, no change, game not over).
- Item 4 hook during GameCopier.makeCopy: written Forge-side EffectRecordGameCopierTest.theHooksACopyCallsWhileItIsBuiltNameTheCopysGame (trigger + replacement listeners; Emperor Crocodile state trigger; non-empty, every call names the copy). forge 1b97ca6f6d7 (test only, no Forge main change).
- Item 5, Forge test notes (forge 9855887c9c8, test only):
  - F2 EffectRecordParamCopyTest.testGameObjectsAreNotCopied, EffectRecordReplacementOutcomeTest.testNoAbilityIsReportedWhenNoneRan, EffectRecordOutcomeHookTest.testAParameterBuildThatThrowsIsContained — already gone
  - F2 FlipCoinEffectOutcomeIntegrationTest branch tests — strengthened: the plain branch's win/lose sub-abilities gain 5/2 life, the note's result is checked against the branch run (a swapped win/loss now fails); NoCall/ForEachPlayer kept vocabulary-only (no observable branch effect on those shapes)
  - F2 EffectRecordClauseHookIntegrationTest.thePointerIsPutBackWhenTheLineHasResolved — deleted, its one assertion folded into eachClauseIsNamedWhileItsOwnOutcomeIsPublished
  - F2 EffectRecordDelayedTriggerTest.theReadingIsACopyThatCannotBeWritten — strengthened/trimmed (Guava half removed) -> theReadingIsACopyCleanupDoesNotReach
  - F3 CardWithoutStaticTest.withoutAStaticThatNeverTouchedTheCardNothingChanges — strengthened (a real static, Levitation)
  - F3 bug-1 test (copy's untap keeps the original's hidden keyword) — done-before (WP1, GameCopierFidelityTest ~1412)
  - F1 GameCopierFidelityTest.anEffectWhoseSourceIsGoneNamesNoOriginalCard — done-before (WP1: asserts not the original's, copy's game and owner)
  - F1 hooks during makeCopy — written (above)
  - F3 EffectRecordPrioritySettledTest.aLoopThatEndsTheGameSettlesNothing vs EffectRecordStateBasedLoopTest.theEndStillComesWhenAPassEndsTheGame — KEPT both (the first asserts no settled signal, a listener the second never installs); merging the two files and folding EffectRecordCombatLegalityTest into CombatLegalityObserverTest — skipped (tidying, churn)
  - F2/F3 "untested behaviour" lists (real-game replacement shapes, trigger limits, CantBlock blocker in the observer, trample/multi-block assignment snapshots, failed payment moments) — not done (budget); see "not done"
- PatchHooksTest.theSubAbilityHookReadsNullOutsideAResolution: strengthened -> theSubAbilityHookReadsNullOnceAResolutionIsOver (a Bolt resolves through the hooks, its damage attributed root, pointer null after). b349ae74
