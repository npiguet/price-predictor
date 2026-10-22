# Task 6 report: dead code in effects/

Branch `t6-effects-dead-code` (from master ff900672; harness HEAD was stale 505fb65c), worktree
`C:\Users\nicol\IdeaProjects\price-predictor\.claude\worktrees\agent-a0175a964d7a26dff`.
Commits: 29120b58 (E1), 53444e22 (E2), 1f30dfdf (E3).
(Report written inside the worktree because the shared path was refused; the file is untracked, not committed.)

Grep evidence: every member was grepped across `forge-connector/src/main` and `src/test`; Python `src/` has no
reference to any of them (they are Java-internal; the Python side only knows main classes and the JSON). No
reflective lookup (`getDeclaredMethod`/string names) exists for any deleted member.

## E1 (deleted, zero callers anywhere)
RandomChoices.random(), EffectRecord.abilityUnresolved() (field + JSON kept), AttributionMode.reset,
ForkCollector.chooseModes (tests used only RandomChoices.chooseModes), SnapshotBuilder 4-arg public toJson.

## E2
- variantOf()/synthetic() setters and fields deleted; toJson writes `"variant_of":null` and `"synthetic":false`
  literally (same bytes).
- ability(List<ProvenanceKey>) deleted; CollectorTest converted to Resolved. The "several keys for a merged line"
  test was deleted: Resolved carries one key, so production cannot emit several through this API.
- costPayload(SpellAbility,String) deleted; its CostPayloadTest "two forms agree" test deleted (nothing left to assert).

## E3
Deleted: ForkCollector.chooseTargets/announceX, runAsForkForTest, discardedForks (+ the whole `discarded` counter,
7 increments, which nothing read), HeldProbe 7-arg ctor (tests pass TURN_UNKNOWN to the 8-arg); recordsWritten on
all collectors and CollectorContext (+ counter); PatchedCollectors.manaReservoirSlot, heldProbeCount,
installedHooks (+ `installed` list, Listener.name; install() counts locally), resetUnknownOutcomeTypeLatchForTest,
allowLegalityRecord (inlined into emitLegality, doc moved to a comment); BusBracketCollector modeHalvesOpen,
unresolvedActivations, abandonedActivations (+ field), combatParticipants (+ the three writes); RecordShardWriter
isValidLifetime/LIFETIME; ForkEventSink.isEmpty; RewriteTally.rewritten; ProbeQueue.size (became unused).

Test rework:
- ForkCollectorTest: counts now read the shard (`recordsInShard`); chooseTargets/announceX cases: the two
  duplicates deleted (RandomChoicesTest already covers a short-of-candidates draw and a null-payer floor), the
  two unique ones moved into RandomChoicesTest with production-shaped args.
- PatchedCollectorTest: mana reservoir tests use ManaReservoir directly (sampler seeded via scramble, as production);
  held-probe tests assert through shard contents (mirror_of ids); legality test calls allowDistinctRecord;
  latch reset done reflectively on the private static; fork-running flag set reflectively on FORK_RUNNING.
- `installedHooks().contains("clause-events")` gates (ShardRun, ModalResolution, RecordCharacterization,
  StateBasedDeath) became `PatchHooks.CLAUSE_LISTENER.present()` (equivalent).
- RecordShardIdentityTest: validity asserts replaced by a regex check on mintLifetime output.
- RewriteTally: the `rewritten` delta assertion dropped (summary tests already assert the "rewrote in place" count).

## Kept
None kept for size; nothing exceeded the 60-line limit.

## Tests
Unit: 935 run, 0 failures. Integration: 59 run, 0 failures. Characterization tests green, expected files untouched.

## Concerns
- Mid-run "held" assertions (recordsWritten between steps) in BusBracketCollectorTest/StateBasedDeathTest were
  dropped, since the gzip shard is only readable after close; final shard-content assertions remain.
- Two tests now use reflection on private statics (latch, FORK_RUNNING) instead of production seams.
- `EventBuffer.file` return value is now unused by BusBracketCollector (left, out of scope).
- Did not touch ApiEvents *ForTest seams or any off-limits file.


## Fix round 1

- BusBracketCollectorTest: aCastIsHeld... and aSpellGoneFromTheStack... now assert collector.finishGame(Set.of(id)) returns 1 (the cast is still held), no new seams.
- StateBasedDeathTest: the mid-run held-half counts have no public observable other than the shard (unreadable until close), so they stay dropped; final shard assertions cover content and order.
- everyReplacementReachesTheWorkerTally: summary tests (theSummaryLeadsWithTheVolumeAndTheResults: 10 records, 1 rewrote in place) cover that rewritten=true counts and false does not at RewriteTally level; the live wiring Updated->rewritten is no longer asserted at the collector level.
- Nits: FORK_RUNNING link made plain text, stray space removed, double blank line in ForkCollectorTest removed, no unused imports (script check over changed files).
- Forge grep: grep -rnE with all deleted member names over C:/Users/nicol/IdeaProjects/forge --include=*.java matched only Forge own chooseModes/announceX in forge-ai simulation (GameSimulator, SpellAbilityChoicesIterator, SpellAbilityPicker, GameSimulatorSpellChoiceTest), unrelated to ForkCollector (those are Forge methods; ForkCollector delegates to RandomChoices). No reflective lookups of connector members.
- Suites: unit 935/0 failures, integration 59/0 failures.
