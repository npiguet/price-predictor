# INT-FIX and GAP-4: the verifier's report (2026-11-15)

State at hand-back:
- Both checkouts are clean at HEAD (`97ff9b11` here, `16697120173` in `../forge`). Only the untracked files that were already there remain (`diaries/`, `grep.exe.stackdump`, Forge's own).
- The installed Forge is the committed one. The installed `forge-ai` and `forge-game` jars hold the INT-FIX classes, and `GameCopier.class` and `TokenInfo.class` are byte-identical to `target/classes` after my restore run.
- The connector fat jar (04:29:42) is untouched, and five restored connector classes are byte-identical to its copies.
- No JVM is left (reaped), and the lock is removed.

What I ran, all under the lock, one job at a time:
- Forge: `EffectRecordGameCopierTest` mutated, then restored (2 runs).
- Connector: five test classes mutated, then restored. The restored run also played the two GAP-4 forks with `-Ddirected.records.dump`.
- A scratch probe test for the leak.
- Three integration tests.
- Python tagging of the dumps and of two gen-2 shards.

Every scratch file is `IV-*` in the coordinator's scratchpad.

## Findings, most severe first

### 1. The leak is still open on one exit path: a fork thrown away at its score check (blocks the pilot's heap verdict)

Confidence: high. Evidence: a probe test went red.

`Forks.copyAndVerify` calls the fidelity audit (`ForkFidelity.differences` → `SnapshotBuilder.forFork(fork, live).ofBoard()` → `LibraryCapture.of(fork, ...)`) before the score check. That audit adds the fork to `BUILT`, and the entry's value holds the fork's game. Two exits then return `Optional.empty()` before `ForkRun.take` sets `forked[0]`:
- the score check fails (a mismatch, or it throws);
- the perspective is null.

On both, the `finally` releases `null`, so the fork's built cards, and with them the whole copied game, stay in the heap for the JVM's life. This is the exact defect INT-FIX fixed, on the one path it missed.

The probe was a fork through the real `ForkRun.take` with a null perspective. Afterwards `LibraryCapture.gamesHeld()` was one higher than before, net of the live game ("expected 0 but was 1").

How fast this leaks depends on how often score checks fail. The 15-minute integration run did not count them.

Fix: in `copyAndVerify`, release the fork from both registries (`LibraryCapture.release(fork)`, `FloatingEffects.release(fork)`) on every exit after `makeCopy` returned. That covers a failed or throwing score check, a null perspective, and the copy guard's catch when `fork` is set. Or return the fork to `take` so its `finally` releases it. Then add a `GameRetentionTest` case for a discarded fork.

### 2. The unpaid-cost refusal turns away every crew line, not only a vehicle already crewed this turn (should fix)

Confidence: high. Evidence: a probe test went red.

`InterventionFork.readsAnUnpaidCost` tests `host.getCrewedByThisTurn() != null`. That getter never returns null: it returns `CardCollection.EMPTY` when the field is null (`Card.getCrewedByThisTurn`). So every crew activation is refused as `intervention reads a cost it never paid`.

The probe: a fresh Smuggler's Copter on the battlefield, never crewed. Its crew line has an empty, non-null crewed list, and `readsAnUnpaidCost` returned true.

A first crew forced unpaid does not throw. `becomesCrewed` stores the null paid list into a null field. Only a second crew in the turn throws, at `addAll(null)`. So all crew interventions are lost (crew is the keyword of every vehicle, in many sealed sets), against what `forks.md` and the commit say.

The saddle half is correct: `getSaddledByThisTurn()` returns the raw field.

Fix: test `!host.getCrewedByThisTurn().isEmpty()`, and add a test case for a never-crewed vehicle, which must not be refused.

### 3. 510.2's new near miss tags the wrong records on real data (should fix)

Confidence: high. Evidence: two gen-2 shards tagged with the real `tag_record`.

Rule 510.2 says all combat damage is dealt at once. Its contrast case, the near miss, is a creature killed by a first striker before it can deal its damage. The new near miss is "a creature dealt combat damage dies in the record having dealt none of its damage events".

Every near miss tagged in two `records-gen2/full-strength` shards (8 of 8) was something else:
- Seven were 0-power creatures dying in the regular damage step (Lobber Crew 0/4, Gatecreeper Vine 0/2 four times, Pit Imp 0/1). They had no damage to deal, so whether damage is dealt at once does not matter.
- One was an animated Rakdos Keyrune, a 3/1 with first strike, killed by Desecration Demon in the regular step. It had dealt its damage in the first-strike step, which is a separate record.

None was a first striker's victim. Each shard gave 62 and 99 fires.

The counts still lack does-not-fire (0 of 456k). A recount with this declaration would fill that side with these false near misses: the rule would leave the directed list and the corpus would carry mislabelled contrasts.

The played fork does tag as intended: the knight_in_the_bears_place fork is tagged does_not_fire, the live trade fires.

Fix: count a victim only when it had power above 0 in the record (base, boosts and counters), and only in the first-strike damage step (`state.global.combat_substep`, which reads `regular` on all eight). Or require a first-strike or double-strike source among the step's damage events. Then pin the 0-power case and the "first striker dies in the regular step" case as neither side.

### 4. A line refused for an unpaid cost still uses up the boundary and one of the game's planned interventions (should fix)

Confidence: high. Evidence: read only.

`InterventionScheduler.force` does three things before it calls `intervene`: marks the line as seen, calls `gate.forced()` (the `SpreadGate` then counts it as forced and drops a pending intervention), and returns `DONE`.

So the refusal spends no `ForkBudget`, but it takes one of the game's planned interventions (2 by default) and ends the boundary. No other line is tried there. Combined with finding 2, any vehicle on the battlefield can take half a game's interventions.

The game-restart filter shows the right place for a deterministic, known-in-advance refusal: `worthIntervening` already turns away `restartsTheGame` lines, so no gate slot is spent.

Fix: test `readsAnUnpaidCost` in `worthIntervening`, or as a `continue` before `seen.add` / `gate.forced()`. Count the refusal there once per line (mark it seen).

### 5. 702.1c's narrowed declaration (minor)

Confidence: high on the data, medium on the reading.

Rule 702.1c covers an effect that says "the same is true for" a list of keywords: it grants every variant of a listed keyword, such as each protection or landwalk ability.

- On real gen-2 data the declaration already tags both sides. In one shard, Sproutwatch Dryad's trigger (`PumpAll`, `SharedKeywordsZone$ Hand,Battlefield`) was tagged 11 fires and 5 does_not_fire. The checked-in counts (55,564 fires, 0 does_not_fire) predate the narrowing, so after a recount 702.1c may no longer lack a side, or need a directed fork at all. The commit's "on the fixtures it now tags nothing" is true of the fixtures only.
- Fires is wider than the rule's point: any shared grant counts (Sproutwatch's flying, say), though none of Sproutwatch's keywords has a variant.
- `api: PumpAll` leaves out three `Pump` carriers (Majestic Myriarch, Mutable Pupa, Priest of Possibility). It also leaves out Kathril, Aspect Warper (`PutCounter`), whose keyword counters the rule names outright.

The played fork is tagged as intended: live fires, fork (Hill Giant) does_not_fire.

Fix: recount 702.1c before relying on its listing. Consider admitting `Pump` and `PutCounter` with `SharedKeywordsZone`.

### 6. `Couldn't map` can still be thrown by one other kind of static effect (minor)

Confidence: medium. Evidence: read only.

`outsideTheCopy` skips an effect only when its source has no counterpart in the copy. A copied source whose effect lists an uncopied card still throws in `mapCollection`. That is a command-zone effect from Wish, Sovereign's Realm or Frontier Explorer (`AffectedZone$ Sideboard`, "play from outside the game").

The fix reaches the one throw site (`GameCopier.find`), but only for effects whose source is uncopied. All three cards the integration run named fit the sideboard pattern: Serpent of the Endless Sea and Zurgo's Vanguard have power set by a characteristic-defining ability, and Cryptic Cruiser has devoid. So the measured one-in-seven is covered.

Fix (if it shows up): map the affected cards leniently in the copier's undo step (counterpart or skip).

### 7. Nothing tests the `FloatingEffects` release (minor)

Confidence: medium. Evidence: read only.

`GameRetentionTest` resolves every clause, so no open note holds the game, and the weak key frees the entry without the release. Removing `FloatingEffects.release` would, by reading, leave every test green. The `LibraryCapture` release is tested: both `GameRetentionTest` cases went red with it removed.

Fix: a case whose body throws mid-clause (an open note), then check that a weak reference to the game clears.

### 8. Two wording points (minor)

Evidence: read only.

- **The dry run's made roles.** FR-010b and `forks.md` say that, for a card the edit makes, the dry run checks "the rest" of the play on the live board. When the made card is the play's source, `possible()` returns true at once, so the live targets are not checked either. Nothing can be checked without the line. Fix: say so ("the whole play when that card is its source").
- **The refusal reason's constant.** `intervention reads a cost it never paid` is listed only in `forks.md`. The contracts enumerate no intervention refusal reasons, and the Python reader (`ForkRefusalCounts`, `FORK_REFUSALS_LINE`) reads any key, so nothing is missing. Its constant sits in `InterventionFork`, as `NEVER_RESOLVED` does, not in `ForkRefusal`.

## The heap: what a pilot should watch

The fixed leak is real: a `WeakHashMap` entry whose value strongly holds its key is never collected. Every recorded game and every fork that took a snapshot stayed in the heap. The fix is the likely cause of the slowdowns, but the run logs that would prove it were not checked.

The OutOfMemoryErrors in the integration run had a second plausible contributor. Both JVMs spent their last 6 to 7 minutes in a repeating Forge AI `TimeoutException`. On Java 25 the timed-out "Game AI Eval" thread is not stopped (INT-FIX's own brief fact), and a thread that keeps running holds its simulation copies.

Per worker JVM, a pilot should watch:
- Heap after GC, game by game (`-Xlog:gc*:file=...`). It should stay flat across a lifetime of 30 to 40 games.
- The number of live `forge.game.Game` instances (`jcmd <pid> GC.class_histogram`). It should stay near one plus the forks in flight, not grow with games played.
- Game length against the game's index in the lifetime. It should not climb.
- The count of "Game AI Eval" threads (`jcmd <pid> Thread.print`).
- `worker_stops.crashed` and `jvm_error` in `run.json`, against the fork score-check discard lines (finding 1).

## Checked and found correct

- **A, the copy fix.** It is sound. An effect is skipped only when its source is uncopied, it affects no player, and none of the cards it affects is copied. Then nothing in the copy carries it. `remove()` touches only the source (`IgnoreEffectCost`), the affected players and the affected cards, and a static that changes players or rules without listing them is re-read from the copy's own cards.
  - "All" zones exclude the sideboard. The copier holds phased-out, face-down, exile and command cards, and tokens, so none of those is skipped.
  - The change is private to `GameCopier`'s copy path. It changes no live game; Forge's AI simulation only throws less. `makeOneToken(Player, int)` behaves as before.
  - Both Forge tests are real. Seen red with each fix reverted: `Couldn't map Serpent of the Endless Sea`, and the token's paper card null. Green restored.
- **B, the token paper card.** Within the effects code, only `ScriptLocator` reads it for tokens (`LibraryCapture` skips tokens). With the fix, a fork's token resolves through the same `PaperToken` as the live token. `getRules` now matches the live token's, and `EffectRecordOutcomes.tokenKind` batches as live. The wire is unchanged: the full suite with the characterization goldens is green, and `CopiedTokenProvenanceTest` still compares the fork's lines block with the live one. Its change from `assertNull` to `assertSame` strengthens the test.
- **C, the leak fix.**
  - Static registries in the connector: only `LibraryCapture.BUILT` and `FloatingEffects.BY_GAME` are keyed by game. The others are counts, reflection caches, name sets, or thread-locals restored in a `finally`. Forge's `TokenInfo.TOKEN_EDITION_PINS` values hold no game, and the hook listeners are single and uninstalled at close.
  - Release runs in the innermost `finally` of `GameRecording.close` (after the marker) and of `ForkRun.take`. It covers a game that throws, a refused fork, a body that throws, a cut game and a worker stop, all of which reach `close`. The one gap is finding 1.
  - Nothing is released too early. Every `SnapshotBuilder` binds its `FloatingEffects` at construction, and nothing snapshots a fork after its played-forward recording closes: directed and protection bodies return `play(...)` last.
- **D, the refusal.** With the refusal's call reverted, `ninjutsuIsRefusedBeforeItIsForced` went red (the forced body threw). It writes nothing, and spends no `ForkBudget` (but see finding 4).
  - Other readers of costs paid are guarded: `getPaidCards` callers check for null, X is read behind null checks, `retainPaidList` iterates a concatenation. The only unguarded dereferences are ninjutsu and sneak (`Returned`) and a second crew or saddle. So apart from finding 2 the predicate is neither too wide nor too narrow.
- **E, the integration modules.** No assertion was dropped or loosened. The early returns became a `break` then `stop_worker`, and the same skips and asserts follow. Three passed in 68 s, JVMs reaped: `test_effects_steered_game.py`, `test_effects_patched.py::test_a_patched_checkout_stamps_every_record_patched` and `test_effects_coverage_isolation.py::test_a_named_deck_reaches_the_records` (two workers).
- **F, the causes left unfixed.** All three are consistent with the code, by reading.
  - Forcefield: `ReplaceDamage` runs in place, and for that API Forge runs the sub-ability (the exile) as the buffered ability, so `replaced_by` and `buffered` truthfully name the same printed line. The buffered run sums `PreventedAmount` from the run-parameters map every replacement on the event shares. A second shield preventing 0 overwrites it and the exile never runs. One more stock path does the same: a single shield against 1 damage prevents 0.
  - `play_round`'s overshoot: the pool polls the stop condition every 2 s (`STOP_POLL_INTERVAL`), and every in-flight game is finished. The docstring is right.
  - `GamePlayer.closeRecording` catches only `RuntimeException`: an `Error` from close after a game failure masks that failure. The registries are released anyway.
- **G, GAP-4.**
  - The five precondition guards: each reverted, and the named `RegisteredRulesTest` case went red. The other 230 cases stayed green.
  - The dry-run change: reverted, and `theDryRunSkipsOnlyWhatARoleTheEditMakesNames` went red.
  - Both forks played and tagged with the real `tag_record`. 510.2: live combat fires, fork does_not_fire. 702.1c: live resolution fires, fork does_not_fire.
  - `GAP1-tag.py` reads `directed-fork-counts.json`, which predates GAP-4's narrowings. I used my own `IV-tag.py`.
  - FR-010b, FR-014b, `contracts/catalogue-extensions.md` and `forks.md` agree with the code, except the wording in finding 8.
- **H, the docs.** The `forks.md` bullets (copy, token, refusal) and `collection-internals.md` § State kept per game are true to the code, except the crew claim (finding 2) and the discarded-fork path (finding 1). The only persistent behaviour INT-FIX added is the new refusal key in `run.json` `fork_refusals`, which the contract covers generically.

## Not checked, and why

- **Heap in a real run.** No long run is allowed. The probe and `GameRetentionTest` show retention, not the run's slowdowns.
- **`test_effects_collection.py` and the other three `test_effects_patched.py` cases.** I ran one or two tests per module, as the brief asked.
- **A forced crew of a vehicle already crewed, played to the throw.** The throw was read, not played.
- **Forcefield on stock Forge.** The cause was read only.
- **The full gen-2 or gen-3 corpora.** Two gen-2 shards only, to stay light.
- **Whether Judgment is sealed-eligible** (finding 6).
