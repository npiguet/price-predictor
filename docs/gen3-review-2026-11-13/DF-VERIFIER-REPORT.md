# Verifier report: DF-A (763b3f13, 4750926b, a3653f86) and DF-B (e21afb05, 70eee3f1, c3c77731)

Tree at 10f217b9. I left it clean: every mutation was reverted with `git checkout`, `git status` shows only the untracked diaries and `grep.exe.stackdump` that were already there, no JVM is left running and the lock is gone. Scratch files are in `C:\Users\nicol\AppData\Local\Temp\claude\01313e7323e07cb5\62b11185-3729-4588-b601-3f0efb5b2b2a\scratchpad\`:
- `VER-dump\`: my own `DirectedForkPlayedForwardTest` run with `-Ddirected.records.dump`, 42 cases passed, 55 games.
- `VER-tag.py` and `VER-tag.out`: those games tagged with the real `tag_record`, per listed rule, beside the rule's counts and the side the counts lack.
- `VER-mutate1.py` and `VER-mutate.py`: the temporary mutations.
- `VER-mut1.log`, `VER-mut2.log`, `VER-mut3.log`, `VER-iso.log`: the Maven runs.

"Lacking" below is what `scripts/rules_catalogue.py` computes from `rules/directed-fork-counts.json`: the `fires` side when a rule has fewer than 100 `fires` records, the `does_not_fire` side when it has none. The command's own output marks every variant of a lacking polarity with `*`; a rule with no `*` has no registered-able variant that makes its lacking side.

## Findings, most severe first

### 1. Nine rules the packages touched fork a variant that makes a side the counts already hold, not the side they lack
Severity: blocks the pilot. Confidence: high (played and tagged).

FR-014a (spec.md:793), `contracts/catalogue-extensions.md` ("A registered variant makes the side the rule's counts lack") and `src/effects/docs/forks.md`, all written by a3653f86, say the opposite of what these forks do. Each of these rules takes an equal share of `--directed-forks-per-game`, and all that share buys is records of the side the counts already have. Fork tags are from VER-tag.out; counts are (fires, does_not_fire).

| Rule | Counts | Lacking side | What the fork yields | Fix |
|---|---|---|---|---|
| 702.103e | (0, 1050) | fires | does_not_fire (variant `target-survives`), which the counts hold 1050 of | The fires side needs a play made in response, which nothing provides (the DF-B brief itself says so). Take it off the list with that reason, as 707.10d. |
| 702.105a | (91, 66) | fires | does_not_fire (`less-life`) | A fires variant is already classified: `tied` (operation `life`, marked `*`). Register it in `Dethrone`. |
| 701.12g | (42, 3) | fires | does_not_fire (`cannot_gain_life`) | A fires variant is already classified: `opponent_above` (operation `life`, marked `*`). Register it in `LifeExchange`. |
| 119.7 | (3, 42) | fires | does_not_fire (`exchange-lowers`) | Invert: bind Tree of Redemption under a can't-gain-life effect with life at or above its toughness, and set the life below. This needs a new variant. |
| 701.39a | (70, 8) | fires | does_not_fire (`no_creature`) | Invert: bind the bolster spell with no creature, `add_permanent` a creature, keep the forced cast. |
| 701.12e | (0, 7) | fires | does_not_fire (`no_aura_in_hand`) | Invert: bind a non-Aura in hand and `move_card` Holy Strength in, keep the forced swap. |
| 702.65a | (0, 7) | fires | does_not_fire (`no_aura_in_hand`) | Same inversion as 701.12e. |
| 702.170c | (42, 9) | fires | does_not_fire (`card_too_expensive`) | Invert: bind with only a too-expensive card and replace it with a cheap one. |
| 608.2p | (1, 1) | fires | does_not_fire (`not-final-chapter`); fires stays at 1 | Invert: bind one lore counter short, add the counter. No forced play is needed. |

The does_not_fire records are not worthless for the four rules whose does_not_fire count is also tiny (608.2p, 701.12e, 702.65a, 701.12g). But they are not the lacking side, and fires stays at 0 to 3 for 608.2p, 701.12e and 702.65a.

Per your note, 118.11 is the same case (lacking does_not_fire, fork gives fires, counts hold 23,335) and is already decided.

### 2. Listed rules the packages did not touch whose fork also adds nothing the counts lack
Severity: should fix. Confidence: high for the first three (played), medium for the last three (read from the script's output only).

- **122.1d.** Its fork's records carry no 122.1d tag at all. The does_not_fire side of its `rewrite_event` detector is a rewrite record whose result is not `replaced`. Without a stun counter no replacement applies, so no rewrite record is written. This is exactly the reason 400.6 and 704.7 were taken off the list, and 122.1d should leave with it.
- **510.2.** Lacking does_not_fire. The fork (`knight_in_the_bears_place`) is tagged **fires** by 510.2-1: the first-strike step's damage plus a death matches the "simultaneous" detector. The detector needs to exclude a first-strike step, or the rule leaves the list.
- **702.19e.** Counts (0, 44), lacking fires. The fork gives does_not_fire.
- **701.42b, 701.44d, 702.19c.** `directed-rules` marks no variant of their lacking polarity (no `*`). Not played.

### 3. Four removed rules could be rescued by a forced play
Severity: should fix, or the user decides. Confidence: medium (reasoned from the detectors and the code, not played).

The `NOT_DIRECTED` reasons for 205.1a, 701.23f and 702.70b, and the DF-A brief's premise that variants of polarity `other` "count on neither side", treat a variant's label as the tag. But tags come from the detector reading the record ("polarity comes from the board", `rule_detectors/__init__.py`). A fork whose rule's fact still happens is tagged fires whatever its variant is labelled. With the forced plays DF-B built:
- **205.1a** (0 fires, 1.2M does_not_fire). Its reason ends "no board edit makes a seat cast Song of the Dryads", which a forced cast now does. With the `counter-kept` edit, the fork would record the type-setting Aura's contribution, the fires side.
- **702.70b** (0, 0). A forced attack by the poisonous creature, with `one_instance` ("still fires once"), gives the fires side. Only its near miss needs a block.
- **701.23f** (2, 0). Its own reason says Archivist's trigger "fires with or without Aven Mindcensor's limit". With an opponent's search forced, the fork gives fires.
- **111.10p** (45, 0), lacking both sides. With Ellivere's cast forced plus the `two_enchantments` add_permanent variant, a Virtuous Role is created and boosts, which gives fires. The reason (and the experiment record) only argues the does_not_fire side.
- Also: 702.133a's reason addresses only 702.133a-1. 702.133a-2's `cast_from_hand` does_not_fire variant (operation `move_card`) is not argued.

Each needs a played check before it goes back on the list.

### 4. 118.12b's precondition almost never binds
Severity: should fix. Confidence: high on the condition; I did not measure how often it binds.

`OptionalSearch.bind` requires `matching.size() == 1`: exactly one card of the searched type (a basic land, for Elfhame Sanctuary) left in the library at upkeep. A limited deck's library holds several basics until very late in a game, so this fork is close to dead. It costs no budget, but the rule is listed for almost nothing. A variant that leaves the search no card in one edit does not exist on a normal library, so this needs a rethink or a reason.

### 5. Several guards a negative test should catch stay green when reverted
Severity: should fix. Confidence: high.

Each was reverted alone or in a disjoint batch, then `ForcedPlayTest`, `ForkPlayForwardTest`, `DirectedForkPlayedForwardTest`, `ForkRunTest`, `InterventionSchedulerTest` and `RegisteredRulesTest` were run:
- **The scheduler's dry run** (`DirectedForkScheduler`, `if (refused.isPresent())` → `if (false)`): all green. `aTargetTheLineCannotTakeIsRefusedBeforeTheFork` asserts only "refusal counted, nothing written", which the in-fork refusal in `ForcedPlayRun.start` also satisfies. FR-010b's "spending none of `--directed-forks-per-game` and leaving the rule's one fork of the game untaken" is untested. The test should assert the budget, the tally and a second boundary's fork.
- **`forced.finish()` deleted** (`ForkPlayForward.java:134`): all green. It is the only check for an answer with no step (118.12b's kind) that is never asked, and for a refusal flagged in the last step before the loop exits.
- **The AI-affordability pre-check** (`ForcedPlayRun.playChosen`, `!affordable ||` removed): green.
- **The forced "yes, pay" refusal** (`ForcingSeatController.payCostToPreventEffect`, `forced.costNotPaid()` removed): green.
- **The attack re-target branch** (`ForcedPlayRun.declared`, `declared != defender` → `declared == null`): green. Nothing tests an AI attacking the wrong defender with the forced creature.
- **The restart checks at copy time** (`Forks.java:483`) **and in `InterventionFork`** (line 383): green. Both are belts: every boundary a fork is taken at has an empty stack. Either test them or say in forks.md that they are belts.

### 6. A forced "yes" to a "you may pay" is not forced
Severity: should fix (latent: no registered rule uses it today). Confidence: high.

`ForcingSeatController.payCostToPreventEffect` answers a forced yes by calling `super.payCostToPreventEffect`. Forge's `PlayerControllerAi.payCostToPreventEffect` (forge-ai `PlayerControllerAi.java:1320`) first asks `willPayUnlessCost` and returns false when the AI would not pay. So a forced yes pays only when the AI would have paid anyway, and is otherwise refused as `forced play cost not paid`. FR-010b and the contract promise "a 'you may' answered yes or no". A forced yes should go straight to `canPayCost` and `CostPayment.payComputerCosts`. `RandomSeatController` does not consult the forced play in `payCostToPreventEffect` at all.

### 7. ForcedPlayRun state may be reached from the AI's evaluation thread
Severity: minor. Confidence: low.

`AiController` runs `canPlayAndPayFor` on a "Game AI Eval" thread (forge-ai `AiController.java:1606-1720`). On a timeout under Java 20+, that thread keeps running because `Thread.stop` is gone. Any AI logic that calls `player.getController().confirmAction(...)` during evaluation reaches `ForcedPlayRun.answer`, whose fields (`made`, `refusal`, `pending`) are plain and unsynchronised. It also consumes the play on the first confirmation of any ability hosted by the source, not necessarily the named "you may". I found no registered rule where this triggers (118.12b's source has no activated line). Possible fixes: guard on `Thread.currentThread()` being the game thread, and match the trigger or line rather than only the host.

### 8. Documentation slips
Severity: minor. Confidence: high.
- forks.md and the contract say a forced play's stream is "drawn only by a fork that forces, so the other forks' seeds stay as they were". `forkSeeds` is sequential per game, so every fork after a forcing one in the same game draws shifted seeds. This is harmless for determinism, but the sentence is false as worded.
- The contract says `forced play cost not paid` means "the AI's mana payment or a named cost card failed". It is also raised before any payment is tried, when `ComputerUtilCost.canPayCost` says no at the first priority where the rules allow the line.

### 9. Unused API and a silent override
Severity: minor. Confidence: high.
- `KeptRecords.and` and `KeptRecords.isEmpty` are unused in main code.
- `ForcedPlay.withX` and `ForcedPlay.at` are used by no registered rule (`at` only by a test).
- `Application.keeping` replaces the kept set instead of combining with it, so `keepingNegativesOf(a).keepingCandidatesOf(b)` silently drops the first.
- `ForcedPlayRun.aim` sets the drawn charm modes, but `RandomSpellPlayer.play` then calls `CharmEffect.makeChoices`, which re-asks the seat. A forced charm's drawn modes and named targets would be overwritten. No rule forces a charm today.

### 10. ForkIsolationTest's forced-play coverage rests on 118.11
Severity: minor, but must be done together with the 118.11 removal. Confidence: high.
- ForkIsolationTest proves "forks with a forced play leave the live game byte-identical" only through a forced 118.11 fork (Siege-Gang Commander, a plain-AI forcing seat, `"aimed_at":["118.11"]` asserted). When 118.11 leaves the list, its class and this assertion go too. The implementer must switch the test to another forcing rule that a match deck reaches.
- The random-seat `ForcedPlaySeat` path is covered only by the unit test `aRandomSeatPlaysTheForcedLineFirst`, not by an isolation match.

## Checked and found correct

- **Two-sided pairs in played games.** All 42 `DirectedForkPlayedForwardTest` cases ran with my own dump, and every fork was tagged with the real `tag_record`. These rules' forks supply their lacking side: 104.2b, 104.3d, 104.4a, 111.10v, 118.12b, 405.6d, 601.2c, 607.2k, 614.17d, 701.24e, 701.42a, 701.44b, 701.50f, 701.57b, 702.19f, 702.53a, 702.65b, 702.71a, 702.84a (fixed through 702.84a-4), 702.118b, 702.124j, 702.126a, 702.127a, 702.131a, 702.187b, 702.188a, 704.5b, 712.4a, 712.14c, 718.3a. 704.5f is always listed and its fork gives does_not_fire. DF-A kept 701.42a, 712.4a and 712.14c rather than dropping them, and the played forks back that.
- **Precondition negative tests.** 11 mutations in one batch (ReplacedSacrificeCost, MutagenToken, LifeTotalWin, CantEnterBattlefield, PartnerWithSearch, Dethrone, CantGainLifeExchange, Champion, Bolster, BestowIllegalTarget, AscendBlessing): every one turned its `BindsNothing…` test red, 12 failures in all.
- **Core guards that are tested.** These went red when reverted: kept candidates (`PriorityCandidates`, 7 kept cases), kept negatives (`TriggerChannel`: 608.2p, 702.105a), the per-step restart check (the Karn test), the `canTarget` check (`aTargetTheLineCannotTake…`), and the named-cost count in `RoleCostDecision.pick`.
- **Forced versus live play.** The same line played live and forced in one game gives the same record sequence and identical cost halves (Demonic Tutor in 701.24e: mana B:2; Fleshwrither's transfigure in 702.71a: B:3 and a sacrifice). The only differences come from the edited board. `RandomSpellPlayer.play` matches `ComputerUtil.handlePlayingSpellAbility` in the current ../forge line for line, apart from the cost decision. The line is chosen only when `PlayabilityVerdict.playable()` holds (timing, affordability by the rules, a legal target), targets pass `canTarget` and `isTargetNumberValid`, and the cost goes through `CostPayment.payComputerCosts`.
- **validate-corpus on the 25 forced-play games.** Link ids 35/35 paired, every cost half met by its effect half, gen-3 members complete. Its 4 failures (record_id repeats, a single outcome value, a 13% duplicate kind, the trigger ratio on 32 records) come from concatenating test games that share one game id, not from the forced plays.
- **Refusals.** `ForkRefusal` is caught in `Forks.ForkRun.take` and counted by `ForkRefusals`, never by `HookFailures`. The three forced-play refusals, `game restart` and the edit refusals all throw before `writeForkLines`, so the held lines are dropped. The tests assert the count and that nothing was written.
- **Restart.** Only Karn Liberated has `RestartGame` in cardsfolder. A stack item resolves at the earliest in the main-loop step after the one that put it there, and the check runs before every step. The check is also made at copy time and before `InterventionFork`'s `resolveStack`, which gives no priority. Probes resolve nothing. I found no path by which a restart resolves inside a fork.
- **Isolation.** ForkIsolationTest and ForcedPlayTest are green on the clean tree (13 tests). The dry run reads the live board with its own `Random(0)` and on a copy of the line. The forcing seat exists in the copy only, and the live random seat's new confirm overrides delegate unchanged when nothing is forced.
- **Kept records (FR-010c).** They are written through `writer.write`, so they count toward the per-game cap. They enter neither the trigger reservoir nor the mana reservoir, and use only the fork's own sampling seed. The per-line cap of 24 and deduplication still apply (`TriggerChannel`). This matches the spec, the contract and forks.md.
- **The list.** `directed-rules --counts rules/directed-fork-counts.json` regenerates it unchanged: 49 rules, git clean. `test_directed_fork_flags.py` and `test_rules_catalogue.py`: 94 passed.
- **Python consumers.** `rule_test_eligible` excludes fork, aimed and directed records. `rule-counts` counts `aimed_at` records as aimed. `validate-corpus` reads forced-play records as ordinary fork records. None needs to know a play was forced, as the contract says.
- **Removed-rule reasons I found true** against the code and Forge (several rest on the standing brief's Forge facts): 104.6, 400.6, 704.7, 702.35a, 702.167a, 111.10n, 704.5v, 730.2g, 707.10d, 712.11d, 701.12h (trigger targets are not forceable), 701.24g, 119.8, 702.62c, 702.154a, 702.85a, 702.102c.
- **Spec and docs.** FR-010, FR-010b, FR-010c, FR-010d, FR-011, FR-014b (the list, its reasons and the count of 49), record-schema-delta, the contract's forced-play catalogue and the experiment record's known-limits entry (22 rules in four groups, matching `NOT_DIRECTED`) all agree with the code. The exceptions are finding 1 (FR-014a and the contract sentence) and the doc slips in finding 8.
- **Reflection.** `NOTE_CONFIRM` fails loudly at use (an `IllegalStateException`, counted as a body failure), not silently.

## Not checked, and why

- **How often each precondition binds in real sealed games.** No game-state corpus exists to replay them against. Finding 4 is argued from the condition, not measured.
- **The rescues in finding 3.** Not played: each needs a new rule class or a re-registration.
- **701.42b, 701.44d, 702.19c.** Not played: no played-forward case exists for them, and the packages did not touch them.
- **The thread race in finding 7.** Not reproduced.
- **A forced play inside a full sealed match through a random seat.** Only unit-tested.
