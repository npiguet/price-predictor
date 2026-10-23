package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.ability.AbilityFactory;
import forge.game.ability.AbilityUtils;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static com.pricepredictor.connector.effects.RecordLines.events;
import static com.pricepredictor.connector.effects.RecordLines.params;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The outcome hook: an effect's own account of what it computed, the event
 * that account becomes, and which game it is admitted from.
 */
@ExtendWith(ForgeExtension.class)
class OutcomeHandlerTest {

    @TempDir
    Path tempDir;

    /**
     * Static, and shared with the production path: without a reset here,
     * whichever test trips it first leaves every test after it in this class
     * unable to see its own "did this print" outcome.
     */
    @BeforeEach
    void resetUnknownOutcomeTypeLatch() {
        TestLatches.reset(ClauseOutcomeRouter.class, "UNKNOWN_OUTCOME_TYPE_REPORTED");
    }

    private static final Method ON_OUTCOME =
            HookShapes.method(HookShapes.Outcome.class, "onOutcome");

    // ── the argument order ──────────────────────────────────────────────

    /**
     * The hook binds by argument position, and no compiler checks that. A
     * reorder in Forge is invisible to both compilers and must fail here
     * rather than in a corpus -- which is why this uses a real, distinct
     * {@link SpellAbility} rather than null for position 0: a null would not
     * notice a swap between the ability and either of the other two
     * arguments, only {@code PatchHooksTest.eachStandInIsForgesOwnDeclaration}
     * would, and that test is reflective and skips on an unpatched checkout.
     */
    @Test
    void anOutcomeReportBecomesAnEventOfThatType() throws Throwable {
        SpellAbility sa = TestCards.scriptedAbility("Alchemist's Gambit", "AddTurn");

        Map<String, Object> written = onlyEvent(
                new Object[]{sa, "vote_taken", Map.of("tally", Map.of("beast", 2))});

        assertEquals("vote_taken", written.get("type"), written.toString());
        assertEquals(Map.of("beast", 2L), params(written).get("tally"), written.toString());
    }

    /**
     * An outcome that names no subjects has none: a coin flip or a token
     * made is not something done to the card whose ability did it, so the
     * acting host is not a default.
     */
    @Test
    void anOutcomeNamingNoSubjectsHasNone() throws Throwable {
        SpellAbility sa = TestCards.scriptedAbility("Alchemist's Gambit", "AddTurn");

        Map<String, Object> written =
                onlyEvent(new Object[]{sa, "coin_flipped", Map.of("won", true)});

        assertEquals("coin_flipped", written.get("type"), written.toString());
        assertEquals(List.of(), written.get("subjects"),
                "the acting host is not a subject: " + written);
    }

    /**
     * A {@code token_created} note hands over a sample token, which becomes
     * the token's script id and characteristics as the board would render
     * that token -- keywords included, spelled the way an entity's are --
     * and never reaches the event itself.
     */
    @Test
    void aTokenSampleBecomesItsScriptIdAndCharacteristics() throws Throwable {
        Card bat = TestCards.token("b_1_1_bat_flying");
        SpellAbility sa = TestCards.scriptedAbility("Alchemist's Gambit", "AddTurn");

        Map<String, Object> written = onlyEvent(
                new Object[]{sa, "token_created", Map.of("sample", bat, "count", 2)});

        assertEquals("token_created", written.get("type"), written.toString());
        Map<String, Object> params = params(written);
        assertEquals(Set.of("token_script_id", "characteristics", "count"),
                params.keySet(), written.toString());
        assertEquals("b_1_1_bat_flying", SnapshotBuilder.tokenScriptIdOf(bat));
        assertEquals("b_1_1_bat_flying", params.get("token_script_id"), written.toString());
        assertEquals(2L, params.get("count"), written.toString());
        assertEquals(List.of("flying"), SnapshotBuilder.keywordNames(bat.getKeywords()));
        assertEquals(Map.of("power", 1L, "toughness", 1L,
                        "types", bat.getType().toString(), "keywords", List.of("flying")),
                params.get("characteristics"), written.toString());
        assertFalse(written.toString().contains("sample"), written.toString());
        assertEquals(List.of(), written.get("subjects"), written.toString());
    }

    /**
     * {@code damage_prevented} names who the damage would have been dealt
     * to, a player or a creature, as the event's subject.
     */
    @Test
    void damagePreventedNamesItsRecipient() throws Throwable {
        Card creature = TestCards.build("Grizzly Bears");
        Player player = new Player("recipient", TestCards.game(), 93201);

        List<Map<String, Object>> events = outcomeEvents(
                new Object[]{null, "damage_prevented",
                        Map.of("amount", 2, "subjects", List.of(creature))},
                new Object[]{null, "damage_prevented",
                        Map.of("amount", 2, "subjects", List.of(player))},
                new Object[]{null, "damage_prevented",
                        Map.of("amount", 2, "subjects", List.of())});

        assertEquals(3, events.size(), events.toString());
        assertEquals(List.of(SnapshotBuilder.entityId(creature)), events.get(0).get("subjects"),
                events.get(0).toString());
        assertEquals(Set.of("amount"), params(events.get(0)).keySet(), events.get(0).toString());
        assertEquals(List.of(SnapshotBuilder.playerId(player)), events.get(1).get("subjects"),
                events.get(1).toString());
        assertEquals(List.of(), events.get(2).get("subjects"), events.get(2).toString());
    }

    /**
     * A reveal from a library names the players it came from; any other
     * reveal arrives with an empty list and names nobody.
     */
    @Test
    void cardRevealedNamesTheLibrariesItCameFrom() throws Throwable {
        Player first = new Player("first", TestCards.game(), 93202);
        Player second = new Player("second", TestCards.game(), 93203);

        List<Map<String, Object>> events = outcomeEvents(
                new Object[]{null, "card_revealed",
                        Map.of("count", 2, "from_zone", "Library",
                                "subjects", List.of(first, second))},
                new Object[]{null, "card_revealed",
                        Map.of("count", 1, "from_zone", "Hand", "subjects", List.of())});

        assertEquals(2, events.size(), events.toString());
        Map<String, Object> fromLibraries = events.get(0);
        assertEquals(List.of(SnapshotBuilder.playerId(first), SnapshotBuilder.playerId(second)),
                fromLibraries.get("subjects"), fromLibraries.toString());
        assertEquals(Set.of("count", "from_zone"), params(fromLibraries).keySet(),
                fromLibraries.toString());
        assertEquals(List.of(), events.get(1).get("subjects"), events.get(1).toString());
    }

    /**
     * {@code "subjects"} in the params map is not a generic parameter, even
     * though Detain/Goad/MustBlock report it as a params entry (see
     * {@code EffectRecordOutcomes.note} call sites in
     * {@code GoadEffect}/{@code DetainEffect}/{@code MustBlockEffect}): the
     * schema's {@code restriction_change} row normalizes only
     * {@code restriction} and {@code value}, so a raw {@code subjects} entry
     * left in {@code params} would fail the per-type param allow-list on the
     * Python side. The handler converts it to the event's own top-level
     * subject refs instead.
     *
     * <p>Exact match, not a containment check: {@code restriction_change}'s
     * acting card (the host, "Alchemist's Gambit" here) must not appear in
     * the subject list alongside the two cards actually goaded, and
     * containment cannot tell "exactly these two" from "these two, plus an
     * extra false one".
     */
    @Test
    void restrictionChangeSubjectsBecomeEventSubjectsNotAParam() throws Throwable {
        Card goaded1 = TestCards.build("Sol Ring");
        Card goaded2 = TestCards.build("Mox Pearl");
        SpellAbility sa = TestCards.scriptedAbility("Alchemist's Gambit", "AddTurn");

        Map<String, Object> written = onlyEvent(new Object[]{sa, "restriction_change",
                Map.of("restriction", "goad", "value", true,
                        "subjects", List.of(goaded1, goaded2))});

        assertEquals("restriction_change", written.get("type"), written.toString());
        assertEquals(Map.of("restriction", "goad", "value", true), params(written),
                "subjects must not leak into params: " + written);
        assertEquals(List.of(SnapshotBuilder.entityId(goaded1), SnapshotBuilder.entityId(goaded2)),
                written.get("subjects"),
                "exactly the two goaded cards -- not the acting host as well: " + written);
    }

    /**
     * An outcome type outside {@link ClauseOutcomeRouter#outcomeHandler()}'s
     * known set writes no event and is reported, once. The completeness
     * guard's textual scan is satisfied by a type named in a javadoc comment
     * as well as by real code, so without this report a deleted {@code
     * note(...)} call or a misspelled event name would go dark with nothing
     * saying so.
     */
    @Test
    void anUnrecognizedOutcomeTypeWritesNoEventAndIsReportedOnce() throws Throwable {
        StringBuilder stderr = new StringBuilder();
        Path shard = ShardRun.shardOf(tempDir, TestCards.game(), (bracket, collectors) -> {
            stderr.append(StdStreams.captureStderr(() -> {
                collectors.clauses().outcomeHandler().invoke(null, ON_OUTCOME,
                        new Object[]{null, "dungeon_entered", Map.of()});
                collectors.clauses().outcomeHandler().invoke(null, ON_OUTCOME,
                        new Object[]{null, "dungeon_entered", Map.of()});
                collectors.clauses().outcomeHandler().invoke(null, ON_OUTCOME,
                        new Object[]{null, "another_unknown_type", Map.of()});
            }));
            bracket.finishGame(Set.of());
        });

        assertFalse(Files.exists(shard), "an unrecognized type must not become an event");
        String output = stderr.toString();
        long lineCount = output.isBlank() ? 0 : output.strip().lines().count();
        assertEquals(1, lineCount, "exactly one report despite three unrecognized types: " + output);
        assertTrue(output.contains("dungeon_entered"), "the report names the first unrecognized type: " + output);
    }

    /**
     * {@code damage_prevented}'s {@code "source"} is converted to an entity
     * ref, matching every other producer of a {@code source} key
     * ({@code damage_dealt}, and the trigger-derived {@code damage_prevented}
     * via {@code subjectOf}). Forge passes the {@code Card} itself rather
     * than a display name for exactly this conversion -- a display name
     * would silently fail a join against those refs, and collide two sources
     * sharing a name.
     */
    @Test
    void damagePreventedSourceBecomesAnEntityRef() throws Throwable {
        Card source = TestCards.build("Grizzly Bears");

        Map<String, Object> written = onlyEvent(new Object[]{null, "damage_prevented",
                Map.of("amount", 3, "source", source)});

        assertEquals("damage_prevented", written.get("type"), written.toString());
        assertEquals(3L, params(written).get("amount"), written.toString());
        assertEquals(SnapshotBuilder.entityId(source), params(written).get("source"),
                "a ref like damage_dealt's, not a display name: " + written);
    }

    // ── another game's ability must not reach this game's bracket ────────

    /**
     * An outcome reported for an ability that belongs to a DIFFERENT
     * {@code Game} -- exactly the shape {@code ForkCollector.forceResolution}
     * produces, since {@code GameSimulator.resolveStack} runs the real
     * resolution pipeline against a {@code GameCopier} clone under
     * {@code AIOption.USE_FULL_SIMULATION} regardless of how this game's own
     * lobby seats were registered -- must not reach the live
     * game's bracket. {@code GameCopier} builds every copied {@code Card}
     * against a new {@code Game} object, so a fork's ability's own host card
     * already carries a different {@code Game} than this collector's.
     *
     * <p>{@code vote_taken} and {@code coin_flipped} are both outcome-hook
     * types, so this exercises {@link ClauseOutcomeRouter#outcomeHandler()}
     * rather than the clause hook -- {@code ClauseContractTest} covers the
     * clause hook's own half of the same check.
     *
     * <p>The same report for a live ability follows it into the same bracket
     * and is written, so the fork's is refused by the game-identity check
     * rather than lost to some other failure this test would then wrongly
     * credit to it.
     */
    @Test
    void anOutcomeFromAnotherGameDoesNotReachThisGamesBracket() throws Throwable {
        Card forkHost = TestBoard.cardInAnotherGame("Grizzly Bears");
        SpellAbility forkAbility = AbilityFactory.getAbility(
                "SP$ AddTurn | ValidTgts$ Player | NumTurns$ 1", forkHost);

        SpellAbility liveAbility = TestCards.scriptedAbility("Alchemist's Gambit", "AddTurn");
        List<String> records = ShardRun.records(tempDir, TestCards.game(), (bracket, collectors) -> {
            bracket.beginBracket(liveAbility);

            collectors.clauses().outcomeHandler().invoke(null, ON_OUTCOME,
                    new Object[]{forkAbility, "vote_taken", Map.of("tally", Map.of("beast", 2))});
            collectors.clauses().outcomeHandler().invoke(null, ON_OUTCOME,
                    new Object[]{liveAbility, "vote_taken", Map.of("tally", Map.of("cat", 1))});

            bracket.endBracket(liveAbility.getId(), false);
            bracket.onNextAction();
        });

        String shard = String.join("\n", records);
        assertFalse(shard.contains("beast"),
                "a fork's outcome must not land in the live game's record: " + shard);
        assertTrue(shard.contains("\"tally\":{\"cat\":1}"),
                "the live ability's own outcome is written beside it: " + shard);
    }

    // ── the shard, read back ────────────────────────────────────────────

    /**
     * The one event the outcome hook filed for {@code report}, as the shard
     * records it.
     */
    private Map<String, Object> onlyEvent(Object[] report) throws Throwable {
        List<Map<String, Object>> events = outcomeEvents(report);
        assertEquals(1, events.size(), events.toString());
        return events.get(0);
    }

    /**
     * The events the outcome hook filed for these reports, in order, as the
     * shard records them.
     *
     * <p>No resolution is open around them, as around the engine's own choke
     * points: the events wait as the game's orphans and the end of the game
     * writes them in a record of their own, with no bracket to stamp a cause
     * on them.
     */
    private List<Map<String, Object>> outcomeEvents(Object[]... reports) throws Throwable {
        List<String> records = ShardRun.records(tempDir, TestCards.game(), (bracket, collectors) -> {
            for (Object[] report : reports) {
                collectors.clauses().outcomeHandler().invoke(null, ON_OUTCOME, report);
            }
            bracket.finishGame(Set.of());
        });
        assertEquals(1, records.size(), records.toString());
        return events(records.get(0));
    }

    // ── a null-ability outcome's own game signal, at the shapes ──────────
    // ── production reaches ───────────────────────────────────────────────

    /**
     * {@code GameAction.reveal}'s {@code card_revealed} and {@code
     * ReplacementHandler}'s {@code damage_prevented} report through {@code
     * EffectRecordOutcomes.note} with a literal {@code null} ability,
     * <em>always</em> -- so a game-identity gate on {@code args[0] instanceof
     * SpellAbility} alone would drop both channels on the live game, not just
     * on a fork.
     *
     * <p>Asserted on what reaches the shard, as every test here is: an event
     * read off the handler before its delivery gate cannot see a channel that
     * dies at the gate.
     *
     * <p>Real combat damage reaches this method with <em>no</em> bracket open
     * at all, which is the shape used here, rather than a bracket open for an
     * ability nothing is resolving ("pointer null" and "bracket open"), a
     * composite production never produces.
     */
    @Test
    void aNullAbilityOutcomeDuringLiveCombatDamageReachesTheShard() throws Throwable {
        // Genuinely mid combat-damage step: the one production shape a live
        // damage_prevented reaches with neither the fork flag nor the
        // resolving-clause pointer set, since Combat.dealAssignedDamage is a
        // turn-based action and no resolution bracket is ever open around it.
        // Its own game rather than the shared TestCards.game(): the phase is
        // what routes an event into the combat bracket, and a shared game left
        // mid combat-damage would route every other test's events there too.
        Game game = TestBoard.twoPlayerGame(PhaseType.COMBAT_DAMAGE);
        List<String> records = ShardRun.records(tempDir, game, (bracket, collectors) -> {
            collectors.clauses().outcomeHandler().invoke(null, ON_OUTCOME,
                    new Object[]{null, "damage_prevented", Map.of("amount", 3)});

            // Flushes the combat bracket the null-ability report above opened,
            // the way BusBracketCollectorTest.theCombatThatEndedTheGameIsWritten
            // does for the same reason: a game ending in combat damage never
            // reaches the ordinary phase-boundary flush.
            bracket.finishGame(Set.of());
        });

        String shard = String.join("\n", records);
        assertTrue(shard.contains("damage_prevented"),
                "a live combat-damage prevention must still reach the shard: " + shard);
    }

    /**
     * Pins the resolving-clause pointer branch directly. The combat-damage
     * test above leaves the pointer null by construction, and the probe test
     * below only exercises the flag, so without this one a single-signal
     * {@code return !ForkCollector.isForkRunningOnThisThread();} would pass
     * every test in this class, as would dropping the {@link
     * LiveGame#owns(CardTraitBase)} delegate and returning {@code true}
     * whenever the pointer is merely non-null. This resolves a real ability -- through {@code
     * AbilityUtils.resolve}, not a synthesized call -- so the pointer is
     * genuinely set the way {@code ForkCollector.forceResolution}'s own
     * forced ability sets it, and installs a real {@code
     * EffectRecordClauseListener} (the same reflective seam {@link
     * PatchedCollectors#install()} itself uses) to fire the null-ability
     * report from {@code onClauseResolving}, while the pointer names the
     * fork's own ability.
     *
     * <p>The live game's own bracket is opened for an unrelated live ability
     * around the fork's resolution, purely so the buffered event has
     * somewhere to flush to -- {@code BusBracketCollector} only ever writes
     * {@code bracketEvents} on a matching {@code endBracket}, discarding them
     * unflushed otherwise, and an empty shard would read the same under the
     * correct implementation and the naive one. This
     * mirrors {@code anOutcomeFromAnotherGameDoesNotReachThisGamesBracket}'s
     * own live bracket, opened for the same reason. It answers a different
     * question than which ability the pointer names: the pointer is set (and
     * read) entirely inside {@code AbilityUtils.resolve(forkAbility)}, which
     * runs nested inside this bracket but neither reads nor writes it.
     */
    @Test
    void aForksAbilitysNullAbilityOutcomeIsDroppedWhileItIsGenuinelyResolving() throws Throwable {
        Card forkHost = TestBoard.cardInAnotherGame("Grizzly Bears");
        Game fork = forkHost.getGame();
        Player forkActor = new Player("fork-actor", fork, 93001);
        Player forkTarget = new Player("fork-target", fork, 93002);
        SpellAbility forkAbility = AbilityFactory.getAbility(
                "SP$ AddTurn | ValidTgts$ Player | NumTurns$ 1", forkHost);
        forkAbility.setActivatingPlayer(forkActor);
        forkAbility.resetTargets();
        forkAbility.getTargets().add(forkTarget);
        SpellAbility flushAbility = TestCards.scriptedAbility("Alchemist's Gambit", "AddTurn");

        List<String> records = ShardRun.records(tempDir, TestCards.game(), (bracket, collectors) -> {
            bracket.beginBracket(flushAbility);

            resolveReportingANullAbilityOutcome(collectors, forkAbility);
            bracket.endBracket(flushAbility.getId(), false);
            bracket.onNextAction();
        });

        String shard = String.join("\n", records);
        assertFalse(shard.contains("damage_prevented"),
                "a fork ability's own null-ability outcome, reported while it "
                        + "is genuinely resolving, must not reach the live shard: " + shard);
    }

    /**
     * Resolve {@code ability} through {@code AbilityUtils.resolve}, as the
     * engine resolves one, with a clause listener installed that reports a
     * null-ability {@code damage_prevented} from {@code onClauseResolving}:
     * the report arrives while the resolving-clause pointer names {@code
     * ability}. Skipped on a checkout without the clause hook.
     */
    private static void resolveReportingANullAbilityOutcome(
            Channels collectors, SpellAbility ability) {
        boolean installed = PatchHooks.install(PatchHooks.ABILITY_UTILS,
                "setEffectRecordClauseListener", (proxy, method, args) -> {
                    if ("onClauseResolving".equals(method.getName())) {
                        try {
                            collectors.clauses().outcomeHandler().invoke(null, ON_OUTCOME,
                                    new Object[]{null, "damage_prevented", Map.of("amount", 3)});
                        } catch (Throwable t) {
                            throw new RuntimeException(t);
                        }
                    }
                    return null;
                });
        Assumptions.assumeTrue(installed, "the checkout lacks the clause hook");
        try {
            AbilityUtils.resolve(ability);
        } finally {
            PatchHooks.uninstall(PatchHooks.ABILITY_UTILS, "setEffectRecordClauseListener");
        }
    }

    /**
     * The admitting half of the same pin: a <em>live</em> ability's own
     * null-ability outcome, reported while the pointer genuinely names it
     * (not null, not a fork), must be admitted. Where the fork test above
     * fires from {@code intervene}'s own phase-boundary timing (no live
     * resolution in flight), this fires while that same live ability's own
     * bracket is open -- production's shape for a live reveal or damage
     * prevention nested inside a live spell's own resolution.
     */
    @Test
    void aLiveAbilitysNullAbilityOutcomeIsAdmittedWhileItIsGenuinelyResolving() throws Throwable {
        SpellAbility liveAbility = AbilityFactory.getAbility(
                "SP$ AddTurn | ValidTgts$ Player | NumTurns$ 1", TestCards.build("Grizzly Bears"));
        Player liveActor = new Player("live-actor", TestCards.game(), 93003);
        Player liveTarget = new Player("live-target", TestCards.game(), 93004);
        liveAbility.setActivatingPlayer(liveActor);
        liveAbility.resetTargets();
        liveAbility.getTargets().add(liveTarget);

        List<String> records = ShardRun.records(tempDir, TestCards.game(), (bracket, collectors) -> {
            bracket.beginBracket(liveAbility);

            resolveReportingANullAbilityOutcome(collectors, liveAbility);
            bracket.endBracket(liveAbility.getId(), false);
            bracket.onNextAction();
        });

        String shard = String.join("\n", records);
        assertTrue(shard.contains("damage_prevented"),
                "a live ability's own null-ability outcome, reported while it "
                        + "is genuinely resolving, must reach the shard: " + shard);
    }

    /**
     * The gate does not fail open on {@code null}: that would readmit
     * exactly what it exists to exclude, because {@code
     * ForkCollector.probe}'s own combat-damage counterfactual fires this same
     * {@code null}-ability {@code damage_prevented} report -- {@code
     * Combat.dealAssignedDamage} always passes a null cause, live or forked.
     * A probe's damage step is a turn-based action with no ability resolving
     * and nothing on the resolving-clause pointer, which is why {@link
     * ForkCollector#isForkRunningOnThisThread()} exists. FORK_RUNNING
     * is set directly here, without needing a real combat to
     * fork. No live bracket is open, matching a probe's own
     * timing: like {@code intervene}, it does not run nested inside a live
     * resolution.
     */
    @Test
    void aNullAbilityOutcomeFromAProbesOwnCombatDoesNotReachTheLiveShard() throws Throwable {
        List<String> records = ShardRun.records(tempDir, TestCards.game(), (bracket, collectors) -> {
            var forkRunning = ForkCollector.class.getDeclaredField("FORK_RUNNING");
            forkRunning.setAccessible(true);
            @SuppressWarnings("unchecked")
            ThreadLocal<Boolean> flag = (ThreadLocal<Boolean>) forkRunning.get(null);
            flag.set(true);
            try {
                collectors.clauses().outcomeHandler().invoke(null, ON_OUTCOME,
                        new Object[]{null, "damage_prevented", Map.of("amount", 3)});
            } finally {
                flag.set(false);
            }

            // The same report with no fork running is admitted, so the one
            // above was built and refused rather than lost on the way.
            collectors.clauses().outcomeHandler().invoke(null, ON_OUTCOME,
                    new Object[]{null, "damage_prevented", Map.of("amount", 4)});

            // The events reach the bracket's buffer, which only a flush
            // writes; without one the shard is empty whether or not the gate
            // exists.
            bracket.finishGame(Set.of());
        });

        String shard = String.join("\n", records);
        assertFalse(shard.contains("\"amount\":3"),
                "a probe's own combat-damage report must not land in the live game's record: "
                        + shard);
        assertTrue(shard.contains("\"amount\":4"),
                "the live report beside it is written: " + shard);
    }
}
