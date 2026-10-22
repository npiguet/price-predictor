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
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

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
        TestLatches.reset(PatchedCollectors.class, "UNKNOWN_OUTCOME_TYPE_REPORTED");
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

        String written = onlyEvent(
                new Object[]{sa, "vote_taken", Map.of("tally", Map.of("beast", 2))});

        assertTrue(written.startsWith("{\"type\":\"vote_taken\""), written);
        assertTrue(written.contains("\"tally\":{\"beast\":2}"), written);
    }

    /**
     * An outcome that names no subjects has none: a coin flip or a token
     * made is not something done to the card whose ability did it, so the
     * acting host is not a default.
     */
    @Test
    void anOutcomeNamingNoSubjectsHasNone() throws Throwable {
        SpellAbility sa = TestCards.scriptedAbility("Alchemist's Gambit", "AddTurn");

        String written = onlyEvent(new Object[]{sa, "coin_flipped", Map.of("won", true)});

        assertTrue(written.startsWith("{\"type\":\"coin_flipped\""), written);
        assertTrue(written.contains("\"subjects\":[]"),
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

        String written = onlyEvent(
                new Object[]{sa, "token_created", Map.of("sample", bat, "count", 2)});

        assertTrue(written.startsWith("{\"type\":\"token_created\""), written);
        assertEquals(Set.of("token_script_id", "characteristics", "count"),
                paramKeys(written), written);
        assertEquals("b_1_1_bat_flying", SnapshotBuilder.tokenScriptIdOf(bat));
        assertTrue(written.contains("\"token_script_id\":\"b_1_1_bat_flying\""), written);
        assertTrue(written.contains("\"count\":2"), written);
        assertEquals(List.of("flying"), SnapshotBuilder.keywordNames(bat.getKeywords()));
        assertTrue(written.contains("\"characteristics\":{\"power\":1,\"toughness\":1,"
                + "\"types\":" + Json.string(bat.getType().toString())
                + ",\"keywords\":[\"flying\"]}"), written);
        assertFalse(written.contains("sample"), written);
        assertTrue(written.contains("\"subjects\":[]"), written);
    }

    /**
     * {@code damage_prevented} names who the damage would have been dealt
     * to, a player or a creature, as the event's subject.
     */
    @Test
    void damagePreventedNamesItsRecipient() throws Throwable {
        Card creature = TestCards.build("Grizzly Bears");
        Player player = new Player("recipient", TestCards.game(), 93201);

        List<String> events = outcomeEvents(
                new Object[]{null, "damage_prevented",
                        Map.of("amount", 2, "subjects", List.of(creature))},
                new Object[]{null, "damage_prevented",
                        Map.of("amount", 2, "subjects", List.of(player))},
                new Object[]{null, "damage_prevented",
                        Map.of("amount", 2, "subjects", List.of())});

        assertEquals(3, events.size(), events.toString());
        assertTrue(events.get(0).contains("\"subjects\":"
                + Json.stringArray(List.of(SnapshotBuilder.entityId(creature)))), events.get(0));
        assertEquals(Set.of("amount"), paramKeys(events.get(0)), events.get(0));
        assertTrue(events.get(1).contains("\"subjects\":"
                + Json.stringArray(List.of(SnapshotBuilder.playerId(player)))), events.get(1));
        assertTrue(events.get(2).contains("\"subjects\":[]"), events.get(2));
    }

    /**
     * A reveal from a library names the players it came from; any other
     * reveal arrives with an empty list and names nobody.
     */
    @Test
    void cardRevealedNamesTheLibrariesItCameFrom() throws Throwable {
        Player first = new Player("first", TestCards.game(), 93202);
        Player second = new Player("second", TestCards.game(), 93203);

        List<String> events = outcomeEvents(
                new Object[]{null, "card_revealed",
                        Map.of("count", 2, "from_zone", "Library",
                                "subjects", List.of(first, second))},
                new Object[]{null, "card_revealed",
                        Map.of("count", 1, "from_zone", "Hand", "subjects", List.of())});

        assertEquals(2, events.size(), events.toString());
        String fromLibraries = events.get(0);
        assertTrue(fromLibraries.contains("\"subjects\":" + Json.stringArray(List.of(
                SnapshotBuilder.playerId(first), SnapshotBuilder.playerId(second)))),
                fromLibraries);
        assertEquals(Set.of("count", "from_zone"), paramKeys(fromLibraries), fromLibraries);
        assertTrue(events.get(1).contains("\"subjects\":[]"), events.get(1));
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

        String written = onlyEvent(new Object[]{sa, "restriction_change",
                Map.of("restriction", "goad", "value", true,
                        "subjects", List.of(goaded1, goaded2))});

        assertTrue(written.startsWith("{\"type\":\"restriction_change\""), written);
        assertEquals(Set.of("restriction", "value"), paramKeys(written),
                "subjects must not leak into params: " + written);
        assertTrue(written.contains("\"restriction\":\"goad\""), written);
        assertTrue(written.contains("\"value\":true"), written);
        assertTrue(written.contains("\"subjects\":" + Json.stringArray(List.of(
                SnapshotBuilder.entityId(goaded1), SnapshotBuilder.entityId(goaded2)))),
                "exactly the two goaded cards -- not the acting host as well: " + written);
    }

    /**
     * An outcome type outside {@link PatchedCollectors#outcomeHandler()}'s
     * known set is exactly the failure this whole plan exists to end,
     * recreated one level up: before this check existed, all nine of this
     * task's types were referenced only inside a javadoc comment, which
     * satisfies the completeness guard's textual scan just as well as real
     * code -- so a deleted {@code note(...)} call or a misspelled event name
     * would go dark and the guard built to notice would say nothing. A
     * recognized type still writes an event; an unrecognized one does not,
     * and is reported exactly once.
     */
    @Test
    void anUnrecognizedOutcomeTypeWritesNoEventAndIsReportedOnce() throws Throwable {
        StringBuilder stderr = new StringBuilder();
        Path shard = ShardRun.shardOf(tempDir, TestCards.game(), (bracket, collectors) -> {
            stderr.append(StdStreams.captureStderr(() -> {
                collectors.outcomeHandler().invoke(null, ON_OUTCOME,
                        new Object[]{null, "dungeon_entered", Map.of()});
                collectors.outcomeHandler().invoke(null, ON_OUTCOME,
                        new Object[]{null, "dungeon_entered", Map.of()});
                collectors.outcomeHandler().invoke(null, ON_OUTCOME,
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

        String written = onlyEvent(new Object[]{null, "damage_prevented",
                Map.of("amount", 3, "source", source)});

        assertTrue(written.startsWith("{\"type\":\"damage_prevented\""), written);
        assertTrue(written.contains("\"amount\":3"), written);
        assertTrue(written.contains("\"source\":" + Json.string(SnapshotBuilder.entityId(source))),
                "a ref like damage_dealt's, not a display name: " + written);
    }

    // ── F2: another game's ability must not reach this game's bracket ────

    /**
     * F2: an outcome reported for an ability that belongs to a DIFFERENT
     * {@code Game} -- exactly the shape {@code ForkCollector.forceResolution}
     * produces, since {@code GameSimulator.resolveStack} runs the real
     * resolution pipeline against a {@code GameCopier} clone under
     * {@code AIOption.USE_FULL_SIMULATION} regardless of how this game's own
     * lobby seats were registered (ruling R31) -- must not reach the live
     * game's bracket. {@code GameCopier} builds every copied {@code Card}
     * against a new {@code Game} object, so a fork's ability's own host card
     * already carries a different {@code Game} than this collector's.
     *
     * <p>{@code vote_taken} and {@code coin_flipped} (the concrete types
     * named in final-fix-1.md F2) are both outcome-hook types, so this
     * exercises {@link PatchedCollectors#outcomeHandler()} rather than the
     * clause hook -- {@code ClauseContractTest} covers the clause hook's own
     * half of the same fix.
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

            collectors.outcomeHandler().invoke(null, ON_OUTCOME,
                    new Object[]{forkAbility, "vote_taken", Map.of("tally", Map.of("beast", 2))});
            collectors.outcomeHandler().invoke(null, ON_OUTCOME,
                    new Object[]{liveAbility, "vote_taken", Map.of("tally", Map.of("cat", 1))});

            bracket.endBracket(liveAbility.getId(), false);
            bracket.flushHeldResolution();
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
    private String onlyEvent(Object[] report) throws Throwable {
        List<String> events = outcomeEvents(report);
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
    private List<String> outcomeEvents(Object[]... reports) throws Throwable {
        List<String> records = ShardRun.records(tempDir, TestCards.game(), (bracket, collectors) -> {
            for (Object[] report : reports) {
                collectors.outcomeHandler().invoke(null, ON_OUTCOME, report);
            }
            bracket.finishGame(Set.of());
        });
        assertEquals(1, records.size(), records.toString());
        return eventsOf(records.get(0));
    }

    /** Each object of a record's {@code events} array, in order. */
    private static List<String> eventsOf(String record) {
        String opening = "\"events\":[";
        int at = record.indexOf(opening);
        assertTrue(at >= 0, record);
        List<String> events = new ArrayList<>();
        int depth = 0;
        int start = -1;
        for (int i = at + opening.length(); i < record.length(); i++) {
            char c = record.charAt(i);
            if (c == '"') {
                i = closingQuote(record, i);
            } else if (c == '{' || c == '[') {
                if (depth++ == 0) {
                    start = i;
                }
            } else if (c == '}' || c == ']') {
                if (depth == 0) {
                    return events;
                }
                if (--depth == 0) {
                    events.add(record.substring(start, i + 1));
                }
            }
        }
        throw new AssertionError("an unterminated events array: " + record);
    }

    /** The keys of an event's {@code params} object. */
    private static Set<String> paramKeys(String event) {
        String opening = "\"params\":{";
        int at = event.indexOf(opening);
        assertTrue(at >= 0, event);
        Set<String> keys = new HashSet<>();
        int depth = 0;
        for (int i = at + opening.length(); i < event.length(); i++) {
            char c = event.charAt(i);
            if (c == '"') {
                int end = closingQuote(event, i);
                if (depth == 0 && event.charAt(end + 1) == ':') {
                    keys.add(event.substring(i + 1, end));
                }
                i = end;
            } else if (c == '{' || c == '[') {
                depth++;
            } else if (c == '}' || c == ']') {
                if (depth-- == 0) {
                    return keys;
                }
            }
        }
        throw new AssertionError("an unterminated params object: " + event);
    }

    /** Where the string literal opening at {@code at} closes. */
    private static int closingQuote(String json, int at) {
        for (int i = at + 1; i < json.length(); i++) {
            if (json.charAt(i) == '\\') {
                i++;
            } else if (json.charAt(i) == '"') {
                return i;
            }
        }
        throw new AssertionError("an unterminated string: " + json);
    }

    // ── final-fix-3.md item 1 / final-fix-4.md item 1: a null-ability ────
    // ── outcome's own game signal, at the shapes production reaches ──────

    /**
     * The critical regression. {@code GameAction.reveal}'s {@code
     * card_revealed} and {@code ReplacementHandler}'s {@code
     * damage_prevented} (rulings R10, R11/R12) report through {@code
     * EffectRecordOutcomes.note} with a literal {@code null} ability,
     * <em>always</em> -- F2's {@code args[0] instanceof SpellAbility} gate
     * (final-fix-1.md) makes that {@code instanceof} false unconditionally,
     * which silently dropped both channels on the live game, not just on a
     * fork.
     *
     * <p>Asserted on what reaches the shard, as every test here is: an event
     * read off the handler before its delivery gate cannot see this class of
     * bug, which is how the channel went dead unnoticed.
     *
     * <p>final-fix-4.md item 1: replaces the previous version of this test,
     * which opened a resolution bracket for an ability nothing was actually
     * resolving -- a synthetic composite ("pointer null" and "bracket open")
     * production never produces. Real combat damage reaches this method with
     * <em>no</em> bracket open at all, which is the shape used here.
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
            collectors.outcomeHandler().invoke(null, ON_OUTCOME,
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
     * final-fix-4.md item 1 (Critical guard gap): pins the resolving-clause
     * pointer branch directly, which nothing did before this test existed.
     * {@code return !ForkCollector.isForkRunningOnThisThread();} -- the exact
     * single-signal version the report built and discarded to demonstrate the
     * naive-fail-open trap -- passed every test in this class, because no
     * test ever set the pointer: the combat-damage test above leaves it null
     * by construction, and the probe test below only ever exercises the flag.
     * So did dropping the {@link PatchedCollectors#belongsToLiveGame(CardTraitBase)}
     * delegate and returning {@code true} whenever the pointer is merely
     * non-null. This resolves a real ability -- through {@code
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
     * unflushed otherwise (confirmed by hand: this test read back an empty
     * shard under both the correct implementation and the naive one before
     * this bracket was added, which is not evidence of anything). This
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

            boolean installed = PatchHooks.install(PatchHooks.ABILITY_UTILS,
                    "setEffectRecordClauseListener", (proxy, method, args) -> {
                        if ("onClauseResolving".equals(method.getName())) {
                            try {
                                collectors.outcomeHandler().invoke(null, ON_OUTCOME,
                                        new Object[]{null, "damage_prevented", Map.of("amount", 3)});
                            } catch (Throwable t) {
                                throw new RuntimeException(t);
                            }
                        }
                        return null;
                    });
            Assumptions.assumeTrue(installed, "the checkout lacks the clause hook");
            try {
                AbilityUtils.resolve(forkAbility);
            } finally {
                PatchHooks.uninstall(PatchHooks.ABILITY_UTILS, "setEffectRecordClauseListener");
            }
            bracket.endBracket(flushAbility.getId(), false);
            bracket.flushHeldResolution();
        });

        String shard = String.join("\n", records);
        assertFalse(shard.contains("damage_prevented"),
                "a fork ability's own null-ability outcome, reported while it "
                        + "is genuinely resolving, must not reach the live shard: " + shard);
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

            boolean installed = PatchHooks.install(PatchHooks.ABILITY_UTILS,
                    "setEffectRecordClauseListener", (proxy, method, args) -> {
                        if ("onClauseResolving".equals(method.getName())) {
                            try {
                                collectors.outcomeHandler().invoke(null, ON_OUTCOME,
                                        new Object[]{null, "damage_prevented", Map.of("amount", 3)});
                            } catch (Throwable t) {
                                throw new RuntimeException(t);
                            }
                        }
                        return null;
                    });
            Assumptions.assumeTrue(installed, "the checkout lacks the clause hook");
            try {
                AbilityUtils.resolve(liveAbility);
            } finally {
                PatchHooks.uninstall(PatchHooks.ABILITY_UTILS, "setEffectRecordClauseListener");
            }
            bracket.endBracket(liveAbility.getId(), false);
            bracket.flushHeldResolution();
        });

        String shard = String.join("\n", records);
        assertTrue(shard.contains("damage_prevented"),
                "a live ability's own null-ability outcome, reported while it "
                        + "is genuinely resolving, must reach the shard: " + shard);
    }

    /**
     * The trap the finding warns against: failing open on {@code null}
     * readmits exactly what the gate exists to exclude, because {@code
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
                collectors.outcomeHandler().invoke(null, ON_OUTCOME,
                        new Object[]{null, "damage_prevented", Map.of("amount", 3)});
            } finally {
                flag.set(false);
            }

            // The same report with no fork running is admitted, so the one
            // above was built and refused rather than lost on the way.
            collectors.outcomeHandler().invoke(null, ON_OUTCOME,
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
