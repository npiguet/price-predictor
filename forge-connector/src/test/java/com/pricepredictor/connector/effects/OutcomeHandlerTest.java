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

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static com.pricepredictor.connector.effects.TestCollectors.recording;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The outcome hook: an effect's own account of what it computed, the event
 * that account becomes, and which game it is admitted from.
 */
@ExtendWith(ForgeExtension.class)
class OutcomeHandlerTest {

    @TempDir
    Path tempDir;

    // ── the argument order ──────────────────────────────────────────────

    /**
     * Forge's outcome listener, copied argument for argument rather than
     * imported: the connector compiles against stock Forge, where
     * {@code EffectRecordOutcomes.EffectRecordOutcomeListener} does not
     * exist, and a test that imported it would stop this module building on
     * an unpatched checkout. See {@code ClauseContractTest.ListenerShape} for
     * the same pattern on the clause hook.
     */
    public interface OutcomeListenerShape {
        void onOutcome(SpellAbility sa, String eventType, Map<String, Object> params);
    }

    @BeforeEach
    void resetUnknownOutcomeTypeLatch() throws ReflectiveOperationException {
        // Static, and shared with the production path: without a reset here,
        // whichever test trips it first leaves every test after it in this
        // class unable to see its own "did this print" outcome.
        var latch = PatchedCollectors.class.getDeclaredField("UNKNOWN_OUTCOME_TYPE_REPORTED");
        latch.setAccessible(true);
        ((java.util.concurrent.atomic.AtomicBoolean) latch.get(null)).set(false);
    }

    private static Method methodNamed(Class<?> shape, String name) {
        for (Method method : shape.getMethods()) {
            if (method.getName().equals(name)) {
                return method;
            }
        }
        throw new AssertionError(shape + " declares no " + name);
    }

    /**
     * The stand-in above is what Forge declares, position by position.
     *
     * <p>Reflective, so an unpatched checkout skips the assertion rather than
     * failing it -- the same degradation every hook lookup makes.
     */
    @Test
    void theOutcomeStandInIsForgesOwnDeclaration() {
        PatchHooks.Lookup setter = PatchHooks.find(
                PatchHooks.EFFECT_RECORD_OUTCOMES, "setEffectRecordOutcomeListener");
        Assumptions.assumeTrue(setter.present(), "the checkout lacks the outcome hook");
        Class<?> declared = setter.method().getParameterTypes()[0];
        Class<?>[] theirs = methodNamed(declared, "onOutcome").getParameterTypes();
        Class<?>[] ours = methodNamed(OutcomeListenerShape.class, "onOutcome").getParameterTypes();
        assertEquals(ours.length, theirs.length, "onOutcome arity");
        for (int i = 0; i < theirs.length; i++) {
            assertEquals(ours[i], theirs[i], "onOutcome argument " + i);
        }
    }

    /**
     * The hook binds by argument position, and no compiler checks that. A
     * reorder in Forge is invisible to both compilers and must fail here
     * rather than in a corpus -- which is why this uses a real, distinct
     * {@link SpellAbility} rather than null for position 0: a null would not
     * notice a swap between the ability and either of the other two
     * arguments, only {@link #theOutcomeStandInIsForgesOwnDeclaration} would,
     * and that test is reflective and skips on an unpatched checkout.
     */
    @Test
    void anOutcomeReportBecomesAnEventOfThatType() throws Throwable {
        SpellAbility sa = TestCards.scriptedAbility("Alchemist's Gambit", "AddTurn");
        PatchedCollectors collectors = recording(tempDir);

        collectors.outcomeHandler().invoke(null,
                methodNamed(OutcomeListenerShape.class, "onOutcome"),
                new Object[]{sa, "vote_taken", Map.of("tally", Map.of("beast", 2))});

        EffectEvent written = collectors.lastClauseEvent();
        assertEquals("vote_taken", written.type());
        assertEquals(Map.of("beast", 2), written.params().get("tally"));
    }

    /**
     * An outcome that names no subjects has none: a coin flip or a token
     * made is not something done to the card whose ability did it, so the
     * acting host is not a default.
     */
    @Test
    void anOutcomeNamingNoSubjectsHasNone() throws Throwable {
        SpellAbility sa = TestCards.scriptedAbility("Alchemist's Gambit", "AddTurn");
        PatchedCollectors collectors = recording(tempDir);

        collectors.outcomeHandler().invoke(null,
                methodNamed(OutcomeListenerShape.class, "onOutcome"),
                new Object[]{sa, "coin_flipped", Map.of("won", true)});

        EffectEvent written = collectors.lastClauseEvent();
        assertEquals("coin_flipped", written.type());
        assertEquals(List.of(), written.subjects(),
                "the acting host is not a subject: " + written.toJson());
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
        PatchedCollectors collectors = recording(tempDir);

        collectors.outcomeHandler().invoke(null,
                methodNamed(OutcomeListenerShape.class, "onOutcome"),
                new Object[]{sa, "token_created", Map.of("sample", bat, "count", 2)});

        EffectEvent written = collectors.lastClauseEvent();
        assertEquals("token_created", written.type());
        assertEquals(Set.of("token_script_id", "characteristics", "count"),
                written.params().keySet(), written.toJson());
        assertEquals("b_1_1_bat_flying", written.params().get("token_script_id"));
        assertEquals(SnapshotBuilder.tokenScriptIdOf(bat), written.params().get("token_script_id"));
        assertEquals(2, written.params().get("count"));
        @SuppressWarnings("unchecked")
        Map<String, Object> traits = (Map<String, Object>) written.params().get("characteristics");
        assertEquals(1, traits.get("power"));
        assertEquals(1, traits.get("toughness"));
        assertEquals(bat.getType().toString(), traits.get("types"));
        assertEquals(List.of("flying"), traits.get("keywords"));
        assertEquals(SnapshotBuilder.keywordNames(bat.getKeywords()), traits.get("keywords"));
        assertFalse(written.toJson().contains("sample"), written.toJson());
        assertEquals(List.of(), written.subjects());
    }

    /**
     * {@code damage_prevented} names who the damage would have been dealt
     * to, a player or a creature, as the event's subject.
     */
    @Test
    void damagePreventedNamesItsRecipient() throws Throwable {
        Card creature = TestCards.build("Grizzly Bears");
        Player player = new Player("recipient", TestCards.game(), 93201);
        PatchedCollectors collectors = recording(tempDir);
        Method onOutcome = methodNamed(OutcomeListenerShape.class, "onOutcome");

        collectors.outcomeHandler().invoke(null, onOutcome,
                new Object[]{null, "damage_prevented",
                        Map.of("amount", 2, "subjects", List.of(creature))});
        assertEquals(List.of(SnapshotBuilder.entityId(creature)),
                collectors.lastClauseEvent().subjects());
        assertNull(collectors.lastClauseEvent().params().get("subjects"));

        collectors.outcomeHandler().invoke(null, onOutcome,
                new Object[]{null, "damage_prevented",
                        Map.of("amount", 2, "subjects", List.of(player))});
        assertEquals(List.of(SnapshotBuilder.playerId(player)),
                collectors.lastClauseEvent().subjects());

        collectors.outcomeHandler().invoke(null, onOutcome,
                new Object[]{null, "damage_prevented",
                        Map.of("amount", 2, "subjects", List.of())});
        assertEquals(List.of(), collectors.lastClauseEvent().subjects());
    }

    /**
     * A reveal from a library names the players it came from; any other
     * reveal arrives with an empty list and names nobody.
     */
    @Test
    void cardRevealedNamesTheLibrariesItCameFrom() throws Throwable {
        Player first = new Player("first", TestCards.game(), 93202);
        Player second = new Player("second", TestCards.game(), 93203);
        PatchedCollectors collectors = recording(tempDir);
        Method onOutcome = methodNamed(OutcomeListenerShape.class, "onOutcome");

        collectors.outcomeHandler().invoke(null, onOutcome,
                new Object[]{null, "card_revealed",
                        Map.of("count", 2, "from_zone", "Library",
                                "subjects", List.of(first, second))});
        EffectEvent fromLibraries = collectors.lastClauseEvent();
        assertEquals(List.of(SnapshotBuilder.playerId(first), SnapshotBuilder.playerId(second)),
                fromLibraries.subjects());
        assertEquals(Set.of("count", "from_zone"), fromLibraries.params().keySet());

        collectors.outcomeHandler().invoke(null, onOutcome,
                new Object[]{null, "card_revealed",
                        Map.of("count", 1, "from_zone", "Hand", "subjects", List.of())});
        assertEquals(List.of(), collectors.lastClauseEvent().subjects());
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
     * <p>Exact match, not {@code containsAll}: {@code restriction_change}'s
     * acting card (the host, "Alchemist's Gambit" here) must not appear in
     * the subject list alongside the two cards actually goaded, and
     * {@code containsAll} cannot tell "exactly these two" from "these two,
     * plus an extra false one".
     */
    @Test
    void restrictionChangeSubjectsBecomeEventSubjectsNotAParam() throws Throwable {
        Card goaded1 = TestCards.build("Sol Ring");
        Card goaded2 = TestCards.build("Mox Pearl");
        SpellAbility sa = TestCards.scriptedAbility("Alchemist's Gambit", "AddTurn");
        PatchedCollectors collectors = recording(tempDir);

        collectors.outcomeHandler().invoke(null,
                methodNamed(OutcomeListenerShape.class, "onOutcome"),
                new Object[]{sa, "restriction_change",
                        Map.of("restriction", "goad", "value", true,
                                "subjects", List.of(goaded1, goaded2))});

        EffectEvent written = collectors.lastClauseEvent();
        assertEquals("restriction_change", written.type());
        assertEquals("goad", written.params().get("restriction"));
        assertEquals(Boolean.TRUE, written.params().get("value"));
        assertNull(written.params().get("subjects"), "subjects must not leak into params");
        assertEquals(List.of(
                SnapshotBuilder.entityId(goaded1), SnapshotBuilder.entityId(goaded2)),
                written.subjects(),
                "exactly the two goaded cards -- not the acting host as well");
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
        PatchedCollectors collectors = recording(tempDir);
        Method onOutcome = methodNamed(OutcomeListenerShape.class, "onOutcome");

        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        PrintStream original = System.err;
        System.setErr(new PrintStream(captured, true));
        try {
            collectors.outcomeHandler().invoke(null, onOutcome,
                    new Object[]{null, "dungeon_entered", Map.of()});
            collectors.outcomeHandler().invoke(null, onOutcome,
                    new Object[]{null, "dungeon_entered", Map.of()});
            collectors.outcomeHandler().invoke(null, onOutcome,
                    new Object[]{null, "another_unknown_type", Map.of()});
        } finally {
            System.setErr(original);
        }

        assertNull(collectors.lastClauseEvent(), "an unrecognized type must not become an event");
        String output = captured.toString();
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
        PatchedCollectors collectors = recording(tempDir);

        collectors.outcomeHandler().invoke(null,
                methodNamed(OutcomeListenerShape.class, "onOutcome"),
                new Object[]{null, "damage_prevented",
                        Map.of("amount", 3, "source", source)});

        EffectEvent written = collectors.lastClauseEvent();
        assertEquals("damage_prevented", written.type());
        assertEquals(3, written.params().get("amount"));
        assertEquals(SnapshotBuilder.entityId(source), written.params().get("source"),
                "a ref like damage_dealt's, not a display name");
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
     * <p>{@code lastClauseEvent()} is asserted non-null first: it is set
     * unconditionally, before the (now gated) delivery to the bracket, so
     * this also proves the event was genuinely built and rejected by the
     * game-identity check -- not merely never reached because of some other,
     * unrelated failure this test would then wrongly credit to the fix.
     */
    @Test
    void anOutcomeFromAnotherGameDoesNotReachThisGamesBracket() throws Throwable {
        Card forkHost = TestBoard.cardInAnotherGame("Grizzly Bears");
        SpellAbility forkAbility = AbilityFactory.getAbility(
                "SP$ AddTurn | ValidTgts$ Player | NumTurns$ 1", forkHost);

        SpellAbility liveAbility = TestCards.scriptedAbility("Alchemist's Gambit", "AddTurn");
        List<String> records = ShardRun.records(tempDir, TestCards.game(), (bracket, collectors) -> {
            bracket.beginBracket(liveAbility);

            collectors.outcomeHandler().invoke(null,
                    methodNamed(OutcomeListenerShape.class, "onOutcome"),
                    new Object[]{forkAbility, "vote_taken", Map.of("tally", Map.of("beast", 2))});

            assertNotNull(collectors.lastClauseEvent(),
                    "the event must still be built -- only its delivery to "
                            + "the live bracket is gated");
            assertEquals("vote_taken", collectors.lastClauseEvent().type());

            bracket.endBracket(liveAbility.getId(), false);
            bracket.flushHeldResolution();
        });

        String shard = String.join("\n", records);
        assertFalse(shard.contains("vote_taken"),
                "a fork's outcome must not land in the live game's record: " + shard);
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
     * <p>Every pre-existing null-ability outcome test (e.g. {@code
     * damagePreventedSourceBecomesAnEntityRef} above) asserts on {@link
     * PatchedCollectors#lastClauseEvent()}, which is set unconditionally
     * above the delivery gate and so cannot see this class of bug -- that is
     * exactly why 689/689 stayed green while the channel was dead. This
     * asserts on what actually reaches the shard instead, the way {@link
     * #anOutcomeFromAnotherGameDoesNotReachThisGamesBracket} already does for
     * the ability-bearing case.
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
            collectors.outcomeHandler().invoke(null,
                    methodNamed(OutcomeListenerShape.class, "onOutcome"),
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
                                collectors.outcomeHandler().invoke(null,
                                        methodNamed(OutcomeListenerShape.class, "onOutcome"),
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
                                collectors.outcomeHandler().invoke(null,
                                        methodNamed(OutcomeListenerShape.class, "onOutcome"),
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
        Path shard = ShardRun.shardOf(tempDir, TestCards.game(), (bracket, collectors) -> {
            var forkRunning = ForkCollector.class.getDeclaredField("FORK_RUNNING");
            forkRunning.setAccessible(true);
            @SuppressWarnings("unchecked")
            ThreadLocal<Boolean> flag = (ThreadLocal<Boolean>) forkRunning.get(null);
            flag.set(true);
            try {
                collectors.outcomeHandler().invoke(null,
                        methodNamed(OutcomeListenerShape.class, "onOutcome"),
                        new Object[]{null, "damage_prevented", Map.of("amount", 3)});
            } finally {
                flag.set(false);
            }

            assertNotNull(collectors.lastClauseEvent(),
                    "the event must still be built -- only its delivery to "
                            + "the live bracket is gated");

            // The event reaches the bracket's buffer, which only a flush
            // writes; without one the shard is empty whether or not the gate
            // exists.
            bracket.finishGame(Set.of());
        });

        assertFalse(Files.exists(shard),
                "a probe's own combat-damage report must not land in the live game's record");
    }
}
