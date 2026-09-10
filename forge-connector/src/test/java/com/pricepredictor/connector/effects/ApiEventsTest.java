package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.ability.AbilityFactory;
import forge.game.player.Player;
import forge.game.spellability.AbilitySub;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * An effect whose outcome the engine broadcasts nowhere still describes itself:
 * its own parameters say what it did. These are the APIs that need nothing else.
 */
@ExtendWith(ForgeExtension.class)
class ApiEventsTest {

    /**
     * The failure latch is static, shared with production: without a reset
     * here, whichever test trips it first leaves every test after it --
     * including ones that never touch the latch, and any later test class in
     * the same JVM -- unable to see its own "did this print" outcome.
     */
    @BeforeEach
    void resetEmitterFailureLatch() {
        ApiEvents.resetEmitterFailureLatchForTest();
    }

    @Test
    void anExtraTurnCarriesItsCount() {
        SpellAbility sa = TestCards.scriptedAbility("Alchemist's Gambit", "AddTurn");

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertNotNull(event);
        assertEquals(EffectEvent.TURN_ADDED, event.type());
        assertEquals(1, event.params().get("count"));
    }

    /**
     * Not Blinding Angel: its {@code SkipPhase} clause is the {@code Execute$}
     * of a triggered ability, which lives on the {@code Trigger} object, not
     * in {@code card.getCurrentState().getSpellAbilities()} or any
     * {@code SubAbility$} chain hung off it -- {@code scriptedAbility} cannot
     * reach it and never will. Moment of Silence casts {@code SP$ SkipPhase}
     * as its own root ability.
     */
    @Test
    void aSkippedPhaseNamesThePhase() {
        SpellAbility sa = TestCards.scriptedAbility("Moment of Silence", "SkipPhase");

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertEquals(EffectEvent.PHASE_SKIPPED, event.type());
        assertNotNull(event.params().get("phase"));
    }

    @Test
    void anApiWithNoEmitterIsSilent() {
        SpellAbility sa = TestCards.scriptedAbility("Lightning Bolt", "DealDamage");

        assertNull(ApiEvents.after(sa, ApiEvents.before(sa)),
                "damage already arrives on the bus; a second event would double-count it");
    }

    /** Meditate: {@code SP$ Draw | ... | SubAbility$ DBSkip}, {@code DB$ SkipTurn}. */
    @Test
    void askippedTurnCarriesItsCount() {
        SpellAbility sa = TestCards.scriptedAbility("Meditate", "SkipTurn");

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertEquals(EffectEvent.TURN_SKIPPED, event.type());
        assertEquals(1, event.params().get("count"));
    }

    /** Full Throttle: {@code SP$ AddPhase | ExtraPhase$ Combat | NumPhases$ 2}. */
    @Test
    void anAddedPhaseNamesThePhaseAndItsCount() {
        SpellAbility sa = TestCards.scriptedAbility("Full Throttle", "AddPhase");

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertEquals(EffectEvent.PHASE_ADDED, event.type());
        assertEquals("Combat", event.params().get("phase"));
        assertEquals(2, event.params().get("count"));
    }

    /**
     * Darkpact's clause carries no {@code DefinedPlayer}, so the new owner
     * falls back to the activator -- {@code null} here, since
     * {@link TestCards#game()} is shared and nothing in this suite ever
     * activates an ability as a player. What this pins is that the fallback
     * path runs end to end rather than that it names anyone in particular;
     * {@link ApiEvents} names the {@code DefinedPlayer}-driven branch's real
     * printed counter-examples (Tempest Efreet, Bronze Tablet) in its own
     * javadoc, since neither is reachable without a live target or a prior
     * {@code Remembered} player, which this harness cannot cheaply build.
     */
    @Test
    void anOwnershipChangeWithNoNamedOwnerFallsBackToTheActivator() {
        SpellAbility sa = TestCards.scriptedAbility("Darkpact", "GainOwnership");

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertEquals(EffectEvent.OWNERSHIP_CHANGE, event.type());
    }

    /**
     * {@code SpellAbilityEffect.getPlayers} ({@code SpellAbilityEffect.java:314})
     * splits a {@code " & "}-joined {@code Defined}/{@code DefinedPlayer} and
     * unions each token; a rule that read the raw string as one token would
     * hand it to {@code AbilityUtils.getDefinedPlayers} unrecognized, which
     * falls through to its catch-all ({@code AbilityUtils.java:1187-1190}:
     * {@code players.addAll(game.getPlayersInTurnOrder())}) -- empty on
     * {@link TestCards#game()}'s zero-player game, so {@code ownerOf} would
     * fall back to the activator instead.
     *
     * <p>{@code You & You} (the first version of this test) could not tell
     * that fallback apart from a correct split, because both halves name the
     * activator either way -- the test passed unchanged with the split
     * deleted. {@code Targeted & You} can: split, the owner is the targeted
     * player; unsplit, the whole string matches nothing and {@code ownerOf}
     * falls back to the activator, a different id. Mutation-tested: deleting
     * the {@code " & "} split in {@code definedPlayers} makes this fail with
     * the activator's id where the targeted player's was expected, then
     * passes again with the split restored.
     */
    @Test
    void ownershipDefinedPlayerSplitsOnAmpersand() {
        SpellAbility sa = AbilityFactory.getAbility(
                "SP$ GainOwnership | ValidTgts$ Card | DefinedPlayer$ Targeted & You",
                TestCards.build("Grizzly Bears"));
        Player activator = new Player("activator", TestCards.game(), 91501);
        Player targeted = new Player("targeted", TestCards.game(), 91502);
        sa.setActivatingPlayer(activator);
        sa.resetTargets();
        sa.getTargets().add(targeted);

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertEquals(SnapshotBuilder.playerId(targeted), event.params().get("owner"));
    }

    // ── subject is who the effect affects, not who cast it ──────────────

    /**
     * Eon Frolicker: {@code DB$ AddTurn | ValidTgts$ Opponent | NumTurns$ 1}
     * -- not reachable through {@link TestCards#scriptedAbility}, the same
     * way Blinding Angel's {@code SkipPhase} is not (it is a triggered
     * ability's {@code Execute$}), so built directly. {@code
     * AddTurnEffect.resolve} iterates {@code getTargetPlayers(sa)}, and
     * {@code ValidTgts$ Opponent} guarantees the targeted player is never the
     * activator: this is the engine's own common case, not an edge one.
     */
    @Test
    void anExtraTurnNamesTheTargetedPlayerNotTheActivator() {
        SpellAbility sa = AbilityFactory.getAbility(
                "SP$ AddTurn | ValidTgts$ Opponent | NumTurns$ 1",
                TestCards.build("Grizzly Bears"));
        Player activator = new Player("activator", TestCards.game(), 91001);
        Player targeted = new Player("targeted", TestCards.game(), 91002);
        sa.setActivatingPlayer(activator);
        sa.resetTargets();
        sa.getTargets().add(targeted);

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertEquals(List.of(SnapshotBuilder.playerId(targeted)), event.subjects());
    }

    /**
     * {@code getTargetPlayers} returns every target, not one: two players
     * targeted must read as two subjects, not the first with the second
     * silently dropped.
     */
    @Test
    void anExtraTurnNamesEveryTargetedPlayerNotJustTheFirst() {
        SpellAbility sa = AbilityFactory.getAbility(
                "SP$ AddTurn | ValidTgts$ Player | NumTurns$ 1",
                TestCards.build("Grizzly Bears"));
        Player first = new Player("first", TestCards.game(), 91101);
        Player second = new Player("second", TestCards.game(), 91102);
        sa.resetTargets();
        sa.getTargets().add(first);
        sa.getTargets().add(second);

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertEquals(
                List.of(SnapshotBuilder.playerId(first), SnapshotBuilder.playerId(second)),
                event.subjects());
    }

    /** Same rule shape as {@code AddTurn}; {@code SkipTurnEffect.resolve} matches it. */
    @Test
    void aSkippedTurnNamesTheTargetedPlayerNotTheActivator() {
        SpellAbility sa = AbilityFactory.getAbility(
                "SP$ SkipTurn | ValidTgts$ Opponent | NumTurns$ 1",
                TestCards.build("Grizzly Bears"));
        Player activator = new Player("activator", TestCards.game(), 91201);
        Player targeted = new Player("targeted", TestCards.game(), 91202);
        sa.setActivatingPlayer(activator);
        sa.resetTargets();
        sa.getTargets().add(targeted);

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertEquals(List.of(SnapshotBuilder.playerId(targeted)), event.subjects());
    }

    /** Same rule shape again; {@code SkipPhaseEffect.resolve} matches it too. */
    @Test
    void aSkippedPhaseNamesTheTargetedPlayerNotTheActivator() {
        SpellAbility sa = AbilityFactory.getAbility(
                "SP$ SkipPhase | ValidTgts$ Opponent | Phase$ BeginCombat",
                TestCards.build("Grizzly Bears"));
        Player activator = new Player("activator", TestCards.game(), 91301);
        Player targeted = new Player("targeted", TestCards.game(), 91302);
        sa.setActivatingPlayer(activator);
        sa.resetTargets();
        sa.getTargets().add(targeted);

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertEquals(List.of(SnapshotBuilder.playerId(targeted)), event.subjects());
    }

    /**
     * {@code AddPhaseEffect.resolve} reads {@code sa.getActivatingPlayer()}
     * exclusively and has no targeting or {@code Defined$} concept -- unlike
     * its three siblings above, the activator genuinely is the subject here,
     * and this pins that the fix was not generalized to a rule that never
     * needed it.
     */
    @Test
    void anAddedPhaseNamesTheActivator() {
        SpellAbility sa = TestCards.scriptedAbility("Full Throttle", "AddPhase");
        Player activator = new Player("activator", TestCards.game(), 91401);
        sa.setActivatingPlayer(activator);

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertEquals(List.of(SnapshotBuilder.playerId(activator)), event.subjects());
    }

    /**
     * {@code affectedPlayers} has two branches; the five subject tests above
     * all exercise {@code usesTargeting()}. The {@code Defined$}-only branch
     * -- the one Ral Zarek, Guest Lecturer actually uses
     * ({@code ral_zarek_guest_lecturer.txt:9}, {@code SVar:DBSkipTurn:DB$
     * SkipTurn | NumTurns$ X | Defined$ Targeted}, no {@code ValidTgts$} on
     * the clause itself) -- had no test asserting <em>who</em>, only the two
     * pre-existing {@code SkipTurn} tests' counts.
     *
     * <p>{@code Defined$ Targeted} on a non-targeting clause has to reach
     * through {@code getAllTargetChoices()} to a <em>different</em> link in
     * the same ability chain that does target -- {@code getAllTargetChoices}
     * walks {@code getRootAbility()} through {@code getSubAbility()} and
     * collects {@code getTargets()} only from links where
     * {@code usesTargeting()} is true -- so this builds the same two-link
     * shape: a targeting root ({@code Pump}, standing in for Ral Zarek's
     * {@code RollDice}) with a non-targeting {@code SkipTurn} wired on as its
     * sub-ability.
     */
    @Test
    void aSkippedTurnThroughDefinedTargetedNamesTheRootsTarget() {
        SpellAbility root = AbilityFactory.getAbility(
                "SP$ Pump | ValidTgts$ Player | NumAtt$ +0 | NumDef$ +0",
                TestCards.build("Grizzly Bears"));
        SpellAbility sub = AbilityFactory.getAbility(
                "DB$ SkipTurn | Defined$ Targeted | NumTurns$ 1", root.getHostCard());
        root.setSubAbility((AbilitySub) sub);
        Player targeted = new Player("targeted", TestCards.game(), 91601);
        root.resetTargets();
        root.getTargets().add(targeted);

        EffectEvent event = ApiEvents.after(sub, ApiEvents.before(sub));

        assertEquals(EffectEvent.TURN_SKIPPED, event.type());
        assertEquals(List.of(SnapshotBuilder.playerId(targeted)), event.subjects());
    }

    // ── the once-latched failure report ──────────────────────────────────

    /**
     * Mirrors {@code EffectRecordClauseHookTest
     * .testAThrowingListenerIsReportedExactlyOnceOnStderr} in the forge repo:
     * capture {@code System.err}, call the package-private seam directly with
     * a synthetic exception rather than contrive a rule that reliably throws,
     * and assert both halves of the promise -- one line naming the event type
     * and the exception, and silence on every call after the first, which is
     * the half inspection of the {@code compareAndSet} alone cannot establish.
     */
    @Test
    void anEmitterFailureIsReportedExactlyOnceOnStderr() {
        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        PrintStream original = System.err;
        System.setErr(new PrintStream(captured, true));
        try {
            ApiEvents.reportEmitterFailure("turn_added", new RuntimeException("boom"));
            ApiEvents.reportEmitterFailure("turn_added", new RuntimeException("boom"));
            ApiEvents.reportEmitterFailure("x_changed", new RuntimeException("boom again"));
        } finally {
            System.setErr(original);
        }

        String output = captured.toString();
        long lineCount = output.isBlank() ? 0 : output.strip().lines().count();
        assertEquals(1, lineCount, "exactly one report despite three throws: " + output);
        assertTrue(output.contains("turn_added"), output);
        assertTrue(output.contains("boom"), output);
    }

    @Test
    void anEmitterThatNeverThrowsReportsNothing() {
        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        PrintStream original = System.err;
        System.setErr(new PrintStream(captured, true));
        try {
            SpellAbility sa = TestCards.scriptedAbility("Alchemist's Gambit", "AddTurn");
            ApiEvents.after(sa, ApiEvents.before(sa));
        } finally {
            System.setErr(original);
        }

        assertEquals("", captured.toString());
    }
}
