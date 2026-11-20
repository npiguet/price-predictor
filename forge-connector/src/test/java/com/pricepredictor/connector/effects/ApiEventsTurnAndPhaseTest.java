package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.ability.AbilityFactory;
import forge.game.player.Player;
import forge.game.spellability.AbilitySub;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.List;

import static com.pricepredictor.connector.effects.ApiClauseObservations.observe;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * The {@link TurnAndPhaseApiRules}: a turn or a phase added or skipped, its
 * count, and who it affects.
 */
@ExtendWith(ForgeExtension.class)
class ApiEventsTurnAndPhaseTest {

    @Test
    void anExtraTurnCarriesItsCount() {
        SpellAbility sa = TestCards.scriptedAbility("Alchemist's Gambit", "AddTurn");

        EffectEvent event = TestCards.onlyEvent(observe(sa));

        assertNotNull(event);
        assertEquals(EventType.TURN_ADDED, event.type());
        assertEquals(1, event.param(EventParam.COUNT));
    }

    /**
     * The count is read off {@code NumTurns$}: Alchemist's Gambit names none,
     * so its 1 is the fallback a rule that never read the parameter also gives.
     */
    @Test
    void anExtraTurnCountIsTheClausesNumTurns() {
        SpellAbility sa = AbilityFactory.getAbility(
                "SP$ AddTurn | Defined$ You | NumTurns$ 3", TestCards.build("Grizzly Bears"));

        assertEquals(3, TestCards.onlyEvent(observe(sa)).param(EventParam.COUNT));
    }

    /** The same for a skipped turn. */
    @Test
    void aSkippedTurnCountIsTheClausesNumTurns() {
        SpellAbility sa = AbilityFactory.getAbility(
                "SP$ SkipTurn | Defined$ You | NumTurns$ 2", TestCards.build("Grizzly Bears"));

        assertEquals(2, TestCards.onlyEvent(observe(sa)).param(EventParam.COUNT));
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

        EffectEvent event = TestCards.onlyEvent(observe(sa));

        assertEquals(EventType.PHASE_SKIPPED, event.type());
        assertEquals("combat_begin", event.param(EventParam.PHASE));
    }

    /** Meditate: {@code SP$ Draw | ... | SubAbility$ DBSkip}, {@code DB$ SkipTurn}. */
    @Test
    void askippedTurnCarriesItsCount() {
        SpellAbility sa = TestCards.scriptedAbility("Meditate", "SkipTurn");

        EffectEvent event = TestCards.onlyEvent(observe(sa));

        assertEquals(EventType.TURN_SKIPPED, event.type());
        assertEquals(1, event.param(EventParam.COUNT));
    }

    /** Full Throttle: {@code SP$ AddPhase | ExtraPhase$ Combat | NumPhases$ 2}. */
    @Test
    void anAddedPhaseNamesThePhaseAndItsCount() {
        SpellAbility sa = TestCards.scriptedAbility("Full Throttle", "AddPhase");

        EffectEvent event = TestCards.onlyEvent(observe(sa));

        assertEquals(EventType.PHASE_ADDED, event.type());
        assertEquals("Combat", event.param(EventParam.PHASE));
        assertEquals(2, event.param(EventParam.COUNT));
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
     * {@code SkipTurnEffect} and {@code SkipPhaseEffect} resolve the same way,
     * so they share the rule shape and this test.
     */
    @ParameterizedTest(name = "{0}")
    @CsvSource(delimiter = ';', value = {
            "an extra turn;     SP$ AddTurn | ValidTgts$ Opponent | NumTurns$ 1;           91001",
            "a skipped turn;    SP$ SkipTurn | ValidTgts$ Opponent | NumTurns$ 1;          91201",
            "a skipped phase;   SP$ SkipPhase | ValidTgts$ Opponent | Phase$ BeginCombat;  91301",
    })
    void aTurnOrPhaseEffectNamesTheTargetedPlayerNotTheActivator(
            String ignoredCase, String script, int firstPlayerId) {
        SpellAbility sa = AbilityFactory.getAbility(script, TestCards.build("Grizzly Bears"));
        Player activator = new Player("activator", TestCards.game(), firstPlayerId);
        Player targeted = new Player("targeted", TestCards.game(), firstPlayerId + 1);
        sa.setActivatingPlayer(activator);
        sa.resetTargets();
        sa.getTargets().add(targeted);

        EffectEvent event = TestCards.onlyEvent(observe(sa));

        assertEquals(List.of(Refs.of(targeted)), event.subjects());
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

        EffectEvent event = TestCards.onlyEvent(observe(sa));

        assertEquals(
                List.of(Refs.of(first), Refs.of(second)),
                event.subjects());
    }

    /**
     * {@code AddPhaseEffect.resolve} reads {@code sa.getActivatingPlayer()}
     * exclusively and has no targeting or {@code Defined$} concept -- unlike
     * its three siblings above, the activator genuinely is the subject here,
     * and this pins that their targeting rule does not reach an API that has
     * none.
     */
    @Test
    void anAddedPhaseNamesTheActivator() {
        SpellAbility sa = TestCards.scriptedAbility("Full Throttle", "AddPhase");
        Player activator = new Player("activator", TestCards.game(), 91401);
        sa.setActivatingPlayer(activator);

        EffectEvent event = TestCards.onlyEvent(observe(sa));

        assertEquals(List.of(Refs.of(activator)), event.subjects());
    }

    /**
     * {@code affectedPlayers} has two branches; the five subject tests above
     * all exercise {@code usesTargeting()}. This one asserts <em>who</em> on
     * the {@code Defined$}-only branch -- the one Ral Zarek, Guest Lecturer
     * uses ({@code SVar:DBSkipTurn:DB$ SkipTurn | NumTurns$ X | Defined$
     * Targeted}, no {@code ValidTgts$} on the clause itself).
     *
     * <p>{@code Defined$ Targeted} on a non-targeting clause has to reach
     * through {@code getAllTargetChoices()} to a <em>different</em> link in
     * the same ability chain that does target -- {@code getAllTargetChoices}
     * walks {@code getRootAbility()} through {@code getSubAbility()} and
     * collects {@code getTargets()} only from links where
     * {@code usesTargeting()} is true -- so this builds the same two-link
     * shape: a targeting root ({@code Pump}, standing in for Ral Zarek's
     * {@code FlipCoin}) with a non-targeting {@code SkipTurn} wired on as its
     * sub-ability.
     *
     * <p>The real card wires its clause with {@code HeadsSubAbility$} rather
     * than {@code SubAbility$}, but {@code setAdditionalAbility} calls the same
     * {@code setParent} this test's {@code setSubAbility} does, so the parent
     * link {@code getAllTargetChoices} walks is the identical one.
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

        EffectEvent event = TestCards.onlyEvent(observe(sub));

        assertEquals(EventType.TURN_SKIPPED, event.type());
        assertEquals(List.of(Refs.of(targeted)), event.subjects());
    }
}
