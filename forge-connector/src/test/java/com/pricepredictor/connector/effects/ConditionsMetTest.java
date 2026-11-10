package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.AbilitySub;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.RecordLines.linkOf;
import static com.pricepredictor.connector.effects.RecordLines.payload;
import static com.pricepredictor.connector.effects.RecordLines.records;
import static com.pricepredictor.connector.effects.TestBoard.castAndResolve;
import static com.pricepredictor.connector.effects.TestBoard.nextStep;
import static com.pricepredictor.connector.effects.TestBoard.spellOf;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A resolution's {@code conditions_met}: for each conditional clause that
 * resolved, whether its condition held, keyed by the chain index its events'
 * {@code attributed_to} uses (FR-034, story 2 acceptance 9).
 *
 * <p>Kami's Flare deals 3 damage to its target and, in its first sub-ability
 * ({@code ConditionPresent$ Creature.YouCtrl+modified}), 2 to the target's
 * controller only while its caster controls a modified creature.
 */
@ExtendWith(ForgeExtension.class)
class ConditionsMetTest {

    private static final String WHY = "to read a clause's condition";

    @TempDir
    Path tempDir;

    /** Kami's Flare in the caster's hand, aimed at {@code target}. */
    private static SpellAbility flareAt(Game game, Player caster, Card target) {
        SpellAbility flare = spellOf(TestBoard.inHand(game, caster, "Kami's Flare"), caster);
        flare.resetTargets();
        flare.getTargets().add(target);
        return flare;
    }

    private static List<Object> conditionsOf(String record) {
        Object conditions = payload(record).get("conditions_met");
        assertTrue(conditions instanceof List<?>, "no conditions_met in " + record);
        return List.copyOf((List<?>) conditions);
    }

    private static String effectHalf(List<String> all) {
        List<String> halves = records(all, "resolution").stream()
                .filter(record -> linkOf(record) != null).toList();
        assertEquals(1, halves.size(), all.toString());
        return halves.get(0);
    }

    /** Story 2 acceptance 9: a segment whose condition did not hold reads {@code met = false}. */
    @Test
    void aConditionThatDidNotHoldIsWrittenUnmet() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card giant = TestBoard.onBattlefield(game, opponent, "Hill Giant");
        SpellAbility flare = flareAt(game, caster, giant);

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            castAndResolve(game, flare);
            nextStep(game);
        });

        assertEquals(20, opponent.getLife(), "the unmet clause dealt nothing");
        assertEquals(List.of(Map.of("segment", "0", "met", false)),
                conditionsOf(effectHalf(records)));
    }

    /** The same clause with its condition holding reads {@code met = true}, and did its damage. */
    @Test
    void aConditionThatHeldIsWrittenMet() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card bears = TestBoard.onBattlefield(game, caster, "Grizzly Bears");
        bears.addCounterInternal(CounterEnumType.P1P1, 1, caster, false, null, Map.of());
        Card giant = TestBoard.onBattlefield(game, opponent, "Hill Giant");
        SpellAbility flare = flareAt(game, caster, giant);

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            castAndResolve(game, flare);
            nextStep(game);
        });

        assertEquals(18, opponent.getLife(), "the met clause dealt its damage");
        assertEquals(List.of(Map.of("segment", "0", "met", true)),
                conditionsOf(effectHalf(records)));
    }

    /** A line no clause of which is conditional writes an empty list, not no field. */
    @Test
    void anUnconditionalLineWritesAnEmptyList() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card giant = TestBoard.onBattlefield(game, game.getPlayers().get(1), "Hill Giant");
        SpellAbility shock = spellOf(TestBoard.inHand(game, caster, "Shock"), caster);
        shock.resetTargets();
        shock.getTargets().add(giant);

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            castAndResolve(game, shock);
            nextStep(game);
        });

        assertEquals(List.of(), conditionsOf(effectHalf(records)));
    }

    /** A forced resolution reads its clauses' conditions as an observed one does. */
    @Test
    void aForcedResolutionCarriesItsConditions() throws java.io.IOException {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card giant = TestBoard.onBattlefield(game, game.getPlayers().get(1), "Hill Giant");
        SpellAbility flare = flareAt(game, caster, giant);
        CollectionCaps caps = TestCaps.caps().manaCap(2000).probes(0).build();

        ShardRun.Fork fork = ShardRun.forkedWithClauseHook(tempDir, game, caps, 5L, flare, WHY);

        assertTrue(fork.wrote(), "the forced resolution observed something");
        List<String> halves = fork.records();
        assertEquals(1, halves.size(), halves.toString());
        assertEquals(List.of(Map.of("segment", "0", "met", false)), conditionsOf(halves.get(0)));
    }

    /** Only the parameters a condition reads make a clause conditional. */
    @Test
    void displayTextAloneIsNoCondition() {
        assertFalse(ConditionsMet.isConditional(Map.of("ConditionDescription", "if you do")));
        assertFalse(ConditionsMet.isConditional(Map.of("NumDmg", "2")));
        assertTrue(ConditionsMet.isConditional(Map.of("Condition", "Kicked")));
        assertTrue(ConditionsMet.isConditional(Map.of("ConditionCheckSVar", "X")));
    }

    /** The segment is the attributed_to spelling: the root, or the index down the chain. */
    @Test
    void theSegmentIsTheClausesAttributedToIndex() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        SpellAbility flare = spellOf(TestBoard.card(game, caster, "Kami's Flare"), caster);
        AbilitySub damage = flare.getSubAbility();

        assertNull(ConditionsMet.of(flare, "RAN"), "the root clause is unconditional");
        ConditionsMet sub = ConditionsMet.of(damage, "CONDITIONS_UNMET");
        assertEquals("0", sub.segment());
        assertEquals("{\"segment\":\"0\",\"met\":false}", sub.toJson());
    }

    /**
     * {@code met} is Forge's own answer, the {@code ClauseRun} the clause hook
     * reports, never a second {@code metConditions()}: a clause whose unless
     * cost decided against it still met its condition, and one that threw, or
     * a jar that reports nothing, gives no entry.
     */
    @Test
    void metIsForgesClauseRun() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        AbilitySub damage =
                spellOf(TestBoard.card(game, caster, "Kami's Flare"), caster).getSubAbility();

        assertTrue(ConditionsMet.of(damage, "RAN").met());
        assertTrue(ConditionsMet.of(damage, "UNLESS_COST_DECIDED").met());
        assertFalse(ConditionsMet.of(damage, "CONDITIONS_UNMET").met());
        assertNull(ConditionsMet.of(damage, "THREW"));
        assertNull(ConditionsMet.of(damage, null));
    }
}
