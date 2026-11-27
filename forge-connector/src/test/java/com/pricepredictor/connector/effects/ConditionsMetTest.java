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

import static com.pricepredictor.connector.effects.RecordAbilities.linkOf;
import static com.pricepredictor.connector.effects.RecordEvents.payload;
import static com.pricepredictor.connector.effects.RecordPicks.records;
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
 * {@code attributed_to} uses (FR-034, Feature 025 story 2 acceptance 9).
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

    /** The opponent's Hill Giant, and Kami's Flare aimed at it cast and resolved: the shard's records. */
    private List<String> flareAtAGiant(Game game, Player caster, Player opponent) throws Throwable {
        Card giant = TestBoard.onBattlefield(game, opponent, "Hill Giant");
        SpellAbility flare = flareAt(game, caster, giant);
        return ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            castAndResolve(game, flare);
            nextStep(game);
        });
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

    /** Feature 025 story 2 acceptance 9: a segment whose condition did not hold reads {@code met = false}. */
    @Test
    void aConditionThatDidNotHoldIsWrittenUnmet() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);

        List<String> records = flareAtAGiant(game, caster, opponent);

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

        List<String> records = flareAtAGiant(game, caster, opponent);

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

    /**
     * A conditional clause of another line of the same card is not the
     * resolution's: Eddymurk Crab's "enters tapped if it's not your turn"
     * is an enters-the-battlefield replacement whose clause resolves while
     * the creature spell resolves, and the spell's effect half writes no
     * condition of it.
     */
    @Test
    void anotherLinesConditionalClauseIsNotTheResolutions() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card crab = TestBoard.inHand(game, caster, "Eddymurk Crab");

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            EffectEndingTheTurnTest.castWithoutTarget(game, crab, caster);
            game.getStack().resolveStack();
            game.getAction().checkStateEffects(false);
            nextStep(game);
        });

        assertTrue(game.getCardsIn(forge.game.zone.ZoneType.Battlefield).stream()
                .anyMatch(card -> card.getId() == crab.getId() && !card.isTapped()),
                "the crab entered untapped on its controller's turn");
        String half = RecordPicks.only(records, "resolution", "cardsfolder/e/eddymurk_crab.txt", "spell");
        assertEquals(List.of(), conditionsOf(half), half);
    }

    /**
     * A chosen charm mode's conditional sub-ability is on the mode's line:
     * Archdruid's Charm's first mode searches out a Forest and its
     * sub-ability puts it onto the battlefield only if it is a land. The
     * mode's half writes that clause's condition, counted down the option
     * line, met.
     */
    @Test
    void aChosenModesConditionalClauseIsTheModesHalfs() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card forest = TestBoard.card(game, caster, "Forest");
        game.getAction().moveToLibrary(forest, null, null);
        Card charmCard = TestBoard.inHand(game, caster, "Archdruid's Charm");

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            Card onStack = game.getAction().moveToStack(charmCard, null);
            SpellAbility charm = TestBoard.charmOf(onStack, caster);
            TestBoard.chain(charm, 0);
            game.getStack().add(charm);
            game.getStack().resolveStack();
            game.getAction().checkStateEffects(false);
            nextStep(game);
        });

        assertTrue(game.getCardsIn(forge.game.zone.ZoneType.Battlefield).stream()
                .anyMatch(card -> card.getId() == forest.getId()), "the Forest entered");
        String half = RecordPicks.only(records, "resolution", "cardsfolder/a/archdruids_charm.txt");
        assertEquals(0L, RecordAbilities.optionOf(half), half);
        assertEquals(List.of(Map.of("segment", "0", "met", true)), conditionsOf(half), half);
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
