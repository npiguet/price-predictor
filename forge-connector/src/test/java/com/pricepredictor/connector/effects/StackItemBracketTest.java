package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.GameObject;
import forge.game.card.Card;
import forge.game.event.GameEventPlayerPriority;
import forge.game.event.GameEventTurnPhase;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.player.PlayerView;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;

import static com.pricepredictor.connector.effects.RecordLines.eventTypes;
import static com.pricepredictor.connector.effects.RecordLines.linkOf;
import static com.pricepredictor.connector.effects.RecordLines.only;
import static com.pricepredictor.connector.effects.RecordLines.outcomeOf;
import static com.pricepredictor.connector.effects.RecordLines.records;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

/**
 * Several things on the stack at once, each with a bracket of its own.
 *
 * <p>A trigger a cast sets off — prowess, "whenever you cast" — and a spell
 * cast in response both go on the stack above the spell and resolve first.
 * The spell's bracket survives them: when it resolves it writes its own
 * linked effect half, as the items above it wrote theirs.
 *
 * <p>Driven through the real engine — {@code MagicStack.add}, the trigger
 * handler, {@code resolveStack} and the state-based check — with the
 * collector on the game's own bus, in the order {@code
 * PhaseHandler.mainLoopStep} runs them.
 */
@ExtendWith(ForgeExtension.class)
class StackItemBracketTest {

    @TempDir
    Path tempDir;

    private static final String BOLT = "cardsfolder/l/lightning_bolt.txt";
    private static final String SHOCK = "cardsfolder/s/shock.txt";
    private static final String SWIFTSPEAR = "cardsfolder/m/monastery_swiftspear.txt";
    private static final String COUNTERSPELL = "cardsfolder/c/counterspell.txt";

    /** The card's own spell, cast by {@code caster} at {@code target}. */
    private static SpellAbility castAt(Game game, Player caster, String name, GameObject target) {
        SpellAbility spell = TestBoard.inHand(game, caster, name).getFirstSpellAbility();
        assertNotNull(spell, "no spell on " + name);
        spell.setActivatingPlayer(caster);
        spell.resetTargets();
        spell.getTargets().add(target);
        return spell;
    }

    /**
     * What {@code PhaseHandler.mainLoopStep} does before a player acts: the
     * priority, the state-based check, and the waiting triggers going on the
     * stack.
     */
    private static void priority(Game game) {
        PlayerView turn = PlayerView.get(game.getPhaseHandler().getPlayerTurn());
        game.fireEvent(new GameEventPlayerPriority(turn, PhaseType.MAIN1, turn));
        game.getAction().checkStateEffects(false);
        game.getStack().addAllTriggeredAbilitiesToStack();
    }

    /** The next real action: a step boundary, which writes the held halves. */
    private static void nextStep(Game game) {
        game.fireEvent(new GameEventTurnPhase(
                game.getPhaseHandler().getPlayerTurn(), PhaseType.MAIN2, ""));
    }

    /** The cost half and the effect half through {@code script}, linked to each other. */
    private static String linkedHalf(List<String> all, String script) {
        String cost = only(all, "activation", script);
        String half = only(all, "resolution", script);
        assertNotNull(linkOf(cost), "the cost half links its effect half: " + cost);
        assertEquals(linkOf(cost), linkOf(half), half);
        return half;
    }

    /**
     * A Lightning Bolt cast with Monastery Swiftspear in play: prowess
     * triggers and resolves above the Bolt. Each writes its own linked
     * effect half, and the Bolt's holds the Bolt's damage and nothing of the
     * pump.
     */
    @Test
    void aSpellUnderItsCastTriggerWritesItsOwnHalf() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        TestBoard.inPlay(game, caster, "Monastery Swiftspear");
        SpellAbility bolt = castAt(game, caster, "Lightning Bolt", opponent);

        List<String> all = ShardRun.observed(tempDir, game, bracket -> {
            game.getStack().add(bolt);
            priority(game);
            assertEquals(2, game.getStack().size(), "prowess went on the stack above the Bolt");
            game.getStack().resolveStack();
            priority(game);
            game.getStack().resolveStack();
            priority(game);
            nextStep(game);
        });

        String boltHalf = linkedHalf(all, BOLT);
        assertEquals(List.of("damage_dealt", "life_change"), eventTypes(boltHalf), boltHalf);
        String prowessHalf = linkedHalf(all, SWIFTSPEAR);
        assertFalse(eventTypes(prowessHalf).contains("damage_dealt"),
                "the Bolt's damage is not the trigger's: " + prowessHalf);
    }

    /**
     * Shock, then Lightning Bolt in response, both at the opponent: the Bolt
     * resolves first, and each spell writes its own linked half with its own
     * damage.
     */
    @Test
    void aSpellAndTheResponseAboveItEachWriteTheirOwnHalf() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        SpellAbility shock = castAt(game, caster, "Shock", opponent);
        SpellAbility bolt = castAt(game, opponent, "Lightning Bolt", caster);

        List<String> all = ShardRun.observed(tempDir, game, bracket -> {
            game.getStack().add(shock);
            game.getStack().add(bolt);
            game.getStack().resolveStack();
            priority(game);
            game.getStack().resolveStack();
            priority(game);
            nextStep(game);
        });

        String boltHalf = linkedHalf(all, BOLT);
        assertEquals(List.of("damage_dealt", "life_change"), eventTypes(boltHalf), boltHalf);
        assertFalse(boltHalf.contains("\"amount\":2"), boltHalf);
        String shockHalf = linkedHalf(all, SHOCK);
        assertEquals(List.of("damage_dealt", "life_change"), eventTypes(shockHalf), shockHalf);
        assertFalse(shockHalf.contains("\"amount\":3"), shockHalf);
    }

    /**
     * Lightning Bolt, then Shock, then a Counterspell at the Bolt. The
     * Counterspell resolves and writes its half; the Bolt, gone from the
     * stack without resolving, is written off as {@code countered} with no
     * half; the Shock, between them, still writes its own.
     */
    @Test
    void aCounteredSpellIsWrittenOffAndTheSpellsAroundItKeepTheirHalves() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        SpellAbility bolt = castAt(game, caster, "Lightning Bolt", opponent);
        SpellAbility shock = castAt(game, caster, "Shock", opponent);

        List<String> all = ShardRun.observed(tempDir, game, bracket -> {
            game.getStack().add(bolt);
            game.getStack().add(shock);
            game.getStack().add(castAt(game, opponent, "Counterspell", bolt));
            game.getStack().resolveStack();
            priority(game);
            assertEquals(1, game.getStack().size(), "the Bolt was countered");
            game.getStack().resolveStack();
            priority(game);
            nextStep(game);
        });

        String boltCost = only(all, "activation", BOLT);
        assertEquals("countered", outcomeOf(boltCost), boltCost);
        assertNull(linkOf(boltCost), boltCost);
        assertEquals(0, records(all, "resolution").stream()
                .filter(r -> RecordLines.scriptOf(r).equals(BOLT)).count(), all.toString());
        linkedHalf(all, COUNTERSPELL);
        String shockHalf = linkedHalf(all, SHOCK);
        assertEquals(List.of("damage_dealt", "life_change"), eventTypes(shockHalf), shockHalf);
    }
}
