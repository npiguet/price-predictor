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
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import static com.pricepredictor.connector.effects.RecordLines.eventTypes;
import static com.pricepredictor.connector.effects.RecordLines.linkOf;
import static com.pricepredictor.connector.effects.RecordLines.only;
import static com.pricepredictor.connector.effects.RecordLines.outcomeOf;
import static com.pricepredictor.connector.effects.RecordLines.records;
import static com.pricepredictor.connector.effects.RecordLines.scriptOf;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Several things on the stack at once, each with a bracket of its own.
 *
 * <p>A trigger a cast sets off — prowess, storm — and a spell cast in
 * response both go on the stack above the spell and resolve first. The
 * spell's bracket survives them: when it resolves it writes its own linked
 * effect half, as the items above it wrote theirs, and nothing of theirs is
 * in it — not even what {@code MagicStack.finishResolving} does after the
 * item above has published its resolution, which is to move the card off the
 * stack.
 *
 * <p>Driven through the real engine — a card moved to the stack zone and then
 * onto the stack, as the engine's own play path does, the trigger handler,
 * {@code resolveStack} and the state-based check — with the collector on the
 * game's own bus, in the order {@code PhaseHandler.mainLoopStep} runs them.
 */
@ExtendWith(ForgeExtension.class)
class StackItemBracketTest {

    @TempDir
    Path tempDir;

    private static final String BOLT = "cardsfolder/l/lightning_bolt.txt";
    private static final String SHOCK = "cardsfolder/s/shock.txt";
    private static final String SWIFTSPEAR = "cardsfolder/m/monastery_swiftspear.txt";
    private static final String COUNTERSPELL = "cardsfolder/c/counterspell.txt";
    private static final String GRAPESHOT = "cardsfolder/g/grapeshot.txt";
    private static final String EXPERTISE = "cardsfolder/k/kari_zevs_expertise.txt";

    /** A card in {@code caster}'s hand, to be cast with {@link #cast}. */
    private static Card inHand(Game game, Player caster, String name) {
        return TestBoard.inHand(game, caster, name);
    }

    /**
     * Cast a card's spell at a target: onto the stack zone first, then onto
     * the stack, the order the engine's own play path takes.
     */
    private static SpellAbility cast(Game game, Card card, Player caster, GameObject target) {
        Card onStack = game.getAction().moveToStack(card, null);
        SpellAbility spell = onStack.getFirstSpellAbility();
        assertNotNull(spell, "no spell on " + card);
        spell.setActivatingPlayer(caster);
        spell.resetTargets();
        spell.getTargets().add(target);
        game.getStack().add(spell);
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

    /** Resolve the top of the stack and run the priority after it. */
    private static void resolve(Game game) {
        game.getStack().resolveStack();
        priority(game);
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

    /** Every effect half through {@code script}, in the order written. */
    private static List<String> halvesThrough(List<String> all, String script) {
        List<String> found = new ArrayList<>();
        for (String record : records(all, "resolution")) {
            if (record.contains("\"script_file\":\"") && scriptOf(record).equals(script)) {
                found.add(record);
            }
        }
        return found;
    }

    private static final List<String> DAMAGE = List.of("damage_dealt", "life_change");

    // ── a trigger above the spell ───────────────────────────────────────

    /**
     * A Lightning Bolt cast with Monastery Swiftspear in play: prowess
     * triggers and resolves above the Bolt. Each writes its own linked
     * effect half: the pump in the trigger's, the Bolt's damage and the
     * Bolt's own move to the graveyard nowhere but where they belong.
     */
    @Test
    void aSpellUnderItsCastTriggerWritesItsOwnHalf() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card bolt = inHand(game, caster, "Lightning Bolt");

        List<String> all = ShardRun.observed(tempDir, game, collector -> {
            // Entered under the collector's eye, so the pump is a change it
            // has a reading to compare against (FirstStatChangeTest).
            TestBoard.inPlay(game, caster, "Monastery Swiftspear");
            cast(game, bolt, caster, opponent);
            priority(game);
            assertEquals(2, game.getStack().size(), "prowess went on the stack above the Bolt");
            resolve(game);
            resolve(game);
            nextStep(game);
        });

        assertEquals(DAMAGE, eventTypes(linkedHalf(all, BOLT)), all.toString());
        assertEquals(List.of("pt_change"),
                eventTypes(linkedHalf(all, SWIFTSPEAR)), all.toString());
    }

    /**
     * Maro's power and toughness are the cards in its controller's hand, so
     * casting the last of them leaves a 0/0 that dies in the state-based
     * check before prowess goes on the stack. The death is the cast's
     * consequence and stays in the Bolt's half when the trigger goes above
     * it.
     */
    @Test
    void whatASpellGatheredBeforeATriggerWentAboveItStaysInItsHalf() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        TestBoard.inPlay(game, caster, "Monastery Swiftspear");
        Card maro = TestBoard.inPlay(game, caster, "Maro");
        Card bolt = inHand(game, caster, "Lightning Bolt");

        List<String> all = ShardRun.observed(tempDir, game, collector -> {
            cast(game, bolt, caster, opponent);
            priority(game);
            assertEquals(ZoneType.Graveyard, game.getZoneOf(game.getCardState(maro)).getZoneType(),
                    "Maro died with no card left in hand");
            assertEquals(2, game.getStack().size(), "prowess went on the stack above the Bolt");
            resolve(game);
            resolve(game);
            nextStep(game);
        });

        String boltHalf = linkedHalf(all, BOLT);
        assertTrue(boltHalf.contains("\"subjects\":[\"E" + maro.getId()
                + "\"],\"params\":{\"from_zone\":\"battlefield\",\"to_zone\":\"graveyard\""), boltHalf);
        assertEquals(List.of("zone_change", "damage_dealt", "life_change"), eventTypes(boltHalf),
                boltHalf);
    }

    /**
     * Grapeshot cast after a Lightning Bolt: storm copies it once, and the
     * copy resolves above the original. Each Grapeshot writes its own half
     * with its own damage; the copy ceasing to exist as it leaves the stack
     * is in neither.
     */
    @Test
    void aStormCopyAboveItsOriginalLeavesTheOriginalItsOwnHalf() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card bolt = inHand(game, caster, "Lightning Bolt");
        Card grapeshot = inHand(game, caster, "Grapeshot");

        List<String> all = ShardRun.observed(tempDir, game, collector -> {
            cast(game, bolt, caster, opponent);
            priority(game);
            resolve(game);
            cast(game, grapeshot, caster, opponent);
            priority(game);
            assertEquals(2, game.getStack().size(), "storm went on the stack above Grapeshot");
            resolve(game);
            assertEquals(2, game.getStack().size(), "the copy is above the original");
            resolve(game);
            resolve(game);
            nextStep(game);
        });

        List<String> halves = halvesThrough(all, GRAPESHOT);
        assertEquals(3, halves.size(), "the storm trigger, the copy, the original: " + all);
        for (String half : halves) {
            assertNotNull(linkOf(half), half);
        }
        assertEquals(DAMAGE, eventTypes(halves.get(1)), "the copy: " + halves.get(1));
        assertEquals(DAMAGE, eventTypes(halves.get(2)), "the original: " + halves.get(2));
    }

    // ── a response above the spell ──────────────────────────────────────

    /**
     * Shock, then Lightning Bolt in response, both at the opponent: the Bolt
     * resolves first, and each spell writes its own linked half with its own
     * damage. The Bolt's move to the graveyard, which comes after its
     * resolution is published, is the Shock's neither.
     */
    @Test
    void aSpellAndTheResponseAboveItEachWriteTheirOwnHalf() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card shock = inHand(game, caster, "Shock");
        Card bolt = inHand(game, opponent, "Lightning Bolt");

        List<String> all = ShardRun.observed(tempDir, game, collector -> {
            cast(game, shock, caster, opponent);
            cast(game, bolt, opponent, caster);
            resolve(game);
            resolve(game);
            nextStep(game);
        });

        String boltHalf = linkedHalf(all, BOLT);
        assertEquals(DAMAGE, eventTypes(boltHalf), boltHalf);
        assertTrue(boltHalf.contains("\"amount\":3"), boltHalf);
        String shockHalf = linkedHalf(all, SHOCK);
        assertEquals(DAMAGE, eventTypes(shockHalf), shockHalf);
        assertTrue(shockHalf.contains("\"amount\":2"), shockHalf);
    }

    /**
     * Lightning Bolt, then Shock, then a Counterspell at the Bolt. The
     * Counterspell resolves and writes its half with the Bolt's counter; the
     * Bolt, gone from the stack without resolving, is written off as
     * {@code countered} with no half; the Shock, between them, still writes
     * its own, with nothing of the Counterspell's.
     */
    @Test
    void aCounteredSpellIsWrittenOffAndTheSpellsAroundItKeepTheirHalves() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card bolt = inHand(game, caster, "Lightning Bolt");
        Card shock = inHand(game, caster, "Shock");
        Card counterspell = inHand(game, opponent, "Counterspell");

        List<String> all = ShardRun.observed(tempDir, game, collector -> {
            SpellAbility boltSpell = cast(game, bolt, caster, opponent);
            cast(game, shock, caster, opponent);
            cast(game, counterspell, opponent, boltSpell);
            resolve(game);
            assertEquals(1, game.getStack().size(), "the Bolt was countered");
            resolve(game);
            nextStep(game);
        });

        String boltCost = only(all, "activation", BOLT);
        assertEquals("countered", outcomeOf(boltCost), boltCost);
        assertNull(linkOf(boltCost), boltCost);
        assertTrue(halvesThrough(all, BOLT).isEmpty(), all.toString());
        String counterHalf = linkedHalf(all, COUNTERSPELL);
        assertEquals(List.of("zone_change"), eventTypes(counterHalf), counterHalf);
        assertTrue(counterHalf.contains("\"subjects\":[\"E" + bolt.getId()
                + "\"],\"params\":{\"from_zone\":\"stack\",\"to_zone\":\"graveyard\""),
                "the countered Bolt going to the graveyard: " + counterHalf);
        assertEquals(DAMAGE, eventTypes(linkedHalf(all, SHOCK)), all.toString());
    }

    /**
     * Shock, then a lethal Lightning Bolt above it: the win is decided by the
     * state-based check after the Bolt, with the Shock still on the stack,
     * and is written as the orphaned event it is.
     */
    @Test
    void aWinDecidedAboveAWaitingSpellIsStillWritten() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        opponent.setLife(3, null);
        Card shock = inHand(game, caster, "Shock");
        Card bolt = inHand(game, caster, "Lightning Bolt");

        List<String> all = ShardRun.observed(tempDir, game, collector -> {
            SpellAbility shockSpell = cast(game, shock, caster, opponent);
            cast(game, bolt, caster, opponent);
            resolve(game);
            assertTrue(game.isGameOver(), "the Bolt won the game");
            collector.finishGame(Set.of(shockSpell.getId()));
        });

        List<String> won = new ArrayList<>();
        for (String record : all) {
            if (record.contains("\"type\":\"player_won\"")) {
                won.add(record);
            }
        }
        assertEquals(1, won.size(), "the win is written: " + all);
        assertTrue(won.get(0).contains("\"link_id\":null"), "an orphan: " + won.get(0));
        assertEquals(DAMAGE, eventTypes(linkedHalf(all, BOLT)), all.toString());
    }

    /**
     * A Lightning Bolt cast, taken off the stack without anything noticing,
     * and cast again — its card's spell keeps its id across casts. The first
     * cast is written off as {@code countered} when the second arrives; the
     * second resolves with its own link. Driven by hand, because in a game a
     * reconcile point at the second cast would see the first gone only if the
     * second were not already on the stack under the same id.
     */
    @Test
    void aSpellCastAgainWritesOffItsEarlierCast() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        SpellAbility bolt = inHand(game, caster, "Lightning Bolt").getFirstSpellAbility();
        bolt.setActivatingPlayer(caster);

        List<String> all = ShardRun.records(tempDir, game, (bracket, collectors) -> {
            bracket.beginBracket(bolt);
            bracket.beginBracket(bolt);
            bracket.endBracket(bolt.getId(), false);
            bracket.onNextAction();
        });

        List<String> costs = records(all, "activation");
        assertEquals(2, costs.size(), all.toString());
        assertEquals("countered", outcomeOf(costs.get(0)), costs.get(0));
        assertNull(linkOf(costs.get(0)), costs.get(0));
        assertEquals("resolved", outcomeOf(costs.get(1)), costs.get(1));
        assertEquals(linkOf(costs.get(1)), linkOf(only(all, "resolution", BOLT)));
    }

    /**
     * A response's payment, made while a Shock waits on the stack — a land
     * tapped, an optional cost turned down — is the response's cost, not
     * the Shock's: the Shock keeps what it gathered since its cast, and none
     * of that payment. Driven by hand, as {@code CostPayment.payCost} wraps
     * each part it pays.
     */
    @Test
    void aResponsesPaymentIsNotTheWaitingSpells() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        SpellAbility shock = inHand(game, caster, "Shock").getFirstSpellAbility();
        shock.setActivatingPlayer(caster);
        SpellAbility bolt = inHand(game, caster, "Lightning Bolt").getFirstSpellAbility();
        bolt.setActivatingPlayer(caster);

        List<String> all = ShardRun.records(tempDir, game, (bracket, collectors) -> {
            bracket.beginBracket(shock);
            bracket.recordClauseEvent(new EffectEvent(EffectEvent.KEYWORD_CHANGE));
            game.costPaymentStack.push(
                    bolt.getPayCosts().getCostParts().get(0),
                    new forge.game.cost.CostPayment(bolt.getPayCosts(), bolt));
            bracket.recordClauseEvent(new EffectEvent(EffectEvent.TAPPED));
            bracket.noteDeclined();
            game.costPaymentStack.pop();
            bracket.beginBracket(bolt);
            bracket.endBracket(bolt.getId(), false);
            bracket.endBracket(shock.getId(), false);
            bracket.onNextAction();
        });

        String shockCost = only(all, "activation", SHOCK);
        assertEquals("resolved", outcomeOf(shockCost), shockCost);
        assertEquals(List.of("keyword_change"), eventTypes(linkedHalf(all, SHOCK)),
                "what the Shock gathered, without the Bolt's payment: " + all);
    }

    // ── a spell that casts another ──────────────────────────────────────

    /**
     * Kari Zev's Expertise takes a creature and casts the Lightning Bolt in
     * its caster's hand. The Bolt resolves after the Expertise; the
     * Expertise's own move to the graveyard, after its resolution is
     * published, is in neither half.
     */
    @Test
    void aSpellCastBySorceryGetsNoneOfTheSorcerysMoveOffTheStack() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card bears = TestBoard.inPlay(game, opponent, "Grizzly Bears");
        Card expertise = inHand(game, caster, "Kari Zev's Expertise");
        Card bolt = inHand(game, caster, "Lightning Bolt");

        List<String> all = ShardRun.observed(tempDir, game, collector -> {
            cast(game, expertise, caster, bears);
            game.getStack().resolveStack();
            assertEquals(ZoneType.Stack, game.getZoneOf(game.getCardState(bolt)).getZoneType(),
                    "the Expertise cast the Bolt");
            priority(game);
            resolve(game);
            nextStep(game);
        });

        assertEquals(DAMAGE, eventTypes(linkedHalf(all, BOLT)),
                "nothing of the Expertise's is the Bolt's: " + all);
        String expertiseHalf = linkedHalf(all, EXPERTISE);
        assertFalse(expertiseHalf.contains("\"subjects\":[\"E" + expertise.getId() + "\"]"),
                "the Expertise's own move off the stack is nobody's effect: " + expertiseHalf);
        assertFalse(eventTypes(expertiseHalf).contains("damage_dealt"), expertiseHalf);
    }
}
