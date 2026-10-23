package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
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
import static com.pricepredictor.connector.effects.RecordLines.withEventOfType;
import static com.pricepredictor.connector.effects.TestBoard.cast;
import static com.pricepredictor.connector.effects.TestBoard.enterStep;
import static com.pricepredictor.connector.effects.TestBoard.nextStep;
import static com.pricepredictor.connector.effects.TestBoard.priorityWithTriggers;
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

    /** Resolve the top of the stack and run the priority after it. */
    private static void resolve(Game game) {
        game.getStack().resolveStack();
        priorityWithTriggers(game);
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
            if (script.equals(scriptOf(record))) {
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
            priorityWithTriggers(game);
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
            priorityWithTriggers(game);
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
            priorityWithTriggers(game);
            resolve(game);
            cast(game, grapeshot, caster, opponent);
            priorityWithTriggers(game);
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

        List<String> won = withEventOfType(all, "player_won");
        assertEquals(1, won.size(), "the win is written: " + all);
        assertNull(linkOf(won.get(0)), "an orphan: " + won.get(0));
        assertEquals(DAMAGE, eventTypes(linkedHalf(all, BOLT)), all.toString());
    }

    /**
     * The same ability cast twice while its first cast is still held: the
     * first is written off as {@code countered} when the second arrives, and
     * the second resolves with its own link. A defensive guard, driven by
     * hand because no real game reaches it: a card that leaves the stack is
     * rebuilt by {@code CardCopyService.copyCard} with fresh spell ability
     * ids, so a card cast again always casts a new id.
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
            bracket.recordClauseEvent(new EffectEvent(EventType.KEYWORD_CHANGE));
            game.costPaymentStack.push(
                    bolt.getPayCosts().getCostParts().get(0),
                    new forge.game.cost.CostPayment(bolt.getPayCosts(), bolt));
            bracket.recordClauseEvent(new EffectEvent(EventType.TAPPED));
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

    // ── what a cast sets off before it is announced ─────────────────────

    /**
     * Maro's power and toughness are the cards in its controller's hand, and
     * that controller answers a Shock with the Lightning Bolt that is the
     * last of them. Moving the Bolt to the stack zone shrinks Maro before the
     * Bolt is cast — {@code GameAction.changeZone} runs the static check
     * then — and the 0/0 dies in the check after the cast. Both are the
     * Bolt's consequences, in the Bolt's half; the Shock's is its damage.
     */
    @Test
    void whatAResponsesMoveOntoTheStackSetsOffIsTheResponses() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card shock = inHand(game, caster, "Shock");
        Card bolt = inHand(game, opponent, "Lightning Bolt");
        Card[] maro = new Card[1];

        List<String> all = ShardRun.observed(tempDir, game, collector -> {
            // Entered under the collector's eye, so its shrink has a reading
            // to differ from (FirstStatChangeTest).
            maro[0] = TestBoard.inPlay(game, opponent, "Maro");
            game.getAction().checkStateEffects(false);
            cast(game, shock, caster, opponent);
            cast(game, bolt, opponent, caster);
            priorityWithTriggers(game);
            assertEquals(ZoneType.Graveyard,
                    game.getZoneOf(game.getCardState(maro[0])).getZoneType(),
                    "Maro died with no card left in hand");
            resolve(game);
            resolve(game);
            nextStep(game);
        });

        assertEquals(DAMAGE, eventTypes(linkedHalf(all, SHOCK)), all.toString());
        String boltHalf = linkedHalf(all, BOLT);
        String maroRef = "\"subjects\":[\"E" + maro[0].getId() + "\"]";
        assertTrue(boltHalf.contains("{\"type\":\"pt_change\"," + maroRef),
                "the shrink as the Bolt left the hand: " + boltHalf);
        assertTrue(boltHalf.contains("{\"type\":\"zone_change\"," + maroRef
                + ",\"params\":{\"from_zone\":\"battlefield\",\"to_zone\":\"graveyard\""),
                "the death in the check after the cast: " + boltHalf);
        assertEquals(List.of("pt_change", "zone_change", "damage_dealt", "life_change"),
                eventTypes(boltHalf), boltHalf);
    }

    // ── a player's action at priority ───────────────────────────────────

    /**
     * Battering Craghorn, face down, is turned face up in answer to a Shock
     * — a special action, which {@code MagicStack.add} resolves in place with
     * no cast event, paid and taken through the AI's own play path. What it
     * does is nobody's record: the Shock's half is its damage.
     */
    @Test
    void unmorphingInResponseIsNotTheWaitingSpells() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card shock = inHand(game, caster, "Shock");
        Card[] craghorn = new Card[1];

        List<String> all = ShardRun.observed(tempDir, game, collector -> {
            craghorn[0] = faceDownCraghorn(game, caster);
            SpellAbility unmorph = turnFaceUpOf(craghorn[0], caster);
            cast(game, shock, caster, opponent);
            assertTrue(forge.ai.ComputerUtil.handlePlayingSpellAbility(caster, unmorph, null),
                    "the AI turned the Craghorn face up");
            assertFalse(craghorn[0].isFaceDown(), "the Craghorn is face up");
            priorityWithTriggers(game);
            resolve(game);
            nextStep(game);
        });

        assertEquals(DAMAGE, eventTypes(linkedHalf(all, SHOCK)), all.toString());
    }

    /**
     * A Shivan Reef tapped for red at priority, with a Shock waiting: the
     * damage the Reef deals its controller is the mana ability's, which no
     * record holds, and not the Shock's.
     */
    @Test
    void aManaAbilityAtPriorityIsNotTheWaitingSpells() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card shock = inHand(game, caster, "Shock");
        Card reef = TestBoard.inPlay(game, caster, "Shivan Reef");
        SpellAbility painful = null;
        for (SpellAbility mana : reef.getManaAbilities()) {
            if (mana.getSubAbility() != null) {
                painful = mana;
            }
        }
        assertNotNull(painful, "the Reef's coloured ability, which deals damage");
        painful.setActivatingPlayer(caster);
        SpellAbility tap = painful;

        List<String> all = ShardRun.observed(tempDir, game, collector -> {
            cast(game, shock, caster, opponent);
            assertTrue(forge.ai.ComputerUtil.handlePlayingSpellAbility(caster, tap, null),
                    "the AI activated the Reef");
            assertEquals(19, caster.getLife(), "the Reef dealt 1 damage to its controller");
            priorityWithTriggers(game);
            resolve(game);
            nextStep(game);
        });

        String shockHalf = linkedHalf(all, SHOCK);
        assertEquals(DAMAGE, eventTypes(shockHalf), shockHalf);
        assertFalse(shockHalf.contains("\"subjects\":[\"P0\"]"),
                "nothing dealt to the Reef's controller: " + shockHalf);
    }

    /**
     * Battering Craghorn face down on {@code owner}'s battlefield, with the
     * three Mountains that pay for turning it face up, after the check that
     * follows.
     */
    private static Card faceDownCraghorn(Game game, Player owner) {
        for (int i = 0; i < 3; i++) {
            TestBoard.inPlay(game, owner, "Mountain");
        }
        Card craghorn = TestBoard.inPlay(game, owner, "Battering Craghorn");
        craghorn.turnFaceDown(true);
        game.getAction().checkStateEffects(false);
        return craghorn;
    }

    /** The special action that turns a face-down card face up. */
    private static SpellAbility turnFaceUpOf(Card card, Player activator) {
        // Offered while face down, so it is the face-up state's, among the
        // abilities a player could play now.
        for (SpellAbility sa : card.getAllPossibleAbilities(activator, false)) {
            if (sa.isTurnFaceUp()) {
                sa.setActivatingPlayer(activator);
                return sa;
            }
        }
        throw new AssertionError("no turn-face-up ability on " + card);
    }

    // ── a cast the AI abandoned ─────────────────────────────────────────

    /**
     * What {@code ComputerUtil.handlePlayingSpellAbility} leaves behind when
     * a charm choice, a target check or a payment fails: the card moved to the
     * stack zone, no stack item, and nothing moving it back. Nothing after it
     * is that cast's: a lethal Bolt still writes its win.
     */
    @Test
    void aCardLeftInTheStackZoneDoesNotSwallowAWin() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        opponent.setLife(3, null);
        Card stuck = inHand(game, caster, "Grizzly Bears");
        Card bolt = inHand(game, caster, "Lightning Bolt");

        List<String> all = ShardRun.observed(tempDir, game, collector -> {
            game.getAction().moveToStack(stuck, null);
            priorityWithTriggers(game);
            cast(game, bolt, caster, opponent);
            resolve(game);
            assertTrue(game.isGameOver(), "the Bolt won the game");
            collector.finishGame(Set.of());
        });

        assertEquals(1, withEventOfType(all, "player_won").size(), "the win is written: " + all);
        assertEquals(DAMAGE, eventTypes(linkedHalf(all, BOLT)), all.toString());
    }

    /**
     * The Maro response with an abandoned cast's card in the stack zone: the
     * Bolt's half holds the shrink and the death exactly as it does without
     * it ({@link #whatAResponsesMoveOntoTheStackSetsOffIsTheResponses}).
     */
    @Test
    void aCardLeftInTheStackZoneDoesNotSwallowADeath() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card stuck = inHand(game, caster, "Grizzly Bears");
        Card shock = inHand(game, caster, "Shock");
        Card bolt = inHand(game, opponent, "Lightning Bolt");
        Card[] maro = new Card[1];

        List<String> all = ShardRun.observed(tempDir, game, collector -> {
            maro[0] = TestBoard.inPlay(game, opponent, "Maro");
            game.getAction().checkStateEffects(false);
            game.getAction().moveToStack(stuck, null);
            cast(game, shock, caster, opponent);
            cast(game, bolt, opponent, caster);
            priorityWithTriggers(game);
            assertEquals(ZoneType.Graveyard,
                    game.getZoneOf(game.getCardState(maro[0])).getZoneType(),
                    "Maro died with no card left in hand");
            resolve(game);
            resolve(game);
            nextStep(game);
        });

        assertEquals(DAMAGE, eventTypes(linkedHalf(all, SHOCK)), all.toString());
        assertEquals(List.of("pt_change", "zone_change", "damage_dealt", "life_change"),
                eventTypes(linkedHalf(all, BOLT)), all.toString());
    }

    // ── an action is in no record ───────────────────────────────────────

    /**
     * Battering Craghorn turned face up with nothing cast after it before the
     * game ends: its stat changes are in no record, the game-end orphan
     * record included.
     */
    @Test
    void anActionIsInNoRecordEvenAtTheEndOfTheGame() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card[] craghorn = new Card[1];

        List<String> all = ShardRun.observed(tempDir, game, collector -> {
            craghorn[0] = faceDownCraghorn(game, caster);
            // A cast and its resolution empty the orphans of the board's setup.
            cast(game, inHand(game, caster, "Shock"), caster, game.getPlayers().get(1));
            resolve(game);
            nextStep(game);
            SpellAbility unmorph = turnFaceUpOf(craghorn[0], caster);
            assertTrue(forge.ai.ComputerUtil.handlePlayingSpellAbility(caster, unmorph, null),
                    "the AI turned the Craghorn face up");
            assertFalse(craghorn[0].isFaceDown(), "the Craghorn is face up");
            priorityWithTriggers(game);
            enterStep(game, PhaseType.COMBAT_BEGIN);
            collector.finishGame(Set.of());
        });

        String ref = "\"E" + craghorn[0].getId() + "\"";
        for (String record : all) {
            String payload = record.substring(record.indexOf("\"payload\":"));
            assertFalse(payload.contains("\"subjects\":[" + ref + "]"),
                    "an action's events written in a record: " + record);
        }
    }

    /**
     * An action at priority that files nothing of its own — Rod of Ruin's
     * activation named to the stack, one bus event that no record takes, and
     * the stack let go — still ends the waiting Shock's consequences: the
     * Grizzly Bears that die in the state-based check after it are not in the
     * Shock's half.
     */
    @Test
    void anActionThatFilesNothingStillEndsTheWaitingSpellsConsequences() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card[] bears = new Card[1];

        List<String> all = ShardRun.observed(tempDir, game, collector -> {
            Card rod = TestBoard.inPlay(game, caster, "Rod of Ruin");
            bears[0] = TestBoard.inPlay(game, opponent, "Grizzly Bears");
            cast(game, inHand(game, caster, "Shock"), caster, opponent);
            priorityWithTriggers(game);
            SpellAbility activation = null;
            for (SpellAbility sa : rod.getSpellAbilities()) {
                if (sa.isActivatedAbility()) {
                    activation = sa;
                }
            }
            assertNotNull(activation, "Rod of Ruin's activated ability");
            activation.setActivatingPlayer(caster);
            // ComputerUtil.handlePlayingSpellAbility names the action first.
            game.getStack().freezeStack(activation);
            game.fireEvent(new forge.game.event.GameEventAddLog(
                    forge.game.GameLogEntryType.STACK_ADD, "an action that files nothing"));
            game.getStack().unfreezeStack();
            bears[0].setDamage(2);
            game.getAction().checkStateEffects(false);
            assertEquals(ZoneType.Graveyard,
                    game.getZoneOf(game.getCardState(bears[0])).getZoneType(),
                    "the Bears died in the check");
            resolve(game);
            nextStep(game);
        });

        String shockHalf = linkedHalf(all, SHOCK);
        assertFalse(shockHalf.contains("\"subjects\":[\"E" + bears[0].getId() + "\"]"),
                "the Bears' death after the action is not the Shock's: " + shockHalf);
        assertEquals(DAMAGE, eventTypes(shockHalf), shockHalf);
    }

    /**
     * Bone Splinters cast in the combat damage step, its additional cost a
     * sacrifice: the Swamp's tap and the Elves' sacrifice pay the Splinters
     * and are not the combat's.
     */
    @Test
    void aPaymentInTheDamageStepIsNotTheCombats() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.COMBAT_DAMAGE);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card splinters = inHand(game, caster, "Bone Splinters");
        Card swamp = TestBoard.inPlay(game, caster, "Swamp");
        Card elves = TestBoard.inPlay(game, caster, "Llanowar Elves");
        Card angel = TestBoard.inPlay(game, opponent, "Serra Angel");
        SpellAbility spell = splinters.getFirstSpellAbility();
        spell.setActivatingPlayer(caster);
        spell.resetTargets();
        spell.getTargets().add(angel);

        List<String> all = ShardRun.observed(tempDir, game, collector -> {
            // Something dealt in the step, so the step has a combat record.
            game.fireEvent(new forge.game.event.GameEventPlayerLivesChanged(opponent, 20, 18));
            assertTrue(forge.ai.ComputerUtil.handlePlayingSpellAbility(caster, spell, null),
                    "the AI cast the Splinters");
            assertEquals(ZoneType.Graveyard,
                    game.getZoneOf(game.getCardState(elves)).getZoneType(),
                    "the Elves were sacrificed");
            enterStep(game, PhaseType.COMBAT_END);
        });

        List<String> combat = all.stream().filter(r -> r.contains("\"kind\":\"combat\"")).toList();
        assertEquals(1, combat.size(), all.toString());
        for (Card paid : List.of(swamp, elves)) {
            assertFalse(combat.get(0).contains("\"subjects\":[\"E" + paid.getId() + "\"]"),
                    "the Splinters' payment is in the combat record: " + combat.get(0));
        }
        assertEquals(List.of("life_change"), eventTypes(combat.get(0)), combat.get(0));
    }

    // ── a forced cast ───────────────────────────────────────────────────

    /**
     * A forced Lightning Bolt cast from the hand of Maro's controller: moving
     * it to the stack shrinks Maro, and the forced record holds the shrink,
     * as the live Bolt's half does.
     */
    @Test
    void aForcedCastKeepsWhatItsMoveOntoTheStackSetOff() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card maro = TestBoard.inPlay(game, caster, "Maro");
        TestBoard.inPlay(game, caster, "Grizzly Bears");
        SpellAbility bolt = inHand(game, caster, "Lightning Bolt").getFirstSpellAbility();
        bolt.setActivatingPlayer(caster);
        game.getAction().checkStateEffects(false);
        CollectionCaps caps = new CollectionCaps(2000, 0.1, 2, 0, List.of(), 0.0, 0.0);

        List<String> records = ShardRun.forked(tempDir, game, caps, 5L, bolt).records();

        assertEquals(1, records.size(), records.toString());
        assertTrue(records.get(0).contains(
                        "{\"type\":\"pt_change\",\"subjects\":[\"E" + maro.getId() + "\"]"),
                "the shrink the forced Bolt's move from hand set off: " + records.get(0));
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
            priorityWithTriggers(game);
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
