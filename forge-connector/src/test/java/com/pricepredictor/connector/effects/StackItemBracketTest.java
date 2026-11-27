package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.event.GameEventGameFinished;
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

import static com.pricepredictor.connector.effects.RecordEvents.eventTypes;
import static com.pricepredictor.connector.effects.RecordEvents.eventsNaming;
import static com.pricepredictor.connector.effects.RecordAbilities.linkOf;
import static com.pricepredictor.connector.effects.RecordPicks.only;
import static com.pricepredictor.connector.effects.RecordAbilities.outcomeOf;
import static com.pricepredictor.connector.effects.RecordPicks.records;
import static com.pricepredictor.connector.effects.RecordAbilities.scriptOf;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static com.pricepredictor.connector.effects.RecordMoves.damageTo;
import static com.pricepredictor.connector.effects.RecordMoves.deathsOf;
import static com.pricepredictor.connector.effects.RecordMoves.eventsOn;
import static com.pricepredictor.connector.effects.RecordMoves.moves;
import static com.pricepredictor.connector.effects.RecordMoves.movesOffTheStack;
import static com.pricepredictor.connector.effects.RecordPicks.withEventOfType;
import static com.pricepredictor.connector.effects.TestBoard.cast;
import static com.pricepredictor.connector.effects.TestBoard.enterStep;
import static com.pricepredictor.connector.effects.TestBoard.nextStep;
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
 * {@code resolveStack} and the state-based loop — with the game recorded as a
 * worker on the patched checkout records it ({@link ShardRun#hooked}), in the
 * order {@code PhaseHandler.mainLoopStep} runs them.
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
    private static final String WHY = "for the resolution starts and the state-based loop's signals";

    /** A card in {@code caster}'s hand, to be cast with {@link #cast}. */
    private static Card inHand(Game game, Player caster, String name) {
        return TestBoard.inHand(game, caster, name);
    }

    /** The events of {@code type} the recording has counted as reaching no record in the first main phase. */
    private static long dropped(GameRecording recording, String type) {
        return recording.droppedEvents().counts()
                .getOrDefault(new DroppedEvents.Key(type, "main1"), 0L);
    }

    /** The moves the recording has counted as reaching no record in the first main phase. */
    private static long droppedMoves(GameRecording recording) {
        return dropped(recording, "zone_change");
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
    /** A burn spell's half: its damage, then its card's move off the stack (FR-042b). */
    private static final List<String> SPELL = List.of("damage_dealt", "life_change", "zone_change");

    /** The effect half through {@code script}, of an item nobody cast: a trigger, a copy. */
    private static String unlinkedHalf(List<String> all, String script) {
        String half = only(all, "resolution", script);
        assertNull(linkOf(half), "no cost half, no link: " + half);
        return half;
    }

    // ── a trigger above the spell ───────────────────────────────────────

    /**
     * A Lightning Bolt cast with Monastery Swiftspear in play: prowess
     * triggers and resolves above the Bolt. Each writes its own effect half,
     * the trigger's unlinked, having no cost half: the pump in the trigger's,
     * the Bolt's damage and the Bolt's own move to the graveyard in the
     * Bolt's.
     */
    @Test
    void aSpellUnderItsCastTriggerWritesItsOwnHalf() throws Throwable {
        Duel duel = Duel.inMainPhase();
        Game game = duel.game();
        Card bolt = inHand(game, duel.caster(), "Lightning Bolt");

        List<String> all = ShardRun.hooked(tempDir, game, WHY, recording -> {
            // Entered under the collector's eye, so the pump is a change it
            // has a reading to compare against (FirstStatChangeTest).
            TestBoard.inPlay(game, duel.caster(), "Monastery Swiftspear");
            cast(game, bolt, duel.caster(), duel.opponent());
            duel.priority();
            assertEquals(2, game.getStack().size(), "prowess went on the stack above the Bolt");
            duel.resolve();
            duel.resolve();
            nextStep(game);
        });

        assertEquals(SPELL, eventTypes(linkedHalf(all, BOLT)), all.toString());
        assertEquals(List.of("pt_change"),
                eventTypes(unlinkedHalf(all, SWIFTSPEAR)), all.toString());
    }

    /**
     * Maro's power and toughness are the cards in its controller's hand, so
     * casting the last of them shrinks it as the Bolt leaves the hand, before
     * prowess goes on the stack. The shrink precedes the Bolt's effect half's
     * board and is no record's: it is counted as dropped (FR-042a, FR-042b).
     * The 0/0 then dies in the state-based check with no resolution held and
     * no damage step open, so that death reaches no record and is counted
     * (FR-029). The Bolt's half is its damage and its move off the stack.
     */
    @Test
    void whatACastSetOffBeforeItsTriggerWentAboveItIsCountedAsDropped() throws Throwable {
        Duel duel = Duel.inMainPhase();
        Game game = duel.game();
        TestBoard.inPlay(game, duel.caster(), "Monastery Swiftspear");
        Card bolt = inHand(game, duel.caster(), "Lightning Bolt");
        Card[] maro = new Card[1];

        List<String> all = ShardRun.hooked(tempDir, game, WHY, recording -> {
            // Entered under the collector's eye, so its shrink has a reading
            // to differ from (FirstStatChangeTest).
            maro[0] = TestBoard.inPlay(game, duel.caster(), "Maro");
            duel.stateBasedLoop();
            cast(game, bolt, duel.caster(), duel.opponent());
            assertTrue(dropped(recording, "pt_change") >= 1,
                    "the cast's shrink is counted as dropped (A4): "
                            + recording.droppedEvents().counts());
            long movesDroppedBefore = droppedMoves(recording);
            duel.priority();
            assertEquals(ZoneType.Graveyard,
                    game.getZoneOf(game.getCardState(maro[0])).getZoneType(),
                    "Maro died with no card left in hand");
            assertEquals(2, game.getStack().size(), "prowess went on the stack above the Bolt");
            assertEquals(1L, droppedMoves(recording) - movesDroppedBefore,
                    "the death is counted: " + recording.droppedEvents().counts());
            duel.resolve();
            duel.resolve();
            nextStep(game);
        });

        String boltHalf = linkedHalf(all, BOLT);
        assertEquals(SPELL, eventTypes(boltHalf), boltHalf);
    }

    /**
     * Grapeshot cast after a Lightning Bolt: storm copies it once, and the
     * copy resolves above the original. Each Grapeshot writes its own half
     * with its own damage; the storm trigger and the copy, nobody's casts,
     * have no cost half and no link, and the original's half ends with its
     * card's move to the graveyard.
     */
    @Test
    void aStormCopyAboveItsOriginalLeavesTheOriginalItsOwnHalf() throws Throwable {
        Duel duel = Duel.inMainPhase();
        Game game = duel.game();
        Card bolt = inHand(game, duel.caster(), "Lightning Bolt");
        Card grapeshot = inHand(game, duel.caster(), "Grapeshot");

        List<String> all = ShardRun.hooked(tempDir, game, WHY, recording -> {
            cast(game, bolt, duel.caster(), duel.opponent());
            duel.priority();
            duel.resolve();
            cast(game, grapeshot, duel.caster(), duel.opponent());
            duel.priority();
            assertEquals(2, game.getStack().size(), "storm went on the stack above Grapeshot");
            duel.resolve();
            assertEquals(2, game.getStack().size(), "the copy is above the original");
            duel.resolve();
            duel.resolve();
            nextStep(game);
        });

        List<String> halves = halvesThrough(all, GRAPESHOT);
        assertEquals(3, halves.size(), "the storm trigger, the copy, the original: " + all);
        assertNull(linkOf(halves.get(0)), "the storm trigger: " + halves.get(0));
        assertNull(linkOf(halves.get(1)), "the copy: " + halves.get(1));
        assertNotNull(linkOf(halves.get(2)), "the original: " + halves.get(2));
        assertEquals(DAMAGE, eventTypes(halves.get(1)).subList(0, 2), "the copy: " + halves.get(1));
        assertEquals(SPELL, eventTypes(halves.get(2)), "the original: " + halves.get(2));
    }

    // ── a response above the spell ──────────────────────────────────────

    /**
     * Shock, then Lightning Bolt in response, both at the opponent: the Bolt
     * resolves first, and each spell writes its own linked half with its own
     * damage and its own move to the graveyard, which comes after its
     * resolution is published.
     */
    @Test
    void aSpellAndTheResponseAboveItEachWriteTheirOwnHalf() throws Throwable {
        Duel duel = Duel.inMainPhase();
        Game game = duel.game();
        Card shock = inHand(game, duel.caster(), "Shock");
        Card bolt = inHand(game, duel.opponent(), "Lightning Bolt");

        List<String> all = ShardRun.hooked(tempDir, game, WHY, recording -> {
            cast(game, shock, duel.caster(), duel.opponent());
            cast(game, bolt, duel.opponent(), duel.caster());
            duel.resolve();
            duel.resolve();
            nextStep(game);
        });

        String boltHalf = linkedHalf(all, BOLT);
        assertEquals(SPELL, eventTypes(boltHalf), boltHalf);
        assertEquals(List.of(3L), damageTo(boltHalf, Refs.of(duel.caster())), boltHalf);
        String shockHalf = linkedHalf(all, SHOCK);
        assertEquals(SPELL, eventTypes(shockHalf), shockHalf);
        assertEquals(1, movesOffTheStack(shockHalf, shock).size(), shockHalf);
        assertEquals(List.of(), eventsNaming(shockHalf, Refs.of(bolt)), shockHalf);
        assertEquals(List.of(2L), damageTo(shockHalf, Refs.of(duel.opponent())), shockHalf);
    }

    /**
     * Lightning Bolt, then Shock, then a Counterspell at the Bolt. The
     * Counterspell resolves and writes its half with the Bolt's move off the
     * stack, the {@code spell_countered} the counter hook reports, and its
     * own move off the stack (FR-042b); the Bolt, gone from the stack without
     * resolving, has its cost half from its cast and writes no effect half;
     * the Shock, between them, still writes its own, with nothing of the
     * Counterspell's.
     */
    @Test
    void aCounteredSpellIsWrittenOffAndTheSpellsAroundItKeepTheirHalves() throws Throwable {
        Duel duel = Duel.inMainPhase();
        Game game = duel.game();
        Card bolt = inHand(game, duel.caster(), "Lightning Bolt");
        Card shock = inHand(game, duel.caster(), "Shock");
        Card counterspell = inHand(game, duel.opponent(), "Counterspell");

        List<String> all = ShardRun.hooked(tempDir, game, WHY, recording -> {
            SpellAbility boltSpell = cast(game, bolt, duel.caster(), duel.opponent());
            cast(game, shock, duel.caster(), duel.opponent());
            cast(game, counterspell, duel.opponent(), boltSpell);
            duel.resolve();
            assertEquals(1, game.getStack().size(), "the Bolt was countered");
            duel.resolve();
            nextStep(game);
        });

        String boltCost = only(all, "activation", BOLT);
        assertFalse(RecordEvents.payload(boltCost).containsKey("outcome"), boltCost);
        assertTrue(halvesThrough(all, BOLT).isEmpty(), all.toString());
        String counterHalf = linkedHalf(all, COUNTERSPELL);
        assertEquals(List.of("zone_change", "spell_countered", "zone_change"),
                eventTypes(counterHalf), counterHalf);
        assertEquals(1, moves(counterHalf, bolt, "stack", "graveyard").size(),
                "the countered Bolt going to the graveyard: " + counterHalf);
        assertEquals(1, eventsOn(counterHalf, bolt, "spell_countered").size(), counterHalf);
        assertEquals(1, moves(counterHalf, counterspell, "stack", "graveyard").size(), counterHalf);
        assertEquals(SPELL, eventTypes(linkedHalf(all, SHOCK)), all.toString());
    }

    /**
     * Shock, then a lethal Lightning Bolt above it: the win is decided by the
     * state-based check after the Bolt, with the Shock still on the stack,
     * and is written as the orphaned event it is.
     */
    @Test
    void aWinDecidedAboveAWaitingSpellIsStillWritten() throws Throwable {
        Duel duel = Duel.inMainPhase();
        Game game = duel.game();
        duel.opponent().setLife(3, null);
        Card shock = inHand(game, duel.caster(), "Shock");
        Card bolt = inHand(game, duel.caster(), "Lightning Bolt");

        List<String> all = ShardRun.hooked(tempDir, game, WHY, recording -> {
            cast(game, shock, duel.caster(), duel.opponent());
            cast(game, bolt, duel.caster(), duel.opponent());
            duel.resolve();
            assertTrue(game.isGameOver(), "the Bolt won the game");
            assertEquals(1, game.getStack().size(), "the Shock still waits");
            game.fireEvent(new GameEventGameFinished());
        });

        List<String> won = withEventOfType(all, "player_won");
        assertEquals(1, won.size(), "the win is written: " + all);
        assertNull(linkOf(won.get(0)), "an orphan: " + won.get(0));
        assertEquals(SPELL, eventTypes(linkedHalf(all, BOLT)), all.toString());
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
        Duel duel = Duel.inMainPhase();
        Game game = duel.game();
        SpellAbility shock = inHand(game, duel.caster(), "Shock").getFirstSpellAbility();
        shock.setActivatingPlayer(duel.caster());
        SpellAbility bolt = inHand(game, duel.caster(), "Lightning Bolt").getFirstSpellAbility();
        bolt.setActivatingPlayer(duel.caster());

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

        String shockHalf = only(all, "resolution", SHOCK);
        assertEquals("resolved", outcomeOf(shockHalf), shockHalf);
        assertEquals(List.of("keyword_change"), eventTypes(linkedHalf(all, SHOCK)),
                "what the Shock gathered, without the Bolt's payment: " + all);
    }

    // ── what a cast sets off before it is announced ─────────────────────

    /**
     * Maro's power and toughness are the cards in its controller's hand, and
     * that controller answers a Shock with the Lightning Bolt that is the
     * last of them. Moving the Bolt to the stack zone shrinks Maro before the
     * Bolt is cast — {@code GameAction.changeZone} runs the static check
     * then — and the 0/0 dies in the check after the cast. The shrink
     * precedes the Bolt's effect half's board and is in neither half
     * (FR-042a, FR-042b); the Shock's is its damage. The
     * death is a state-based one with no resolution held and no damage step
     * open, so it reaches no record and is counted (FR-029).
     */
    @Test
    void whatAResponsesMoveOntoTheStackSetsOffIsTheResponses() throws Throwable {
        Duel duel = Duel.inMainPhase();
        Game game = duel.game();
        Card shock = inHand(game, duel.caster(), "Shock");
        Card bolt = inHand(game, duel.opponent(), "Lightning Bolt");
        Card[] maro = new Card[1];

        List<String> all = ShardRun.hooked(tempDir, game, WHY, recording -> {
            // Entered under the collector's eye, so its shrink has a reading
            // to differ from (FirstStatChangeTest).
            maro[0] = TestBoard.inPlay(game, duel.opponent(), "Maro");
            duel.stateBasedLoop();
            cast(game, shock, duel.caster(), duel.opponent());
            cast(game, bolt, duel.opponent(), duel.caster());
            long movesDroppedBefore = droppedMoves(recording);
            duel.priority();
            assertEquals(ZoneType.Graveyard,
                    game.getZoneOf(game.getCardState(maro[0])).getZoneType(),
                    "Maro died with no card left in hand");
            assertEquals(1L, droppedMoves(recording) - movesDroppedBefore,
                    "the death is counted: " + recording.droppedEvents().counts());
            duel.resolve();
            duel.resolve();
            nextStep(game);
        });

        assertEquals(SPELL, eventTypes(linkedHalf(all, SHOCK)), all.toString());
        String boltHalf = linkedHalf(all, BOLT);
        assertEquals(SPELL, eventTypes(boltHalf), boltHalf);
        for (String record : all) {
            assertEquals(List.of(), deathsOf(record, maro[0]), "the death reached a record: " + record);
        }
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
        Duel duel = Duel.inMainPhase();
        Game game = duel.game();
        Card shock = inHand(game, duel.caster(), "Shock");
        Card[] craghorn = new Card[1];

        List<String> all = ShardRun.hooked(tempDir, game, WHY, recording -> {
            craghorn[0] = faceDownCraghorn(duel);
            SpellAbility unmorph = turnFaceUpOf(craghorn[0], duel.caster());
            cast(game, shock, duel.caster(), duel.opponent());
            assertTrue(forge.ai.ComputerUtil.handlePlayingSpellAbility(duel.caster(), unmorph, null),
                    "the AI turned the Craghorn face up");
            assertFalse(craghorn[0].isFaceDown(), "the Craghorn is face up");
            duel.priority();
            duel.resolve();
            nextStep(game);
        });

        assertEquals(SPELL, eventTypes(linkedHalf(all, SHOCK)), all.toString());
    }

    /**
     * A Shivan Reef tapped for red at priority, with a Shock waiting: the
     * damage the Reef deals its controller is the mana ability's, which no
     * record holds, and not the Shock's.
     */
    @Test
    void aManaAbilityAtPriorityIsNotTheWaitingSpells() throws Throwable {
        Duel duel = Duel.inMainPhase();
        Game game = duel.game();
        Player caster = duel.caster();
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

        List<String> all = ShardRun.hooked(tempDir, game, WHY, recording -> {
            cast(game, shock, caster, duel.opponent());
            assertTrue(forge.ai.ComputerUtil.handlePlayingSpellAbility(caster, tap, null),
                    "the AI activated the Reef");
            assertEquals(19, caster.getLife(), "the Reef dealt 1 damage to its controller");
            duel.priority();
            duel.resolve();
            nextStep(game);
        });

        String shockHalf = linkedHalf(all, SHOCK);
        assertEquals(SPELL, eventTypes(shockHalf), shockHalf);
        assertEquals(List.of(), eventsNaming(shockHalf, Refs.of(caster)),
                "nothing dealt to the Reef's controller: " + shockHalf);
    }

    /**
     * Battering Craghorn face down on the caster's battlefield, with the
     * three Mountains that pay for turning it face up, after the state-based
     * loop that follows.
     */
    private static Card faceDownCraghorn(Duel duel) {
        for (int i = 0; i < 3; i++) {
            TestBoard.inPlay(duel.game(), duel.caster(), "Mountain");
        }
        Card craghorn = TestBoard.inPlay(duel.game(), duel.caster(), "Battering Craghorn");
        craghorn.turnFaceDown(true);
        duel.stateBasedLoop();
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
        Duel duel = Duel.inMainPhase();
        Game game = duel.game();
        duel.opponent().setLife(3, null);
        Card stuck = inHand(game, duel.caster(), "Grizzly Bears");
        Card bolt = inHand(game, duel.caster(), "Lightning Bolt");

        List<String> all = ShardRun.hooked(tempDir, game, WHY, recording -> {
            game.getAction().moveToStack(stuck, null);
            duel.priority();
            cast(game, bolt, duel.caster(), duel.opponent());
            duel.resolve();
            assertTrue(game.isGameOver(), "the Bolt won the game");
            game.fireEvent(new GameEventGameFinished());
        });

        assertEquals(1, withEventOfType(all, "player_won").size(), "the win is written: " + all);
        assertEquals(SPELL, eventTypes(linkedHalf(all, BOLT)), all.toString());
    }

    /**
     * The Maro response with an abandoned cast's card in the stack zone: the
     * Bolt's half is exactly as it is without it, and the
     * death is counted rather than swallowed by the stuck card's cast
     * ({@link #whatAResponsesMoveOntoTheStackSetsOffIsTheResponses}).
     */
    @Test
    void aCardLeftInTheStackZoneDoesNotSwallowADeath() throws Throwable {
        Duel duel = Duel.inMainPhase();
        Game game = duel.game();
        Card stuck = inHand(game, duel.caster(), "Grizzly Bears");
        Card shock = inHand(game, duel.caster(), "Shock");
        Card bolt = inHand(game, duel.opponent(), "Lightning Bolt");
        Card[] maro = new Card[1];

        List<String> all = ShardRun.hooked(tempDir, game, WHY, recording -> {
            maro[0] = TestBoard.inPlay(game, duel.opponent(), "Maro");
            duel.stateBasedLoop();
            game.getAction().moveToStack(stuck, null);
            cast(game, shock, duel.caster(), duel.opponent());
            cast(game, bolt, duel.opponent(), duel.caster());
            long movesDroppedBefore = droppedMoves(recording);
            duel.priority();
            assertEquals(ZoneType.Graveyard,
                    game.getZoneOf(game.getCardState(maro[0])).getZoneType(),
                    "Maro died with no card left in hand");
            assertEquals(1L, droppedMoves(recording) - movesDroppedBefore,
                    "the death is counted: " + recording.droppedEvents().counts());
            duel.resolve();
            duel.resolve();
            nextStep(game);
        });

        assertEquals(SPELL, eventTypes(linkedHalf(all, SHOCK)), all.toString());
        assertEquals(SPELL, eventTypes(linkedHalf(all, BOLT)), all.toString());
    }

    // ── an action is in no record ───────────────────────────────────────

    /**
     * Battering Craghorn turned face up with nothing cast after it before the
     * game ends: its stat changes are in no record, the game-end orphan
     * record included.
     */
    @Test
    void anActionIsInNoRecordEvenAtTheEndOfTheGame() throws Throwable {
        Duel duel = Duel.inMainPhase();
        Game game = duel.game();
        Card[] craghorn = new Card[1];

        List<String> all = ShardRun.hooked(tempDir, game, WHY, recording -> {
            craghorn[0] = faceDownCraghorn(duel);
            // A cast and its resolution empty the orphans of the board's setup.
            cast(game, inHand(game, duel.caster(), "Shock"), duel.caster(), duel.opponent());
            duel.resolve();
            nextStep(game);
            SpellAbility unmorph = turnFaceUpOf(craghorn[0], duel.caster());
            assertTrue(forge.ai.ComputerUtil.handlePlayingSpellAbility(duel.caster(), unmorph, null),
                    "the AI turned the Craghorn face up");
            assertFalse(craghorn[0].isFaceDown(), "the Craghorn is face up");
            duel.priority();
            enterStep(game, PhaseType.COMBAT_BEGIN);
            game.fireEvent(new GameEventGameFinished());
        });

        String ref = Refs.of(craghorn[0]);
        for (String record : all) {
            assertEquals(List.of(), eventsNaming(record, ref),
                    "an action's events written in a record: " + record);
        }
    }

    /**
     * An action at priority that files nothing of its own — Rod of Ruin's
     * activation named to the stack, one bus event that no record takes, and
     * the stack let go — still ends the waiting Shock's consequences: the
     * Grizzly Bears that die in the state-based loop after it are not in the
     * Shock's half.
     */
    @Test
    void anActionThatFilesNothingStillEndsTheWaitingSpellsConsequences() throws Throwable {
        Duel duel = Duel.inMainPhase();
        Game game = duel.game();
        Card[] bears = new Card[1];

        List<String> all = ShardRun.hooked(tempDir, game, WHY, recording -> {
            Card rod = TestBoard.inPlay(game, duel.caster(), "Rod of Ruin");
            bears[0] = TestBoard.inPlay(game, duel.opponent(), "Grizzly Bears");
            cast(game, inHand(game, duel.caster(), "Shock"), duel.caster(), duel.opponent());
            duel.priority();
            SpellAbility activation = null;
            for (SpellAbility sa : rod.getSpellAbilities()) {
                if (sa.isActivatedAbility()) {
                    activation = sa;
                }
            }
            assertNotNull(activation, "Rod of Ruin's activated ability");
            activation.setActivatingPlayer(duel.caster());
            // ComputerUtil.handlePlayingSpellAbility names the action first.
            game.getStack().freezeStack(activation);
            game.fireEvent(new forge.game.event.GameEventAddLog(
                    forge.game.GameLogEntryType.STACK_ADD, "an action that files nothing"));
            game.getStack().unfreezeStack();
            bears[0].setDamage(2);
            duel.stateBasedLoop();
            assertEquals(ZoneType.Graveyard,
                    game.getZoneOf(game.getCardState(bears[0])).getZoneType(),
                    "the Bears died in the check");
            duel.resolve();
            nextStep(game);
        });

        String shockHalf = linkedHalf(all, SHOCK);
        assertEquals(List.of(), eventsNaming(shockHalf, Refs.of(bears[0])),
                "the Bears' death after the action is not the Shock's: " + shockHalf);
        assertEquals(SPELL, eventTypes(shockHalf), shockHalf);
    }

    /**
     * Bone Splinters cast in the combat damage step, its additional cost a
     * sacrifice: the Swamp's tap and the Elves' sacrifice pay the Splinters
     * and are not the combat's.
     */
    @Test
    void aPaymentInTheDamageStepIsNotTheCombats() throws Throwable {
        Duel duel = Duel.at(PhaseType.COMBAT_DAMAGE);
        Game game = duel.game();
        Player caster = duel.caster();
        Card splinters = inHand(game, caster, "Bone Splinters");
        Card swamp = TestBoard.inPlay(game, caster, "Swamp");
        Card elves = TestBoard.inPlay(game, caster, "Llanowar Elves");
        Card angel = TestBoard.inPlay(game, duel.opponent(), "Serra Angel");
        SpellAbility spell = splinters.getFirstSpellAbility();
        spell.setActivatingPlayer(caster);
        spell.resetTargets();
        spell.getTargets().add(angel);

        List<String> all = ShardRun.hooked(tempDir, game, WHY, recording -> {
            // Something dealt in the step, so the step has a combat record.
            game.fireEvent(new forge.game.event.GameEventPlayerLivesChanged(duel.opponent(), 20, 18));
            assertTrue(forge.ai.ComputerUtil.handlePlayingSpellAbility(caster, spell, null),
                    "the AI cast the Splinters");
            assertEquals(ZoneType.Graveyard,
                    game.getZoneOf(game.getCardState(elves)).getZoneType(),
                    "the Elves were sacrificed");
            enterStep(game, PhaseType.COMBAT_END);
        });

        List<String> combat = all.stream().filter(r -> "combat".equals(parse(r).get("kind"))).toList();
        assertEquals(1, combat.size(), all.toString());
        for (Card paid : List.of(swamp, elves)) {
            assertEquals(List.of(), eventsNaming(combat.get(0), Refs.of(paid)),
                    "the Splinters' payment is in the combat record: " + combat.get(0));
        }
        assertEquals(List.of("life_change"), eventTypes(combat.get(0)), combat.get(0));
    }

    // ── a forced cast ───────────────────────────────────────────────────

    /**
     * A forced Lightning Bolt cast from the hand of Maro's controller: moving
     * it to the stack shrinks Maro before the forced resolution starts, so
     * the forced record, read from there, holds no shrink, as the live Bolt's
     * half holds none (FR-042b).
     */
    @Test
    void aForcedCastsMoveOntoTheStackIsNotItsEffect() throws Throwable {
        Duel duel = Duel.inMainPhase();
        Game game = duel.game();
        Card maro = TestBoard.inPlay(game, duel.caster(), "Maro");
        TestBoard.inPlay(game, duel.caster(), "Grizzly Bears");
        SpellAbility bolt = inHand(game, duel.caster(), "Lightning Bolt").getFirstSpellAbility();
        bolt.setActivatingPlayer(duel.caster());
        duel.stateBasedLoop();
        CollectionCaps caps = TestCaps.caps().manaCap(2000).probes(0).build();

        List<String> records = ShardRun.forkedWithClauseHook(
                tempDir, new ShardRun.ForkRun(game, caps, 5L, bolt), "for the fork's own resolution").records();

        assertEquals(1, records.size(), records.toString());
        assertEquals(List.of(), eventsOn(records.get(0), maro, "pt_change"),
                "the shrink the forced Bolt's move from hand set off: " + records.get(0));
    }

    // ── a spell that casts another ──────────────────────────────────────

    /**
     * Kari Zev's Expertise takes a creature and casts the Lightning Bolt in
     * its caster's hand. The Bolt resolves after the Expertise; the
     * Expertise's own move to the graveyard, after its resolution is
     * published, ends the Expertise's half and is nothing of the Bolt's.
     */
    @Test
    void aSpellCastBySorceryGetsNoneOfTheSorcerysMoveOffTheStack() throws Throwable {
        Duel duel = Duel.inMainPhase();
        Game game = duel.game();
        Card bears = TestBoard.inPlay(game, duel.opponent(), "Grizzly Bears");
        Card expertise = inHand(game, duel.caster(), "Kari Zev's Expertise");
        Card bolt = inHand(game, duel.caster(), "Lightning Bolt");

        List<String> all = ShardRun.hooked(tempDir, game, WHY, recording -> {
            cast(game, expertise, duel.caster(), bears);
            game.getStack().resolveStack();
            assertEquals(ZoneType.Stack, game.getZoneOf(game.getCardState(bolt)).getZoneType(),
                    "the Expertise cast the Bolt");
            duel.priority();
            duel.resolve();
            nextStep(game);
        });

        String boltHalf = linkedHalf(all, BOLT);
        assertEquals(SPELL, eventTypes(boltHalf), "nothing of the Expertise's is the Bolt's: " + all);
        assertEquals(List.of(), eventsNaming(boltHalf, Refs.of(expertise)), boltHalf);
        String expertiseHalf = linkedHalf(all, EXPERTISE);
        List<String> types = eventTypes(expertiseHalf);
        assertEquals("zone_change", types.get(types.size() - 1), expertiseHalf);
        assertEquals(1, moves(expertiseHalf, expertise, "stack", "graveyard").size(),
                "the Expertise's own move off the stack ends its half: " + expertiseHalf);
        assertFalse(types.contains("damage_dealt"), expertiseHalf);
    }
}
