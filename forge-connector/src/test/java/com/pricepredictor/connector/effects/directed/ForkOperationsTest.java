package com.pricepredictor.connector.effects.directed;

import forge.ai.simulation.GameCopier;
import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.BoardStates;
import com.pricepredictor.connector.effects.ForkRefusal;
import com.pricepredictor.connector.effects.TestBoard;
import com.pricepredictor.connector.effects.directed.operations.AddPermanentOperation;
import com.pricepredictor.connector.effects.directed.operations.BoardEdit;
import com.pricepredictor.connector.effects.directed.operations.ColourTypeOperation;
import com.pricepredictor.connector.effects.directed.operations.CountersOperation;
import com.pricepredictor.connector.effects.directed.operations.KeywordOperation;
import com.pricepredictor.connector.effects.directed.operations.LifeOperation;
import com.pricepredictor.connector.effects.directed.operations.MoveCardOperation;
import com.pricepredictor.connector.effects.directed.operations.PtOperation;
import com.pricepredictor.connector.effects.directed.operations.RemovePermanentOperation;
import com.pricepredictor.connector.effects.directed.operations.ReplaceCardOperation;
import com.pricepredictor.connector.effects.directed.operations.TapOperation;
import forge.card.ColorSet;
import forge.game.Game;
import forge.game.ability.AbilityKey;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.keyword.Keyword;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import static com.pricepredictor.connector.effects.directed.BoardChanges.changes;
import static com.pricepredictor.connector.effects.directed.BoardChanges.entity;
import static com.pricepredictor.connector.effects.directed.BoardChanges.id;
import static com.pricepredictor.connector.effects.directed.BoardChanges.indexed;
import static com.pricepredictor.connector.effects.directed.BoardChanges.links;
import static com.pricepredictor.connector.effects.directed.BoardChanges.renamed;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Each fork operation changes exactly its one fact (T086), and the three
 * board-membership operations change it silently (T086a, story 3 acceptance
 * 6): no trigger fires from the edit, and no per-turn count moves.
 *
 * <p>Every check compares the board-only snapshot a record would carry,
 * before and after the edit, member by member: {@link BoardChanges#changes} names each
 * entity, player or global member that differs, each entity that appeared or
 * went, and each link that appeared or went. The edits run on a fresh
 * {@code TestBoard} game, which is what a fork is to its edit.
 *
 * <p>A trigger is caught where Forge first records one: run, it lands among
 * the stack's simultaneous entries. Each silence test has a twin making the
 * same change the way the game does, which fires: the board can tell.
 */
@ExtendWith(ForgeExtension.class)
class ForkOperationsTest {

    // ── one fact each (T086) ────────────────────────────────────────────

    @Test
    void ptSetsTheBaseToughnessAlone() {
        Board board = Board.main();
        Card bears = board.creature(board.you, "Grizzly Bears");
        assertEquals(Set.of(id(bears) + ".base_pt"),
                board.edit(new PtOperation(bears, null, 3)));
        assertEquals(3, bears.getNetToughness());
        assertEquals(2, bears.getNetPower());
    }

    @Test
    void keywordAddedChangesTheLinesAlone() {
        Board board = Board.main();
        Card bears = board.creature(board.you, "Grizzly Bears");
        // A keyword no static grants reads as granted until the turn ends:
        // the line and the temporary grant name the one keyword.
        assertEquals(Set.of(id(bears) + ".lines", id(bears) + ".granted_temporary"),
                board.edit(KeywordOperation.adding(bears, "Flying")));
        assertTrue(bears.hasKeyword(Keyword.FLYING));
    }

    @Test
    void keywordRemovedByTheCorpusSpelling() {
        Board board = Board.main();
        Card angel = board.creature(board.you, "Serra Angel");
        assertEquals(Set.of(id(angel) + ".lines", id(angel) + ".granted_temporary"),
                board.edit(KeywordOperation.removing(angel, "flying")));
        assertFalse(angel.hasKeyword(Keyword.FLYING));
        assertTrue(angel.hasKeyword(Keyword.VIGILANCE));
    }

    @Test
    void removingAKeywordTheCardLacksIsRefused() {
        Board board = Board.main();
        Card bears = board.creature(board.you, "Grizzly Bears");
        assertThrows(com.pricepredictor.connector.effects.ForkRefusal.class,
                () -> KeywordOperation.removing(bears, "flying").apply(board.game));
    }

    @Test
    void colourSetChangesTheColoursAlone() {
        Board board = Board.main();
        Card bears = board.creature(board.you, "Grizzly Bears");
        assertEquals(Set.of(id(bears) + ".colors"),
                board.edit(ColourTypeOperation.colours(bears, ColorSet.fromNames("blue"))));
        assertTrue(bears.getColor().hasBlue());
        assertFalse(bears.getColor().hasGreen());
    }

    @Test
    void typeAddedChangesTheTypesAlone() {
        Board board = Board.main();
        Card bears = board.creature(board.you, "Grizzly Bears");
        assertEquals(Set.of(id(bears) + ".types"),
                board.edit(ColourTypeOperation.addingType(bears, "Artifact")));
        assertTrue(bears.isArtifact());
        assertTrue(bears.isCreature());
    }

    @Test
    void countersOnACardChangeItsCountersAndTheBoostTheyGive() {
        Board board = Board.main();
        Card bears = board.creature(board.you, "Grizzly Bears");
        assertEquals(Set.of(id(bears) + ".counters", id(bears) + ".pt_boosts"),
                board.edit(new CountersOperation(bears, CounterEnumType.P1P1, 2)));
        assertEquals(4, bears.getNetPower());
    }

    @Test
    void countersOnAPlayerChangeThatCountAlone() {
        Board board = Board.main();
        assertEquals(Set.of("P1.poison"),
                board.edit(new CountersOperation(board.opponent, CounterEnumType.POISON, 3)));
    }

    @Test
    void tapChangesTheTappedFlagAlone() {
        Board board = Board.main();
        Card bears = board.creature(board.you, "Grizzly Bears");
        assertEquals(Set.of(id(bears) + ".tapped"), board.edit(new TapOperation(bears, true)));
        assertTrue(bears.isTapped());
    }

    @Test
    void lifeChangesTheTotalAloneAndFiresNoLifeGainTrigger() {
        Board board = Board.main();
        board.creature(board.opponent, "Ajani's Pridemate");
        assertEquals(Set.of("P1.life"), board.edit(new LifeOperation(board.opponent, 25)));
        assertEquals(25, board.opponent.getLife());
        assertFalse(board.triggered(), "Ajani's Pridemate triggered on a silent life edit");
        assertEquals(0, board.opponent.getLifeGainedThisTurn());
    }

    @Test
    void theLifeBoardCanTellAGainThatFires() {
        Board board = Board.main();
        board.creature(board.opponent, "Ajani's Pridemate");
        board.opponent.gainLife(5, null, null);
        assertTrue(board.triggered());
    }

    @Test
    void replacingABlockerCarriesItsStateAndChangesTheCardAlone() {
        Board board = Board.blocked();
        board.bears.setDamage(1);
        board.bears.setCounters(CounterEnumType.P1P1, 1);
        Map<String, Object> before = BoardStates.of(board.game);

        new ReplaceCardOperation(board.bears, "Centaur Courser").apply(board.game);

        Card courser = board.opponent.getCardsIn(ZoneType.Battlefield).stream()
                .filter(c -> c.getName().equals("Centaur Courser")).findFirst().orElseThrow();
        Map<String, Object> after = BoardStates.of(board.game);
        // The before board with the courser's id wherever the bears' stood:
        // what is left differs only in what makes the card that card.
        Set<String> left = changes(renamed(before, id(board.bears), id(courser)), after);
        Set<String> identity = Set.of("name", "colors", "subtypes", "mana_value", "mana_cost",
                "base_pt", "lines");
        assertTrue(left.contains(id(courser) + ".name"), "the card itself changed: " + left);
        for (String change : left) {
            assertTrue(change.startsWith(id(courser) + ".")
                            && identity.contains(change.substring(change.indexOf('.') + 1)),
                    "replacing changed " + change + "; all: " + left);
        }
        assertNull(board.bears.getZone());
        assertTrue(board.game.getCombat().isBlocking(courser, board.giant));
        assertEquals(courser, board.holy.getEntityAttachedTo());
        assertEquals(1, courser.getDamage());
        assertFalse(board.triggered());
    }

    // ── board membership, silently (T086a) ──────────────────────────────

    @Test
    void removingABlockerWithAnAuraTakesItAloneAndFiresNothing() {
        Board board = Board.blocked();
        Map<String, Object> before = BoardStates.of(board.game);
        String bears = id(board.bears);

        Set<String> changed = board.edit(new RemovePermanentOperation(board.bears));

        // The Aura's static affects nothing now (its line's affected list),
        // and each later permanent moves up one place in timestamp order.
        Set<String> expected = new TreeSet<>(Set.of(
                "-" + bears, id(board.holy) + ".attached_to", id(board.holy) + ".lines",
                id(board.giant) + ".combat"));
        Object rank = entity(before, bears).get("order_rank");
        for (Map.Entry<String, Map<String, Object>> other : indexed(before, "entities").entrySet()) {
            if ((Long) other.getValue().get("order_rank") > (Long) rank) {
                expected.add(other.getKey() + ".order_rank");
            }
        }
        for (String link : links(before)) {
            if (link.contains("=" + bears + ",") || link.contains("=" + bears + "}")) {
                expected.add("-link " + link);
            }
        }
        assertEquals(expected, new TreeSet<>(changed));
        Map<String, Object> holy = entity(BoardStates.of(board.game), id(board.holy));
        assertNull(holy.get("attached_to"));
        assertFalse(board.game.getCombat().isBlocking(board.bears));
        assertTrue(board.game.getCombat().getBlockers(board.giant).isEmpty());
        assertNull(board.bears.getZone());
        assertFalse(board.triggered(), "a dies or leaves trigger fired on a silent removal");
    }

    @Test
    void theRemovalBoardCanTellADeathThatFires() {
        Board board = Board.blocked();
        board.game.getAction().moveToGraveyard(board.bears, null, null);
        assertTrue(board.triggered());
    }

    @Test
    void addingAnElvishVisionaryDrawsNothingAndArrivesAsAsked() {
        Board board = Board.main();
        board.library(board.you, "Forest", "Island");
        Map<String, Object> before = BoardStates.of(board.game);

        new AddPermanentOperation("Elvish Visionary", board.you, board.you, true, true,
                Map.of(CounterEnumType.P1P1, 1)).apply(board.game);

        Card visionary = board.you.getCardsIn(ZoneType.Battlefield).stream()
                .filter(c -> c.getName().equals("Elvish Visionary")).findFirst().orElseThrow();
        Map<String, Object> after = BoardStates.of(board.game);
        String added = id(visionary);
        for (String change : changes(before, after)) {
            assertTrue(change.equals("+" + added)
                            || (change.startsWith("+link ") && change.contains("=" + added)),
                    "adding changed " + change);
        }
        Map<String, Object> entity = entity(after, added);
        assertEquals(true, entity.get("tapped"));
        assertEquals(true, entity.get("sick"));
        assertEquals(List.of(1L), new ArrayList<>(((Map<?, ?>) entity.get("counters")).values()));
        assertEquals(0, board.you.getCardsIn(ZoneType.Hand).size());
        assertFalse(board.triggered(), "Elvish Visionary's enters trigger fired on a silent add");
    }

    @Test
    void addingUnderAnotherControllerPutsItOnTheirBattlefield() {
        Board board = Board.main();
        new AddPermanentOperation("Grizzly Bears", board.you, board.opponent, false, false,
                Map.of()).apply(board.game);
        Card bears = board.opponent.getCardsIn(ZoneType.Battlefield).get(0);
        assertEquals(board.you, bears.getOwner());
        assertEquals(board.opponent, bears.getController());
        assertFalse(bears.isSick());
        assertEquals(Boolean.FALSE, ((Map<?, ?>) entity(BoardStates.of(board.game), id(bears)).get("history"))
                        .get("entered_this_turn"),
                "a permanent that is not summoning-sick has been here since an earlier turn");
    }

    // ── on a real copy, through the first state-based check ─────────────

    /**
     * A permanent the edit adds and the first state-based check takes away
     * (a 0/0 Walking Ballista) is seen dying: Blood Artist triggers. The
     * copy's last-known battlefield was taken before the edit, so without
     * the edit taking it anew the death looked back at nothing.
     */
    @Test
    void anAddedPermanentTheFirstCheckKillsIsSeenDying() {
        Game live = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player you = live.getPlayers().get(0);
        TestBoard.inPlay(live, you, "Blood Artist");
        live.getTriggerHandler().resetActiveTriggers(false, null);
        live.copyLastState();
        GameCopier copier = new GameCopier(live);
        Game fork = copier.makeCopy();

        AddPermanentOperation.of("Walking Ballista", copier.find(you), false).apply(fork);
        fork.getAction().checkStateEffects(true);
        fork.getTriggerHandler().runWaitingTriggers();

        assertTrue(copier.find(you).getCardsIn(ZoneType.Battlefield).stream()
                .noneMatch(card -> card.getName().equals("Walking Ballista")), "the 0/0 did not die");
        assertTrue(fork.getStack().hasSimultaneousStackEntries() || !fork.getStack().isEmpty(),
                "Blood Artist did not see the Ballista die");
    }

    /**
     * A replacement is the same permanent in every respect but the card: a
     * resolved pump and a keyword granted until end of turn, its soulbond
     * pair and what it attacked this turn carry over; an anthem's rows are
     * the anthem's to apply anew, and a noncreature it no longer applies to
     * keeps none of them.
     */
    @Test
    void aReplacementCarriesThePermanentsEffectsPairAndHistory() {
        Board board = Board.main();
        Card anthem = board.permanent(board.you, "Glorious Anthem");
        Card bears = board.creature(board.you, "Grizzly Bears");
        Card partner = board.creature(board.you, "Hill Giant");
        bears.addPTBoost(3, 3, board.game.getNextTimestamp(), 0);
        bears.addChangedCardKeywords(List.of("Flying"), null, false, board.game.getNextTimestamp(), null);
        bears.setPairedWith(partner);
        partner.setPairedWith(bears);
        bears.getDamageHistory().setCreatureAttackedThisCombat(board.opponent, 0);
        board.game.getAction().checkStaticAbilities();
        assertEquals(2 + 1 + 3, bears.getNetPower());

        board.edit(new ReplaceCardOperation(bears, "Hill Giant"));
        Card giant = board.you.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card != partner && card.getName().equals("Hill Giant")).findFirst().orElseThrow();

        assertEquals(3 + 1 + 3, giant.getNetPower(), "the pump and the anthem, once each");
        assertTrue(giant.hasKeyword("Flying"));
        assertEquals(partner, giant.getPairedWith());
        assertEquals(giant, partner.getPairedWith());
        assertTrue(giant.getDamageHistory().hasAttackedThisTurn(board.opponent));

        board.edit(new ReplaceCardOperation(giant, "Sol Ring"));
        Card ring = board.you.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals("Sol Ring")).findFirst().orElseThrow();
        long anthemRows = ring.getPTBoostTable().cellSet().stream()
                .filter(cell -> cell.getColumnKey() != 0L).count();
        assertEquals(0, anthemRows, "the anthem's row stayed on a noncreature: " + ring.getPTBoostTable()
                + " (anthem " + anthem + ")");
    }

    /**
     * A keyword the card cannot have (a can't-have effect) is refused rather
     * than silently not added, so a fork tagged with the edit is discarded.
     */
    @Test
    void aKeywordTheCardCannotHaveIsRefused() {
        Board board = Board.main();
        Card bears = board.creature(board.you, "Grizzly Bears");
        bears.addCantHaveKeyword(board.game.getNextTimestamp(), 0L,
                List.of(forge.game.keyword.Keyword.FLYING));
        bears.updateKeywordsCache();

        assertThrows(com.pricepredictor.connector.effects.ForkRefusal.class,
                () -> KeywordEdit.add(board.game, bears, "Flying"));
    }

    @Test
    void theAddBoardCanTellAnEnteringThatFires() {
        Board board = Board.main();
        board.library(board.you, "Forest", "Island");
        Card visionary = TestBoard.inHand(board.game, board.you, "Elvish Visionary");
        board.settled();
        board.game.getAction().moveToPlay(visionary, board.you, null, null);
        assertTrue(board.triggered());
    }

    @Test
    void movingFromLibraryToHandDrawsNothingAndKeepsTheLibraryOrder() {
        Board board = Board.main();
        List<Card> library = board.library(board.you, "Forest", "Island", "Mountain");
        Card island = library.get(1);
        Map<String, Object> before = BoardStates.of(board.game);

        Set<String> changed = board.edit(MoveCardOperation.move(island, ZoneType.Hand, null));

        for (String change : changed) {
            assertTrue(Set.of("P0.hand", "P0.library", "+" + id(island)).contains(change)
                            || (change.startsWith("+link ") && change.contains("=" + id(island))),
                    "moving changed " + change + "; all: " + changed);
        }
        assertTrue(changed.containsAll(Set.of("P0.hand", "P0.library", "+" + id(island))));
        assertEquals(List.of(library.get(0), library.get(2)),
                List.copyOf(board.you.getCardsIn(ZoneType.Library)));
        assertEquals(0, board.you.getNumDrawnThisTurn());
        assertEquals(before.get("global"), BoardStates.of(board.game).get("global"));
        assertFalse(board.triggered());
    }

    @Test
    void movingFromHandToGraveyardFiresNoDiscardTriggerAndNoDescent() {
        Board board = Board.main();
        board.permanent(board.opponent, "Megrim");
        Card bears = TestBoard.inHand(board.game, board.you, "Grizzly Bears");
        board.settled();

        Set<String> changed = board.edit(MoveCardOperation.move(bears, ZoneType.Graveyard, null));

        for (String change : changed) {
            assertTrue(Set.of("P0.hand", "P0.graveyard").contains(change)
                            || change.startsWith(id(bears) + "."),
                    "moving changed " + change + "; all: " + changed);
        }
        assertEquals(Set.of(id(bears) + ".zone"), only(changed, id(bears) + "."));
        assertEquals(0, board.you.getDescended());
        assertFalse(board.triggered(), "Megrim triggered on a silent move to the graveyard");
    }

    @Test
    void theDiscardBoardCanTellADiscardThatFires() {
        Board board = Board.main();
        board.permanent(board.opponent, "Megrim");
        Card bears = TestBoard.inHand(board.game, board.you, "Grizzly Bears");
        board.settled();
        board.you.discard(bears, null, false, AbilityKey.newMap());
        assertTrue(board.triggered());
    }

    @Test
    void replacingTheTopLibraryCardPutsTheNewCardOnTop() {
        Board board = Board.main();
        List<Card> library = board.library(board.you, "Forest", "Island");
        Card forest = library.get(0);
        Map<String, Object> before = BoardStates.of(board.game);

        Set<String> changed = board.edit(MoveCardOperation.replace(forest, "Lightning Bolt"));

        Card top = board.you.getCardsIn(ZoneType.Library).get(0);
        assertEquals("Lightning Bolt", top.getName());
        assertEquals(List.of(top, library.get(1)), List.copyOf(board.you.getCardsIn(ZoneType.Library)));
        assertNull(forest.getZone());
        Map<String, Object> entity = entity(BoardStates.of(board.game), id(top));
        assertEquals("library", entity.get("zone"));
        assertEquals(0L, entity.get("order_rank"));
        for (String change : changed) {
            assertTrue(change.equals("-" + id(forest)) || change.equals("+" + id(top))
                            || change.startsWith("-link ") && change.contains("=" + id(forest))
                            || change.startsWith("+link ") && change.contains("=" + id(top)),
                    "replacing changed " + change + "; all: " + changed);
        }
        assertEquals(entity(before, id(forest)).get("order_rank"), entity.get("order_rank"));
        assertFalse(board.triggered());
    }

    @Test
    void aPlaceBeyondTheZoneFails() {
        Board board = Board.main();
        Card island = board.library(board.you, "Island").get(0);
        Card bears = TestBoard.inHand(board.game, board.you, "Grizzly Bears");
        assertThrows(IllegalStateException.class,
                () -> MoveCardOperation.move(bears, ZoneType.Library, 3).apply(board.game));
        assertEquals(List.of(island), List.copyOf(board.you.getCardsIn(ZoneType.Library)));
    }

    // ── the board ───────────────────────────────────────────────────────

    /**
     * A started two-seat game: you (P0) and the opponent (P1), with whatever
     * each test puts on it, settled so no trigger is waiting before the edit.
     */
    // ── replace_card: what names the old card, and what it refuses ─────

    /** The Hill Giant that replaced {@code old} on {@code owner}'s battlefield. */
    private static Card replacementOf(Board board, Player owner, Card... others) {
        List<Card> skip = List.of(others);
        return owner.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals("Hill Giant") && !skip.contains(card))
                .findFirst().orElseThrow();
    }

    /**
     * Giant Growth on a creature, then replace_card: the boost the new card
     * took over ends on it at cleanup, the until-end-of-turn command having
     * been rewritten to name it.
     */
    @Test
    void aPumpTheReplacementTookOverEndsAtCleanup() {
        Board board = Board.main();
        Card bears = board.creature(board.you, "Grizzly Bears");
        SpellAbility growth = TestBoard.inHand(board.game, board.you, "Giant Growth").getFirstSpellAbility();
        growth.setActivatingPlayer(board.you);
        growth.getTargets().add(bears);
        board.game.getStack().addAndUnfreeze(growth);
        board.game.getStack().resolveStack();
        assertEquals(5, bears.getNetPower());

        board.edit(new ReplaceCardOperation(bears, "Hill Giant"));
        Card giant = replacementOf(board, board.you);
        assertEquals(6, giant.getNetPower(), "the pump carried");
        board.game.getPhaseHandler().devModeSet(PhaseType.MAIN2, board.you, board.game.getPhaseHandler().getTurn());
        board.game.getPhaseHandler().devAdvanceToPhase(PhaseType.CLEANUP);

        assertEquals(3, giant.getNetPower(), "the pump ended at cleanup");
    }

    /** A delayed trigger and a card remembering the old card remember the new one. */
    @Test
    void whatRemembersTheReplacedCardRemembersTheReplacement() {
        Board board = Board.main();
        Card bears = board.creature(board.you, "Grizzly Bears");
        Card holder = board.creature(board.opponent, "Gray Ogre");
        holder.addRemembered(bears);
        forge.game.trigger.Trigger delayed = forge.game.trigger.TriggerHandler.parseTrigger(
                "Mode$ Phase | Phase$ End of Turn | TriggerDescription$ test", bears, false);
        delayed.addRemembered(bears);
        board.game.getTriggerHandler().registerDelayedTrigger(delayed);

        board.edit(new ReplaceCardOperation(bears, "Hill Giant"));
        Card giant = replacementOf(board, board.you);

        assertTrue(holder.isRemembered(giant) && !holder.isRemembered(bears), holder.getRemembered().toString());
        assertSame(giant, delayed.getHostCard());
        assertEquals(List.of(giant), delayed.getTriggerRemembered());
    }

    /** Goad, monstrous and renowned are the permanent's designations: carried. */
    @Test
    void aReplacementCarriesGoadMonstrousAndRenowned() {
        Board board = Board.main();
        Card bears = board.creature(board.you, "Grizzly Bears");
        bears.addGoad(board.game.getNextTimestamp(), board.opponent);
        bears.setMonstrous(true);
        bears.setRenowned(true);

        board.edit(new ReplaceCardOperation(bears, "Hill Giant"));
        Card giant = replacementOf(board, board.you);

        assertTrue(giant.isGoadedBy(board.opponent));
        assertTrue(giant.isMonstrous());
        assertTrue(giant.isRenowned());
    }

    private static void assertRefused(Board board, Card old, String reason) {
        ForkRefusal refusal = assertThrows(ForkRefusal.class,
                () -> new ReplaceCardOperation(old, "Hill Giant").apply(board.game));
        assertEquals(reason, refusal.reason());
        assertTrue(old.getZone() != null && old.getZone().is(ZoneType.Battlefield), "the refusal changed the board");
    }

    @Test
    void aFaceDownPermanentIsRefused() {
        Board board = Board.main();
        Card bears = TestBoard.card(board.game, board.you, "Grizzly Bears");
        bears.turnFaceDownNoUpdate();
        board.you.getZone(ZoneType.Battlefield).add(bears);
        assertRefused(board, bears, ReplaceCardOperation.FACE_DOWN);
    }

    @Test
    void aClassAboveItsFirstLevelIsRefused() {
        Board board = Board.main();
        Card ranger = board.permanent(board.you, "Ranger Class");
        ranger.setClassLevel(2);
        assertRefused(board, ranger, ReplaceCardOperation.CLASS_LEVEL);
    }

    @Test
    void aPermanentHoldingAChosenValueIsRefused() {
        Board board = Board.main();
        Card bears = board.creature(board.you, "Grizzly Bears");
        bears.setChosenType("Elf");
        assertRefused(board, bears, ReplaceCardOperation.CHOSEN_VALUES);
    }

    @Test
    void aPermanentRememberingACardIsRefused() {
        Board board = Board.main();
        Card bears = board.creature(board.you, "Grizzly Bears");
        bears.addRemembered(board.creature(board.opponent, "Gray Ogre"));
        assertRefused(board, bears, ReplaceCardOperation.LINKED_CARDS);
    }

    @Test
    void aPermanentPhasedOutIsRefused() {
        Board board = Board.main();
        Card bears = board.creature(board.you, "Grizzly Bears");
        bears.setPhasedOut(board.you);
        assertRefused(board, bears, ReplaceCardOperation.PHASED_OUT);
    }

    /** A pending command that cannot be read (it holds an object it cannot write) leaves the edit refused. */
    @Test
    void aCommandThatCannotBeRewrittenRefusesTheEdit() {
        Board board = Board.main();
        Card bears = board.creature(board.you, "Grizzly Bears");
        Object unwritable = new Object();
        board.game.getEndOfTurn().addUntil(() -> unwritable.hashCode());
        assertRefused(board, bears, ReplaceCardOperation.UNREWRITABLE_COMMAND);
    }

    /**
     * C4 #1: the damage the replaced creature carries is lethal to a smaller
     * card, which the fork's first check would put into a graveyard; the edit
     * refuses rather than fork that board. Hill Giant with 2 damage becomes
     * Grizzly Bears with 2 damage.
     */
    @Test
    void aReplacementTheCarriedDamageKillsIsRefused() {
        Board board = Board.main();
        Card giant = board.creature(board.you, "Hill Giant");
        giant.setDamage(2);
        assertTrue(ReplaceCardOperation.replacementDies(giant, "Grizzly Bears"), "the preconditions' reading");
        ForkRefusal refusal = assertThrows(ForkRefusal.class,
                () -> new ReplaceCardOperation(giant, "Grizzly Bears").apply(board.game));
        assertEquals(ReplaceCardOperation.DIES, refusal.reason());
    }

    @Test
    void aReplacementTheCarriedCountersKillIsRefused() {
        Board board = Board.main();
        Card giant = board.creature(board.you, "Hill Giant");
        giant.setCounters(CounterEnumType.M1M1, 2);
        assertTrue(ReplaceCardOperation.replacementDies(giant, "Grizzly Bears"), "the preconditions' reading");
        ForkRefusal refusal = assertThrows(ForkRefusal.class,
                () -> new ReplaceCardOperation(giant, "Grizzly Bears").apply(board.game));
        assertEquals(ReplaceCardOperation.DIES, refusal.reason());
    }

    /** A resolved -2/-2 until end of turn leaves Hill Giant a 1/1 and would leave Grizzly Bears a 0/0. */
    @Test
    void aReplacementTheCarriedBoostKillsIsRefused() {
        Board board = Board.main();
        Card giant = board.creature(board.you, "Hill Giant");
        giant.addPTBoost(-2, -2, board.game.getNextTimestamp(), 0);
        assertTrue(ReplaceCardOperation.replacementDies(giant, "Grizzly Bears"), "the preconditions' reading");
        assertFalse(ReplaceCardOperation.replacementDies(giant, "Craw Wurm"), "a 6/4 lives at 2 toughness");
        ForkRefusal refusal = assertThrows(ForkRefusal.class,
                () -> new ReplaceCardOperation(giant, "Grizzly Bears").apply(board.game));
        assertEquals(ReplaceCardOperation.DIES, refusal.reason());
    }

    @Test
    void aReplacementThatSurvivesTheCarriedDamageGoesThrough() {
        Board board = Board.main();
        Card bears = board.creature(board.you, "Grizzly Bears");
        bears.setDamage(1);
        assertFalse(ReplaceCardOperation.replacementDies(bears, "Hill Giant"), "the preconditions' reading");
        new ReplaceCardOperation(bears, "Hill Giant").apply(board.game);
        assertTrue(board.you.getCardsIn(ZoneType.Battlefield).stream()
                .anyMatch(card -> card.getName().equals("Hill Giant") && card.getDamage() == 1));
    }

    private static final class Board {
        final Game game;
        final Player you;
        final Player opponent;
        Card giant;
        Card bears;
        Card holy;

        private Board(PhaseType phase) {
            game = TestBoard.startedTwoPlayerGame(phase);
            you = game.getPlayers().get(0);
            opponent = game.getPlayers().get(1);
        }

        static Board main() {
            return new Board(PhaseType.MAIN1);
        }

        /**
         * Your Hill Giant attacks; the opponent's Grizzly Bears, enchanted by
         * the opponent's Holy Strength, blocks it; a Doomed Dissenter and a
         * Blood Artist watch.
         */
        static Board blocked() {
            Board board = new Board(PhaseType.COMBAT_DECLARE_BLOCKERS);
            board.giant = board.creature(board.you, "Hill Giant");
            board.bears = board.creature(board.opponent, "Grizzly Bears");
            board.holy = board.permanent(board.opponent, "Holy Strength");
            board.holy.attachToEntity(board.bears, null);
            board.creature(board.opponent, "Doomed Dissenter");
            board.creature(board.opponent, "Blood Artist");
            TestBoard.blockedCombat(board.game, board.you, board.opponent, board.giant, board.bears);
            board.settled();
            return board;
        }

        /**
         * A permanent since the turn began, with a timestamp of its own as
         * one entering in play has (a test board's are all unset).
         */
        Card creature(Player owner, String name) {
            Card card = TestBoard.inPlay(game, owner, name);
            card.setGameTimestamp(game.getNextTimestamp());
            settled();
            return card;
        }

        Card permanent(Player owner, String name) {
            return creature(owner, name);
        }

        /** Cards put on the bottom in turn, so the list reads top first. */
        List<Card> library(Player owner, String... names) {
            List<Card> cards = new ArrayList<>();
            for (String name : names) {
                cards.add(game.getAction().moveToLibrary(
                        TestBoard.card(game, owner, name), -1, null, null));
            }
            settled();
            return cards;
        }

        /**
         * Nothing waiting, nothing on the stack, and the last-state battlefield
         * current, so a dies trigger can look back at it.
         */
        void settled() {
            game.getTriggerHandler().runWaitingTriggers();
            game.getStack().clearSimultaneousStack();
            game.getTriggerHandler().resetActiveTriggers(false, null);
            game.copyLastState();
        }

        /** Whether a trigger has fired since the board was settled. */
        boolean triggered() {
            game.getTriggerHandler().runWaitingTriggers();
            return game.getStack().hasSimultaneousStackEntries() || !game.getStack().isEmpty();
        }

        /** Apply {@code edit} and name what changed on the board. */
        Set<String> edit(BoardEdit edit) {
            Map<String, Object> before = BoardStates.of(game);
            edit.apply(game);
            return changes(before, BoardStates.of(game));
        }
    }

    // ── comparing snapshots ─────────────────────────────────────────────

    /** The changes of one entity or player, by their {@code prefix}. */
    private static Set<String> only(Set<String> changed, String prefix) {
        Set<String> kept = new TreeSet<>();
        for (String change : changed) {
            if (change.startsWith(prefix)) {
                kept.add(change);
            }
        }
        return kept;
    }
}
