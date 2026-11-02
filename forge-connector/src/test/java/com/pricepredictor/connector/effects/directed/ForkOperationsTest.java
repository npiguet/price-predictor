package com.pricepredictor.connector.effects.directed;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.BoardStates;
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
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Each fork operation changes exactly its one fact (T086), and the three
 * board-membership operations change it silently (T086a, story 3 acceptance
 * 6): no trigger fires from the edit, and no per-turn count moves.
 *
 * <p>Every check compares the board-only snapshot a record would carry,
 * before and after the edit, member by member: {@link #changes} names each
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
    void removingAKeywordTheCardLacksFails() {
        Board board = Board.main();
        Card bears = board.creature(board.you, "Grizzly Bears");
        assertThrows(IllegalStateException.class,
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

    private static String id(Card card) {
        return "E" + card.getId();
    }

    /**
     * What differs between two snapshots: {@code E5.tapped} for an entity
     * member, {@code P1.life} for a player member, {@code global.turn} for a
     * global one, {@code +E9} / {@code -E5} for an entity that appeared or
     * went, and {@code +link …} / {@code -link …} for a link.
     */
    static Set<String> changes(Map<String, Object> before, Map<String, Object> after) {
        Set<String> changed = new TreeSet<>();
        members("global", asMap(before.get("global")), asMap(after.get("global")), changed);
        byId("players", before, after, changed);
        byId("entities", before, after, changed);
        Set<String> was = links(before);
        Set<String> is = links(after);
        for (String link : was) {
            if (!is.contains(link)) {
                changed.add("-link " + link);
            }
        }
        for (String link : is) {
            if (!was.contains(link)) {
                changed.add("+link " + link);
            }
        }
        return changed;
    }

    private static void byId(String block, Map<String, Object> before, Map<String, Object> after,
                             Set<String> changed) {
        Map<String, Map<String, Object>> was = indexed(before, block);
        Map<String, Map<String, Object>> is = indexed(after, block);
        for (Map.Entry<String, Map<String, Object>> entry : was.entrySet()) {
            Map<String, Object> now = is.get(entry.getKey());
            if (now == null) {
                changed.add("-" + entry.getKey());
            } else {
                members(entry.getKey(), entry.getValue(), now, changed);
            }
        }
        for (String id : is.keySet()) {
            if (!was.containsKey(id)) {
                changed.add("+" + id);
            }
        }
    }

    private static void members(String owner, Map<String, Object> was, Map<String, Object> is,
                                Set<String> changed) {
        Set<String> keys = new HashSet<>(was.keySet());
        keys.addAll(is.keySet());
        for (String key : keys) {
            if (!java.util.Objects.equals(was.get(key), is.get(key))) {
                changed.add(owner + "." + key);
            }
        }
    }

    private static Map<String, Map<String, Object>> indexed(Map<String, Object> state, String block) {
        Map<String, Map<String, Object>> byId = new LinkedHashMap<>();
        for (Object item : (List<?>) state.get(block)) {
            Map<String, Object> map = asMap(item);
            byId.put((String) map.get("id"), map);
        }
        return byId;
    }

    /** Each link as one string, its members in name order: {@code {from=P0, to=E3, type=controls}}. */
    private static Set<String> links(Map<String, Object> state) {
        Set<String> links = new TreeSet<>();
        for (Object link : (List<?>) state.get("links")) {
            links.add(new TreeMap<>(asMap(link)).toString());
        }
        return links;
    }

    private static Map<String, Object> entity(Map<String, Object> state, String id) {
        Map<String, Object> entity = indexed(state, "entities").get(id);
        assertTrue(entity != null, "no entity " + id);
        return entity;
    }

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

    /** {@code value} with every string equal to {@code from} read as {@code to}. */
    @SuppressWarnings("unchecked")
    private static <T> T renamed(T value, String from, String to) {
        if (value instanceof Map<?, ?> map) {
            Map<String, Object> copy = new LinkedHashMap<>();
            map.forEach((key, entry) -> copy.put((String) key, renamed(entry, from, to)));
            return (T) copy;
        }
        if (value instanceof List<?> list) {
            return (T) list.stream().map(entry -> renamed(entry, from, to)).toList();
        }
        if (from.equals(value)) {
            return (T) to;
        }
        return value;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> asMap(Object value) {
        return (Map<String, Object>) value;
    }
}
