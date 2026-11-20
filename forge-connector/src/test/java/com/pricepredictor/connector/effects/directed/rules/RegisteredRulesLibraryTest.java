package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.TestBoard;
import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;
import java.util.Set;

import static com.pricepredictor.connector.effects.directed.BoardChanges.changes;
import static com.pricepredictor.connector.effects.directed.BoardChanges.id;
import static com.pricepredictor.connector.effects.directed.rules.RuleTwinChecks.assertOnlyAdded;
import static com.pricepredictor.connector.effects.directed.rules.RuleTwinChecks.assertOnlyChanged;
import static com.pricepredictor.connector.effects.directed.rules.RuleTwinChecks.assertOnlySwapped;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The registered directed rules of the library: searches (118.12b,
 * 701.23f, 702.53a transmute, 702.71a transfigure, 702.124j partner
 * with), a shuffle of one card (701.24e), and exploring, conniving and
 * discovering against a library or a draw that gives nothing (701.44b,
 * 701.50f, 701.57b). Each rule's scenario base binds, a board missing it
 * binds nothing, and the fork differs from its twin by the variant's one
 * fact, as {@link RegisteredRulesTest} sets out.
 */
@ExtendWith(ForgeExtension.class)
class RegisteredRulesLibraryTest {

    // ── 118.12b, an optional search: add_permanent ──────────────────────

    /** A's upkeep begins over A's library, listed top card first. */
    private static RuleBoard upkeepBoard(String... library) {
        RuleBoard board = new RuleBoard(PhaseType.UPKEEP);
        board.library(board.you, library);
        return board;
    }

    @Test
    void optionalSearchBindsTheActivePlayerWithABasicLandInItsLibrary() {
        RuleBoard board = upkeepBoard("Shock", "Forest", "Forest");
        assertEquals(board.you, OptionalSearch.bind(board.game).orElseThrow().player("player"));
    }

    @Test
    void optionalSearchBindsNothingWithoutABasicLandToFind() {
        assertTrue(OptionalSearch.bind(upkeepBoard("Shock", "Lightning Bolt").game).isEmpty());
    }

    /** Elfhame Sanctuary finds a basic land card: a nonbasic land in the library is none. */
    @Test
    void optionalSearchBindsNothingWithOnlyANonbasicLandToFind() {
        assertTrue(OptionalSearch.bind(upkeepBoard("Shock", "Evolving Wilds").game).isEmpty());
    }

    @Test
    void optionalSearchBindsNothingAtAnotherBoundary() {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
        board.library(board.you, "Forest");
        assertTrue(OptionalSearch.bind(board.game).isEmpty());
    }

    /** The added Sanctuary is bound on the copy for the forced answer, which no live card maps to. */
    @Test
    void optionalSearchForksElfhameSanctuaryAndBindsItForTheAnswer() {
        RuleBoard board = upkeepBoard("Shock", "Forest");
        RuleTwin twin = board.twin(OptionalSearch.rule());
        assertOnlyAdded(twin, twin.fork().player("player"), OptionalSearch.SANCTUARY);
        Application application = OptionalSearch.rule().variants().get(OptionalSearch.SANCTUARY_ADDED);
        Card sanctuary = application.afterEdit(twin.fork(), twin.forkGame()).card("sanctuary");
        assertEquals(OptionalSearch.SANCTUARY, sanctuary.getName());
        assertTrue(sanctuary.isInPlay());
    }

    // ── 701.23f, a search limited to the top cards: add_permanent ───────

    /** A's Archivist of Oghma and B's untapped {@code permanent}, as A's first main phase begins. */
    private static RuleBoard archivistBoard(String watcher, String permanent) {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
        board.other = board.card(board.you, watcher);
        board.subject = board.card(board.opponent, permanent);
        board.library(board.opponent, "Grizzly Bears", "Forest");
        return board;
    }

    @Test
    void searchUnderMindcensorBindsTheWatcherTheSearcherAndTheFetch() {
        RuleBoard board = archivistBoard("Archivist of Oghma", "Evolving Wilds");
        Binding binding = SearchUnderMindcensor.bind(board.game).orElseThrow();
        assertEquals(board.you, binding.player("watcher"));
        assertEquals(board.opponent, binding.player("searcher"));
        assertEquals(board.subject, binding.card("permanent"));
    }

    @Test
    void searchUnderMindcensorBindsNothingWithoutASearchToWatch() {
        assertTrue(SearchUnderMindcensor.bind(archivistBoard("Grizzly Bears", "Evolving Wilds").game).isEmpty());
    }

    @Test
    void searchUnderMindcensorBindsNothingWithoutASearchingPermanent() {
        assertTrue(SearchUnderMindcensor.bind(archivistBoard("Archivist of Oghma", "Forest").game).isEmpty());
    }

    /**
     * Prime Speaker Vannifar's search is activated only as a sorcery: on its
     * controller's opponent's turn the forced activation could never be made.
     */
    @Test
    void searchUnderMindcensorBindsNothingForASorcerySpeedSearch() {
        RuleBoard board = archivistBoard("Archivist of Oghma", "Prime Speaker Vannifar");
        board.card(board.opponent, "Grizzly Bears");
        assertTrue(SearchUnderMindcensor.bind(board.game).isEmpty());
    }

    /** A tapped Evolving Wilds cannot pay its own tap, whatever other lands its controller has. */
    @Test
    void searchUnderMindcensorBindsNothingForATappedFetch() {
        RuleBoard board = archivistBoard("Archivist of Oghma", "Evolving Wilds");
        board.card(board.opponent, "Forest");
        board.subject.setTapped(true);
        assertTrue(SearchUnderMindcensor.bind(board.game).isEmpty());
    }

    /** Archivist watches the opponents' searches only: its controller's own fetch is no search it watches. */
    @Test
    void searchUnderMindcensorBindsNothingForTheWatchersOwnFetch() {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
        board.card(board.you, "Archivist of Oghma");
        board.card(board.you, "Evolving Wilds");
        assertTrue(SearchUnderMindcensor.bind(board.game).isEmpty());
    }

    /**
     * Besides the Mindcensor, the searcher's Keyword Effects card holds the
     * limit the Mindcensor's static gives it ({@code LimitSearchLibrary}).
     */
    @Test
    void searchUnderMindcensorForksAvenMindcensor() {
        RuleBoard board = archivistBoard("Archivist of Oghma", "Evolving Wilds");
        RuleTwin twin = board.twin(SearchUnderMindcensor.rule());
        Card mindcensor = twin.fork().player("watcher").getCardsIn(ZoneType.Battlefield).stream()
                .filter(c -> c.getName().equals(SearchUnderMindcensor.MINDCENSOR)).findFirst().orElseThrow();
        Card effects = twin.fork().player("searcher").getKeywordCard();
        Set<String> named = Set.of(id(mindcensor), id(effects));
        for (String change : twin.changes()) {
            assertTrue(change.equals("global.floating_effects")
                            || named.stream().anyMatch(id -> change.equals("+" + id) || change.contains("=" + id)),
                    "adding changed " + change + "; all: " + twin.changes());
        }
        assertTrue(twin.changes().contains("+" + id(mindcensor)));
        assertTrue(twin.fork().player("searcher").hasKeyword("LimitSearchLibrary"));
    }

    // ── 701.24e, shuffling a library of one card: move_card ─────────────

    /** The opponent's Cosi's Trickster; your Demonic Tutor over {@code library}, top card first. */
    private static RuleBoard tricksterBoard(String... library) {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
        board.card(board.opponent, "Cosi's Trickster");
        board.card(board.you, "Swamp");
        board.card(board.you, "Swamp");
        TestBoard.inHand(board.game, board.you, "Demonic Tutor");
        for (String name : library) {
            board.subject = board.game.getAction().moveToLibrary(
                    TestBoard.card(board.game, board.you, name), -1, null, null);
        }
        return board;
    }

    /** The searcher binds over a library of common cards, whatever its size: the edit empties it. */
    @Test
    void smallLibraryShuffleBindsTheSearcherOverAnyLibraryOfOneCardOrMore() {
        for (String[] library : new String[][]{{"Forest"}, {"Forest", "Grizzly Bears", "Island"}}) {
            RuleBoard board = tricksterBoard(library);
            Binding binding = SmallLibraryShuffle.bind(board.game).orElseThrow();
            assertEquals(board.you, binding.player("searcher"));
            assertEquals("Demonic Tutor", binding.card("spell").getName());
        }
    }

    /** An empty library is the rare board itself: the edit would add nothing to it. */
    @Test
    void smallLibraryShuffleBindsNothingOverAnEmptyLibrary() {
        assertTrue(SmallLibraryShuffle.bind(tricksterBoard().game).isEmpty());
    }

    @Test
    void smallLibraryShuffleForksTheLibraryEmptyAndTheGraveyardAsItWas() {
        RuleBoard board = tricksterBoard("Forest", "Grizzly Bears", "Island");
        Card top = board.you.getCardsIn(ZoneType.Library).get(0);
        RuleTwin twin = board.twin(SmallLibraryShuffle.rule());
        // The snapshot shows a library by its count and its top card alone.
        assertOnlyChanged(twin, top, top, "P0.library");
        Player searcher = twin.fork().player("searcher");
        assertTrue(searcher.getCardsIn(ZoneType.Library).isEmpty());
        assertTrue(searcher.getCardsIn(ZoneType.Graveyard).isEmpty(), "no card went to the graveyard");
    }

    // ── 701.44b, exploring with an empty library: move_card, inverted ──

    /** A's Merfolk Branchwalker in hand over two Forests, Wildgrowth Walker, A's library {@code library}. */
    private static RuleBoard branchwalkerBoard(String... library) {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
        board.cards(board.you, "Forest", 2);
        return watchedCast(board, "Wildgrowth Walker", SeveralExplore.BRANCHWALKER, library);
    }

    /**
     * {@code board} with {@code card} the subject in your hand beside your
     * {@code watcher}, your library {@code library}, top card first.
     */
    private static RuleBoard watchedCast(RuleBoard board, String watcher, String card, String... library) {
        board.card(board.you, watcher);
        board.subject = TestBoard.inHand(board.game, board.you, card);
        board.library(board.you, library);
        return board;
    }

    @Test
    void exploreEmptyLibraryBindsTheExplorerOverALibraryOfCommonCards() {
        RuleBoard board = branchwalkerBoard("Forest", "Grizzly Bears");
        Binding binding = ExploreEmptyLibrary.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("card"));
        assertEquals(board.you, binding.player("player"));
    }

    /** An empty library is the rare board itself: the edit would add nothing to it. */
    @Test
    void exploreEmptyLibraryBindsNothingOverAnEmptyLibrary() {
        assertTrue(ExploreEmptyLibrary.bind(branchwalkerBoard().game).isEmpty());
    }

    @Test
    void exploreEmptyLibraryForksTheLibraryEmpty() {
        RuleBoard board = branchwalkerBoard("Forest", "Grizzly Bears");
        Card top = board.you.getCardsIn(ZoneType.Library).get(0);
        RuleTwin twin = board.twin(ExploreEmptyLibrary.rule());
        // The snapshot shows a library by its count and its top card alone.
        assertOnlyChanged(twin, top, top, "P0.library");
        assertTrue(twin.fork().player("player").getCardsIn(ZoneType.Library).isEmpty());
    }

    // ── 701.50f, conniving when drawing is impossible: add_permanent, inverted

    /**
     * A holds Echo Inspector over four Islands and Glorious Purpose, and has
     * drawn its one card this turn (a Forest) when {@code drawn}; B's Spirit
     * of the Labyrinth, when {@code spirit}.
     */
    private static RuleBoard inspectorBoard(boolean spirit, boolean drawn) {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
        for (int i = 0; i < 4; i++) {
            board.card(board.you, "Island");
        }
        board.card(board.you, "Glorious Purpose");
        if (spirit) {
            board.card(board.opponent, ConniveWithoutDraw.SPIRIT);
        }
        board.subject = TestBoard.inHand(board.game, board.you, "Echo Inspector");
        board.library(board.you, "Forest", "Island");
        if (drawn) {
            board.you.drawCard();
        }
        return board;
    }

    @Test
    void conniveWithoutDrawBindsTheConniverOfAPlayerWhoDrewAndCanDrawAgain() {
        RuleBoard board = inspectorBoard(false, true);
        Binding binding = ConniveWithoutDraw.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("conniver"));
        assertEquals(board.you, binding.player("player"));
        assertEquals(board.opponent, binding.player("opponent"));
    }

    /** Under the Spirit the board is the rare one already: the edit would add nothing to it. */
    @Test
    void conniveWithoutDrawBindsNothingWhenThePlayerCannotDraw() {
        assertTrue(ConniveWithoutDraw.bind(inspectorBoard(true, true).game).isEmpty());
    }

    /** Before the turn's first draw the Spirit added would stop no draw of the connive. */
    @Test
    void conniveWithoutDrawBindsNothingBeforeTheTurnsFirstDraw() {
        assertTrue(ConniveWithoutDraw.bind(inspectorBoard(false, false).game).isEmpty());
    }

    @Test
    void conniveWithoutDrawForksTheSpiritOfTheLabyrinthUnderTheOpponent() {
        RuleTwin twin = inspectorBoard(false, true).twin(ConniveWithoutDraw.rule());
        assertOnlyAdded(twin, twin.fork().player("opponent"), ConniveWithoutDraw.SPIRIT);
        assertFalse(twin.fork().player("player").canDraw(), "the Spirit stops the connive's draw");
        assertTrue(twin.fork().player("opponent").getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals(ConniveWithoutDraw.SPIRIT)).allMatch(Card::isTapped),
                "the Spirit is tapped, so it blocks no attacker of the fork's turn");
    }

    // ── 701.57b, a discovery that finds nothing: move_card, inverted ────

    /** A's Geological Appraiser over four Mountains and Val, Marooned Surveyor; A's library top card first. */
    private static RuleBoard appraiserBoard(String... library) {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
        board.cards(board.you, "Mountain", 4);
        return watchedCast(board, "Val, Marooned Surveyor", "Geological Appraiser", library);
    }

    @Test
    void discoverNothingFoundBindsTheDiscovererOverALibraryWithACardToFind() {
        RuleBoard board = appraiserBoard("Forest", "Grizzly Bears", "Island");
        Binding binding = DiscoverNothingFound.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("discoverer"));
        assertEquals(board.you, binding.player("player"));
    }

    /** An empty library is the rare board itself: the edit would add nothing to it. */
    @Test
    void discoverNothingFoundBindsNothingOverAnEmptyLibrary() {
        assertTrue(DiscoverNothingFound.bind(appraiserBoard().game).isEmpty());
    }

    @Test
    void discoverNothingFoundForksTheLibraryEmpty() {
        RuleBoard board = appraiserBoard("Forest", "Grizzly Bears", "Island");
        Card top = board.you.getCardsIn(ZoneType.Library).get(0);
        RuleTwin twin = board.twin(DiscoverNothingFound.rule());
        // The snapshot shows a library by its count and its top card alone.
        assertOnlyChanged(twin, top, top, "P0.library");
        assertTrue(twin.fork().player("player").getCardsIn(ZoneType.Library).isEmpty());
    }

    // ── 702.53a, transmute: move_card ───────────────────────────────────

    /** A's Dimir Infiltrator in hand over an Island, a Swamp and a Forest; A's library top card first. */
    private static RuleBoard infiltratorBoard(String... library) {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
        for (String land : List.of("Island", "Swamp", "Forest")) {
            board.card(board.you, land);
        }
        board.other = TestBoard.inHand(board.game, board.you, "Dimir Infiltrator");
        board.subject = board.library(board.you, library).stream()
                .filter(card -> card.getName().equals("Grizzly Bears")).findFirst().orElse(null);
        return board;
    }

    @Test
    void transmuteBindsTheTransmuterAndTheOneCardOfItsManaValue() {
        RuleBoard board = infiltratorBoard("Forest", "Shock", "Grizzly Bears", "Island");
        Binding binding = ManaValueSearch.transmute().precondition(board.game).orElseThrow();
        assertEquals(board.other, binding.card("source"));
        assertEquals(board.subject, binding.card("card"));
    }

    @Test
    void transmuteBindsNothingWithTwoCardsOfItsManaValue() {
        assertTrue(ManaValueSearch.transmute().precondition(infiltratorBoard("Grizzly Bears", "Grizzly Bears").game).isEmpty());
    }

    @Test
    void transmuteForksACentaurCourserInTheCardsPlace() {
        RuleBoard board = infiltratorBoard("Forest", "Shock", "Grizzly Bears", "Island");
        RuleTwin twin = board.twin(ManaValueSearch.transmute());
        List<Card> library = List.copyOf(twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Library));
        assertEquals(List.of("Forest", "Shock", Cards.COURSER, "Island"),
                library.stream().map(Card::getName).toList());
        // A library card the board shows by count alone: the swap the list
        // above shows changes no visible member.
        assertTrue(twin.changes().isEmpty(), "visible changes: " + twin.changes());
    }

    // ── 702.71a, transfigure: move_card ─────────────────────────────────

    /** A's Fleshwrither (mana value 4) over three Swamps; A's library top card first. */
    private static RuleBoard fleshwritherBoard(String... library) {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
        for (int i = 0; i < 3; i++) {
            board.card(board.you, "Swamp");
        }
        board.other = board.card(board.you, "Fleshwrither");
        board.subject = board.library(board.you, library).get(0);
        return board;
    }

    @Test
    void transfigureBindsTheCreatureAndTheOneCreatureCardOfItsManaValue() {
        RuleBoard board = fleshwritherBoard("Hill Giant", "Grizzly Bears");
        Binding binding = ManaValueSearch.transfigure().precondition(board.game).orElseThrow();
        assertEquals(board.other, binding.card("source"));
        assertEquals(board.subject, binding.card("card"));
    }

    @Test
    void transfigureBindsNothingWithTwoCreatureCardsOfItsManaValue() {
        assertTrue(ManaValueSearch.transfigure().precondition(fleshwritherBoard("Hill Giant", "Hill Giant").game).isEmpty());
    }

    @Test
    void transfigureForksACrawWurmInTheCardsPlace() {
        RuleBoard board = fleshwritherBoard("Hill Giant", "Grizzly Bears");
        RuleTwin twin = board.twin(ManaValueSearch.transfigure());
        List<Card> library = List.copyOf(twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Library));
        assertEquals(List.of(Cards.WURM, "Grizzly Bears"), library.stream().map(Card::getName).toList());
        assertOnlyChanged(twin, board.subject, library.get(0));
    }

    // ── 702.124j, partner with's search: move_card ──────────────────────

    /** A's Pir, Imaginative Rascal in hand over three Forests; A's library top card first, the top card the subject. */
    private static RuleBoard pirBoard(String... library) {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
        for (int i = 0; i < 3; i++) {
            board.card(board.you, "Forest");
        }
        board.other = TestBoard.inHand(board.game, board.you, "Pir, Imaginative Rascal");
        board.subject = board.library(board.you, library).get(0);
        return board;
    }

    @Test
    void partnerWithSearchBindsTheTopCardOfALibraryWithoutThePartner() {
        RuleBoard board = pirBoard("Forest", "Forest", "Forest");
        Binding binding = PartnerWithSearch.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("card"));
        assertEquals(board.other, binding.card("partner with"));
    }

    @Test
    void partnerWithSearchBindsNothingWithThePartnerInTheLibrary() {
        assertTrue(PartnerWithSearch.bind(pirBoard("Forest", "Toothy, Imaginary Friend", "Forest").game).isEmpty());
    }

    @Test
    void partnerWithSearchForksThePartnerOnTop() {
        RuleBoard board = pirBoard("Forest", "Forest", "Forest");
        RuleTwin twin = board.twin(PartnerWithSearch.rule());
        List<Card> library = List.copyOf(twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Library));
        assertEquals(List.of("Toothy, Imaginary Friend", "Forest", "Forest"),
                library.stream().map(Card::getName).toList());
        // The top card is on the board as the library's visible card.
        assertOnlySwapped(twin, board.subject, library.get(0));
    }
}
