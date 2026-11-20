package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.RecordLines.events;
import static com.pricepredictor.connector.effects.RecordLines.params;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The forced-play cases (FR-010b) whose variant forces a cast, played
 * forward: they find the play their variant forces made, and what it did
 * (the spell's resolution, or a watcher's trigger on what it did).
 */
@ExtendWith(ForgeExtension.class)
class DirectedForkPlayedForwardForcedCastTest extends DirectedForkPlayedForwardSupport {

    // ── forced plays (FR-010b) ──────────────────────────────────────────
    //
    // Each case's variant forces the play its point follows, which Forge's
    // AI makes rarely or never on the fork's board; the fork's records show
    // the play made and what it did.

    /**
     * {@code add_permanent}, 701.39a, inverted: Cached Defenses over three
     * Forests, A controlling no creature. The fork adds Grizzly Bears and A
     * is made to cast it: the bolster puts its counters on the Bears.
     */
    @Test
    void aBolsterWithACreaturePutsItsCounters() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        lands(game, active, "Forest", 3);
        Card defenses = TestBoard.inHand(game, active, "Cached Defenses");

        List<String> fork = forkOf(game, PhaseType.MAIN1, "701.39a", "has_creature");

        assertTrue(resolutionsOf(fork, defenses).stream()
                        .anyMatch(record -> RecordLines.eventTypes(record).contains("counter_change")),
                "Cached Defenses puts its counters in the fork");
    }

    /**
     * {@code counters}, 205.1a, forcing a cast: A holds Song of the Dryads
     * and three Forests, and B controls Grizzly Bears, as A's first main
     * phase begins. The fork's Bears has a +1/+1 counter, and the forced Song
     * makes it a colorless Forest land: the continuous record of the Song's
     * static removes every card type it had.
     */
    @Test
    void aForcedSongOfTheDryadsSetsTheBearsType() throws Throwable {
        Game game = empty(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        lands(game, active, "Forest", 3);
        Card bears = permanent(game, game.getPlayers().get(1), "Grizzly Bears");
        TestBoard.inHand(game, active, "Song of the Dryads");
        library(game, active, "Forest", "Forest", "Forest");
        library(game, game.getPlayers().get(1), "Forest", "Forest", "Forest");

        List<String> fork = forkOf(game, PhaseType.MAIN1, "205.1a", "counter-kept");

        assertTrue(fork.stream().anyMatch(record -> "continuous".equals(parse(record).get("kind"))
                        && record.contains("-all-card-types") && record.contains(ref(bears))),
                "the Song's static sets the Bears' card type in the fork: " + holding(fork, "attached"));
    }

    /**
     * {@code add_permanent}, 111.10p, forcing a cast: A holds Ellivere of
     * the Wild Court and four lands, beside its Grizzly Bears, as its first
     * main phase begins. The fork's A also controls Seal of Fire; the forced
     * Ellivere's enters trigger attaches a Virtuous Role to the Bears, whose
     * static boosts it.
     */
    @Test
    void aForcedElliveresRoleBoostsTheBears() throws Throwable {
        Game game = empty(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        lands(game, active, "Forest", 3);
        lands(game, active, "Plains", 1);
        permanent(game, active, "Grizzly Bears");
        TestBoard.inHand(game, active, "Ellivere of the Wild Court");
        library(game, active, "Forest", "Forest", "Forest");
        library(game, game.getPlayers().get(1), "Forest", "Forest", "Forest");

        List<String> fork = forkOf(game, PhaseType.MAIN1, "111.10p", "two_enchantments");

        assertTrue(holding(fork, "token_created").contains("resolution/resolution"),
                "Ellivere's trigger creates the Role in the fork");
        assertTrue(fork.stream().anyMatch(record -> "continuous".equals(parse(record).get("kind"))
                && record.contains("Virtuous")), "the Role's static is recorded in the fork");
    }

    /**
     * {@code move_card}, 702.133a, forcing a cast: Radical Idea in A's
     * graveyard, two Islands, as A's first main phase begins. The fork's
     * Radical Idea is in A's hand and is cast from there; as it resolves it
     * goes to the graveyard, not to exile.
     */
    @Test
    void aJumpStartCardCastFromHandGoesToTheGraveyard() throws Throwable {
        Game game = empty(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        lands(game, active, "Island", 2);
        Card idea = TestBoard.card(game, active, "Radical Idea");
        game.getAction().moveToGraveyard(idea, null);
        library(game, active, "Plains", "Plains", "Plains", "Plains", "Plains", "Plains");
        library(game, game.getPlayers().get(1), "Forest", "Forest", "Forest");

        List<String> fork = forkOf(game, PhaseType.MAIN1, "702.133a", "cast_from_hand");

        List<String> moved = new ArrayList<>();
        for (String record : fork) {
            for (Map<String, Object> event : events(record)) {
                if ("zone_change".equals(event.get("type")) && "stack".equals(params(event).get("from_zone"))) {
                    moved.add(String.valueOf(params(event).get("to_zone")));
                }
            }
        }
        assertTrue(moved.contains("graveyard") && !moved.contains("exile"),
                "the spell cast from hand leaves the stack to the graveyard: " + moved);
    }

    /**
     * {@code remove_permanent}, 607.2k: Changeling Hero over five Plains,
     * Grizzly Bears A's only other creature. The fork's A controls none and
     * is made to cast the Hero: its champion trigger exiles nothing and it is
     * sacrificed.
     */
    @Test
    void aChampionWithNothingToChampionIsSacrificed() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        lands(game, active, "Plains", 5);
        permanent(game, active, "Grizzly Bears");
        Card hero = TestBoard.inHand(game, active, "Changeling Hero");

        List<String> fork = forkOf(game, PhaseType.MAIN1, "607.2k", "no_creature_to_champion");

        assertTrue(eventsOf(fork, "zone_change", ref(hero)).stream()
                        .anyMatch(move -> "graveyard".equals(params(move).get("to_zone"))),
                "the Hero is sacrificed in the fork");
    }

    /**
     * {@code pt}, 704.5f: Disfigure in the hand of A over a Swamp, the Drudge
     * Skeletons of B a 1/1. The fork's Skeletons is a 1/3 and A is made to
     * cast Disfigure at it: it survives the -2/-2.
     */
    @Test
    void aTougherCreatureSurvivesTheForcedDisfigure() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        permanent(game, active, "Swamp");
        Card disfigure = TestBoard.inHand(game, active, "Disfigure");
        Card skeletons = permanent(game, game.getPlayers().get(1), "Drudge Skeletons");

        List<String> fork = forkOf(game, PhaseType.MAIN1, "704.5f", "tougher");

        assertFalse(resolutionsOf(fork, disfigure).isEmpty(), "Disfigure resolves in the fork");
        assertTrue(eventsOf(fork, "zone_change", ref(skeletons)).isEmpty(), "the Skeletons survive");
    }

    /**
     * 701.24e: the Cosi's Trickster of B; the Demonic Tutor of A over two
     * Swamps and a library of three. The fork's library is empty and A is
     * made to cast the Tutor: the empty library is shuffled after the search
     * and the Trickster's trigger is evaluated.
     */
    @Test
    void aTutorShufflesTheEmptiedLibrary() throws Throwable {
        Game game = empty(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        Card trickster = permanent(game, game.getPlayers().get(1), "Cosi's Trickster");
        lands(game, active, "Swamp", 2);
        Card tutor = TestBoard.inHand(game, active, "Demonic Tutor");
        library(game, active, "Forest", "Island", "Grizzly Bears");

        List<String> fork = forkOf(game, PhaseType.MAIN1, "701.24e", "empty_library");

        assertFalse(resolutionsOf(fork, tutor).isEmpty(), "the Tutor resolves in the fork");
        assertFalse(fired(fork, trickster).isEmpty(), "the Trickster triggers on the empty library's shuffle");
    }

    /**
     * 701.44b, inverted: Merfolk Branchwalker over two Forests beside
     * Wildgrowth Walker, the library of A holding cards. The fork's library
     * is empty and A is made to cast the Branchwalker, which explores with
     * nothing to reveal: Wildgrowth Walker's trigger still fires.
     */
    @Test
    void aBranchwalkerExploresAnEmptyLibrary() throws Throwable {
        Game game = empty(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        lands(game, active, "Forest", 2);
        Card walker = permanent(game, active, "Wildgrowth Walker");
        Card branchwalker = TestBoard.inHand(game, active, "Merfolk Branchwalker");
        library(game, active, "Forest", "Grizzly Bears", "Island");

        List<String> fork = forkOf(game, PhaseType.MAIN1, "701.44b", "empty_library");

        assertFalse(resolutionsOf(fork, branchwalker).isEmpty(), "the Branchwalker is cast in the fork");
        assertFalse(fired(fork, walker).isEmpty(), "Wildgrowth Walker triggers in the fork");
        assertTrue(holding(fork, "card_revealed").isEmpty(), "the fork's explore reveals nothing");
    }

    /**
     * 701.50f, inverted: Echo Inspector in the hand of A, A having drawn,
     * beside Glorious Purpose. The fork's B controls Spirit of the Labyrinth
     * and A is made to cast the Inspector, which connives without drawing:
     * Glorious Purpose's trigger still fires.
     */
    @Test
    void anInspectorConnivesUnderTheAddedSpirit() throws Throwable {
        Game game = empty(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        lands(game, active, "Island", 4);
        Card purpose = permanent(game, active, "Glorious Purpose");
        Card inspector = TestBoard.inHand(game, active, "Echo Inspector");
        library(game, active, "Forest", "Island", "Island");
        active.drawCard();

        List<String> fork = forkOf(game, PhaseType.MAIN1, "701.50f", "spirit_added");

        assertFalse(resolutionsOf(fork, inspector).isEmpty(), "the Inspector is cast in the fork");
        assertFalse(fired(fork, purpose).isEmpty(), "Glorious Purpose triggers in the fork");
        assertTrue(holding(fork, "card_drawn").isEmpty(), "the fork's connive draws nothing");
    }

    /**
     * 701.57b, inverted: Geological Appraiser over four Mountains beside Val,
     * Marooned Surveyor, Grizzly Bears in the library. The fork's library is
     * empty and A is made to cast the Appraiser, whose discovery finds
     * nothing: Val's trigger still fires.
     */
    @Test
    void anAppraiserDiscoversAnEmptyLibrary() throws Throwable {
        Game game = empty(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        lands(game, active, "Mountain", 4);
        Card val = permanent(game, active, "Val, Marooned Surveyor");
        Card appraiser = TestBoard.inHand(game, active, "Geological Appraiser");
        library(game, active, "Forest", "Grizzly Bears", "Island", "Island", "Island");

        List<String> fork = forkOf(game, PhaseType.MAIN1, "701.57b", "empty_library");

        assertFalse(resolutionsOf(fork, appraiser).isEmpty(), "the Appraiser is cast in the fork");
        assertFalse(fired(fork, val).isEmpty(), "Val triggers in the fork");
    }

    /**
     * {@code replace_card}, 702.103e, inverted: Hopeful Eidolon over four
     * Plains, Grizzly Bears the battlefield's one creature. The fork's is
     * Skulking Ghost and A is made to cast the Eidolon bestowed, its target
     * drawn, the Ghost: the Ghost is sacrificed as it becomes the target, and
     * the Eidolon resolves as a creature, attached to nothing.
     */
    @Test
    void aBestowedEidolonWhoseTargetIsSacrificedEntersAsACreature() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        lands(game, active, "Plains", 4);
        permanent(game, active, "Grizzly Bears");
        Card eidolon = TestBoard.inHand(game, active, "Hopeful Eidolon");

        List<String> fork = forkOf(game, PhaseType.MAIN1, "702.103e", "skulking_ghost");

        List<String> resolved = resolutionsOf(fork, eidolon);
        assertTrue(resolved.stream().anyMatch(record -> RecordLines.eventTypes(record).contains("zone_change")),
                "the Eidolon enters in the fork");
        assertTrue(resolved.stream().noneMatch(record -> RecordLines.eventTypes(record).contains("attached")),
                "attached to nothing");
    }

    /**
     * {@code move_card}, 702.170c, inverted: Kellan Joins Up over a Forest, a
     * Plains and an Island, Hill Giant the hand's other card. The fork's is
     * Grizzly Bears and A is made to cast Kellan: its trigger exiles the
     * Bears, which become plotted.
     */
    @Test
    void kellanPlotsTheBears() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        for (String land : List.of("Forest", "Plains", "Island")) {
            permanent(game, active, land);
        }
        Card kellan = TestBoard.inHand(game, active, "Kellan Joins Up");
        TestBoard.inHand(game, active, "Hill Giant");

        List<String> fork = forkOf(game, PhaseType.MAIN1, "702.170c", "small_card_in_hand");

        assertTrue(resolutionsOf(fork, kellan).stream().anyMatch(record -> events(record).stream()
                        .anyMatch(event -> "zone_change".equals(event.get("type"))
                                && "exile".equalsIgnoreCase(String.valueOf(params(event).get("to_zone"))))),
                "Kellan's trigger exiles a card in the fork");
    }
}
