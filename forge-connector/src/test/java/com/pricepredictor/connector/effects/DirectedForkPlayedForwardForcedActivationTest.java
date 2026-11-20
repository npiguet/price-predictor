package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The forced-play cases (FR-010b) whose variant forces an activation or an
 * answer to a "you may", played forward: they find the play their variant
 * forces made, and what it did.
 */
@ExtendWith(ForgeExtension.class)
class DirectedForkPlayedForwardForcedActivationTest extends DirectedForkPlayedForwardSupport {

    // ── forced plays (FR-010b) ──────────────────────────────────────────
    //
    // Each case's variant forces the play its point follows, which Forge's
    // AI makes rarely or never on the fork's board; the fork's records show
    // the play made and what it did.

    /**
     * {@code add_permanent}, 111.10v, inverted: a Mutagen token beside
     * Crustacean Commando, A's one Island tapped. The fork adds an untapped
     * Island and forces the token's activation, which puts its counter.
     */
    @Test
    void anIslandLetsTheMutagenTokenPutItsCounter() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        permanent(game, active, "Crustacean Commando");
        Card token = game.getAction().moveToPlay(forge.game.card.CardFactory.getCard(
                forge.StaticData.instance().getAllTokens().getToken("c_a_mutagen_sac"), active, game),
                active, null, null);
        token.setGameTimestamp(game.getNextTimestamp());
        token.setSickness(false);
        permanent(game, active, "Island").setTapped(true);

        List<String> fork = forkOf(game, PhaseType.MAIN1, "111.10v", "mana_available");

        assertTrue(resolutionsOf(fork, token).stream()
                .anyMatch(record -> RecordLines.eventTypes(record).contains("counter_change")),
                "the token's ability puts its counter in the fork");
    }

    /**
     * {@code add_permanent}, 119.7, inverted: Tree of Redemption, A at 5
     * life. The fork adds Leyline of Punishment and A is made to activate
     * the Tree: the exchange, which would raise A's life, does not happen.
     */
    @Test
    void anExchangeAPlayerWhoCannotGainLifeWouldGainFromChangesNothing() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        Card tree = permanent(game, active, "Tree of Redemption");
        active.setLife(5, null);

        List<String> fork = forkOf(game, PhaseType.MAIN1, "119.7", "leyline");

        List<String> resolved = resolutionsOf(fork, tree);
        assertFalse(resolved.isEmpty(), "the exchange resolves in the fork");
        assertTrue(resolved.stream().noneMatch(record -> RecordLines.eventTypes(record).contains("life_change")),
                "and changes no life");
    }

    /**
     * {@code life}, 701.12g: Tree of Perdition, B at 7. The fork's B has 20
     * and A is made to activate the Tree at B: the exchange lowers B's life.
     */
    @Test
    void anExchangeThatLowersTheOpponentsLifeHappens() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        Card tree = permanent(game, active, "Tree of Perdition");
        game.getPlayers().get(1).setLife(7, null);

        List<String> fork = forkOf(game, PhaseType.MAIN1, "701.12g", "opponent_above");

        assertTrue(resolutionsOf(fork, tree).stream()
                        .anyMatch(record -> RecordLines.eventTypes(record).contains("life_change")),
                "the exchange changes the life of B in the fork");
    }

    /**
     * {@code move_card}, 702.71a: Fleshwrither over three Swamps, Hill Giant
     * the library's one creature of mana value 4. The fork's is Craw Wurm,
     * and its A is made to transfigure: Fleshwrither is sacrificed and
     * nothing enters.
     */
    @Test
    void aTransfigureFindingNothingPutsNothingOntoTheBattlefield() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        lands(game, active, "Swamp", 3);
        Card fleshwrither = permanent(game, active, "Fleshwrither");
        library(game, active, "Hill Giant");

        List<String> fork = forkOf(game, PhaseType.MAIN1, "702.71a", "no_matching_value");

        assertEquals(List.of(ref(fleshwrither)), ForcedPlayTest.costCards(fork, "sacrificed"),
                "Fleshwrither is sacrificed for the cost");
        assertTrue(fork.stream().noneMatch(record -> record.contains("\"from_zone\":\"library\",\"to_zone\":\"battlefield\"")),
                "nothing enters from the library");
    }

    /**
     * {@code move_card}, 702.53a: Dimir Infiltrator over an Island, a Swamp
     * and a Forest, Grizzly Bears the library's one card of mana value 2. The
     * fork's is Centaur Courser, and its A is made to transmute: the
     * Infiltrator is discarded and the search finds nothing.
     */
    @Test
    void aTransmuteFindingNothingPutsNothingIntoTheHand() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        for (String land : List.of("Island", "Swamp", "Forest")) {
            permanent(game, active, land);
        }
        Card infiltrator = TestBoard.inHand(game, active, "Dimir Infiltrator");
        library(game, active, "Grizzly Bears");

        List<String> fork = forkOf(game, PhaseType.MAIN1, "702.53a", "no_match");

        assertEquals(List.of(ref(infiltrator)), ForcedPlayTest.costCards(fork, "discarded"),
                "the Infiltrator is discarded for the cost");
        assertTrue(fork.stream().noneMatch(record -> record.contains("\"from_zone\":\"library\",\"to_zone\":\"hand\"")),
                "the search finds nothing");
    }

    /**
     * {@code add_permanent}, 118.12b, inverted, forcing a "you may" answer:
     * A's upkeep begins with a basic land in its library and no Elfhame
     * Sanctuary. Live, A draws in its draw step; the fork's Sanctuary
     * triggers, the forced yes searches, and the draw step is skipped.
     */
    @Test
    void anAddedSanctuarysForcedSearchSkipsTheDraw() throws Throwable {
        Game game = empty(PhaseType.UPKEEP);
        Player active = game.getPlayers().get(0);
        library(game, active, "Shock", "Forest", "Shock", "Shock", "Shock", "Shock");
        library(game, game.getPlayers().get(1), "Shock", "Shock", "Shock");

        List<String> fork = forkOf(game, PhaseType.UPKEEP, "118.12b", "sanctuary_added");

        assertTrue(holding(fork, "phase_skipped").contains("resolution/resolution"),
                "the search's resolution skips the draw step in the fork");
    }

    /**
     * {@code add_permanent}, 701.23f, inverted, forcing an activation: A
     * controls Archivist of Oghma and B an untapped Evolving Wilds as A's
     * first main phase begins. The fork's A also controls Aven Mindcensor;
     * B is made to activate the Wilds, searches the top four cards of its
     * library, and Archivist's trigger fires.
     */
    @Test
    void aSearchLimitedByTheMindcensorStillFiresArchivist() throws Throwable {
        Game game = empty(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        Player searcher = game.getPlayers().get(1);
        Card archivist = permanent(game, active, "Archivist of Oghma");
        permanent(game, searcher, "Evolving Wilds");
        library(game, active, "Island", "Island", "Island");
        library(game, searcher, "Grizzly Bears", "Grizzly Bears", "Grizzly Bears", "Grizzly Bears", "Forest");

        List<String> fork = forkOf(game, PhaseType.MAIN1, "701.23f", "mindcensor_added");

        assertFalse(fired(fork, archivist).isEmpty(), "Archivist's trigger fires on the limited search");
    }

    /** Arcanum Wings on Grizzly Bears over three Islands, {@code inHand} in the hand of A. */
    private static Game wingsBoard(String inHand) {
        Game game = bare(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        Card bears = permanent(game, active, "Grizzly Bears");
        permanent(game, active, "Arcanum Wings").attachToEntity(bears, null);
        lands(game, active, "Island", 3);
        TestBoard.inHand(game, active, inHand);
        return game;
    }

    /** The fork's resolutions of the Aura swap of the Arcanum Wings on the battlefield of {@code game}. */
    private static List<String> swaps(Game game, List<String> fork) {
        Card wings = game.getCardsIn(forge.game.zone.ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals("Arcanum Wings")).findFirst().orElseThrow();
        return resolutionsOf(fork, wings);
    }

    /**
     * {@code move_card}, 701.12e and 702.65a, inverted: Arcanum Wings on
     * Grizzly Bears, no Aura in hand. The fork's hand holds Holy Strength in
     * the hand's card's place, and its A is made to activate the swap: Holy
     * Strength moves from the hand and is attached to the Bears.
     */
    @Test
    void aSwapWithAnAuraInHandExchangesAndAttaches() throws Throwable {
        for (String rule : List.of("701.12e", "702.65a")) {
            Game game = wingsBoard(rule.equals("701.12e") ? "Giant Growth" : "Grizzly Bears");

            List<String> fork = forkOf(game, PhaseType.MAIN1, rule, "aura_in_hand");

            List<String> swapped = swaps(game, fork);
            assertTrue(swapped.stream().anyMatch(record -> RecordLines.eventTypes(record).contains("attached")),
                    rule + ": the swap attaches Holy Strength in the fork");
            assertTrue(swapped.stream().anyMatch(record -> RecordLines.eventTypes(record).contains("zone_change")),
                    rule + ": and moves it from the hand");
        }
    }

    /**
     * {@code move_card}, 702.65b: Arcanum Wings on Grizzly Bears, Wild Growth
     * (enchant land) in hand. The fork's is Holy Strength, and its A is made
     * to activate the swap: the exchange happens.
     */
    @Test
    void aSwapWithAnAuraThatFitsExchanges() throws Throwable {
        Game game = wingsBoard("Wild Growth");

        List<String> fork = forkOf(game, PhaseType.MAIN1, "702.65b", "can_enchant");

        assertTrue(swaps(game, fork).stream().anyMatch(record -> RecordLines.eventTypes(record).contains("zone_change")),
                "the swap moves the Auras in the fork");
    }
}
