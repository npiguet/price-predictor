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
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Rules whose point the beginning phase makes, played forward: a trigger at
 * the upkeep (702.1c, 104.2b, 104.4a) and the draw step's draw (704.5b).
 * 104.4a reads a record no other kind of point writes: the game's end at a
 * state-based check (two losses live, and the fork's one beside a win).
 */
@ExtendWith(ForgeExtension.class)
class DirectedForkPlayedForwardUpkeepTest extends DirectedForkPlayedForwardSupport {

    /**
     * {@code replace_card}, 702.1c: Concerted Effort beside White Knight and
     * Grizzly Bears as the upkeep begins. The live trigger gives the Bears the
     * Knight's first strike and protection from black; the fork's Hill Giant
     * in the Knight's place has no listed keyword, so the trigger resolves
     * and grants nothing.
     */
    @Test
    void aGiantInTheKnightsPlaceLeavesConcertedEffortNothingToShare() throws Throwable {
        Game game = bare(PhaseType.UNTAP);
        Player active = game.getPlayers().get(0);
        Card effort = permanent(game, active, "Concerted Effort");
        permanent(game, active, "White Knight");
        Card bears = permanent(game, active, "Grizzly Bears");

        // The live game reaches the upkeep itself, so its trigger fires there too.
        List<String> all = played(game, null, "702.1c", Decisions.SAMPLED);

        List<String> live = all.stream().filter(record -> !Boolean.TRUE.equals(parse(record).get("fork"))).toList();
        assertFalse(eventsOf(live, "keyword_change", ref(bears)).isEmpty(),
                "the live Bears share the Knight's keywords");
        List<String> fork = forkOf(all, "702.1c", "no_keywords");
        List<String> resolved = resolutionsOf(fork, effort);
        assertFalse(resolved.isEmpty(), "Concerted Effort's trigger resolves in the fork");
        assertTrue(resolved.stream().noneMatch(record -> RecordLines.eventTypes(record).contains("keyword_change")),
                "and grants no keyword there");
    }

    /**
     * {@code life}, 104.2b, inverted: Felidar Sovereign's controller at 39
     * life as the upkeep begins. The live trigger does not fire; the fork's
     * 40 life fires it, it resolves and wins the game, the {@code player_won}
     * in the resolution's effect half 104.2b-1's declaration reads.
     */
    @Test
    void fortyLifeWinsTheGameInTheFork() throws Throwable {
        Game game = board(PhaseType.UPKEEP);
        Player active = game.getPlayers().get(0);
        permanent(game, active, "Felidar Sovereign");
        active.setLife(39, null);

        List<String> fork = forkOf(game, PhaseType.UPKEEP, "104.2b", "forty");

        List<String> won = new ArrayList<>();
        for (String record : fork) {
            if (RecordLines.eventTypes(record).contains("player_won")) {
                won.add(parse(record).get("kind") + "/" + parse(record).get("moment"));
            }
        }
        assertEquals(List.of("resolution/resolution"), won, "the win is in the trigger's effect half");
        assertFalse(game.isGameOver(), "the live game goes on");
    }

    /**
     * {@code move_card}, 704.5b: the active player's library holds one card
     * as its upkeep begins. The live draw step draws it and the game goes
     * on; the fork's card is in the graveyard, so the draw step's draw is
     * attempted from the empty library and the player loses the game.
     */
    @Test
    void aDrawStepFromAnEmptyLibraryLosesTheGame() throws Throwable {
        Game game = board(PhaseType.UPKEEP);
        Player active = game.getPlayers().get(0);
        List<Card> library = List.copyOf(active.getCardsIn(forge.game.zone.ZoneType.Library));
        for (Card card : library.subList(1, library.size())) {
            game.getAction().exile(card, null, null);
        }
        assertEquals(1, active.getCardsIn(forge.game.zone.ZoneType.Library).size());

        List<String> fork = forkOf(game, PhaseType.UPKEEP, "704.5b", "empty");

        List<String> kinds = new ArrayList<>();
        Map<String, Object> lost = null;
        for (String record : fork) {
            for (Map<String, Object> event : events(record)) {
                if ("player_lost".equals(event.get("type"))
                        && List.of(Refs.of(active)).equals(event.get("subjects"))) {
                    lost = event;
                    kinds.add(parse(record).get("kind") + "/" + parse(record).get("moment"));
                }
            }
        }
        assertTrue(lost != null, "the active player loses in the fork: " + fork);
        assertEquals("Milled", params(lost).get("reason"), lost.toString());
        assertFalse(active.hasLost(), "the live player draws its last card and plays on");
        assertEquals(List.of("resolution/resolution"), kinds,
                "the loss is in an effect half, where 704.5b-1's declaration reads it");
    }

    /** The subjects of every event of {@code record} of {@code type}. */
    private static List<Object> subjectsOf(String record, String type) {
        List<Object> found = new ArrayList<>();
        for (Map<String, Object> event : events(record)) {
            if (type.equals(event.get("type"))) {
                found.add(event.get("subjects"));
            }
        }
        return found;
    }

    /**
     * {@code life}, 104.4a, inverted: Plague Spitter's controller and its
     * opponent at 20 life as the untap step begins. In the live upkeep the
     * trigger takes 1 life from each and nobody loses; the fork, taken as
     * the upkeep begins, has both players at 1, and its game ends with two
     * losses and no winner, a draw, in the trigger's resolution's effect
     * half, where 104.4a-2's declaration reads it.
     */
    @Test
    void theForksPlayersAtOneLifeBothLoseToPlagueSpitter() throws Throwable {
        Game game = board(PhaseType.UNTAP);
        Player active = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        permanent(game, active, "Plague Spitter");

        List<String> records = played(game, null, "104.4a", Decisions.SAMPLED);

        List<List<Object>> ends = new ArrayList<>();
        for (String record : records) {
            List<Object> lost = subjectsOf(record, "player_lost");
            if (!lost.isEmpty()) {
                Map<String, Object> parsed = parse(record);
                assertEquals("resolution/resolution", parsed.get("kind") + "/" + parsed.get("moment"), record);
                ends.add(List.of(Boolean.TRUE.equals(parsed.get("fork")), lost, subjectsOf(record, "player_won")));
            }
        }
        assertEquals(List.of(
                List.of(true, List.of(List.of(Refs.of(active)), List.of(Refs.of(opponent))), List.of())),
                ends, "the fork's game ends in a draw, the live game goes on");
    }
}
