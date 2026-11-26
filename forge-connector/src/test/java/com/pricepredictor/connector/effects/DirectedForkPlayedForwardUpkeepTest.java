package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;

import static com.pricepredictor.connector.effects.RecordLines.parse;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A rule whose point the beginning phase makes, played forward: a trigger at
 * the upkeep (702.1c).
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
        assertTrue(resolved.stream().noneMatch(record -> RecordEvents.eventTypes(record).contains("keyword_change")),
                "and grants no keyword there");
    }
}
