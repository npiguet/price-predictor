package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.RecordLines.parse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The inverted cases (FR-014a) played forward: they bind the common board
 * and find the rare fact's record in the fork.
 */
@ExtendWith(ForgeExtension.class)
class DirectedForkPlayedForwardInvertedTest extends DirectedForkPlayedForwardSupport {

    // ── inverted variants (DF-A item 3) ─────────────────────────────────

    /**
     * {@code replace_card}, 701.44d, inverted: Hakbal of the Surging Soul is
     * A's only Merfolk as its first main phase begins, beside A's Grizzly
     * Bears. Live, Hakbal's beginning-of-combat trigger has Hakbal alone
     * explore and reveal one card; the fork's Merfolk Branchwalker in the
     * Bears' place explores too, so the trigger's resolution reveals two.
     */
    @Test
    void aBranchwalkerInTheBearsPlaceExploresBesideHakbal() throws Throwable {
        Game game = board(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        permanent(game, active, "Hakbal of the Surging Soul");

        List<String> records = played(game, PhaseType.MAIN1, "701.44d", Decisions.SAMPLED);

        assertEquals(List.of(1), revealsOfExplores(records, false), "Hakbal explores alone live");
        assertEquals(List.of(2), revealsOfExplores(records, true), "two explore in the fork");
    }

    /**
     * Per resolution revealing cards among the fork's or the live game's
     * records (on this board only Hakbal's trigger reveals), the cards it
     * revealed.
     */
    private static List<Integer> revealsOfExplores(List<String> records, boolean fork) {
        List<Integer> found = new ArrayList<>();
        for (String record : records) {
            Map<String, Object> parsed = parse(record);
            if (Boolean.TRUE.equals(parsed.get("fork")) != fork || !"resolution".equals(parsed.get("moment"))) {
                continue;
            }
            List<String> types = RecordLines.eventTypes(record);
            if (types.contains("card_revealed")) {
                found.add((int) types.stream().filter("card_revealed"::equals).count());
            }
        }
        return found;
    }

    /**
     * {@code add_permanent}, 614.17d, inverted: Zombify in hand and Craw Wurm
     * in A's graveyard, no Grafdigger's Cage; the fork adds the Cage and
     * forces the cast of Zombify (FR-010b), which Forge's AI does not make
     * under the Cage: the Wurm's move to the battlefield is prevented.
     */
    @Test
    void theCagePreventsTheForcedReanimation() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        lands(game, active, "Swamp", 4);
        TestBoard.inHand(game, active, "Zombify");
        TestBoard.inGraveyard(game, active, "Craw Wurm");
        permanent(game, game.getPlayers().get(1), "Hill Giant");

        List<String> all = played(game, PhaseType.MAIN1, "614.17d", Decisions.EVERY);

        List<String> fork = forkOf(all, "614.17d", "cage");
        assertTrue(fork.stream().anyMatch(record -> "rewrite".equals(parse(record).get("kind"))
                        && record.contains("\"result\":\"prevented\"")),
                "the move to the battlefield is prevented in the fork");
    }

    /**
     * {@code move_card}, 702.124j, inverted: Pir, Imaginative Rascal in hand
     * over a library of Forests. The fork's top card is Toothy, Imaginary
     * Friend, which Pir's partner-with trigger finds, Pir's cast forced.
     */
    @Test
    void thePartnerOnTopIsFound() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        lands(game, active, "Forest", 3);
        TestBoard.inHand(game, active, "Pir, Imaginative Rascal");

        List<String> fork = forkOf(game, PhaseType.MAIN1, "702.124j", "named_card_in_library");

        assertTrue(fork.stream().anyMatch(record -> "resolution".equals(parse(record).get("kind"))
                        && record.contains("\"from_zone\":\"library\",\"to_zone\":\"hand\"")),
                "the partner-with trigger finds Toothy in the fork");
    }

    /**
     * {@code add_permanent}, 702.131a, inverted: nine permanents and Secrets
     * of the Golden City in hand. The fork adds a tapped Island and forces
     * the cast; the spell resolving gives the city's blessing.
     */
    @Test
    void aTenthPermanentGivesTheBlessing() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        lands(game, active, "Island", 8);
        permanent(game, active, "Grizzly Bears");
        TestBoard.inHand(game, active, "Secrets of the Golden City");

        List<String> fork = forkOf(game, PhaseType.MAIN1, "702.131a", "ten_permanents");

        assertTrue(holding(fork, "state_flag_change").contains("resolution/resolution"),
                "the spell's resolution gives the blessing in the fork");
    }

    /**
     * {@code counters}, 104.3d, inverted: B with no poison counter and Ichor
     * Rats in A's hand. The fork's B has 9 and A is made to cast the Rats,
     * whose enters trigger gives B its tenth: B loses.
     */
    @Test
    void aNinthPoisonCounterLetsTheNextOneKill() throws Throwable {
        Game game = bare(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        lands(game, active, "Swamp", 3);
        TestBoard.inHand(game, active, "Ichor Rats");

        List<String> fork = forkOf(game, PhaseType.MAIN1, "104.3d", "nine");

        assertTrue(holding(fork, "player_lost").contains("resolution/resolution"),
                "B's tenth poison counter loses it the game in the fork");
    }

    /**
     * {@code replace_card}, 701.42a, 712.4a and 712.14c, inverted, one
     * registration whichever of the three is listed: Graf Rats beside Hill
     * Giant as the first main phase begins. The fork's Giant is
     * Midnight Scavengers, so at the beginning of combat Graf Rats' trigger
     * melds the two into Chittering Host, with no seat's choice.
     */
    @Test
    void thePartnerInTheGiantsPlaceMelds() throws Throwable {
        for (String rule : List.of("701.42a", "712.4a", "712.14c")) {
            Game game = bare(PhaseType.MAIN1);
            Player active = game.getPlayers().get(0);
            permanent(game, active, "Graf Rats");
            permanent(game, active, "Hill Giant");

            List<String> fork = forkOf(game, PhaseType.MAIN1, rule, "meld_pair");

            assertFalse(fork.stream().filter(record -> record.contains("\"to_zone\":\"exile\"")).toList().isEmpty(),
                    rule + ": the pair is exiled to meld");
        }
    }
}
