package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.directed.KeywordEdit;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.combat.Combat;
import forge.game.event.GameEventGameFinished;
import forge.game.keyword.Keyword;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
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
 * One registered rule per operation family played forward (review TC3's
 * "real fix"): a live game set up as the rule's base, recorded by a
 * {@code GameRecording} that takes the rule's directed fork at the boundary
 * ({@code DirectedFork.take}), the live turn played out and the fork played
 * to its end; the fork's records then show the variant's point happening,
 * not only the edited board. Each case picks a rule whose point is the
 * engine's own (a turn-based action, a trigger, combat damage, the
 * collector's own reading of a priority stop's candidates), so neither
 * seat's choices decide it.
 *
 * <p>Two cases read a record no other kind of point writes: the engine's
 * own question before a declaration of blockers (702.118b's blockers
 * record), and the game's end at a state-based check (104.4a's two losses
 * live, and the fork's one beside a win).
 *
 * <p>{@code pt} is played forward by
 * {@code ForkRecordsTest.theRegisteredToughnessZeroRuleForksATougherCreatureToTheEndOfTheTurn}.
 * The families left out ({@code tap}, {@code colour_type}) have no
 * registered rule whose point happens without a seat's choice: a spell
 * cast, an attack, a block.
 */
@ExtendWith(ForgeExtension.class)
class DirectedForkPlayedForwardTest {

    @TempDir
    Path tempDir;

    /** A started two-seat game in {@code step} of the first seat's third turn. */
    private static Game board(PhaseType step) {
        Game game = ForkPlayForwardTest.board(TestBoard.twoPlayerGame(PhaseType.MAIN1));
        game.getPhaseHandler().devModeSet(step, game.getPlayers().get(0), 3);
        return game;
    }

    /** A permanent since the turn began, with a timestamp of its own. */
    private static Card permanent(Game game, Player owner, String name) {
        Card card = TestBoard.inPlay(game, owner, name);
        card.setGameTimestamp(game.getNextTimestamp());
        card.setSickness(false);
        return card;
    }

    /**
     * Record the game from the boundary beginning {@code step}, taking
     * {@code rule}'s directed fork there, and play the live turn out; the
     * fork's records, each checked to name the rule.
     */
    private List<String> forkOf(Game game, PhaseType step, String rule, String variant) throws Throwable {
        return forkOf(game, step, rule, variant, false);
    }

    /** {@link #forkOf(Game, PhaseType, String, String)}, every decision candidate written when {@code decisions}. */
    private List<String> forkOf(Game game, PhaseType step, String rule, String variant, boolean decisions)
            throws Throwable {
        List<String> forked = new ArrayList<>();
        for (String record : played(game, step, rule, decisions)) {
            Map<String, Object> parsed = parse(record);
            if (Boolean.TRUE.equals(parsed.get("fork"))) {
                assertEquals(List.of(rule), parsed.get("aimed_at"), record);
                assertEquals(variant, RecordLines.asMap(parsed.get("directed")).get("variant"), record);
                forked.add(record);
            }
        }
        assertFalse(forked.isEmpty(), rule + ": the fork was kept and recorded");
        return forked;
    }

    /**
     * Every record of the game recorded from the boundary beginning
     * {@code step}, {@code rule}'s directed fork taken there, and the live
     * turn played out; with {@code step} null, the live game reaches the
     * boundary itself, its turn-based actions included. A live game the turn
     * ended writes its end, as {@code Match.startGame} has it do.
     */
    private List<String> played(Game game, PhaseType step, String rule, boolean decisions) throws Throwable {
        TestCaps built = TestCaps.caps().manaCap(2000).interventions(0).probes(0);
        if (decisions) {
            built = built.everyDecision();
        }
        CollectionCaps caps = built.directed(1, 1.0, rule).build();
        RecordShardWriter shard = ShardRun.writer(tempDir);
        try (GameRecording recording = new GameRecording(game, shard, ShardRun.GAME_ID, caps, 7L, null)) {
            recording.open();
            game.getTriggerHandler().resetActiveTriggers();
            if (step != null) {
                TestBoard.enterStep(game, step);
            }
            ForkPlayForwardTest.playTheTurnOut(game);
            if (game.isGameOver()) {
                game.fireEvent(new GameEventGameFinished());
            }
        } finally {
            shard.close();
        }
        return ShardRun.readShard(shard.path());
    }

    /** Every event of {@code records} of {@code type} naming {@code ref}. */
    private static List<Map<String, Object>> eventsOf(List<String> records, String type, String ref) {
        List<Map<String, Object>> found = new ArrayList<>();
        for (String record : records) {
            for (Map<String, Object> event : events(record)) {
                if (type.equals(event.get("type")) && event.get("subjects") instanceof List<?> subjects
                        && subjects.contains(ref)) {
                    found.add(event);
                }
            }
        }
        return found;
    }

    private static String ref(Card card) {
        return "E" + card.getId();
    }

    /**
     * {@code counters}, 122.1d: a tapped Grizzly Bears with a stun counter
     * as the untap step begins. The fork takes the counter away, so the
     * step untaps the Bears, where the live game removes the counter instead.
     */
    @Test
    void aStunCounterTakenAwayLetsThePermanentUntap() throws Throwable {
        Game game = board(PhaseType.UNTAP);
        Card bears = game.getCardsIn(forge.game.zone.ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals("Grizzly Bears")).findFirst().orElseThrow();
        bears.setTapped(true);
        bears.setCounters(CounterEnumType.STUN, 1);

        List<String> fork = forkOf(game, PhaseType.UNTAP, "122.1d", "no_stun");

        assertFalse(eventsOf(fork, "untapped", ref(bears)).isEmpty(), "the Bears untap in the fork");
    }

    /**
     * {@code keyword}, 702.63c: Maelstrom Djinn with vanishing twice and two
     * time counters as the upkeep begins. The live game's two triggers take
     * both counters and sacrifice it; the fork's one instance takes one, and
     * the Djinn stays.
     */
    @Test
    void oneInstanceOfVanishingTakesOneCounter() throws Throwable {
        Game game = board(PhaseType.UPKEEP);
        Player active = game.getPlayers().get(0);
        Card djinn = permanent(game, active, "Maelstrom Djinn");
        KeywordEdit.add(game, djinn, "Vanishing");
        KeywordEdit.add(game, djinn, "Vanishing");
        djinn.setCounters(CounterEnumType.TIME, 2);
        assertEquals(2, djinn.getAmountOfKeyword(Keyword.VANISHING));

        List<String> fork = forkOf(game, PhaseType.UPKEEP, "702.63c", "one_instance");

        List<Map<String, Object>> removed = eventsOf(fork, "counter_change", ref(djinn));
        assertEquals(1, removed.size(), "one trigger, one counter: " + removed);
        assertTrue(eventsOf(fork, "sacrificed", ref(djinn)).isEmpty(), "the Djinn stays in the fork");
    }

    /**
     * {@code life}, 104.2b: Felidar Sovereign's controller at 40 life as the
     * upkeep begins. The live game ends in its win; the fork's 39 life
     * fires no win.
     */
    @Test
    void oneLifeShortOfFortyWinsNothing() throws Throwable {
        Game game = board(PhaseType.UPKEEP);
        Player active = game.getPlayers().get(0);
        permanent(game, active, "Felidar Sovereign");
        active.setLife(40, null);

        List<String> fork = forkOf(game, PhaseType.UPKEEP, "104.2b", "one_short");

        for (String record : fork) {
            assertFalse(RecordLines.eventTypes(record).contains("player_won"), record);
        }
    }

    /**
     * {@code remove_permanent}, 400.6: Rest in Peace as the first main phase
     * begins, and Sneak Attack's Hill Giant to be sacrificed at the end step.
     * The live Giant is exiled instead of put into a graveyard; the fork has
     * no Rest in Peace, and its Giant goes to the graveyard.
     */
    @Test
    void withoutTheReplacementTheSacrificeReachesTheGraveyard() throws Throwable {
        Game game = board(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        Card giant = ForkPlayForwardTest.sneakAttack(game);
        permanent(game, active, "Rest in Peace");

        List<String> fork = forkOf(game, PhaseType.MAIN1, "400.6", "no_replacement");

        List<Map<String, Object>> moves = eventsOf(fork, "zone_change", ref(giant));
        assertTrue(moves.stream().anyMatch(move -> "graveyard".equals(params(move).get("to_zone"))),
                "the fork's Giant goes to the graveyard: " + moves);
        assertTrue(moves.stream().noneMatch(move -> "exile".equals(params(move).get("to_zone"))), moves.toString());
    }

    /**
     * {@code replace_card}, 510.2: A's Grizzly Bears blocked by B's Grizzly
     * Bears as the first-strike damage step begins. The live creatures trade
     * in the regular step; the fork's blocker is Youthful Knight, which kills
     * the attacker in the first-strike step and takes no damage.
     */
    @Test
    void aFirstStrikeBlockerInTheBlockersPlaceKillsFirst() throws Throwable {
        Game game = board(PhaseType.COMBAT_FIRST_STRIKE_DAMAGE);
        Player active = game.getPlayers().get(0);
        Player defending = game.getPlayers().get(1);
        Card attacker = game.getCardsIn(forge.game.zone.ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals("Grizzly Bears")).findFirst().orElseThrow();
        attacker.setSickness(false);
        Card blocker = permanent(game, defending, "Grizzly Bears");
        TestBoard.blockedCombat(game, active, defending, attacker, blocker);

        List<String> fork = forkOf(game, PhaseType.COMBAT_FIRST_STRIKE_DAMAGE, "510.2",
                "knight_in_the_bears_place");

        List<Map<String, Object>> moves = eventsOf(fork, "zone_change", ref(attacker));
        assertTrue(moves.stream().anyMatch(move -> "graveyard".equals(params(move).get("to_zone"))),
                "the attacker dies in the fork: " + moves);
        String knight = null;
        for (String record : fork) {
            for (Map<String, Object> event : events(record)) {
                if ("damage_dealt".equals(event.get("type"))
                        && List.of(ref(attacker)).equals(event.get("subjects"))) {
                    knight = (String) params(event).get("source");
                }
            }
        }
        assertTrue(knight != null && !knight.equals(ref(blocker)), "a new card dealt the attacker's damage");
        assertTrue(eventsOf(fork, "damage_dealt", knight).isEmpty(), "the Knight is dealt no damage");
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

    /**
     * {@code add_permanent}, 601.2c: the active player holds Undo ("return
     * two target creatures") with Grizzly Bears the only creature as its
     * first main phase begins. The live Undo has no legal choice of targets;
     * the fork's opponent also controls Craw Wurm, and the decision records
     * of the fork's priority stops show Undo with a legal choice of targets.
     */
    @Test
    void aSecondCreatureGivesTheTwoTargetSpellALegalChoice() throws Throwable {
        Game game = board(PhaseType.MAIN1);
        Player active = game.getPlayers().get(0);
        for (int i = 0; i < 3; i++) {
            permanent(game, active, "Island");
        }
        TestBoard.inHand(game, active, "Undo");

        List<String> fork = forkOf(game, PhaseType.MAIN1, "601.2c", "two-creatures", true);

        List<Object> verdicts = new ArrayList<>();
        for (String record : fork) {
            if (!"playability".equals(parse(record).get("kind"))
                    || !(RecordLines.payload(record).get("candidates") instanceof List<?> candidates)) {
                continue;
            }
            for (Object candidate : candidates) {
                Map<String, Object> read = RecordLines.asMap(candidate);
                if (read.get("ability") instanceof List<?> keys && keys.stream().anyMatch(key ->
                        "cardsfolder/u/undo.txt".equals(RecordLines.asMap(key).get("script_file")))) {
                    verdicts.add(RecordLines.asMap(read.get("verdict")).get("has_legal_target"));
                }
            }
        }
        assertFalse(verdicts.isEmpty(), "a fork decision record names Undo");
        assertTrue(verdicts.contains(true), "Undo has a legal choice of targets in the fork: " + verdicts);
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
     * {@code life}, 104.4a: Plague Spitter's controller and its opponent at
     * 1 life as the untap step begins. In the live upkeep the trigger takes
     * both players' last life, and the game ends with two losses and no
     * winner, a draw; the fork, taken as the upkeep begins, has the opponent
     * at 2, and its game ends with the active player's loss beside the
     * opponent's win. Both ends are in a resolution's effect half, where
     * 104.4a-1's declaration reads them.
     */
    @Test
    void bothPlayersAtOneLifeDrawAndTheForksOpponentAtTwoWins() throws Throwable {
        Game game = board(PhaseType.UNTAP);
        Player active = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        permanent(game, active, "Plague Spitter");
        active.setLife(1, null);
        opponent.setLife(1, null);

        List<String> records = played(game, null, "104.4a", false);

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
                List.of(true, List.of(List.of(Refs.of(active))), List.of(List.of(Refs.of(opponent)))),
                List.of(false, List.of(List.of(Refs.of(active)), List.of(Refs.of(opponent))), List.of())),
                ends, "the fork's end has a winner, the live end none");
    }

    /**
     * {@code replace_card}, 702.118b: Furtive Homunculus (2/1, skulk)
     * attacks and B's Centaur Courser could block it but for its power, as
     * the declare-blockers step begins. The fork's Courser is Llanowar
     * Elves, which the blockers record of the engine's question before the
     * declaration lists among the Homunculus's legal blockers, whatever B
     * then blocks with.
     */
    @Test
    void anElfOfLesserPowerInTheCoursersPlaceIsALegalBlocker() throws Throwable {
        Game game = board(PhaseType.COMBAT_DECLARE_BLOCKERS);
        Player active = game.getPlayers().get(0);
        Player defending = game.getPlayers().get(1);
        Card homunculus = permanent(game, active, "Furtive Homunculus");
        Card courser = permanent(game, defending, "Centaur Courser");
        Combat combat = new Combat(active);
        combat.addAttacker(homunculus, defending);
        game.getPhaseHandler().setCombat(combat);

        List<String> fork = forkOf(game, PhaseType.COMBAT_DECLARE_BLOCKERS, "702.118b", "lesser_power");

        List<Object> legal = new ArrayList<>();
        for (String record : fork) {
            if ("blockers".equals(parse(record).get("subkind"))
                    && ref(homunculus).equals(RecordLines.payload(record).get("anchor_attacker"))) {
                legal.add(RecordLines.payload(record).get("legal_blockers"));
            }
        }
        assertEquals(1, legal.size(), "one blockers record for the Homunculus: " + legal);
        assertTrue(legal.get(0) instanceof List<?> blockers && blockers.size() == 1
                && !blockers.contains(ref(courser)), "the Elves, a new card, may block: " + legal);
    }
}
