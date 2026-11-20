package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.CollectionCaps;
import com.pricepredictor.connector.effects.RecordShardWriter;
import com.pricepredictor.connector.effects.ShardId;
import com.pricepredictor.connector.effects.TestCaps;
import com.pricepredictor.connector.effects.TestInterventions;
import forge.deck.Deck;
import forge.game.Game;
import forge.game.Match;
import forge.game.phase.PhaseType;
import forge.util.MyRandom;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Where real games spend their interventions (D3): spread over the game
 * rather than at its start, and each forced line at a step's first priority,
 * on a board the rules give priority on. Seeded burn and boros mirrors played
 * through {@link GamePlayer}, the scheduler's forced lines read off its
 * watch and their records off the games' shards.
 */
@ExtendWith(ForgeExtension.class)
class InterventionSpreadTest {

    @TempDir
    Path dir;

    @AfterEach
    void stopWatching() {
        TestInterventions.watch(null);
    }

    /** One forced line: the game turn and step it was forced at. */
    private record Forced(int turn, PhaseType phase, String line) {
        @Override
        public String toString() {
            return turn + ":" + phase + ":" + line;
        }
    }

    /**
     * The steps a turn with no attacker skips (508.8). Its declare-attackers
     * and end-of-combat steps still give priority. The seeded games force at
     * none of them, so the check on these steps is made by {@code
     * ScheduledInterventionsTest} on a real declared attack.
     */
    private static final Set<String> NEEDS_AN_ATTACKER = Set.of(
            "combat_declare_blockers", "combat_first_strike_damage", "combat_damage");

    /** Every forced record the games played so far wrote, parsed. */
    private final List<Map<String, Object>> records = new ArrayList<>();

    /** One game's forced lines, and how many turns it lasted. */
    private record Played(List<Forced> forced, int turns) {
    }

    private static Deck deck(Object... cardsAndCounts) {
        Deck deck = new Deck();
        for (int i = 0; i < cardsAndCounts.length; i += 2) {
            deck.getMain().add((String) cardsAndCounts[i], (Integer) cardsAndCounts[i + 1]);
        }
        return deck;
    }

    private static Deck burn() {
        return deck("Mountain", 16, "Lightning Bolt", 24);
    }

    /** Sorceries, creatures and an aura beside instants: lines with both timings. */
    private static Deck boros() {
        return deck("Mountain", 9, "Plains", 8, "Pacifism", 2, "Serra Angel", 2, "Arc Trail", 2,
                "Shock", 2, "Youthful Knight", 2, "Lightning Bolt", 2, "Boros Charm", 2, "Fireball", 1,
                "Healing Salve", 1, "Raise the Alarm", 1, "Prodigal Pyromancer", 1, "Holy Day", 1,
                "Fencing Ace", 2, "Mogg Fanatic", 1, "Hill Giant", 1);
    }

    private static final int BUDGET = 2;

    private List<Played> play(String name, Deck deck, long... seeds) throws Exception {
        CollectionCaps caps = TestCaps.caps().interventions(BUDGET).probes(0).build();
        List<Played> played = new ArrayList<>();
        for (long seed : seeds) {
            Map<Game, List<Forced>> byGame = new IdentityHashMap<>();
            TestInterventions.watch((line, holder) -> {
                Game game = holder.getGame();
                byGame.computeIfAbsent(game, g -> new ArrayList<>()).add(new Forced(
                        game.getPhaseHandler().getTurn(), game.getPhaseHandler().getPhase(),
                        line.getHostCard().getName()));
            });
            Path out = Files.createDirectory(dir.resolve(name + seed));
            RecordShardWriter writer = new RecordShardWriter(out, new ShardId("run", 0, "l1"));
            Random was = MyRandom.getRandom();
            MyRandom.setRandom(new Random(seed));
            List<Game> games = new ArrayList<>();
            try {
                new GamePlayer(new GamePlayer.Config(1, writer, caps, WorkerStop.NEVER, (match, game) -> {
                    games.add(game);
                    match.startGame(game);
                }, () -> 1_700_000_000_000L + seed * 7919)).playMatch(deck, deck);
            } finally {
                MyRandom.setRandom(was);
                writer.close();
            }
            records.addAll(TestInterventions.forcedRecords(writer.path()));
            for (Game game : games) {
                played.add(new Played(byGame.getOrDefault(game, List.of()),
                        game.getPhaseHandler().getTurn()));
            }
        }
        return played;
    }

    private static double median(List<Double> values) {
        List<Double> sorted = values.stream().sorted().toList();
        int n = sorted.size();
        return n % 2 == 1 ? sorted.get(n / 2) : (sorted.get(n / 2 - 1) + sorted.get(n / 2)) / 2;
    }

    /** The last forced line's game turn as a share of the game's turns, for each game that forced one. */
    private static List<Double> lastShares(List<Played> games) {
        List<Double> shares = new ArrayList<>();
        for (Played game : games) {
            if (!game.forced().isEmpty()) {
                shares.add(game.forced().get(game.forced().size() - 1).turn() / (double) game.turns());
            }
        }
        return shares;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> map(Object value) {
        return (Map<String, Object>) value;
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> maps(Object value) {
        return (List<Map<String, Object>>) value;
    }

    private static Map<String, Object> global(Map<String, Object> record) {
        return map(map(record.get("state")).get("global"));
    }

    /** The entity the record's {@code refs.source} names, or null. */
    private static Map<String, Object> source(Map<String, Object> record) {
        Map<String, Object> state = map(record.get("state"));
        Object ref = map(state.get("refs")).get("source");
        return maps(state.get("entities")).stream()
                .filter(entity -> entity.get("id").equals(ref)).findFirst().orElse(null);
    }

    /**
     * A forced record's board is one its step gives priority on: a step that
     * needs an attacker shows one, at a draw step the player whose turn it is
     * has drawn, and a spell without flash other than an instant is
     * resolving in a main phase of its caster's turn (none of these decks'
     * spells has flash).
     */
    private static void assertTheBoardGivesPriority(Map<String, Object> record) {
        Map<String, Object> state = map(record.get("state"));
        Object phase = global(record).get("phase");
        Map<String, Object> source = source(record);
        if (source != null && "stack".equals(source.get("zone"))
                && !((List<?>) source.get("types")).contains("instant")) {
            assertTrue(("main1".equals(phase) || "main2".equals(phase))
                            && global(record).get("active").equals(record.get("actor_player")),
                    "a sorcery-speed spell forced outside its caster's main phase: " + record);
        }
        if (NEEDS_AN_ATTACKER.contains(phase)) {
            assertTrue(maps(state.get("entities")).stream()
                            .anyMatch(entity -> entity.get("combat") != null
                                    && map(entity.get("combat")).get("attacking") != null),
                    "a combat step's forced record with no attacker: " + record);
        }
        if ("draw".equals(phase)) {
            Map<String, Object> active = maps(state.get("players")).stream()
                    .filter(player -> player.get("id").equals(global(record).get("active")))
                    .findFirst().orElseThrow();
            assertTrue(((Number) map(active.get("turn_counts")).get("cards_drawn")).longValue() > 0,
                    "a draw step's forced record before the draw: " + record);
        }
    }

    @Test
    void realGamesSpendTheirInterventionsOverTheGameAndOnlyWhereTheLineIsPlayable() throws Exception {
        List<Played> mixed = play("mixed", boros(), 21, 22, 23, 24, 25, 26, 27, 28);
        List<Played> burn = play("burn", burn(), 21, 22, 23, 24, 25);
        System.out.println("InterventionSpreadTest mixed: " + mixed);
        System.out.println("InterventionSpreadTest burn: " + burn);

        for (Map<String, Object> record : records) {
            assertTheBoardGivesPriority(record);
        }
        assertTrue(records.size() >= mixed.size(), "too few forced records to check: " + records.size());
        long spent = mixed.stream().filter(game -> game.forced().size() == BUDGET).count();
        assertTrue(spent * 2 > mixed.size(), "the budget is spent in most mixed games: " + mixed);
        // A burn mirror holds one line, Lightning Bolt, forced once a game at most.
        long bolted = burn.stream().filter(game -> game.forced().size() == 1).count();
        assertTrue(bolted * 2 > burn.size(), "the one line is forced in most burn games: " + burn);
        for (Played game : burn) {
            assertTrue(game.forced().size() <= 1, burn.toString());
        }
        List<Double> shares = lastShares(mixed);
        shares.addAll(lastShares(burn));
        assertTrue(median(shares) > 0.5,
                "the last intervention lies past the game's midpoint, at the median: " + shares);
    }
}
