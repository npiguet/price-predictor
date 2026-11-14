package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.CollectionCaps;
import com.pricepredictor.connector.effects.RecordShardWriter;
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

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Where real games spend their interventions (D3): spread over the game
 * rather than at its start, and each forced line at a boundary where its
 * holder could play it. Seeded burn and boros mirrors played through
 * {@link GamePlayer}, the scheduler's forced lines read off its watch.
 */
@ExtendWith(ForgeExtension.class)
class InterventionSpreadTest {

    @TempDir
    Path dir;

    @AfterEach
    void stopWatching() {
        TestInterventions.watch(null);
    }

    /** One forced line: the game turn and step it was forced at, and whether its timing allowed it. */
    private record Forced(int turn, PhaseType phase, boolean timely, String line) {
        @Override
        public String toString() {
            return turn + ":" + phase + (timely ? "" : "!") + ":" + line;
        }
    }

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
                PhaseType phase = game.getPhaseHandler().getPhase();
                boolean sorcerySpeed = (line.isSpell() && !line.withFlash(line.getHostCard(), holder))
                        || line.isPwAbility() || line.getRestrictions().isSorcerySpeed();
                boolean priority = phase != PhaseType.UNTAP && phase != PhaseType.CLEANUP;
                boolean timely = priority && (!sorcerySpeed || (phase.isMain()
                        && game.getPhaseHandler().isPlayerTurn(holder) && game.getStack().isEmpty()));
                byGame.computeIfAbsent(game, g -> new ArrayList<>()).add(new Forced(
                        game.getPhaseHandler().getTurn(), game.getPhaseHandler().getPhase(), timely,
                        line.getHostCard().getName()));
            });
            Path out = Files.createDirectory(dir.resolve(name + seed));
            RecordShardWriter writer = new RecordShardWriter(out, "run", 0, "l1");
            Random was = MyRandom.getRandom();
            MyRandom.setRandom(new Random(seed));
            List<Game> games = new ArrayList<>();
            try {
                new GamePlayer(1, writer, caps, WorkerStop.NEVER, (match, game) -> {
                    games.add(game);
                    match.startGame(game);
                }, () -> 1_700_000_000_000L + seed * 7919).playMatch(deck, deck);
            } finally {
                MyRandom.setRandom(was);
                writer.close();
            }
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

    @Test
    void realGamesSpendTheirInterventionsOverTheGameAndOnlyWhereTheLineIsPlayable() throws Exception {
        List<Played> mixed = play("mixed", boros(), 21, 22, 23, 24, 25, 26, 27, 28);
        List<Played> burn = play("burn", burn(), 21, 22, 23, 24, 25);
        System.out.println("InterventionSpreadTest mixed: " + mixed);
        System.out.println("InterventionSpreadTest burn: " + burn);

        for (Played game : mixed) {
            for (Forced forced : game.forced()) {
                assertTrue(forced.timely(), "a line forced where its holder could not play it: " + mixed);
            }
        }
        for (Played game : burn) {
            for (Forced forced : game.forced()) {
                assertTrue(forced.timely(), "a line forced where its holder could not play it: " + burn);
            }
        }
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
