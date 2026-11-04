package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.DirectedRules;
import forge.game.Game;
import forge.game.phase.PhaseType;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import java.util.Set;

/**
 * When a directed fork is taken, and for which rule (FR-010a).
 *
 * <p>At each phase boundary, every enabled rule that has not forked in this
 * game and whose board precondition holds draws against
 * {@code --directed-fork-probability}, from a generator seeded per game. The
 * rules that draw a fork are tried in ascending order of the directed forks
 * each has taken in the worker JVM's lifetime, ties broken by their order in
 * {@code --directed-fork-rules}, until {@code --directed-forks-per-game} is
 * spent. A rule forks at most once a game, and a fork counts against the
 * budget, the rule's one fork and the lifetime tally the moment it is taken,
 * kept or discarded: otherwise a copy that systematically fails its score
 * check would retry at every boundary until the game ended.
 *
 * <p>The probability is what spreads a rule's one fork over a game: a
 * precondition that holds at many boundaries would otherwise fork at the
 * first every time, and the corpus would see only that board.
 *
 * <p>Prior art: {@link InterventionScheduler} (once per line per game) and
 * {@link ProbeScheduler} (its per-game keyword rotation), both run from the
 * same phase bridge.
 */
final class DirectedForkScheduler {

    /** Takes one fork; the scheduler has already spent its budget. */
    @FunctionalInterface
    interface Fork {
        /** @return whether the fork was kept */
        boolean take(DirectedRule rule, String variant, Application application, Binding binding);
    }

    /** Kept apart from the other streams the same game seed seeds. */
    private static final long SEED_SALT = 0x2545F4914F6CDD1DL;

    private final Game game;
    /** The run's rules that are registered, in {@code --directed-fork-rules} order. */
    private final List<DirectedRule> enabled;
    private final double probability;
    private final ForkBudget budget;
    private final DirectedForkTally tally;
    private final Random random;
    private final Fork fork;
    private final Set<String> forkedThisGame = new HashSet<>();

    /**
     * @param registry the registered rules; a rule the run names and the
     *                 registry lacks takes no fork (the supervisor refuses
     *                 one before any worker starts)
     * @param seed     the game's seed
     */
    DirectedForkScheduler(
            Game game, CollectionCaps caps, DirectedRules registry, ForkBudget budget,
            DirectedForkTally tally, long seed, Fork fork) {
        this.game = game;
        this.enabled = caps.directedForkRules().stream()
                .distinct()
                .map(registry::find)
                .flatMap(Optional::stream)
                .toList();
        this.probability = caps.directedForkProbability();
        this.budget = budget;
        this.tally = tally;
        this.random = new Random(GameRecording.scramble(seed ^ SEED_SALT));
        this.fork = fork;
    }

    /** One enabled rule whose precondition held and whose draw forks. */
    private record Match(DirectedRule rule, Binding binding) {
    }

    /**
     * Take this boundary's directed forks.
     *
     * <p>Never at the beginning of combat: the score check reads a combat
     * Forge has not built yet there, so the copy always disagrees, and a fork
     * taken there would spend the rule's one fork of the game on nothing. A
     * rule about combat forks as attackers are about to be declared instead.
     *
     * @param phase the phase this boundary begins
     * @return how many forks were taken, kept or discarded
     */
    int collect(PhaseType phase) {
        if (phase == PhaseType.COMBAT_BEGIN || game.isGameOver() || !budget.mayDirect()) {
            return 0;
        }
        List<Match> matching = new ArrayList<>();
        for (DirectedRule rule : enabled) {
            if (forkedThisGame.contains(rule.rule())) {
                continue;
            }
            Optional<Binding> binding = rule.precondition(game);
            // Drawn only where the precondition holds, so a game's draws
            // depend on its own boards and nothing else.
            if (binding.isPresent() && random.nextDouble() < probability) {
                matching.add(new Match(rule, binding.get()));
            }
        }
        // A stable sort: ties keep their --directed-fork-rules order.
        matching.sort(Comparator.comparingInt(match -> tally.taken(match.rule().rule())));
        int taken = 0;
        for (Match match : matching) {
            if (!budget.mayDirect()) {
                break;
            }
            DirectedRule rule = match.rule();
            budget.spendDirected();
            forkedThisGame.add(rule.rule());
            tally.count(rule.rule());
            String variant = variantOf(rule);
            fork.take(rule, variant, rule.variants().get(variant), match.binding());
            taken++;
        }
        return taken;
    }

    /** One of the rule's variants, drawn from the game's generator when it has several. */
    private String variantOf(DirectedRule rule) {
        List<String> labels = List.copyOf(rule.variants().keySet());
        return labels.size() == 1 ? labels.get(0) : labels.get(random.nextInt(labels.size()));
    }
}
