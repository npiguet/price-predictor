package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.DirectedRules;
import forge.game.Game;
import forge.game.phase.PhaseType;

import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import java.util.Set;

/**
 * When a directed fork is taken, and for which rule (FR-010a, D8).
 *
 * <p>Every listed rule has a probability of its own ({@link
 * CollectionCaps#directedForkProbability(String)}). At each phase boundary
 * the enabled rules are considered in descending order of probability, ties
 * in their {@code --directed-fork-rules} order: each that has not forked in
 * this game and whose board precondition holds draws against its
 * probability, from a generator seeded per game, and one whose draw forks
 * takes a fork while {@code --directed-forks-per-game} allows. So the
 * budget goes first to the rule that needs it most, and a rule at
 * probability 0 never forks. A rule forks at most once a game, and a fork
 * counts against the budget and the rule's one fork the moment it is taken,
 * kept or discarded: otherwise a copy that systematically fails its score
 * check would retry at every boundary until the game ended.
 *
 * <p>Rules that share a board, a binding and an edit are one registration
 * ({@link DirectedRule#alsoFor}), considered as one: listed once or several
 * times, it draws at the largest probability among its listed rules, at the
 * place of the first listed, forks at most once a game, spends one slot of
 * the budget, and is counted in the tally under each of its rules.
 *
 * <p>The rules are still considered once the game's budget is spent, each
 * drawing as before, so the tally ({@link DirectedForkTally}) counts how
 * often a rule's board appeared and drew a fork the budget no longer
 * allowed. Nothing else reads the generator once the budget is spent.
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

    /**
     * What the run directs: the caps that list and weigh its rules, the
     * registry that holds them, and the game's seed.
     *
     * @param registry the registered rules; a rule the run names and the
     *                 registry lacks takes no fork (the supervisor refuses
     *                 one before any worker starts)
     * @param seed     the game's seed
     */
    record Directing(CollectionCaps caps, DirectedRules registry, long seed) {
    }

    /** An enabled registration and the probability it forks with. */
    private record Candidate(DirectedRule rule, double probability) {
    }

    private final Game game;
    /** The registrations of the run's rules, in descending probability, ties in list order. */
    private final List<Candidate> enabled;
    private final ForkBudget budget;
    private final DirectedForkTally tally;
    private final Random random;
    private final Fork fork;
    private final Set<DirectedRule> forkedThisGame = new HashSet<>();

    DirectedForkScheduler(
            Game game, Directing directing, ForkBudget budget, DirectedForkTally tally, Fork fork) {
        this.game = game;
        this.enabled = enabled(directing.caps(), directing.registry());
        this.budget = budget;
        this.tally = tally;
        this.random = SeedStreams.random(directing.seed(), SeedStreams.Consumer.DIRECTED_FORKS);
        this.fork = fork;
    }

    /**
     * The registrations of the run's rules, each once, at its first listed
     * rule's place, in descending probability; a stable sort, so ties keep
     * their {@code --directed-fork-rules} order.
     */
    private static List<Candidate> enabled(CollectionCaps caps, DirectedRules registry) {
        return caps.directedForkRules().stream()
                .map(registry::find)
                .flatMap(Optional::stream)
                .distinct()
                .map(rule -> new Candidate(rule, probability(caps, rule)))
                .sorted(Comparator.comparingDouble(Candidate::probability).reversed())
                .toList();
    }

    /**
     * The largest probability among the registration's listed rules: its one
     * fork serves each of them, so a rule whose side is short is not held
     * back by another already full.
     */
    private static double probability(CollectionCaps caps, DirectedRule rule) {
        return rule.rules().stream()
                .filter(caps.directedForkRules()::contains)
                .mapToDouble(caps::directedForkProbability)
                .max().orElseThrow();
    }

    /**
     * Take this boundary's directed forks.
     *
     * <p>Never where the score check cannot pass ({@link Forks#scoreCheckable}):
     * a fork taken there would spend the rule's one fork of the game on
     * nothing. A rule about combat forks as attackers are about to be
     * declared instead.
     *
     * @param phase the phase this boundary begins
     * @return how many forks were taken, kept or discarded
     */
    int collect(PhaseType phase) {
        if (!Forks.scoreCheckable(phase) || game.isGameOver()) {
            return 0;
        }
        int taken = 0;
        for (Candidate candidate : enabled) {
            Optional<Binding> drawn = draw(candidate);
            if (drawn.isPresent() && take(candidate.rule(), drawn.get())) {
                taken++;
            }
        }
        return taken;
    }

    /**
     * The registration's binding when it has not forked this game, its
     * precondition holds and its draw forks, else empty; counted in the
     * tally under each of its rules.
     */
    private Optional<Binding> draw(Candidate candidate) {
        DirectedRule rule = candidate.rule();
        if (forkedThisGame.contains(rule)) {
            return Optional.empty();
        }
        Optional<Binding> binding = rule.precondition(
                game, failure -> tally.preconditionThrew(rule.rules(), failure));
        if (binding.isEmpty()) {
            return binding;
        }
        count(rule, DirectedForkTally.Count.MATCHED);
        // Drawn only where the precondition holds, so a game's draws depend
        // on its own boards and nothing else.
        if (random.nextDouble() >= candidate.probability()) {
            return Optional.empty();
        }
        count(rule, DirectedForkTally.Count.DRAWN);
        return binding;
    }

    /** One more {@code count} under each of the registration's rules. */
    private void count(DirectedRule rule, DirectedForkTally.Count count) {
        rule.rules().forEach(id -> tally.add(id, count));
    }

    /**
     * Take the registration's fork when the game's budget allows and its forced
     * play, if any, can be made on the live board.
     *
     * @return whether the fork was taken, kept or discarded
     */
    private boolean take(DirectedRule rule, Binding binding) {
        if (!budget.mayDirect()) {
            count(rule, DirectedForkTally.Count.BUDGET_BLOCKED);
            return false;
        }
        String variant = variantOf(rule);
        Application application = rule.variants().get(variant);
        // A forced play the live board cannot offer (FR-010b) is refused
        // before the fork is taken, so the budget is not spent on it.
        Optional<String> refused = ForcedPlayRun.dryRun(application, binding, game);
        if (refused.isPresent()) {
            ForkRefusals.count(DirectedFork.KIND, refused.get());
            return false;
        }
        budget.spendDirected();
        forkedThisGame.add(rule);
        count(rule, DirectedForkTally.Count.TAKEN);
        if (fork.take(rule, variant, application, binding)) {
            count(rule, DirectedForkTally.Count.KEPT);
        }
        return true;
    }

    /** One of the rule's variants, drawn from the game's generator when it has several. */
    private String variantOf(DirectedRule rule) {
        List<String> labels = List.copyOf(rule.variants().keySet());
        return labels.size() == 1 ? labels.get(0) : labels.get(random.nextInt(labels.size()));
    }
}
