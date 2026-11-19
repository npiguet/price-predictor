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

    /** An enabled rule and the probability it forks with. */
    private record Candidate(DirectedRule rule, double probability) {
        String id() {
            return rule.rule();
        }
    }

    private final Game game;
    /** The run's rules that are registered, in descending probability, ties in list order. */
    private final List<Candidate> enabled;
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
        // A stable sort: ties keep their --directed-fork-rules order.
        this.enabled = caps.directedForkRules().stream()
                .distinct()
                .map(registry::find)
                .flatMap(Optional::stream)
                .map(rule -> new Candidate(rule, caps.directedForkProbability(rule.rule())))
                .sorted(Comparator.comparingDouble(Candidate::probability).reversed())
                .toList();
        this.budget = budget;
        this.tally = tally;
        this.random = SeedStreams.random(seed, SeedStreams.Consumer.DIRECTED_FORKS);
        this.fork = fork;
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
     * The rule's binding when it has not forked this game, its precondition
     * holds and its draw forks, else empty; counted in the tally.
     */
    private Optional<Binding> draw(Candidate candidate) {
        String rule = candidate.id();
        if (forkedThisGame.contains(rule)) {
            return Optional.empty();
        }
        Optional<Binding> binding = candidate.rule().precondition(
                game, failure -> tally.preconditionThrew(rule, failure));
        if (binding.isEmpty()) {
            return binding;
        }
        tally.add(rule, DirectedForkTally.Count.MATCHED);
        // Drawn only where the precondition holds, so a game's draws depend
        // on its own boards and nothing else.
        if (random.nextDouble() >= candidate.probability()) {
            return Optional.empty();
        }
        tally.add(rule, DirectedForkTally.Count.DRAWN);
        return binding;
    }

    /**
     * Take the rule's fork when the game's budget allows and its forced
     * play, if any, can be made on the live board.
     *
     * @return whether the fork was taken, kept or discarded
     */
    private boolean take(DirectedRule rule, Binding binding) {
        if (!budget.mayDirect()) {
            tally.add(rule.rule(), DirectedForkTally.Count.BUDGET_BLOCKED);
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
        forkedThisGame.add(rule.rule());
        tally.add(rule.rule(), DirectedForkTally.Count.TAKEN);
        if (fork.take(rule, variant, application, binding)) {
            tally.add(rule.rule(), DirectedForkTally.Count.KEPT);
        }
        return true;
    }

    /** One of the rule's variants, drawn from the game's generator when it has several. */
    private String variantOf(DirectedRule rule) {
        List<String> labels = List.copyOf(rule.variants().keySet());
        return labels.size() == 1 ? labels.get(0) : labels.get(random.nextInt(labels.size()));
    }
}
