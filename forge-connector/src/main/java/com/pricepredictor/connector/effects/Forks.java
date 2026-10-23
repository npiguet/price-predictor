package com.pricepredictor.connector.effects;

import forge.ai.simulation.GameCopier;
import forge.ai.simulation.GameStateEvaluator;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

import java.util.Optional;
import java.util.function.Consumer;
import java.util.function.Supplier;

/**
 * The machinery shared by stage three's two kinds of fork, which record what
 * observation cannot reach.
 *
 * <p>Two kinds of fork, and the flags tell them apart on their own: an
 * {@link InterventionFork interventional resolution} forces an ability Forge's
 * AI never chose to use, and a {@link DamageStepProbe damage-step probe}
 * re-runs one combat with a keyword stripped. {@link ForkBudget} decides how
 * many of each a game may take. What both need is here: copying the live game
 * and checking the copy, mapping live objects into it, and marking the thread
 * while a fork runs.
 *
 * <p><b>Every fork is score-checked against the live game before any
 * perturbation.</b> A copy that already disagrees with the game it came from is
 * measuring the copier rather than the ability, and a discarded fork still
 * counts against its budget — otherwise a systematically-failing copy would
 * retry forever.
 */
final class Forks {

    private Forks() {
    }

    /**
     * Marks this thread as running one of the two forks -- an
     * {@link InterventionFork#intervene interventional resolution} or a
     * {@link DamageStepProbe#probe damage-step probe} -- for {@code
     * LiveGame.ownsNullAbilityOutcome()}.
     *
     * <p>It covers both methods' <em>entire</em> bodies, not just the forced
     * resolution or {@code probe}'s {@code Combat.dealAssignedDamage}: the
     * resolving-clause pointer is null for every segment of a fork outside
     * {@code AbilityUtils.resolve} -- the copy, the score check and
     * {@link #findInFork}, {@code InterventionFork.forceResolution}'s mode,
     * target and X draws, and {@code GameSimulator.resolveStack}'s own
     * state-based check and trigger pass. A null-ability outcome fired in any
     * of them would otherwise be indistinguishable from the live game's own
     * turn-based actions. So "a fork is running on this thread" is a
     * sufficient answer on its own, and {@code ownsNullAbilityOutcome} checks
     * it first, with the pointer behind it as a second, independent signal.
     *
     * <p>Saved and restored rather than set and cleared, as {@code
     * AbilityUtils}' {@code EFFECT_RECORD_SUB_ABILITY} is: correct without
     * relying on the two methods never nesting.
     */
    private static final ThreadLocal<Boolean> FORK_RUNNING =
            ThreadLocal.withInitial(() -> Boolean.FALSE);

    /** Whether an intervention or a probe is running on this thread. */
    static boolean isForkRunningOnThisThread() {
        return FORK_RUNNING.get();
    }

    /** Run {@code action} with this thread marked as running a fork; see {@link #FORK_RUNNING}. */
    static <T> T asFork(Supplier<T> action) {
        return withThreadLocal(FORK_RUNNING, true, action);
    }

    /**
     * Where a forced resolution's clauses report in, while one is running.
     *
     * <p>The clause hook is a JVM static that {@code ClauseOutcomeRouter} owns,
     * and the live collector drops a fork's clauses on game identity — rightly,
     * for its own bracket. A fork that splits a modal resolution into per-mode
     * halves (FR-029f) needs those same clauses, so the live handler hands a
     * clause it is not keeping to whatever fork is running on its thread, and
     * {@code InterventionFork.forceResolution} installs the consumer for
     * exactly the span of its own resolution. Thread-local like
     * {@link #FORK_RUNNING}, and for the same reason: the engine runs games on
     * more than one thread.
     */
    private static final ThreadLocal<Consumer<SpellAbility>> CLAUSE_LISTENER =
            new ThreadLocal<>();

    /** A clause of some other game is resolving; the running fork may want it. */
    static void noteClauseResolving(SpellAbility clause) {
        Consumer<SpellAbility> listener = CLAUSE_LISTENER.get();
        if (listener != null) {
            listener.accept(clause);
        }
    }

    /** Run {@code action} with {@code listener} receiving the clauses {@link #noteClauseResolving} is handed. */
    static <T> T listeningForClauses(Consumer<SpellAbility> listener, Supplier<T> action) {
        return withThreadLocal(CLAUSE_LISTENER, listener, action);
    }

    /**
     * Run {@code action} with {@code local} set to {@code value}, and restore
     * what it held before however the action ends: correct without relying
     * on two such spans never nesting.
     */
    private static <V, T> T withThreadLocal(ThreadLocal<V> local, V value, Supplier<T> action) {
        V was = local.get();
        local.set(value);
        try {
            return action.get();
        } finally {
            local.set(was);
        }
    }

    // ── the score check ─────────────────────────────────────────────────

    /**
     * Verify a fresh copy scores the same as the game it came from.
     *
     * <p>Run <b>before any perturbation</b>: a copy that already disagrees is
     * measuring the copier, and a record from it would be noise dressed as a
     * counterfactual. A mismatch logs a warning, discards the fork, and
     * <b>still counts against the budget</b> — otherwise a systematically
     * failing copy would retry until the game ended.
     */
    static boolean scoreCheck(Game live, GameCopier copier, Game fork, Player perspective) {
        try {
            GameStateEvaluator evaluator = new GameStateEvaluator();
            int original = evaluator.getScoreForGameState(live, perspective).value;
            Player forkPerspective = mappedPlayer(copier, perspective);
            if (forkPerspective == null) {
                return false;
            }
            int copied = evaluator
                    .getScoreForGameState(fork, forkPerspective).value;
            if (original != copied) {
                System.err.println(
                        "Effect records: discarding a fork whose score "
                                + copied + " disagrees with the live game's "
                                + original + "; it would measure the copier "
                                + "rather than the ability");
                return false;
            }
            return true;
        } catch (RuntimeException e) {
            System.err.println(
                    "Effect records: fork score check failed (" + e.getMessage()
                            + "); discarding");
            return false;
        }
    }

    /**
     * A copy of the live game that scored as the original does.
     *
     * @param copier      maps the live game's objects to the copy's
     * @param perspective the live player the score was checked from
     */
    record VerifiedFork(GameCopier copier, Game fork, Player perspective) {
    }

    /**
     * Copy the live game and score-check the copy, empty when either fails.
     *
     * @param perspective read after the copy, because a game the copier cannot
     *                    read may not answer it either
     */
    static Optional<VerifiedFork> copyAndVerify(Game live, Supplier<Player> perspective) {
        GameCopier copier;
        Game fork;
        try {
            // Both inside the guard: the copier's own constructor reads the
            // game, so a game it cannot read fails here rather than at makeCopy.
            copier = new GameCopier(live);
            fork = copier.makeCopy();
        } catch (RuntimeException e) {
            return Optional.empty();
        }
        Player from = perspective.get();
        if (from == null || !scoreCheck(live, copier, fork, from)) {
            return Optional.empty();
        }
        return Optional.of(new VerifiedFork(copier, fork, from));
    }

    // ── mapping live objects into the copy ──────────────────────────────

    /**
     * The fork's copy of a live ability, or null when it cannot be matched.
     *
     * <p>The host card maps directly through the copier. The ability itself
     * does not, so it is matched by description among the copy's — the same
     * approach Forge's own simulator takes.
     */
    static SpellAbility findInFork(GameCopier copier, SpellAbility ability) {
        Card host;
        try {
            host = copier.find(ability.getHostCard());
        } catch (RuntimeException e) {
            return null;
        }
        if (host == null) {
            return null;
        }
        String description = ability.getDescription();
        for (SpellAbility candidate : host.getSpellAbilities()) {
            if (candidate.getDescription().equals(description)) {
                return candidate;
            }
        }
        return null;
    }

    static Player mappedPlayer(GameCopier copier, Player player) {
        try {
            return copier.find(player);
        } catch (RuntimeException e) {
            return null;
        }
    }

    static Card mappedCard(GameCopier copier, Card card) {
        try {
            return copier.find(card);
        } catch (RuntimeException e) {
            return null;
        }
    }
}
