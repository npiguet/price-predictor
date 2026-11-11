package com.pricepredictor.connector.effects;

import forge.ai.simulation.GameCopier;
import forge.ai.simulation.GameStateEvaluator;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.util.MyRandom;

import java.util.Optional;
import java.util.Random;
import java.util.function.BiFunction;
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
     * {@link #mappedAbility}, {@code InterventionFork.forceResolution}'s mode,
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
     * The fork being played on to the end of the turn on this thread, with
     * collectors of its own, or null.
     *
     * <p>{@link #FORK_RUNNING} says a fork runs, which is all a fork that
     * writes its records itself needs: nothing a live collector sees during
     * it is the live game's. A fork {@code ForkPlayForward} plays on is
     * recorded by collectors bound to it, which do need its turn-based
     * outcomes, and this names the game they belong to.
     */
    private static final ThreadLocal<Game> PLAYING_FORWARD = new ThreadLocal<>();

    /** The fork played on to the end of the turn on this thread, or null. */
    static Game playingForwardOnThisThread() {
        return PLAYING_FORWARD.get();
    }

    /** Run {@code action} with {@code fork} marked as played on on this thread; see {@link #PLAYING_FORWARD}. */
    static <T> T playingForward(Game fork, Supplier<T> action) {
        return withThreadLocal(PLAYING_FORWARD, fork, action);
    }

    /**
     * What a forced resolution hears of the hooks while it runs: the
     * resolutions and clauses of its fork, and the mana its resolutions make.
     *
     * <p>The hooks are JVM statics that the live game's handlers own, and
     * each handler drops a fork's callbacks on game identity — rightly, for
     * its own records. A forced resolution needs the same moments the live
     * bracket takes: its resolution's start (FR-042b), each chosen mode's
     * first clause (FR-029f), whether each conditional clause met its
     * condition (FR-034) and the mana it made (FR-042c). So the live handlers
     * hand what they are not keeping to whatever fork is running on their
     * thread, and {@code InterventionFork.forceResolution} installs one for
     * exactly the span of its own resolution.
     */
    interface ForkHooks {
        /** An item of some other game's stack starts to resolve. */
        default void resolutionStarting(SpellAbility item, boolean fizzled) {
        }

        /** A clause of some other game starts to resolve. */
        default void clauseResolving(SpellAbility clause) {
        }

        /**
         * A clause of some other game has finished its own effect.
         *
         * @param run the {@code ClauseRun} constant Forge reported, or null
         */
        default void clauseResolved(SpellAbility clause, String run) {
        }

        /** Mana some other game's resolving spell or trigger made, as its event. */
        default void manaProduced(SpellAbility producer, EffectEvent produced) {
        }
    }

    /**
     * The forced resolution's {@link ForkHooks} on this thread, or null.
     * Thread-local like {@link #FORK_RUNNING}, and for the same reason: the
     * engine runs games on more than one thread.
     */
    private static final ThreadLocal<ForkHooks> FORK_HOOKS = new ThreadLocal<>();

    /** The running fork's {@link ForkHooks}, or a listener that does nothing. */
    static ForkHooks forkHooks() {
        ForkHooks hooks = FORK_HOOKS.get();
        return hooks != null ? hooks : NO_FORK;
    }

    private static final ForkHooks NO_FORK = new ForkHooks() {
    };

    /** Run {@code action} with {@code hooks} hearing what {@link #forkHooks} is handed. */
    static <T> T listeningForHooks(ForkHooks hooks, Supplier<T> action) {
        return withThreadLocal(FORK_HOOKS, hooks, action);
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
     * Whether a fork taken at the boundary beginning {@code phase} can pass
     * its score check. Not at the beginning of combat: the evaluator plays
     * its own copy on to combat damage, and the phase event comes before
     * Forge builds the turn's combat, so declaring attackers on that copy
     * reads a combat that does not exist and the check always fails. A
     * scheduler skips the boundary rather than spend a fork on it.
     */
    static boolean scoreCheckable(PhaseType phase) {
        return phase != PhaseType.COMBAT_BEGIN;
    }

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
            int original = score(live, perspective);
            Player forkPerspective = mappedPlayer(copier, perspective);
            if (forkPerspective == null) {
                return false;
            }
            int copied = score(fork, forkPerspective);
            if (original != copied) {
                System.err.println(
                        "Effect records: discarding a fork whose score "
                                + copied + " disagrees with the live game's "
                                + original + "; it would measure the copier "
                                + "rather than the ability");
                ScoreMismatchReport.print(live, perspective, fork, forkPerspective);
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

    /** The seed every score-check scoring draws from; any fixed value would do. */
    static final long SCORE_SEED = 0x5C0_2EC4L;

    /**
     * {@code game}'s score for {@code perspective}, every random draw taken
     * from a fresh stream seeded with {@link #SCORE_SEED}.
     *
     * <p>Before the second main phase the evaluator scores the combat to come
     * by having the AI declare attackers and blockers in a copy, and those
     * choices draw from Forge's one random stream ({@code MyRandom}): scored
     * from it, the same board can simulate two different combats, so a live
     * game and its exact copy disagreed by a creature's worth and the fork
     * was discarded as a copier error. Both are scored from the same seeded
     * stream instead, and the live game's stream is put back untouched, so
     * the check no longer shifts what the live game draws next either.
     */
    static int score(Game game, Player perspective) {
        return seeded(() -> new GameStateEvaluator().getScoreForGameState(game, perspective).value);
    }

    /** Run {@code action} drawing from a fresh stream seeded with {@link #SCORE_SEED}; see {@link #score}. */
    static <T> T seeded(Supplier<T> action) {
        Random live = MyRandom.getRandom();
        MyRandom.setRandom(new Random(SCORE_SEED));
        try {
            return action.get();
        } finally {
            MyRandom.setRandom(live);
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
     * Every copy is first audited against the live game's board snapshot
     * ({@link ForkFidelity}), whose differences are reported and decide
     * nothing.
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
        ForkFidelity.report(live, fork, copier.getNoCounterpartTally());
        Player from = perspective.get();
        if (from == null || !scoreCheck(live, copier, fork, from)) {
            return Optional.empty();
        }
        return Optional.of(new VerifiedFork(copier, fork, from));
    }

    // ── one fork, start to finish ───────────────────────────────────────

    /**
     * What a fork does once the copy has passed its score check: map the
     * live objects it needs into the copy and run on it.
     *
     * @param <R> what the run produced, for the fork's writer
     */
    @FunctionalInterface
    interface ForkBody<R> {
        /**
         * @param snapshots reads the copy's state, the live game beside it
         * @return what the run produced, or null when the fork has nothing
         *         to write: an object the copy could not map, or a run that
         *         observed nothing
         */
        R run(VerifiedFork verified, SnapshotBuilder snapshots);
    }

    /**
     * One fork's whole life, in the order every kind of fork takes it: mark
     * the thread ({@link #asFork}), spend the budget, copy and score-check
     * ({@link #copyAndVerify}), run the fork's body under the guard, and hand
     * what it produced to the fork's writer, which stamps its own flags.
     *
     * <p>The budget is spent before the score check, so a discarded fork
     * still counts: otherwise a systematically failing copy would retry
     * until the game ended. The guard catches {@code RuntimeException} and
     * {@code StackOverflowError}: a fork reaches states ordinary play does
     * not (an ability resolving with no legal target, a combat without a
     * stripped keyword), and a card that throws there is one this fork
     * cannot describe. Discarding is right; failing the worker is not. The
     * writer runs outside the guard, so a record that cannot be written
     * fails as a live one does.
     *
     * @param live        the game the fork copies
     * @param spend       spends the fork's budget
     * @param perspective the live player the copy's score is checked from,
     *                    read after the copy
     */
    record ForkRun(Game live, Runnable spend, Supplier<Player> perspective) {

        /**
         * Take the fork.
         *
         * @return what {@code write} returned, or null when the copy failed
         *         its check, the body threw, or the body produced nothing
         */
        <R, W> W take(ForkBody<R> body, BiFunction<VerifiedFork, R, W> write) {
            // The whole span is the fork, not just its run: see FORK_RUNNING
            // for what a narrower scope left uncovered.
            return asFork(() -> {
                spend.run();
                Optional<VerifiedFork> verified = copyAndVerify(live, perspective);
                if (verified.isEmpty()) {
                    return null;
                }
                // The fork's own state, read against the live game it came from.
                SnapshotBuilder snapshots = new SnapshotBuilder(verified.get().fork(), live);
                R produced;
                try {
                    produced = body.run(verified.get(), snapshots);
                } catch (RuntimeException | StackOverflowError e) {
                    return null;
                }
                return produced == null ? null : write.apply(verified.get(), produced);
            });
        }
    }

    // ── mapping live objects into the copy ──────────────────────────────

    /**
     * The fork's copy of a live ability, or null when it cannot be matched:
     * the copier's own structural mapper ({@code GameCopier.counterpartAbility},
     * by the ability's place on its host), the one every fork maps abilities
     * through.
     */
    static SpellAbility mappedAbility(GameCopier copier, SpellAbility ability) {
        try {
            return copier.counterpartAbility(ability);
        } catch (RuntimeException e) {
            return null;
        }
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
