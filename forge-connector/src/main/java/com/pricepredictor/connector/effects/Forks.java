package com.pricepredictor.connector.effects;

import forge.ai.simulation.GameCopier;
import forge.ai.simulation.GameStateEvaluator;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.util.MyRandom;

import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import java.util.TreeMap;
import java.util.function.BiFunction;
import java.util.function.Supplier;

/**
 * The machinery shared by stage three's four kinds of fork, which record what
 * observation cannot reach.
 *
 * <p>Four kinds of fork: an {@link InterventionFork interventional
 * resolution} forces an ability Forge's AI never chose to use; a {@link
 * DamageStepProbe damage-step probe} re-runs one combat with a keyword
 * stripped; a {@link ProtectionFork protection fork} grants a creature a rare
 * protection and plays the turn on; a {@link DirectedFork directed fork}
 * edits the board as a rule's variant does and plays the turn on.
 * {@link ForkBudget} decides how many of each a game may take. What all four
 * need is here: copying the live game and checking the copy, mapping live
 * objects into it, marking the thread while a fork runs, and making sure a
 * fork leaves the live game as it was ({@link LiveMarks}).
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
     * target and X draws, and the forced resolution's own state-based check
     * and trigger pass. A null-ability outcome fired in any
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
     * The running probe's {@link ForkHooks} on this thread, or null.
     * Thread-local like {@link #FORK_RUNNING}, and for the same reason: the
     * engine runs games on more than one thread.
     */
    private static final ThreadLocal<ForkHooks> FORK_HOOKS = new ThreadLocal<>();

    /** The running fork's {@link ForkHooks}, or a listener that does nothing ({@link ForkHooks#NONE}). */
    static ForkHooks forkHooks() {
        ForkHooks hooks = FORK_HOOKS.get();
        return hooks != null ? hooks : ForkHooks.NONE;
    }

    /** Whether a probe on this thread hears the hooks, which some callbacks cost to build for. */
    static boolean forkHooksListening() {
        return FORK_HOOKS.get() != null;
    }

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
     * Whether a fork taken at the phase event beginning {@code phase} can pass
     * its score check (the protection fork, a probe, a directed fork; an
     * intervention is taken at the step's first priority, where Forge has
     * built the combat, and asks nothing). Not at the beginning of combat: the evaluator plays
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
        return scoreCheck("unnamed", new CopyUnderCheck(live, copier, fork, perspective));
    }

    /**
     * A fresh copy beside the game it came from, as a score check compares
     * them.
     *
     * @param live        the game copied
     * @param copier      maps {@code live}'s objects to the copy's
     * @param fork        the copy
     * @param perspective the live player both games are scored for
     */
    record CopyUnderCheck(Game live, GameCopier copier, Game fork, Player perspective) {
    }

    /**
     * The same for a fork of {@code kind}: a check that throws is counted
     * under {@code fork <kind> score check} in the worker's failure report
     * ({@link HookFailures}) besides its line.
     */
    static boolean scoreCheck(String kind, CopyUnderCheck copy) {
        Game live = copy.live();
        Game fork = copy.fork();
        Player perspective = copy.perspective();
        try {
            int original = score(live, perspective);
            Player forkPerspective = mappedPlayer(copy.copier(), perspective);
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
        } catch (RuntimeException | StackOverflowError e) {
            HookFailures.countContained(new HookFailures.FailureKey("fork " + kind + " score check"));
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
     * @param copyDropped what the copier left out for want of a counterpart,
     *                    by kind as a record spells it ({@link #dropped}):
     *                    every record of the fork carries it as
     *                    {@code copy_dropped}; empty for a lossless copy
     */
    record VerifiedFork(GameCopier copier, Game fork, Player perspective,
                        Map<String, Integer> copyDropped) {

        VerifiedFork {
            copyDropped = Map.copyOf(copyDropped);
        }

        /** Stamp {@code record} with what this fork's copy left out, if anything. */
        EffectRecord stamp(EffectRecord record) {
            return record.copyDropped(copyDropped);
        }
    }

    /**
     * The copier's tally of what a copy left out, as a record spells it: each
     * kind with a drop, lower-cased ({@code GameCopier.NoCounterpart}'s
     * names: {@code command}, {@code delayed_trigger}, {@code remembered},
     * {@code link}, {@code attack}, {@code ability}, {@code activation}), to
     * how many, in name order; kinds with none left out.
     */
    static Map<String, Integer> dropped(Map<?, Integer> tally) {
        Map<String, Integer> dropped = new TreeMap<>();
        if (tally != null) {
            tally.forEach((kind, count) -> {
                if (count != null && count > 0) {
                    dropped.put(String.valueOf(kind).toLowerCase(Locale.ROOT), count);
                }
            });
        }
        return dropped;
    }

    /**
     * Copy the live game and score-check the copy, empty when either fails.
     * Every copy is first audited against the live game's board snapshot
     * ({@link ForkFidelity}), whose differences are reported and decide
     * nothing.
     *
     * <p>A copy that is not handed on (its audit or its notes threw, the
     * perspective is null or throws, the score check fails or throws) is
     * released from the registries keyed by game here ({@link #release}):
     * the audit has already snapshotted it, and the caller never learns of
     * it to release it.
     *
     * @param kind        the fork's kind, which names a copy that threw in
     *                    the worker's failure report ({@code "fork <kind> copy"})
     * @param perspective read after the copy, because a game the copier cannot
     *                    read may not answer it either
     */
    static Optional<VerifiedFork> copyAndVerify(String kind, Game live, Supplier<Player> perspective) {
        GameCopier copier;
        Game fork = null;
        Map<String, Integer> copyDropped;
        try {
            // All inside the guard: the copier's own constructor reads the
            // game, so a game it cannot read fails here rather than at makeCopy.
            // A board deep enough to overflow the copier's recursion is one
            // it cannot copy, not one that should end the worker.
            copier = new GameCopier(live);
            fork = copier.makeCopy();
            // Read once, now: mapping objects into the copy later adds nothing to it.
            copyDropped = dropped(copier.getNoCounterpartTally());
            // Before anything reads the fork: every builder over it, the audit's
            // and a played-forward recording's, reads the live game's notes too.
            ClauseNotes.ofFork(fork, live);
            ForkFidelity.report(live, fork, copier.getNoCounterpartTally(), copier.getNoCounterpartExamples());
        } catch (RuntimeException | StackOverflowError e) {
            HookFailures.countContained(new HookFailures.FailureKey("fork " + kind + " copy"), e);
            release(fork);
            return Optional.empty();
        }
        Optional<VerifiedFork> verified = Optional.empty();
        try {
            Player from = perspective.get();
            if (from != null && scoreCheck(kind, new CopyUnderCheck(live, copier, fork, from))) {
                verified = Optional.of(new VerifiedFork(copier, fork, from, copyDropped));
            }
            return verified;
        } finally {
            if (verified.isEmpty()) {
                release(fork);
            }
        }
    }

    /**
     * Forget everything the registries keyed by game hold for {@code fork}
     * ({@link LibraryCapture}, {@link ClauseNotes}): its fork has ended,
     * or was discarded before it began. Each built card or open note holds
     * its game, so a weak key alone would keep the fork for the JVM's life.
     */
    static void release(Game fork) {
        LibraryCapture.release(fork);
        ClauseNotes.release(fork);
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
         * @return what the run produced, or empty when the fork has nothing
         *         to write: an object the copy could not map, or a run that
         *         observed nothing (the one consumer, {@link ForkRun#take},
         *         treats the two alike: the fork is discarded unwritten)
         */
        Optional<R> run(VerifiedFork verified, SnapshotBuilder snapshots);
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
     * cannot describe. Discarding is right; failing the worker is not.
     *
     * <p>A copy (with its audit and its floating effects' notes), a score
     * check, a body or a writer that throws is counted under {@code "fork
     * <kind> copy"}, {@code "fork <kind> score check"}, {@code "fork <kind>
     * body"} or {@code "fork <kind> write"} in the worker's failure report
     * ({@link HookFailures}), the first copy, body or write failure printed
     * with its stack: a systematically failing fork leaves a trace, not just
     * an empty corpus. A body that declines for an expected reason throws a
     * {@link ForkRefusal}, which is counted apart, by reason, in {@link
     * ForkRefusals}, and is no failure.
     *
     * <p>A fork never changes the live game. What of it is cheap to read
     * ({@link LiveMarks}: its timestamp and card id counters, the random
     * stream it draws from, its players' life, zones and controllers, its
     * stack, and Forge's JVM-wide trigger id counter, which only a restart of
     * the game lowers) is read before the fork and after; a difference, or a
     * fork whose game was restarted, is counted under
     * {@code "fork <kind> moved the live game"}, its first printed with what
     * moved.
     *
     * <p>The whole span draws from a random stream of its own ({@link
     * #isolated}), never from the live game's: a fork's AI, its stack and
     * its effects all draw from Forge's one global stream, and a draw a fork
     * took there is one the live game no longer takes.
     *
     * @param kind        the fork's kind, as the failure report names it:
     *                    {@code intervention}, {@code probe}, {@code
     *                    protection} or {@code directed}
     * @param live        the game the fork copies
     * @param spend       spends the fork's budget
     * @param perspective the live player the copy's score is checked from,
     *                    read after the copy
     */
    record ForkRun(String kind, Game live, Runnable spend, Supplier<Player> perspective) {

        /**
         * Take the fork.
         *
         * @return what {@code write} returned, or empty when the copy failed
         *         its check, the body threw or refused, the body produced
         *         nothing or the writer threw: the fork was discarded unwritten
         */
        <R, W> Optional<W> take(ForkBody<R> body, BiFunction<VerifiedFork, R, W> write) {
            LiveMarks before = LiveMarks.of(live);
            Game[] forked = {null};
            try {
                // The whole span is the fork, not just its run: see FORK_RUNNING
                // for what a narrower scope left uncovered.
                return asFork(() -> isolated(SeedStreams.nextForkStream(live), () -> {
                    spend.run();
                    Optional<VerifiedFork> verified = copyAndVerify(kind, live, perspective);
                    if (verified.isEmpty()) {
                        return Optional.<W>empty();
                    }
                    forked[0] = verified.get().fork();
                    return produce(body, verified.get())
                            .flatMap(produced -> written(write, verified.get(), produced));
                }));
            } finally {
                leave(before, forked[0]);
            }
        }

        /**
         * Run {@code body} on the verified copy, empty when it refused (counted
         * by reason in {@link ForkRefusals}) or threw (counted under {@code
         * fork <kind> body}).
         */
        private <R> Optional<R> produce(ForkBody<R> body, VerifiedFork verified) {
            try {
                // A copy whose stack already holds a restart is refused
                // before its body can resolve it (FR-010d): a belt, as
                // the copier copies no stack.
                refuseARestart(verified.fork());
                // The fork's own state, read against the live game it came from.
                SnapshotBuilder snapshots = SnapshotBuilder.forFork(verified.fork(), live);
                return body.run(verified, snapshots);
            } catch (ForkRefusal refusal) {
                ForkRefusals.count(kind, refusal.reason());
                return Optional.empty();
            } catch (RuntimeException | StackOverflowError e) {
                HookFailures.countContained(new HookFailures.FailureKey("fork " + kind + " body"), e);
                return Optional.empty();
            }
        }

        /**
         * Hand what the body produced to the fork's writer, empty when the
         * writer threw (counted under {@code fork <kind> write}).
         */
        private <R, W> Optional<W> written(BiFunction<VerifiedFork, R, W> write, VerifiedFork verified, R produced) {
            try {
                return Optional.ofNullable(write.apply(verified, produced));
            } catch (RuntimeException | StackOverflowError e) {
                HookFailures.countContained(new HookFailures.FailureKey("fork " + kind + " write"), e);
                return Optional.empty();
            }
        }

        /**
         * Compare the live game's marks with {@code before} and release
         * {@code fork}, null when no copy was handed on, however the
         * comparison ends.
         */
        private void leave(LiveMarks before, Game fork) {
            try {
                before.compareAfter("fork " + kind + " moved the live game", live, fork);
            } finally {
                // The fork is done with: the registries keyed by game
                // would otherwise hold it for the JVM's life. A copy
                // discarded before it was handed on is released by
                // copyAndVerify itself.
                release(fork);
            }
        }

    }

    /**
     * End the fork as the refusal {@link ForkRefusal#GAME_RESTART} when its
     * stack holds an ability with a restart clause (Karn Liberated's
     * ultimate), before that ability can resolve (FR-010d).
     *
     * <p>Forge's {@code RestartGameEffect} calls {@code Trigger.resetIDs()},
     * which sets the trigger id counter every game of the JVM shares to
     * 50000; the live game's later triggers would then take ids its own
     * triggers hold, which Forge compares triggers by and the records write
     * ({@code D<id>}). So every fork asks this where its stack can come to
     * resolve: once its copy is made ({@link ForkRun#take}), before each step
     * a fork played on takes ({@link ForkPlayForward}), and before a forced
     * resolution drains its stack ({@link InterventionFork}). A restart put on
     * the stack resolves at the earliest in the main-loop step after the one
     * that put it there, once every other player has passed, so the check
     * before each step comes first, and is the one that acts. The other two
     * are belts no game reaches: {@code GameCopier} copies no stack
     * ({@code GameSimulator.COPY_STACK} is off, and copies spells only when
     * on, while Karn's restart is a loyalty ability), and a forced line is
     * never a restart ({@link InterventionScheduler}). {@link LiveMarks}
     * stays the backstop: it counts a fork whose game was restarted all the
     * same.
     */
    static void refuseARestart(Game fork) {
        for (forge.game.spellability.SpellAbilityStackInstance instance : fork.getStack()) {
            SpellAbility ability = instance.getSpellAbility();
            if (ability != null && InterventionScheduler.restartsTheGame(ability)) {
                throw new ForkRefusal(ForkRefusal.GAME_RESTART);
            }
        }
    }

    /**
     * Run {@code action} with Forge's global random stream ({@code MyRandom})
     * replaced by {@code stream}, the fork's own, and the live game's put back
     * however the action ends, so the fork draws nothing from it. Nothing is
     * lost by the fresh stream's seed: the live stream is an unseeded one,
     * which no rerun reproduces anyway. The fork's stream is the next of its
     * game's ({@link SeedStreams#nextForkStream}), so the n-th fork of two
     * games, or of two worker lifetimes, does not replay the same draws.
     */
    static <T> T isolated(Random stream, Supplier<T> action) {
        Random live = MyRandom.getRandom();
        MyRandom.setRandom(stream);
        try {
            return action.get();
        } finally {
            MyRandom.setRandom(live);
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
