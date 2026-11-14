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
     * What a forced resolution or a probe hears of the hooks while it runs:
     * the resolutions and clauses of its fork, the mana its resolutions make,
     * the events its clauses promise and the outcomes its effects report.
     *
     * <p>The hooks are JVM statics that the live game's handlers own, and
     * each handler drops a fork's callbacks on game identity — rightly, for
     * its own records. A forced resolution needs the same moments the live
     * bracket takes: its resolution's start (FR-042b), each chosen mode's
     * first clause (FR-029f), whether each conditional clause met its
     * condition (FR-034) and the mana it made (FR-042c); and both it and a
     * probe need the events the live record takes from the hooks rather
     * than the bus, the clauses' and the outcomes' (a token made, damage
     * prevented), or a fork's record would lack what its live twin holds.
     * So the live handlers hand what they are not keeping to whatever fork
     * is running on their thread, and {@code InterventionFork.forceResolution}
     * and {@code DamageStepProbe} install one for exactly the span of their
     * own resolution or damage step.
     */
    interface ForkHooks {
        /** An item of some other game's stack starts to resolve. */
        default void resolutionStarting(SpellAbility item, boolean fizzled) {
        }

        /**
         * The same, with the hook's third argument: each chain link's
         * targets before Forge's check for illegal targets, or null.
         */
        default void resolutionStarting(SpellAbility item, boolean fizzled, Object targetsAtCast) {
            resolutionStarting(item, fizzled);
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

        /**
         * The event a clause of some other game promised by its effect API's
         * own parameters, told once the clause ran ({@code ApiEvents}): a
         * turn added, a choice made, a continuous effect created.
         */
        default void clauseEvent(SpellAbility clause, EffectEvent event) {
        }

        /**
         * What an effect of some other game computed and reported through
         * the outcome hook: a coin flipped, a token made, damage prevented.
         *
         * @param ability the ability the effect named, or null for one with
         *                none (combat damage a shield prevented)
         */
        default void outcome(SpellAbility ability, EffectEvent event) {
        }

        /**
         * Some other game's controller turned down a "you may" (the confirm
         * hook answered false).
         *
         * @param offered the ability asked about, or null where the engine
         *                named none
         */
        default void declined(SpellAbility offered) {
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

    /** Whether a forced resolution or a probe on this thread hears the hooks, which some callbacks cost to build for. */
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
        return scoreCheck("unnamed", live, copier, fork, perspective);
    }

    /**
     * The same for a fork of {@code kind}: a check that throws is counted
     * under {@code fork <kind> score check} in the worker's failure report
     * ({@link HookFailures}) besides its line.
     */
    static boolean scoreCheck(String kind, Game live, GameCopier copier, Game fork, Player perspective) {
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
        } catch (RuntimeException | StackOverflowError e) {
            HookFailures.countContained("fork " + kind + " score check");
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
     * @param kind        the fork's kind, which names a copy that threw in
     *                    the worker's failure report ({@code "fork <kind> copy"})
     * @param perspective read after the copy, because a game the copier cannot
     *                    read may not answer it either
     */
    static Optional<VerifiedFork> copyAndVerify(String kind, Game live, Supplier<Player> perspective) {
        GameCopier copier;
        Game fork;
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
            FloatingEffects.ofFork(fork, live);
            ForkFidelity.report(live, fork, copier.getNoCounterpartTally(), copier.getNoCounterpartExamples());
        } catch (RuntimeException | StackOverflowError e) {
            HookFailures.countContained("fork " + kind + " copy", e);
            return Optional.empty();
        }
        Player from = perspective.get();
        if (from == null || !scoreCheck(kind, live, copier, fork, from)) {
            return Optional.empty();
        }
        return Optional.of(new VerifiedFork(copier, fork, from, copyDropped));
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
     * stack) is read before the fork and after; a difference is counted
     * under {@code "fork <kind> moved the live game"}, its first printed
     * with what moved.
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
         * @return what {@code write} returned, or null when the copy failed
         *         its check, the body threw, or the body produced nothing
         */
        <R, W> W take(ForkBody<R> body, BiFunction<VerifiedFork, R, W> write) {
            LiveMarks before = LiveMarks.of(live);
            try {
                // The whole span is the fork, not just its run: see FORK_RUNNING
                // for what a narrower scope left uncovered.
                return asFork(() -> isolated(() -> {
                    spend.run();
                    Optional<VerifiedFork> verified = copyAndVerify(kind, live, perspective);
                    if (verified.isEmpty()) {
                        return null;
                    }
                    R produced;
                    try {
                        // The fork's own state, read against the live game it came from.
                        SnapshotBuilder snapshots = SnapshotBuilder.forFork(verified.get().fork(), live);
                        produced = body.run(verified.get(), snapshots);
                    } catch (ForkRefusal refusal) {
                        ForkRefusals.count(kind, refusal.reason());
                        return null;
                    } catch (RuntimeException | StackOverflowError e) {
                        HookFailures.countContained("fork " + kind + " body", e);
                        return null;
                    }
                    if (produced == null) {
                        return null;
                    }
                    try {
                        return write.apply(verified.get(), produced);
                    } catch (RuntimeException | StackOverflowError e) {
                        HookFailures.countContained("fork " + kind + " write", e);
                        return null;
                    }
                }));
            } finally {
                before.compareAfter("fork " + kind + " moved the live game", live);
            }
        }

    }

    /**
     * What a fork must leave as it was in the live game, the part of it that
     * is cheap to read: its timestamp and card id counters, the random
     * stream Forge draws from, each player's life, poison, controller and
     * zone sizes, and its stack's size.
     */
    record LiveMarks(String marks) {

        private static final java.lang.reflect.Field CARD_IDS = counter("cardIdCounter");
        private static final java.lang.reflect.Field HIDDEN_IDS = counter("hiddenCardIdCounter");

        private static java.lang.reflect.Field counter(String name) {
            try {
                java.lang.reflect.Field field = Game.class.getDeclaredField(name);
                field.setAccessible(true);
                return field;
            } catch (ReflectiveOperationException | RuntimeException e) {
                return null;
            }
        }

        /** {@code live}'s marks, or none for no game. */
        static LiveMarks of(Game live) {
            if (live == null) {
                return new LiveMarks("");
            }
            StringBuilder out = new StringBuilder()
                    .append("timestamp ").append(live.getTimestamp())
                    .append(", card ids ").append(read(CARD_IDS, live))
                    .append(", hidden card ids ").append(read(HIDDEN_IDS, live))
                    .append(", random stream ").append(System.identityHashCode(MyRandom.getRandom()))
                    .append(", stack ").append(live.getStack().size());
            for (Player player : live.getRegisteredPlayers()) {
                out.append(", ").append(Refs.of(player))
                        .append(" life ").append(player.getLife())
                        .append(" poison ").append(player.getPoisonCounters())
                        .append(" controller ").append(System.identityHashCode(player.getController()))
                        .append(" zones");
                for (forge.game.zone.ZoneType zone : MARKED_ZONES) {
                    out.append(' ').append(player.getZone(zone).size());
                }
            }
            return new LiveMarks(out.toString());
        }

        private static final List<forge.game.zone.ZoneType> MARKED_ZONES = List.of(
                forge.game.zone.ZoneType.Hand, forge.game.zone.ZoneType.Library,
                forge.game.zone.ZoneType.Battlefield, forge.game.zone.ZoneType.Graveyard,
                forge.game.zone.ZoneType.Exile, forge.game.zone.ZoneType.Command);

        private static Object read(java.lang.reflect.Field field, Game game) {
            try {
                return field == null ? "?" : field.get(game);
            } catch (ReflectiveOperationException e) {
                return "?";
            }
        }

        /** Count a difference from {@code live}'s marks now under {@code key}, the first printed. */
        void compareAfter(String key, Game live) {
            LiveMarks after;
            try {
                after = of(live);
            } catch (RuntimeException e) {
                return;
            }
            if (!after.equals(this)) {
                HookFailures.countContained(key, "Effect records: " + key + "; before the fork: " + marks
                        + "; after: " + after.marks() + " (later ones are counted, not printed)");
            }
        }
    }

    /** Seeds each fork's own random stream ({@link #isolated}); never the live game's. */
    private static final java.util.concurrent.atomic.AtomicLong FORK_STREAMS =
            new java.util.concurrent.atomic.AtomicLong();

    /**
     * Seed the next fork's stream as this JVM's first fork's: for a test whose
     * fork plays on and whose written records must not depend on how many
     * forks the JVM took before it.
     */
    static void restartStreams() {
        FORK_STREAMS.set(0);
    }

    /**
     * Run {@code action} with Forge's global random stream ({@code MyRandom})
     * replaced by a fresh one of the fork's own, and the live game's put back
     * however the action ends, so the fork draws nothing from it. Nothing is
     * lost by the fresh stream's seed: the live stream is an unseeded one,
     * which no rerun reproduces anyway.
     */
    static <T> T isolated(Supplier<T> action) {
        Random live = MyRandom.getRandom();
        MyRandom.setRandom(new Random(GameRecording.scramble(FORK_STREAMS.incrementAndGet())));
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
