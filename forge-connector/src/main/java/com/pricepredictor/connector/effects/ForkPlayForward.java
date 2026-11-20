package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.effects.directed.KeptRecords;
import forge.game.Game;
import forge.game.event.GameEventTurnPhase;
import forge.game.phase.PhaseHandler;
import forge.game.phase.PhaseType;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Consumer;

/**
 * Plays a fork on to the end of the current turn, recording it (D4).
 *
 * <p>A directed fork changes one fact of the board and then lets the game go
 * on: both seats choose afresh, as their live controllers would, and what
 * the turn does with the changed fact is the record. So the fork is played
 * the way {@code PhaseHandler.mainGameLoop} plays a game, one
 * {@code mainLoopStep} at a time, until its cleanup step leaves nothing to
 * resolve or the game ends.
 *
 * <p>Recorded by collectors of its own. The fork's bus is its own, so a
 * bracket over the fork hears its events; the engine hooks are JVM statics
 * the live game's collectors hold, so for the span of the play they hand
 * every callback to the fork's collectors as well, each keeping its own
 * game's ({@link PatchedCollectors#routingTo}). The fork's recording
 * schedules no intervention, probe or directed fork of its own: a fork never
 * forks.
 *
 * <p>The fork's lines are held until it has played out and written to the
 * shard only then, so a fork that throws part way, which the caller's
 * {@link Forks.ForkRun} discards, leaves no record behind.
 *
 * <p>Not {@code InterventionFork.forceResolution}: that forces the resolution
 * of one chosen ability and stops, where this plays whole turns with both
 * seats choosing.
 */
final class ForkPlayForward {

    /**
     * Main-loop steps a fork may take before it is given up on. A turn takes
     * a few dozen; this bounds an engine that stops making progress, which
     * would otherwise hold the live game's thread for good.
     */
    static final int MAX_STEPS = 5_000;

    private final GameRecording live;
    private final ForkSeats seats;
    /** Each fork's seat and sampling seeds, drawn per game rather than from the match's source. */
    private final Random forkSeeds;

    /**
     * @param live  the live game's recording, whose hooks route to the fork's
     * @param seats attaches the fork's seats to it
     * @param seed  the game's seed
     */
    ForkPlayForward(GameRecording live, ForkSeats seats, long seed) {
        this.live = live;
        this.seats = seats;
        this.forkSeeds = SeedStreams.random(seed, SeedStreams.Consumer.PLAY_FORWARD_SEEDS);
    }

    /**
     * What a fork played on left.
     *
     * @param records  how many records it wrote
     * @param gameOver whether the game ended in the fork before the turn did
     * @param dropped  the fork's events that reached no record
     */
    record Played(int records, boolean gameOver, DroppedEvents dropped) {
    }

    /**
     * Count a kept fork's dropped events into the live game's, under the
     * fork's {@code kind}.
     *
     * @return {@code played}
     */
    Played countDropped(String kind, Played played) {
        live.countForkDropped(kind, played.dropped());
        return played;
    }

    /**
     * Play {@code fork} on to the end of its turn and write its records.
     *
     * <p>The copy was taken at a phase event, which the engine fires ahead
     * of the phase's turn-based actions and its priority, so the fork begins
     * that phase itself before it loops (see {@link PhaseStart}). Before
     * that, it runs the state-based check the fork's edit makes, as its
     * premise ({@link ForkPremise}).
     *
     * @param flags the flags every record of the fork carries
     * @throws RuntimeException when the fork cannot be played out; nothing
     *                          is written then, and the caller's guard
     *                          discards the fork
     */
    Played play(Game fork, Consumer<EffectRecord> flags) {
        return play(fork, flags, KeptRecords.NONE);
    }

    /**
     * {@link #play(Game, Consumer)}, the records {@code kept} names written
     * whatever the fork's samplers draw (FR-010c).
     */
    Played play(Game fork, Consumer<EffectRecord> flags, KeptRecords kept) {
        return play(fork, flags, kept, null);
    }

    /**
     * {@link #play(Game, Consumer, KeptRecords)}, the actor of {@code forced}
     * made to make its play (FR-010b), or with no play forced when it is
     * null. A forced play refused while the fork plays on ends it between two
     * main-loop steps; one never made ends it after the turn; either way
     * before any of its lines is written.
     *
     * @throws ForkRefusal when the forced play was not made
     */
    Played play(Game fork, Consumer<EffectRecord> flags, KeptRecords kept, ForcedPlayRun forced) {
        Random seat = new Random(forkSeeds.nextLong());
        long sampling = forkSeeds.nextLong();
        // Drawn only by a fork that forces: the game's later forks then take
        // the next seeds of the sequence, shifted by one.
        Random forcing = forced == null ? null : new Random(forkSeeds.nextLong());
        List<String> held = new ArrayList<>();
        GameRecording recording = live.forkRecording(fork, new ForkOutput(flags, held::add, kept), sampling);
        try {
            recording.open();
            seats.attach(fork, recording.priorityCandidates(), recording.fruitlessActivations(), seat);
            if (forced != null) {
                forced.attach(forcing);
            }
            live.routingTo(recording, () -> Forks.playingForward(fork, () -> {
                // The edit's own consequences first, as the fork's premise.
                recording.settlePremise();
                playToTheEndOfTheTurn(fork, forced);
                return null;
            }));
        } finally {
            recording.close();
        }
        if (forced != null) {
            forced.finish();
        }
        live.writeForkLines(held);
        return new Played(held.size(), fork.isGameOver(), recording.droppedEvents());
    }

    /**
     * Begin the fork's phase, then step until the turn would end or the game
     * ends.
     *
     * <p>The turn ends in the step after a cleanup that leaves the stack
     * empty, and that one {@code mainLoopStep} also begins the next turn's
     * untap step: its phase event and its turn-based actions (F10). So the
     * fork stops before it, in the cleanup step, its own actions done.
     */
    private static void playToTheEndOfTheTurn(Game fork, ForcedPlayRun forced) {
        PhaseHandler phases = fork.getPhaseHandler();
        int turn = phases.getTurn();
        PhaseStart.begin(phases);
        int steps = 0;
        while (!fork.isGameOver() && phases.getTurn() == turn && !turnWouldEnd(fork)) {
            if (++steps > MAX_STEPS) {
                throw new IllegalStateException(
                        "fork took " + MAX_STEPS + " steps without ending turn " + turn);
            }
            // Before the step that could resolve it (FR-010d).
            Forks.refuseARestart(fork);
            if (forced != null) {
                // A play refused, or whose step passed unmade, plays nothing more.
                forced.check();
            }
            phases.mainLoopStep();
        }
    }

    /**
     * Whether the next main-loop step ends the turn: a cleanup step with
     * nothing left to resolve, and none to repeat. A cleanup in which a
     * player got priority (a state-based action or a trigger, 514.3a) is
     * followed by another cleanup step, whose turn-based actions are the
     * turn's too.
     */
    static boolean turnWouldEnd(Game fork) {
        return fork.getPhaseHandler().is(PhaseType.CLEANUP)
                && fork.getStack().isEmpty()
                && !fork.getStack().hasSimultaneousStackEntries()
                && !PhaseStart.repeatsCleanup(fork.getPhaseHandler());
    }

    /**
     * Begins the phase a copy was taken in, as the live game does: the phase
     * event, priority to the active player, then the phase's turn-based
     * actions and its beginning-of-step triggers.
     *
     * <p>{@code GameCopier} sets the copy's phase without beginning it, and
     * its engine would give no one priority in it either: the copy skips
     * every turn-based action of the step the live game was entering, the
     * beginning-of-end-step triggers among them. {@code onPhaseBegin} and
     * the priority flag are private, so they are reached by reflection; a
     * Forge that renamed either fails the fork rather than the worker.
     */
    static final class PhaseStart {

        private static final Method ON_PHASE_BEGIN;
        private static final Field GIVE_PRIORITY;
        /** {@code PhaseHandler.bRepeatCleanup}: the cleanup step is to be taken again (514.3a). */
        private static final Field REPEAT_CLEANUP;

        static {
            Method begin = null;
            Field give = null;
            Field repeat = null;
            try {
                begin = PhaseHandler.class.getDeclaredMethod("onPhaseBegin");
                begin.setAccessible(true);
                give = PhaseHandler.class.getDeclaredField("givePriorityToPlayer");
                give.setAccessible(true);
                repeat = PhaseHandler.class.getDeclaredField("bRepeatCleanup");
                repeat.setAccessible(true);
            } catch (ReflectiveOperationException | RuntimeException e) {
                begin = null;
                give = null;
                repeat = null;
            }
            ON_PHASE_BEGIN = begin;
            GIVE_PRIORITY = give;
            REPEAT_CLEANUP = repeat;
        }

        private PhaseStart() {
        }

        /**
         * Whether the engine will take the cleanup step again; private in
         * {@code PhaseHandler}, so read by reflection. A Forge without the
         * field fails the fork rather than ending it a step early.
         */
        static boolean repeatsCleanup(PhaseHandler phases) {
            if (REPEAT_CLEANUP == null) {
                throw new IllegalStateException("this Forge's PhaseHandler cannot say whether cleanup repeats");
            }
            try {
                return REPEAT_CLEANUP.getBoolean(phases);
            } catch (IllegalAccessException e) {
                throw new IllegalStateException(e);
            }
        }

        static void begin(PhaseHandler phases) {
            if (ON_PHASE_BEGIN == null || GIVE_PRIORITY == null) {
                throw new IllegalStateException("this Forge's PhaseHandler cannot begin a copied phase");
            }
            // The phase event first, as advanceToNextPhase fires it: the
            // fork's collectors open the step's windows on it (the turn-based
            // actions, the combat damage step), and the copy's own was fired
            // before they listened.
            phases.getPlayerTurn().getGame().fireEvent(
                    new GameEventTurnPhase(phases.getPlayerTurn(), phases.getPhase(), ""));
            phases.setPriority(phases.getPlayerTurn());
            try {
                GIVE_PRIORITY.setBoolean(phases, true);
                ON_PHASE_BEGIN.invoke(phases);
            } catch (IllegalAccessException e) {
                throw new IllegalStateException(e);
            } catch (java.lang.reflect.InvocationTargetException e) {
                if (e.getCause() instanceof RuntimeException runtime) {
                    throw runtime;
                }
                if (e.getCause() instanceof Error error) {
                    throw error;
                }
                throw new IllegalStateException(e.getCause());
            }
        }
    }
}
