package com.pricepredictor.connector.effects;

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

    /** Kept apart from the forced resolutions' stream, which the same game seed also seeds. */
    private static final long SEED_SALT = 0x5DEECE66DL;

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
        this.forkSeeds = new Random(GameRecording.scramble(seed ^ SEED_SALT));
    }

    /**
     * What a fork played on left.
     *
     * @param records  how many records it wrote
     * @param gameOver whether the game ended in the fork before the turn did
     */
    record Played(int records, boolean gameOver) {
    }

    /**
     * Play {@code fork} on to the end of its turn and write its records.
     *
     * <p>The copy was taken at a phase event, which the engine fires ahead
     * of the phase's turn-based actions and its priority, so the fork begins
     * that phase itself before it loops (see {@link PhaseStart}).
     *
     * @param flags the flags every record of the fork carries
     * @throws RuntimeException when the fork cannot be played out; nothing
     *                          is written then, and the caller's guard
     *                          discards the fork
     */
    Played play(Game fork, Consumer<EffectRecord> flags) {
        Random seat = new Random(forkSeeds.nextLong());
        long sampling = forkSeeds.nextLong();
        List<String> held = new ArrayList<>();
        GameRecording recording = live.forkRecording(fork, flags, held::add, sampling);
        try {
            recording.open();
            seats.attach(fork, recording.priorityCandidates(), recording.fruitlessActivations(), seat);
            live.routingTo(recording, () -> Forks.playingForward(fork, () -> {
                playToTheEndOfTheTurn(fork);
                return null;
            }));
        } finally {
            recording.close();
        }
        live.writeForkLines(held);
        return new Played(held.size(), fork.isGameOver());
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
    private static void playToTheEndOfTheTurn(Game fork) {
        PhaseHandler phases = fork.getPhaseHandler();
        int turn = phases.getTurn();
        PhaseStart.begin(phases);
        int steps = 0;
        while (!fork.isGameOver() && phases.getTurn() == turn && !turnWouldEnd(fork)) {
            if (++steps > MAX_STEPS) {
                throw new IllegalStateException(
                        "fork took " + MAX_STEPS + " steps without ending turn " + turn);
            }
            phases.mainLoopStep();
        }
    }

    /** Whether the next main-loop step ends the turn: a cleanup step with nothing left to resolve. */
    static boolean turnWouldEnd(Game fork) {
        return fork.getPhaseHandler().is(PhaseType.CLEANUP)
                && fork.getStack().isEmpty()
                && !fork.getStack().hasSimultaneousStackEntries();
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

        static {
            Method begin = null;
            Field give = null;
            try {
                begin = PhaseHandler.class.getDeclaredMethod("onPhaseBegin");
                begin.setAccessible(true);
                give = PhaseHandler.class.getDeclaredField("givePriorityToPlayer");
                give.setAccessible(true);
            } catch (ReflectiveOperationException | RuntimeException e) {
                begin = null;
                give = null;
            }
            ON_PHASE_BEGIN = begin;
            GIVE_PRIORITY = give;
        }

        private PhaseStart() {
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
