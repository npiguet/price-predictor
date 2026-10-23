package com.pricepredictor.connector.effects;

import com.google.common.eventbus.Subscribe;
import forge.game.Game;
import forge.game.event.GameEventTurnPhase;

import java.util.Optional;
import java.util.function.Supplier;

/**
 * Everything one game's effect records are collected through.
 *
 * <p>A game is recorded by three collectors: the bracket on the event bus,
 * the patched collectors on the engine's static hooks, and stage three's
 * forks. They describe the same game, so they share its id, its attribution
 * mode and the match's random seat, and each is handed those in a context of
 * its own when it is built. Nothing is set after construction, so no
 * collector can be wired in the wrong order or miss a seat another one learnt.
 *
 * <p>{@link #open()} and {@link #close()} are the whole lifecycle. The hooks
 * are JVM statics with no game of their own, so whatever {@code open}
 * installs, {@code close} takes down: a hook left installed would hand the
 * next game's callbacks to this one's collectors.
 */
public final class GameRecording implements AutoCloseable {

    private final Game game;
    private final BusBracketCollector bracket;
    private final PatchedCollectors patched;
    /** Stage three's forced resolutions, when the run takes any. */
    private final Optional<InterventionScheduler> interventions;
    /** Stage three's damage-step probes, when the run takes any. */
    private final Optional<ProbeScheduler> probes;

    /**
     * Build one game's collectors.
     *
     * @param gameId             the game's own id: records join to a
     *                           checkpoint's recorded split by it, so it
     *                           changes when the game does
     * @param seed               seeds the sampling and the forks' choices, so
     *                           a worker's games sample independently of one
     *                           another
     * @param randomSeatPlayerId the match's random seat, as a player id, or
     *                           null when it has none
     */
    public GameRecording(
            Game game, RecordShardWriter writer, String gameId, CollectionCaps caps,
            long seed, String randomSeatPlayerId) {
        this.game = game;
        AttributionMode mode = AttributionMode.detect();
        // One context per collector, because each counts its own records.
        Supplier<CollectorContext> context =
                () -> new CollectorContext(writer, gameId, mode, randomSeatPlayerId);
        // Stage three. Nothing is copied unless a budget says so, because a
        // fork costs a game copy and a stack resolution, and nothing on a
        // stock checkout, where no fork could be attributed. Both kinds share
        // one game's budget and one context.
        boolean forking = mode == AttributionMode.PATCHED
                && (caps.interventionsPerGame() > 0 || caps.probesEnabled());
        CollectorContext forkContext = forking ? context.get() : null;
        ForkBudget budget = new ForkBudget(caps);
        this.interventions = forking
                ? Optional.of(new InterventionScheduler(
                        game, new InterventionFork(game, forkContext, budget, seed)))
                : Optional.empty();
        this.probes = forking
                ? Optional.of(new ProbeScheduler(
                        game, gameId, caps, new DamageStepProbe(game, forkContext, budget)))
                : Optional.empty();
        // A probe forks before the damage step and cannot know which combat
        // record it mirrors until that record is written.
        this.bracket = new BusBracketCollector(game, context.get(),
                recordId -> probes.ifPresent(probe -> probe.writeHeldProbes(recordId)));
        this.patched = new PatchedCollectors(game, context.get(), caps, seed, bracket);
    }

    /**
     * Spread out a seed before it reaches {@link java.util.Random}.
     *
     * <p>Games are seeded consecutively, and a linear congruential generator's
     * first output is close to a linear function of its seed — so consecutive
     * seeds make consecutive games take nearly the same first sampling
     * decisions. That would correlate the mana reservoir and the playability
     * sample across a whole run, in a way no test of one game could see.
     *
     * <p>SplitMix64's finalizer, which is the standard mixer for exactly this.
     */
    public static long scramble(long seed) {
        long z = seed + 0x9E3779B97F4A7C15L;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    /**
     * Start recording: subscribe to the game's bus and install every hook the
     * checkout offers.
     *
     * <p>On a stock checkout no hook is found and the game plays exactly as it
     * would unrecorded; the bracket still records off the bus.
     */
    public void open() {
        game.subscribeToEvents(bracket);
        PatchHooks.CONFIRM_LISTENER.install(bracket.confirmHandler());
        patched.install();
        // Continuous effects are read off the layer tables at a phase
        // boundary, which is the cheapest moment the board is stable.
        game.subscribeToEvents(new PhaseBridge());
    }

    /**
     * Stop recording: write what is still held and take every hook down.
     *
     * <p>Called once the game is over, which is the moment the mana reservoir
     * is decided. Every hook comes down even when writing what is held throws,
     * and after an {@link #open()} that threw part way: taking down a hook that
     * was never installed does nothing.
     */
    @Override
    public void close() {
        try {
            patched.close();
        } finally {
            PatchHooks.CONFIRM_LISTENER.uninstall();
        }
    }

    /** Where the random seat writes the decisions its own draws made (FR-022d). */
    public PlayabilityChannel playability() {
        return patched.playability();
    }

    /**
     * Runs the phase-boundary collection off the event bus.
     *
     * <p>A separate object rather than {@code @Subscribe} on the recording
     * itself, because Forge's bus reflects over every public method of a
     * subscriber.
     */
    public final class PhaseBridge {

        private PhaseBridge() {
        }

        @Subscribe
        public void onPhase(GameEventTurnPhase event) {
            patched.continuous().collect();
            interventions.ifPresent(InterventionScheduler::collect);
            // Before the turn-based action that deals the damage: the phase
            // event fires ahead of it, which is the only moment a probe can
            // fork from a board where nothing has been dealt yet.
            probes.ifPresent(ProbeScheduler::collect);
        }
    }
}
