package com.pricepredictor.connector.effects;

import com.google.common.eventbus.Subscribe;
import com.pricepredictor.connector.effects.directed.DirectedRules;
import forge.game.Game;
import forge.game.event.GameEventTurnPhase;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Consumer;
import java.util.function.Supplier;

/**
 * Everything one game's effect records are collected through.
 *
 * <p>A game is recorded by three collectors: the bracket on the event bus,
 * the patched collectors on the engine's static hooks, and stage three's
 * forks. They describe the same game, so they share its id, its attribution
 * mode, the match's random seat and the rules the run aimed at, and each is
 * handed those in a context of
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
    /** Stage three's forced resolutions and protection forks, when the run takes any. */
    private final Optional<InterventionScheduler> interventions;
    /** Stage three's damage-step probes, when the run takes any. */
    private final Optional<ProbeScheduler> probes;
    /** The directed forks (FR-010), when the run names a rule. */
    private final Optional<DirectedForkScheduler> directed;
    /** Where the game's dropped-event counts go when it ends, or null when the run counts none. */
    private final Path droppedEventsFile;
    /**
     * What a fork's recording is built from: the game's envelope, which a
     * fork's records share.
     */
    private final CollectorContext envelope;
    private final CollectionCaps caps;
    /** Plays forks of this game on to the end of the turn; null on a fork's own recording. */
    private final ForkPlayForward playForward;
    /** Whether this records a fork played on from a live game rather than the live game. */
    private final boolean fork;

    /** Said once per worker: a counts file that cannot be written is no reason to stop collecting. */
    private static final OnceWarning DROPPED_EVENTS_UNWRITABLE = new OnceWarning();

    /**
     * Build one game's collectors, for a match whose seats need no attaching
     * in a fork; see {@link #GameRecording(Game, RecordShardWriter, String,
     * CollectionCaps, long, String, ForkSeats)}.
     */
    public GameRecording(
            Game game, RecordShardWriter writer, String gameId, CollectionCaps caps,
            long seed, String randomSeatPlayerId) {
        this(game, writer, gameId, caps, seed, randomSeatPlayerId, ForkSeats.NONE);
    }

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
     * @param forkSeats          attaches a fork's seats when a fork of this
     *                           game plays on
     */
    public GameRecording(
            Game game, RecordShardWriter writer, String gameId, CollectionCaps caps,
            long seed, String randomSeatPlayerId, ForkSeats forkSeats) {
        this(game, writer, gameId, caps, seed, randomSeatPlayerId, forkSeats,
                DirectedRules.registered(), DirectedForkTally.lifetime());
    }

    /**
     * Build one game's collectors, directed forks taken for {@code
     * directedRules} and counted in {@code tally}; the public constructors
     * pass the registered rules and the JVM lifetime's tally.
     */
    GameRecording(
            Game game, RecordShardWriter writer, String gameId, CollectionCaps caps,
            long seed, String randomSeatPlayerId, ForkSeats forkSeats,
            DirectedRules directedRules, DirectedForkTally tally) {
        this.game = game;
        this.fork = false;
        this.caps = caps;
        AttributionMode mode = AttributionMode.detect();
        // One context per collector, because each counts its own records.
        Supplier<CollectorContext> context =
                () -> new CollectorContext(
                        writer, gameId, mode, randomSeatPlayerId, caps.aimedAt().orElse(null));
        this.envelope = context.get();
        // Stage three. Nothing is copied unless a budget says so, because a
        // fork costs a game copy and a stack resolution, and nothing on a
        // stock checkout, where no fork could be attributed. Both kinds share
        // one game's budget and one context.
        boolean forking = mode == AttributionMode.PATCHED
                && (caps.interventionsPerGame() > 0 || caps.probesEnabled());
        CollectorContext forkContext = forking ? context.get() : null;
        ForkBudget budget = new ForkBudget(caps);
        this.playForward = new ForkPlayForward(this, forkSeats, seed);
        // A protection fork (FR-058) is an intervention played on, so it is
        // taken on a run taking interventions, and only there.
        this.interventions = forking
                ? Optional.of(new InterventionScheduler(
                        game, new InterventionFork(game, forkContext, budget, seed),
                        caps.interventionsPerGame() > 0
                                ? new ProtectionFork(game, budget, playForward, seed) : null))
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
        this.droppedEventsFile = caps.countDroppedEvents() ? droppedEventsFile(writer) : null;
        // Directed forks, on a patched checkout as the other two kinds: the
        // fork plays on through the live game's hooks.
        boolean directing = mode == AttributionMode.PATCHED && caps.directedForksEnabled();
        this.directed = directing
                ? Optional.of(new DirectedForkScheduler(
                        game, caps, directedRules, budget, tally, seed,
                        new DirectedFork(game, playForward, bracket::heldRecordId,
                                () -> game.getPhaseHandler().getPlayerTurn())::take))
                : Optional.empty();
    }

    /**
     * A fork's recording: the live game's collectors, built over the fork
     * and writing through {@code context}, with no fork of its own and no
     * hook installed (D4).
     *
     * <p>No intervention, probe or directed fork is scheduled from a fork,
     * so a fork never forks. Its dropped events are not counted: the counts
     * file has one line per real game.
     *
     * @param seed seeds the fork's sampling stream
     */
    private GameRecording(Game fork, CollectorContext context, CollectionCaps caps, long seed) {
        this.game = fork;
        this.fork = true;
        this.caps = caps;
        this.envelope = context;
        this.interventions = Optional.empty();
        this.probes = Optional.empty();
        this.directed = Optional.empty();
        this.bracket = new BusBracketCollector(fork, context, recordId -> { });
        this.patched = PatchedCollectors.forFork(fork, context, caps, seed, bracket);
        this.droppedEventsFile = null;
        this.playForward = null;
    }

    /**
     * The recording of a fork of this game played on to the end of the turn.
     *
     * @param flags the flags every record of the fork carries
     * @param held  where the fork's finished lines go until the fork is kept
     * @param seed  seeds the fork's sampling stream
     */
    GameRecording forkRecording(
            Game fork, Consumer<EffectRecord> flags, Consumer<String> held, long seed) {
        return new GameRecording(fork, envelope.forFork(flags, held), caps, seed);
    }

    /** Plays forks of this game on to the end of the turn (D4). */
    ForkPlayForward playForward() {
        if (playForward == null) {
            throw new IllegalStateException("a fork's recording plays no fork on");
        }
        return playForward;
    }

    /** Write a kept fork's lines to this game's shard. */
    void writeForkLines(List<String> lines) {
        envelope.writeHeld(lines);
    }

    /**
     * Run {@code span} with this game's hooks also handing their callbacks
     * to {@code forkRecording}'s collectors.
     */
    <T> T routingTo(GameRecording forkRecording, Supplier<T> span) {
        return patched.routingTo(forkRecording.patched, span);
    }

    /**
     * The file a worker lifetime's dropped-event counts are appended to,
     * beside its shard (FR-041).
     */
    public static Path droppedEventsFile(RecordShardWriter writer) {
        return writer.sibling(DroppedEvents.SUFFIX);
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
        if (!fork) {
            // A fork's collectors are handed the live hooks' callbacks
            // instead (routingTo): installing would take them from the
            // live game.
            patched.install();
        }
        // The declaration's taps and its end, which hold its attackers
        // records; it is asked nothing on a checkout without the combat hook.
        game.subscribeToEvents(patched.playability().declaration());
        // Continuous effects are read off the layer tables at a phase
        // boundary, which is the cheapest moment the board is stable.
        game.subscribeToEvents(new PhaseBridge());
    }

    /**
     * Stop recording: write what is still held and take every hook down,
     * then append the game's dropped-event counts when the run counts them.
     *
     * <p>Called once the game is over, which is the moment the mana reservoir
     * is decided. Every hook comes down even when writing what is held throws,
     * and after an {@link #open()} that threw part way: taking down a hook that
     * was never installed does nothing.
     */
    @Override
    public void close() {
        try {
            if (fork) {
                // A fork played on stops where no event closes its last
                // records: the turn's end, or the game's, which a fork never
                // publishes.
                bracket.stopPlaying();
            }
            patched.close();
        } finally {
            appendDroppedEvents();
        }
    }

    /** Append the game's dropped-event counts to the lifetime's file, when the run counts them. */
    private void appendDroppedEvents() {
        if (droppedEventsFile == null) {
            return;
        }
        try {
            bracket.droppedEvents().appendTo(droppedEventsFile);
        } catch (IOException e) {
            DROPPED_EVENTS_UNWRITABLE.raise(() -> "cannot append dropped-event counts to "
                    + droppedEventsFile + ": " + e);
        }
    }

    /**
     * The priority stops' candidates, which the random seat draws from, so
     * its plays and its decision records agree (FR-033, FR-040).
     */
    public PriorityCandidates priorityCandidates() {
        return patched.priorityCandidates();
    }

    /**
     * The free activations that changed nothing this turn, which the random
     * seat does not draw again until the turn ends (FR-040).
     */
    public FruitlessActivations fruitlessActivations() {
        return bracket.fruitlessActivations();
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

        /**
         * Each step of the boundary's collection, guarded on its own: a step
         * that throws costs that step's records at this boundary and not the
         * steps after it (C6).
         */
        @Subscribe
        public void onPhase(GameEventTurnPhase event) {
            eachGuarded(
                    new BoundaryStep("continuous", () -> patched.continuous().collect()),
                    new BoundaryStep("interventions", () -> interventions.ifPresent(
                            scheduler -> scheduler.collect(event.phase()))),
                    // Before the turn-based action that deals the damage: the
                    // phase event fires ahead of it, which is the only moment
                    // a probe can fork from a board where nothing has been
                    // dealt yet.
                    new BoundaryStep("probes", () -> probes.ifPresent(ProbeScheduler::collect)),
                    // Last: a directed fork plays the turn on, and the
                    // boundary's other collection reads the live board before
                    // any fork is taken.
                    new BoundaryStep("directed forks", () -> directed.ifPresent(
                            scheduler -> scheduler.collect(event.phase()))));
        }
    }

    /** One named step of a phase boundary's collection. */
    record BoundaryStep(String name, Runnable body) {
    }

    /** Each step's first failure, reported once per JVM. */
    private static final Map<String, OnceWarning> BOUNDARY_STEP_FAILED = new ConcurrentHashMap<>();

    /**
     * Run every step in order, each whatever the ones before it threw; a
     * step's first failure is printed once.
     */
    static void eachGuarded(BoundaryStep... steps) {
        for (BoundaryStep step : steps) {
            try {
                step.body().run();
            } catch (RuntimeException failure) {
                BOUNDARY_STEP_FAILED.computeIfAbsent(step.name(), name -> new OnceWarning())
                        .raise(() -> "Effect records: the phase boundary's " + step.name()
                                + " step threw, and lost its records at that boundary"
                                + " (later failures of this step are not printed): " + failure);
            }
        }
    }
}
