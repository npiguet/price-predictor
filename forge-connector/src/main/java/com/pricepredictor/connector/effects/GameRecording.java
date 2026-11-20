package com.pricepredictor.connector.effects;

import com.google.common.eventbus.Subscribe;
import forge.game.Game;
import forge.game.event.GameEventTurnPhase;
import forge.game.phase.PhaseType;
import forge.game.spellability.SpellAbility;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
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
     * Build one game's collectors from {@code request}: its game id, its
     * seed and the rest of what the run decides ({@link GameRecordingRequest}).
     */
    public GameRecording(GameRecordingRequest request) {
        this.game = request.game();
        this.fork = false;
        CollectionCaps caps = request.caps();
        long seed = request.seed();
        this.caps = caps;
        SeedStreams.register(game, seed);
        AttributionMode mode = AttributionMode.detect();
        this.envelope = request.newContext(mode);
        // Stage three. Nothing is copied unless a budget says so, because a
        // fork costs a game copy and a stack resolution, and nothing on a
        // stock checkout, where no fork could be attributed. Both kinds share
        // one game's budget.
        boolean forking = mode == AttributionMode.PATCHED
                && (caps.interventionsPerGame() > 0 || caps.probesEnabled());
        ForkBudget budget = new ForkBudget(caps);
        this.playForward = new ForkPlayForward(this, request.forkSeats(), seed);
        this.interventions = when(forking, () -> interventionsOf(request, budget));
        this.probes = when(forking, () -> new ProbeScheduler(
                game, request.gameId(), caps,
                new DamageStepProbe(game, request.newContext(mode), budget)));
        // A probe forks before the damage step and cannot know which combat
        // record it mirrors until that record is written.
        this.bracket = new BusBracketCollector(game, request.newContext(mode),
                recordId -> probes.ifPresent(probe -> probe.writeHeldProbes(recordId)));
        this.patched = new PatchedCollectors(
                game, request.newContext(mode), caps, seed, bracket);
        // Forced at a step's first priority, after the decision records.
        interventions.map(InterventionScheduler::prioritySink).ifPresent(patched::alsoTell);
        this.droppedEventsFile = caps.countDroppedEvents()
                ? droppedEventsFile(request.writer()) : null;
        // Directed forks, on a patched checkout as the other two kinds: the
        // fork plays on through the live game's hooks.
        this.directed = when(mode == AttributionMode.PATCHED && caps.directedForksEnabled(),
                () -> directedOf(request, budget));
    }

    private static <T> Optional<T> when(boolean wanted, Supplier<T> make) {
        return wanted ? Optional.of(make.get()) : Optional.empty();
    }

    /**
     * Forced resolutions and protection forks; a protection fork (FR-058) is
     * an intervention played on, so it is taken on a run taking interventions,
     * and only there.
     */
    private InterventionScheduler interventionsOf(GameRecordingRequest request, ForkBudget budget) {
        long seed = request.seed();
        int perGame = caps.interventionsPerGame();
        return new InterventionScheduler(
                game, new InterventionFork(game, this, budget, seed),
                perGame > 0 ? new ProtectionFork(game, budget, playForward, seed) : null,
                request.decks().steered(),
                InterventionScheduler.spreadOverTheGame(seed, perGame));
    }

    private DirectedForkScheduler directedOf(GameRecordingRequest request, ForkBudget budget) {
        return new DirectedForkScheduler(
                game, new DirectedForkScheduler.Directing(caps, request.directedRules(), request.seed()),
                budget, request.tally(),
                new DirectedFork(game, playForward,
                        () -> game.getPhaseHandler().getPlayerTurn())::take);
    }

    /**
     * A fork's recording: the live game's collectors, built over the fork
     * and writing through {@code context}, with no fork of its own and no
     * hook installed (D4).
     *
     * <p>No intervention, probe or directed fork is scheduled from a fork,
     * so a fork never forks. Its dropped events are not appended: the counts
     * file has one line per real game, and a kept fork's are counted into
     * the live game's, under the fork's kind ({@link #countForkDropped}).
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
     * @param output where the fork's records go and what they carry
     * @param seed   seeds the fork's sampling stream
     */
    GameRecording forkRecording(Game fork, ForkOutput output, long seed) {
        return new GameRecording(fork, envelope.forFork(output), caps, seed);
    }

    /** The link a cast or activated item's halves carry in this game's records. */
    String linkOf(SpellAbility item) {
        return CastTracker.linkOf(envelope.gameId(), item);
    }

    /** Plays forks of this game on to the end of the turn (D4). */
    ForkPlayForward playForward() {
        if (playForward == null) {
            throw new IllegalStateException("a fork's recording plays no fork on");
        }
        return playForward;
    }

    /**
     * A fork's premise, on a fork's recording: the state-based check its
     * edit makes, run with its collectors listening and written as a {@code
     * fork_edit} record ({@link ForkPremise}).
     */
    void settlePremise() {
        ForkPremise premise = new ForkPremise(game, envelope);
        bracket.addTap(premise);
        // The premise's snapshot is its stats baseline, as any record's: the
        // fork's reader has seen no card yet, and the first change of a card
        // it has not seen is read as its baseline, so a creature whose Aura
        // the check drops would show no pt_change.
        bracket.rebaselineAtSnapshot();
        premise.settle();
    }

    /** This recording's events that reached no record so far. */
    DroppedEvents droppedEvents() {
        return bracket.droppedEvents();
    }

    /**
     * Count a kept fork's dropped events into this game's, under the fork's
     * {@code kind}, so the game's counts file holds them.
     */
    void countForkDropped(String kind, DroppedEvents forkDropped) {
        bracket.droppedEvents().addFork(kind, forkDropped);
    }

    /** Write a kept fork's lines to this game's shard. */
    void writeForkLines(List<ShardLine> lines) {
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
        // A subscriber that throws loses the events it was handling, which
        // the bus swallows: counted, for this game's and every fork's.
        HookFailures.watchSubscribers();
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
        boolean flushed = false;
        try {
            if (fork) {
                // A fork played on stops where no event closes its last
                // records: the turn's end, or the game's, whose finished
                // event Forge publishes only from a match. A fork that ends
                // the game does publish its outcome, filed as a live one is.
                bracket.stopPlaying();
            }
            patched.close();
            flushed = true;
        } finally {
            try {
                closeEnvelope(flushed);
            } finally {
                // After the marker, whose board is the last snapshot: the
                // registries keyed by game would otherwise hold it, and with
                // it every card and fork of it, for the rest of the JVM's life.
                LibraryCapture.release(game);
                FloatingEffects.release(game);
            }
        }
    }

    /** The dropped-event counts, then a live game's marker and its shard's game end. */
    private void closeEnvelope(boolean flushed) {
        appendDroppedEvents();
        if (fork) {
            return;
        }
        try {
            // Last of the game's records, and only for a game that ended
            // with all of them written: one that threw mid-match, whose
            // flush threw, or whose outcome reached no record, gets none (D2).
            if (endedWhole(flushed)) {
                writeMarker(envelope.gameId(), () -> GameEndMarker.write(game, envelope));
            }
        } finally {
            envelope.endGame();
        }
    }

    /**
     * Whether the game ended with every end-of-game write done: its flush
     * did not throw, the shard flushed every block it was handed, and the
     * record holding its outcome was written.
     */
    private boolean endedWhole(boolean flushed) {
        return flushed && !envelope.flushFailed() && game.isGameOver()
                && bracket.outcomeIsWritten();
    }

    /** The first marker that could not be written, printed once per worker. */
    private static final OnceWarning MARKER_UNWRITTEN = new OnceWarning();

    /**
     * Write a game's marker, or count the failure and go on: the game then
     * has no marker and reads as cut, and the match and its later games are
     * played on. The failure is counted in {@link HookFailures} as
     * {@code "game-end marker"} and its first one printed.
     *
     * @return whether the marker was written
     */
    static boolean writeMarker(String gameId, Runnable write) {
        try {
            write.run();
            return true;
        } catch (RuntimeException failure) {
            HookFailures.countContained("game-end marker");
            MARKER_UNWRITTEN.raise(() -> "Effect records: the game-end marker of game " + gameId
                    + " could not be written, so the game reads as cut (later failures are"
                    + " counted, not printed): " + failure);
            return false;
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
            boolean collecting = collectsTheBoard(event.phase());
            boolean forking = forksAt(game);
            eachGuarded(
                    new BoundaryStep("continuous", () -> {
                        if (collecting) {
                            patched.continuous().collect();
                        }
                    }),
                    // The step's forced resolution waits for its first
                    // priority (InterventionScheduler.prioritySink); its
                    // protection fork is taken here.
                    new BoundaryStep("interventions", () -> interventions.ifPresent(scheduler -> {
                        scheduler.stepBegan();
                        if (collecting && forking) {
                            scheduler.protect(event.phase());
                        }
                    })),
                    // Before the turn-based action that deals the damage: the
                    // phase event fires ahead of it, which is the only moment
                    // a probe can fork from a board where nothing has been
                    // dealt yet.
                    new BoundaryStep("probes", () -> probes.ifPresent(probe -> {
                        if (forking) {
                            probe.collect();
                        }
                    })),
                    // Last: a directed fork plays the turn on, and the
                    // boundary's other collection reads the live board before
                    // any fork is taken.
                    new BoundaryStep("directed forks", () -> directed.ifPresent(scheduler -> {
                        if (forking) {
                            scheduler.collect(event.phase());
                        }
                    })));
        }
    }

    /**
     * Whether the boundary beginning {@code phase} reads the board for
     * continuous records and forces resolutions: everywhere but the untap
     * step. No player gets priority there (CR 502.4), and the live game has
     * not yet re-applied its statics for the new turn (a "during your turn"
     * keyword), so a continuous record could describe a board the rules do
     * not hold, and a forced resolution would spend the budget at a step
     * nobody can act in; upkeep reads the same statics again. The untap
     * step's turn_action records and its directed forks (a stun counter, a
     * skipped upkeep trigger) are taken there all the same.
     */
    static boolean collectsTheBoard(PhaseType phase) {
        return phase != PhaseType.UNTAP;
    }

    /**
     * Whether a fork may be taken at a phase event now: not while the stack
     * resolves. An effect that ends the turn or the combat phase fires the
     * phase events of the steps it moves to from inside its own resolution,
     * and a copy taken there is a game mid-resolution, which no fork is
     * built to play on.
     */
    static boolean forksAt(Game game) {
        return !game.getStack().isResolving();
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
                HookFailures.countContained("boundary step " + step.name());
                BOUNDARY_STEP_FAILED.computeIfAbsent(step.name(), name -> new OnceWarning())
                        .raise(() -> "Effect records: the phase boundary's " + step.name()
                                + " step threw, and lost its records at that boundary"
                                + " (later failures of this step are not printed): " + failure);
            }
        }
    }
}
