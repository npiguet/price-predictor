package com.pricepredictor.connector.effects;

import forge.game.Game;

import java.util.List;
import java.util.Optional;
import java.util.function.Supplier;

/**
 * The channels the engine patch unlocks, for one game, and the hooks they
 * listen on.
 *
 * <p>{@code rewrite}, {@code trigger}, {@code playability} and {@code mana}
 * records all need a hook at a point the bus never publishes: what a replacement
 * received before it rewrote it, a trigger condition that evaluated to false, a
 * combat legality verdict the rules computed and discarded, the moment priority
 * settles, a mana ability that resolved on the inline path. Each is installed reflectively and simply does not run on
 * an unpatched checkout. {@code continuous} records need the patch's layer
 * accessors rather than a hook, and are collected at each phase boundary.
 * The state-based-loop hook writes no record of its own: {@link PhaseSignals}
 * tells the game's collectors where each loop runs, and when priority settles,
 * which is when {@link PriorityCandidates} writes the decision records.
 *
 * <p>The one owner of a game's listener hooks, the bracket's confirm listener
 * included: they are JVM statics with no game of their own, and installing
 * and uninstalling them in one place is what keeps one from outliving its
 * game.
 *
 * <p>These caps live here rather than in the supervisor, because all are
 * per-worker-process quantities the Python side cannot see: {@code --mana-cap}
 * counts resolution records per unique mana-ability text in this JVM, and
 * {@code --playability-rate}, {@code --zone-candidate-rate} and
 * {@code --plain-candidate-rate} sample the {@code decision}-subkind logging
 * points in two stages (FR-033b; the {@code attackers} and {@code blockers} ones are always logged, because
 * they fire once per combat rather than once per candidate).
 *
 * <p><b>One sampling stream per game.</b> The trigger negatives, the decision
 * sample and the mana reservoir all draw from the one {@code Random} their
 * {@link Channels} are built over, so what one channel draws shifts every
 * later draw of the others. That
 * is why each hook checks the game a report comes from before it can draw: a
 * fork's report that drew and was then dropped would re-randomise the live
 * game's sample. <b>One dedup set per game</b> likewise, shared by every
 * channel that deduplicates.
 */
final class PatchedCollectors implements AutoCloseable {

    private final CollectorContext context;
    private final Channels channels;

    /**
     * The listeners installed whatever the attribution mode: the bracket's
     * confirm listener alone.
     *
     * <p>Decline detection needs nothing the mode is decided by. It reads the
     * live bracket's own view of the stack and the cost-payment stack, never
     * the resolving-clause pointer, so a checkout that offers the confirm hook
     * without {@code SUB_ABILITY} still marks a refused offer {@code declined}
     * on a record the bracket writes off the bus anyway.
     */
    private final List<PatchHooks.Listener> everyMode;

    /**
     * Every listener the channels install, in the order they are installed,
     * and only in patched mode.
     *
     * <p>Each carries the hook it was built from, so the hook is named once,
     * where the channel builds its handler, and cannot be paired here with a
     * handler built for another.
     */
    private final List<PatchHooks.Listener> patchedOnly;

    /**
     * Whether these are a fork's collectors, which install no hook: the
     * live game's collectors hand them the fork's callbacks while it plays
     * on ({@link #routingTo}).
     */
    private final boolean forkBound;

    /**
     * The fork's collectors while one {@link #routingTo plays on} from this
     * game, or null. Volatile: the AI evaluates candidates on a thread of
     * its own, and the fork's callbacks from there must find them too.
     */
    private volatile PatchedCollectors routedFork;

    /**
     * @param setup where the game's collectors write: its context and caps, the
     *              seed of its sampling stream, and the bracket the clause and
     *              outcome hooks file their events in and whose confirm
     *              listener this installs
     */
    PatchedCollectors(Game game, Channels.Setup setup) {
        this(game, setup, false);
    }

    /**
     * The collectors of a fork played on from a live game: they install no
     * hook and take none down, and their game is not one the run counts.
     */
    static PatchedCollectors forFork(Game fork, Channels.Setup setup) {
        return new PatchedCollectors(fork, setup, true);
    }

    private PatchedCollectors(Game game, Channels.Setup setup, boolean forkBound) {
        BusBracketCollector bracket = setup.bracket();
        this.forkBound = forkBound;
        this.context = setup.context();
        this.channels = Channels.of(game, setup);
        this.everyMode = List.of(bracket.confirmHandler());
        this.patchedOnly = List.of(
                bracket.castHandler(),
                bracket.resolutionHandler(),
                channels.rewrites().handler(),
                channels.triggers().handler(),
                channels.mana().paymentHandler(),
                channels.mana().handler(),
                channels.playability().legalityHandler(),
                channels.clauses().clauseHandler(),
                channels.clauses().outcomeHandler(),
                channels.phase().stateBasedLoopHandler(),
                channels.phase().prioritySettledHandler());
    }

    /** The decision and legality channel. */
    PlayabilityChannel playability() {
        return channels.playability();
    }

    /**
     * Tell {@code sink} of this game's phase signals too, after its
     * channels: the intervention scheduler, which forces a line where
     * priority settles once the decision records are written.
     */
    void alsoTell(PhaseSignals.Sink sink) {
        channels.phase().alsoTell(sink);
    }

    /** The priority stops' candidates, which the random seat draws from. */
    PriorityCandidates priorityCandidates() {
        return channels.candidates();
    }

    /** The continuous channel, collected at each phase boundary rather than on a hook. */
    ContinuousChannel continuous() {
        return channels.continuous();
    }

    /**
     * Install every hook the checkout offers: the {@link #everyMode} ones
     * always, the channels' only in patched mode.
     *
     * @return how many were found
     */
    int install() {
        if (forkBound) {
            throw new IllegalStateException("a fork's collectors install no hook");
        }
        int installed = install(everyMode);
        if (context.mode() == AttributionMode.DEGRADED) {
            return installed;
        }
        return installed + install(patchedOnly);
    }

    /**
     * Install each listener's routing handler: the listener's own body, and
     * then the same callback for the fork playing on, if one is.
     *
     * <p>The hooks are JVM statics, so a fork's callbacks reach the live
     * game's handlers. Every channel already keeps only its own game's
     * reports, checking before it draws from the sampler or writes, so
     * offering each callback to both sets of collectors hands each its own:
     * dispatch by game identity, on whatever thread the callback comes.
     */
    private int install(List<PatchHooks.Listener> listeners) {
        int installed = 0;
        for (PatchHooks.Listener listener : listeners) {
            PatchHooks.Listener routing = listener.hook().listener(routing(
                    listener.body(), (callback, args) -> {
                        PatchedCollectors fork = routedFork;
                        if (fork != null) {
                            fork.dispatch(listener.hook(), callback, args);
                        }
                    }));
            if (routing.install()) {
                installed++;
            }
        }
        return installed;
    }

    /**
     * {@code live}'s body and then {@code fork}, whether or not the body
     * threw: the fork's collectors must see the callback the live game's
     * handler failed on. What the body threw still reaches the caller.
     */
    static PatchHooks.ListenerBody routing(PatchHooks.ListenerBody live, PatchHooks.ListenerBody fork) {
        return (callback, args) -> {
            try {
                live.on(callback, args);
            } finally {
                fork.on(callback, args);
            }
        };
    }

    /** Hand one callback of {@code hook} to this fork's own listener for it, if it has one. */
    private void dispatch(PatchHooks.Hook hook, PatchHooks.Callback callback, PatchHooks.Args args) {
        Optional<PatchHooks.Listener> own = listenerFor(everyMode, hook);
        if (own.isEmpty() && context.mode() != AttributionMode.DEGRADED) {
            own = listenerFor(patchedOnly, hook);
        }
        own.ifPresent(listener -> listener.body().on(callback, args));
    }

    private static Optional<PatchHooks.Listener> listenerFor(
            List<PatchHooks.Listener> listeners, PatchHooks.Hook hook) {
        return listeners.stream().filter(listener -> listener.hook() == hook).findFirst();
    }

    /**
     * Run {@code span} with the callbacks this game's hooks receive also
     * handed to {@code fork}'s collectors, and stop handing them afterwards
     * however the span ends.
     */
    <T> T routingTo(PatchedCollectors fork, Supplier<T> span) {
        PatchedCollectors was = routedFork;
        routedFork = fork;
        try {
            return span.get();
        } finally {
            routedFork = was;
        }
    }

    /**
     * Write what is still held and take every hook down.
     *
     * <p>The hooks come down even when the flush throws: they are JVM statics,
     * and one left installed would hand the next game's callbacks to this
     * game's channels.
     */
    @Override
    public void close() {
        if (forkBound) {
            // What the fork's channels hold, and nothing taken down: the
            // hooks are the live game's, still installed.
            channels.close(false);
            return;
        }
        try {
            // Before the hooks come down: the channels write what they hold
            // at the end of the game.
            channels.close();
        } finally {
            // Both lists, whatever the mode: uninstalling a hook that was
            // never installed does nothing.
            patchedOnly.forEach(PatchHooks.Listener::uninstall);
            everyMode.forEach(PatchHooks.Listener::uninstall);
        }
    }
}
