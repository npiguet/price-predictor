package com.pricepredictor.connector.effects;

import forge.game.Game;

import java.util.List;

/**
 * The channels the engine patch unlocks, for one game, and the hooks they
 * listen on.
 *
 * <p>{@code rewrite}, {@code trigger}, {@code playability} and {@code mana}
 * records all need a hook at a point the bus never publishes: what a replacement
 * received before it rewrote it, a trigger condition that evaluated to false, a
 * legality verdict the rules computed and discarded, a mana ability that resolved
 * on the inline path. Each is installed reflectively and simply does not run on
 * an unpatched checkout. {@code continuous} records need the patch's layer
 * accessors rather than a hook, and are collected at each phase boundary.
 *
 * <p>The one owner of a game's listener hooks, the bracket's confirm listener
 * included: they are JVM statics with no game of their own, and installing
 * and uninstalling them in one place is what keeps one from outliving its
 * game.
 *
 * <p>Two caps live here rather than in the supervisor, because both are
 * per-worker-process quantities the Python side cannot see: {@code --mana-cap}
 * counts resolution records per unique mana-ability text in this JVM, and
 * {@code --playability-rate} samples the {@code decision}-subkind logging points
 * (the {@code attackers} and {@code blockers} ones are always logged, because
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
     * @param seed    seeds the game's sampling stream
     * @param bracket where the clause and outcome hooks file their events,
     *                and whose confirm listener this installs
     */
    PatchedCollectors(
            Game game, CollectorContext context, CollectionCaps caps, long seed,
            BusBracketCollector bracket) {
        this.context = context;
        this.channels = Channels.of(game, context, caps, seed, bracket);
        this.everyMode = List.of(bracket.confirmHandler());
        this.patchedOnly = List.of(
                channels.rewrites().handler(),
                channels.triggers().handler(),
                channels.playability().decisionHandler(),
                channels.mana().handler(),
                channels.playability().legalityHandler(),
                channels.clauses().clauseHandler(),
                channels.clauses().outcomeHandler());
    }

    /** The decision and legality channel, which the random seat also writes through. */
    PlayabilityChannel playability() {
        return channels.playability();
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
        int installed = install(everyMode);
        if (context.mode() == AttributionMode.DEGRADED) {
            return installed;
        }
        return installed + install(patchedOnly);
    }

    private static int install(List<PatchHooks.Listener> listeners) {
        int installed = 0;
        for (PatchHooks.Listener listener : listeners) {
            if (listener.install()) {
                installed++;
            }
        }
        return installed;
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
