package com.pricepredictor.connector.effects;

import forge.game.Game;

import java.lang.reflect.InvocationHandler;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;

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
 * <p>Two caps live here rather than in the supervisor, because both are
 * per-worker-process quantities the Python side cannot see: {@code --mana-cap}
 * counts resolution records per unique mana-ability text in this JVM, and
 * {@code --playability-rate} samples the {@code decision}-subkind logging points
 * (the {@code attackers} and {@code blockers} ones are always logged, because
 * they fire once per combat rather than once per candidate).
 *
 * <p><b>One sampling stream per game.</b> The trigger negatives, the decision
 * sample and the mana reservoir all draw from the one {@link Random} built
 * here, so what one channel draws shifts every later draw of the others. That
 * is why each hook checks the game a report comes from before it can draw: a
 * fork's report that drew and was then dropped would re-randomise the live
 * game's sample. <b>One dedup set per game</b> likewise, shared by every
 * channel that deduplicates.
 */
final class PatchedCollectors implements AutoCloseable {

    private final AttributionMode mode;
    private final RewriteChannel rewrites;
    private final TriggerChannel triggers;
    private final PlayabilityChannel playability;
    private final ManaChannel mana;
    private final ClauseOutcomeRouter clauses;
    private final ContinuousChannel continuous;

    /** A hook this collector installs a listener on. */
    private record Listener(PatchHooks.Hook hook, Supplier<InvocationHandler> handler) {
    }

    /** Every listener this collector installs, in the order it installs them. */
    private final List<Listener> listeners;

    /**
     * @param seed    seeds the game's sampling stream
     * @param bracket where the clause and outcome hooks file their events
     */
    PatchedCollectors(
            Game game, CollectorContext context, CollectionCaps caps, long seed,
            BusBracketCollector bracket) {
        LiveGame live = new LiveGame(game);
        RecordDedup dedup = new RecordDedup();
        Random sampler = new Random(GameRecording.scramble(seed));
        this.mode = context.mode();
        this.rewrites = new RewriteChannel(live, context, dedup);
        this.triggers = new TriggerChannel(
                live, context, dedup, new TriggerNegativeSampler(sampler));
        this.playability = new PlayabilityChannel(
                live, context, dedup, sampler, caps.playabilityRate());
        this.mana = new ManaChannel(
                live, context, new ManaReservoir(caps.manaCap(), sampler));
        this.clauses = new ClauseOutcomeRouter(live, bracket);
        this.continuous = new ContinuousChannel(live, context, dedup);
        this.listeners = List.of(
                new Listener(PatchHooks.REPLACEMENT_LISTENER, rewrites::handler),
                new Listener(PatchHooks.TRIGGER_LISTENER, triggers::handler),
                new Listener(PatchHooks.PLAYABILITY_LISTENER, playability::decisionHandler),
                new Listener(PatchHooks.MANA_LISTENER, mana::handler),
                new Listener(PatchHooks.COMBAT_LISTENER, playability::legalityHandler),
                new Listener(PatchHooks.CLAUSE_LISTENER, clauses::clauseHandler),
                new Listener(PatchHooks.OUTCOME_LISTENER, clauses::outcomeHandler));
    }

    RewriteChannel rewrites() {
        return rewrites;
    }

    TriggerChannel triggers() {
        return triggers;
    }

    /** The decision and legality channel, which the random seat also writes through. */
    PlayabilityChannel playability() {
        return playability;
    }

    ManaChannel mana() {
        return mana;
    }

    ClauseOutcomeRouter clauses() {
        return clauses;
    }

    /** The continuous channel, collected at each phase boundary rather than on a hook. */
    ContinuousChannel continuous() {
        return continuous;
    }

    /**
     * Install every hook the checkout offers.
     *
     * @return how many were found
     */
    int install() {
        if (mode == AttributionMode.DEGRADED) {
            return 0;
        }
        int installed = 0;
        for (Listener listener : listeners) {
            if (listener.hook().install(listener.handler().get())) {
                installed++;
            }
        }
        return installed;
    }

    @Override
    public void close() {
        // Before the hooks come down: this is the last moment the game can be
        // said to have ended, and the mana reservoir is only decided then.
        mana.flush();
        for (Listener listener : listeners) {
            listener.hook().uninstall();
        }
        // Last, and outside the hook teardown: the question the tally answers
        // is about the run, so it is asked once per game whatever the game did.
        RewriteChannel.JVM_REWRITES.endOfGame();
    }
}
