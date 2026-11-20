package com.pricepredictor.connector.effects;

import forge.game.Game;

import java.util.List;
import java.util.Random;

/**
 * One game's patched channels, built over one game, one dedup set and one
 * sampling stream.
 *
 * <p>The one place they are assembled. The trigger negatives, the decision
 * sample and the mana reservoir share the sampler, and construction order is
 * part of that sharing, so a second assembly — in a test that drives a
 * channel by hand, say — would be a second sampling stream that only looks
 * like the first.
 *
 * <p>Closing writes what the attackers declaration and the mana channel hold
 * and ends the game for the rewrite tally. It installs no hook, so it takes
 * none down; that is {@link PatchedCollectors}' part.
 */
record Channels(
        RewriteChannel rewrites,
        TriggerChannel triggers,
        PlayabilityChannel playability,
        PriorityCandidates candidates,
        ManaChannel mana,
        ClauseOutcomeRouter clauses,
        ContinuousChannel continuous,
        PhaseSignals phase) implements AutoCloseable {

    /**
     * What the channels are assembled from, beside the game.
     *
     * @param seed    seeds the game's sampling stream
     * @param bracket where the clause and outcome hooks file their events,
     *                and the state-based loop's signals go; the
     *                priority-settled signal goes to the candidates
     */
    record Setup(CollectorContext context, CollectionCaps caps, long seed, BusBracketCollector bracket) {
    }

    static Channels of(Game game, Setup setup) {
        CollectorContext context = setup.context();
        CollectionCaps caps = setup.caps();
        BusBracketCollector bracket = setup.bracket();
        LiveGame live = new LiveGame(game);
        RecordDedup dedup = new RecordDedup();
        Random sampler = SeedStreams.random(setup.seed(), SeedStreams.Consumer.SAMPLER);
        PlayabilityChannel playability =
                new PlayabilityChannel(live, context, dedup, sampler, caps);
        PriorityCandidates candidates = new PriorityCandidates(live, playability);
        RewriteChannel rewrites = new RewriteChannel(live, context, dedup);
        ManaChannel mana = new ManaChannel(
                live, context, new RecordReservoir(caps.manaCap(), sampler), bracket);
        if (bracket != null) {
            // A buffered ability's run and a mana ability's own effects are
            // those channels' records, never the open bracket's.
            bracket.addTap(rewrites);
            bracket.addTap(mana);
        }
        return new Channels(
                rewrites,
                new TriggerChannel(live, context, dedup, new TriggerNegativeSampler(sampler),
                        new RecordReservoir(TriggerNegativeSampler.POSITIVES_PER_LINE, sampler)),
                playability,
                candidates,
                mana,
                new ClauseOutcomeRouter(live, bracket, mana),
                new ContinuousChannel(live, context, dedup),
                new PhaseSignals(live, PhaseSignals.Sink.both(
                        PhaseSignals.Sink.both(bracket.phaseSink(), candidates.sink()),
                        rewrites.phaseSink())));
    }

    @Override
    public void close() {
        close(true);
    }

    /**
     * @param countGame whether this was a game of the run's, which the
     *                  rewrite tally counts; a fork played on from one is not
     */
    void close(boolean countGame) {
        RuntimeException failed = null;
        // Each in its own try, so one that throws costs its own records and
        // not the others' (V3 #5); the first failure is thrown once all ran,
        // and the game then gets no end marker.
        for (Runnable step : List.<Runnable>of(
                // A declaration of attackers the game ended before it was made.
                playability::close,
                // A buffered ability that never ran.
                rewrites::writeWaitingAtGameEnd,
                // This is the last moment the game can be said to have ended,
                // and the reservoirs are only decided then.
                triggers::flush,
                mana::flush)) {
            try {
                step.run();
            } catch (RuntimeException failure) {
                if (failed == null) {
                    failed = failure;
                } else {
                    failed.addSuppressed(failure);
                }
            }
        }
        // Whatever the steps did: the question the tally answers is about the
        // run, so it is asked once per game whatever the game did.
        if (countGame) {
            RewriteChannel.JVM_REWRITES.endOfGame();
        }
        if (failed != null) {
            throw failed;
        }
    }
}
