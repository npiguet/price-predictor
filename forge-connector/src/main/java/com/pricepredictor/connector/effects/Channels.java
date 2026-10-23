package com.pricepredictor.connector.effects;

import forge.game.Game;

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
 * <p>Closing writes what the mana channel holds and ends the game for the
 * rewrite tally. It installs no hook, so it takes none down; that is
 * {@link PatchedCollectors}' part.
 */
record Channels(
        RewriteChannel rewrites,
        TriggerChannel triggers,
        PlayabilityChannel playability,
        ManaChannel mana,
        ClauseOutcomeRouter clauses,
        ContinuousChannel continuous) implements AutoCloseable {

    /**
     * @param seed    seeds the game's sampling stream
     * @param bracket where the clause and outcome hooks file their events
     */
    static Channels of(
            Game game, CollectorContext context, CollectionCaps caps, long seed,
            BusBracketCollector bracket) {
        LiveGame live = new LiveGame(game);
        RecordDedup dedup = new RecordDedup();
        Random sampler = new Random(GameRecording.scramble(seed));
        return new Channels(
                new RewriteChannel(live, context, dedup),
                new TriggerChannel(live, context, dedup, new TriggerNegativeSampler(sampler)),
                new PlayabilityChannel(live, context, dedup, sampler, caps.playabilityRate()),
                new ManaChannel(live, context, new ManaReservoir(caps.manaCap(), sampler)),
                new ClauseOutcomeRouter(live, bracket),
                new ContinuousChannel(live, context, dedup));
    }

    @Override
    public void close() {
        try {
            // This is the last moment the game can be said to have ended, and
            // the mana reservoir is only decided then.
            mana.flush();
        } finally {
            // In the finally: the question the tally answers is about the run,
            // so it is asked once per game whatever the game did.
            RewriteChannel.JVM_REWRITES.endOfGame();
        }
    }
}
