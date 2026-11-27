package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

import java.nio.file.Path;

/**
 * Forks over a real, started board, for the tests of what a fork spends and
 * writes: the copy is the one production takes, so a fork that is discarded
 * is discarded for a reason the board gives.
 */
final class RealForks implements AutoCloseable {

    private static final long SEED = 42L;

    private final Game game;
    private final GameRecording live;
    private final RecordShardWriter shard;
    /** Forced resolutions over the board, writing through {@link #shard}. */
    final InterventionFork fork;
    /** The player whose hand holds {@link #bolt}. */
    final Player holder;
    /** A Lightning Bolt in {@link #holder}'s hand: an ability the copy has a counterpart for. */
    final SpellAbility bolt;

    private RealForks(Path dir, ForkBudget budget, CollectionCaps caps) {
        game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        holder = game.getPlayers().get(0);
        TestBoard.inPlay(game, game.getPlayers().get(1), "Grizzly Bears");
        bolt = TestBoard.inHand(game, holder, "Lightning Bolt").getFirstSpellAbility();
        shard = ShardRun.writer(dir);
        live = new GameRecording(
                GameRecordingRequest.of(game, shard, ShardRun.GAME_ID, caps).seeded(1L));
        fork = new InterventionFork(game, live, budget, SEED);
    }

    /** Interventions over a Lightning Bolt in a hand and a Grizzly Bears to aim it at. */
    static RealForks interventions(Path dir, ForkBudget budget, CollectionCaps caps) {
        return new RealForks(dir, budget, caps);
    }

    /** An ability of a card of this game that sits in no zone: the copy holds no counterpart of it. */
    SpellAbility abilityOfACardInNoZone() {
        Card unplaced = TestBoard.card(game, holder, "Lightning Bolt");
        return unplaced.getFirstSpellAbility();
    }

    /** The shard, closed so what the forks wrote is on disk; absent when nothing was written. */
    Path closedShard() {
        return ShardRun.closed(shard);
    }

    @Override
    public void close() {
        live.close();
        shard.close();
    }
}
