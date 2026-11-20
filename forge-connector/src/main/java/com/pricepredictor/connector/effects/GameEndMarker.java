package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.json.Json;
import forge.game.Game;
import forge.game.player.Player;

import java.util.List;

/**
 * The record that says a game's records are complete (D2): a
 * {@code turn_action} of subkind {@code game_end}, written once per game as
 * its last record — after the bracket's end-of-game records, the trigger and
 * mana reservoirs' flushes and every kept fork's lines — and only for a game
 * that ended. A worker killed or a game that threw mid-match writes none, so
 * the corpus build can drop a game whose shard holds no marker for its
 * {@code game_id}.
 *
 * <p>A turn action because it is a record no ability caused, which no curated
 * stratum holds, and whose payload readers already decode: {@code events}
 * (empty), and {@code records_withheld}, the records
 * {@code --max-records-per-game} kept out of the shard for this game. Its
 * {@code actor_player} is the player whose turn the game ended in, its
 * {@code state} the board at the end. Written past the record cap.
 */
final class GameEndMarker {

    private GameEndMarker() {
    }

    /** Write {@code game}'s marker through {@code context}. */
    static void write(Game game, CollectorContext context) {
        LiveGame live = new LiveGame(game);
        context.emitAtGameEnd(context.newTurnAction(TurnActionSubkind.GAME_END)
                .actor(actorOf(game, live))
                .state(live.snapshots().ofBoard().state())
                .payload(RecordPayload.withoutEvents(payload(context.withheld()))));
    }

    /** {@code {"events": [], "records_withheld": n}}. */
    static String payload(int withheld) {
        return Json.object()
                .raw("events", "[]")
                .value("records_withheld", withheld)
                .render();
    }

    /** The player whose turn the game ended in, else its first registered player. */
    private static String actorOf(Game game, LiveGame live) {
        String active = live.activePlayerId();
        if (active != null) {
            return active;
        }
        List<Player> players = game.getRegisteredPlayers();
        return players.isEmpty() ? null : Refs.of(players.get(0));
    }
}
