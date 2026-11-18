package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.json.Json;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.player.Player;
import forge.game.zone.ZoneType;

import java.util.List;

/**
 * A snapshot's {@code state.players} entry for one player: life, zone sizes,
 * this turn's tallies and the land plays it allows, the mana it holds and could still make, and the
 * player-wide designations (speed, the city's blessing, rad counters, the
 * monarch, the initiative).
 */
final class PlayerJson {

    private PlayerJson() {
    }

    /**
     * One player, as {@code state.players} carries it, on {@code board}: a
     * continuous record's {@code untapped_production} leaves out what its
     * static does ({@link UntappedProduction}).
     */
    static String render(Game game, Player player, StaticSuppression board) {
        return Json.object()
                .value("id", Refs.of(player))
                .value("life", player.getLife())
                .value("hand", player.getCardsIn(ZoneType.Hand).size())
                .value("library", player.getCardsIn(ZoneType.Library).size())
                .value("graveyard", player.getCardsIn(ZoneType.Graveyard).size())
                .value("poison", player.getPoisonCounters())
                .value("energy", player.getCounters(CounterEnumType.ENERGY))
                .raw("this_turn", Json.object()
                        .value("creatures_died", creaturesDied(game, player))
                        .value("spells_cast", player.getSpellsCastThisTurn())
                        .value("lands_played", player.getLandsPlayedThisTurn())
                        .render())
                // What bounds this_turn.lands_played: 1 plus every
                // AdjustLandPlays$ raise in force, null when unlimited.
                .value("max_land_plays", player.getMaxLandPlaysInfinite()
                        ? null : player.getMaxLandPlays())
                // Every symbol, zeros included, in ManaCounts' fixed order.
                .raw("floating_mana",
                        ManaCounts.of(player.getManaPool()::getAmountOfColor).toJson())
                .raw("untapped_production", UntappedProduction.of(game, player, board).toJson())
                .value("speed", player.getSpeed())
                .value("city_blessing", player.hasBlessing())
                .value("rad", player.getCounters(CounterEnumType.RAD))
                .value("monarch", game.getMonarch() == player)
                .value("initiative", player.hasInitiative())
                .raw("turn_counts", turnCountsJson(player))
                .render();
    }

    /**
     * What this player's turn has held so far, beside {@code this_turn}'s
     * three counts (FR-024, {@code in.history}): many conditions ask about
     * the turn rather than the board ("if you gained life this turn",
     * revolt, descend), and the board alone cannot answer them.
     *
     * <p>{@code permanents_to_graveyard} counts the cards that came to this
     * player's graveyard from the battlefield this turn, tokens included;
     * {@code descended} is Forge's count of permanent cards put into it from
     * anywhere. {@code descended} and {@code dungeons_completed} are
     * whole-game counts, as Forge keeps them; every other count resets at the
     * turn.
     */
    private static String turnCountsJson(Player player) {
        return Json.object()
                .value("cards_drawn", player.getNumDrawnThisTurn())
                .value("life_gained", player.getLifeGainedThisTurn())
                .value("life_lost", player.getLifeLostThisTurn())
                .value("damage_taken", player.getAssignedDamage())
                .value("permanents_to_graveyard", player.getZone(ZoneType.Graveyard)
                        .getCardsAddedThisTurn(ZoneType.Battlefield).size())
                .value("spells_cast_last_turn", player.getSpellsCastLastTurn())
                .value("dungeons_completed", player.getCompletedDungeons().size())
                .value("descended", player.getDescended())
                .render();
    }

    /**
     * Creatures that died under this player's control this turn: put into a
     * graveyard from the battlefield (700.4), as morbid counts them.
     *
     * <p>Read off every graveyard's cards put there from the battlefield this
     * turn, the list {@code permanents_to_graveyard} counts for its owner:
     * Forge keeps each as it last was on the battlefield, so an animated
     * land that died counts, a token counts though it has ceased to exist,
     * and a stolen creature counts for the player who controlled it, whose
     * graveyard did not receive it. A bounced, blinked or exiled creature
     * left without dying and does not count.
     *
     * <p>The controller is matched by id, as Forge matches players: a fork's
     * copy carries the live game's last-known copies, whose controller is the
     * live game's player.
     */
    private static int creaturesDied(Game game, Player player) {
        int count = 0;
        for (Player owner : game.getRegisteredPlayers()) {
            count += diedUnder(player, owner.getZone(ZoneType.Graveyard)
                    .getCardsAddedThisTurn(ZoneType.Battlefield));
        }
        return count;
    }

    /** How many of {@code died} were creatures {@code player} controlled. */
    private static int diedUnder(Player player, List<Card> died) {
        int count = 0;
        for (Card card : died) {
            if (card.isCreature() && player.equals(card.getController())) {
                count++;
            }
        }
        return count;
    }
}
