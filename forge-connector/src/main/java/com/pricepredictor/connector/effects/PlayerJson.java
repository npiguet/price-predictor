package com.pricepredictor.connector.effects;

import forge.card.MagicColor;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.List;

/**
 * A snapshot's {@code state.players} entry for one player: life, zone sizes,
 * this turn's tallies, and the mana it holds and could still make.
 */
final class PlayerJson {

    private PlayerJson() {
    }

    /** One player, as {@code state.players} carries it. */
    static String render(Game game, Player player) {
        return Json.object()
                .value("id", Refs.of(player))
                .value("life", player.getLife())
                .value("hand", player.getCardsIn(ZoneType.Hand).size())
                .value("library", player.getCardsIn(ZoneType.Library).size())
                .value("graveyard", player.getCardsIn(ZoneType.Graveyard).size())
                .value("poison", player.getPoisonCounters())
                .value("energy", player.getCounters(CounterEnumType.ENERGY))
                .raw("this_turn", Json.object()
                        .value("creatures_died", creaturesLost(game, player))
                        .value("spells_cast", player.getSpellsCastThisTurn())
                        .value("lands_played", player.getLandsPlayedThisTurn())
                        .render())
                // Every symbol, zeros included, in ManaCounts' fixed order.
                .raw("floating_mana",
                        ManaCounts.of(player.getManaPool()::getAmountOfColor).toJson())
                .raw("untapped_production", untappedProduction(player).toJson())
                .render();
    }

    /**
     * How much of each colour this player could still make, per colour.
     *
     * <p>Counted over untapped permanents' mana abilities, one per ability that
     * can produce the colour. It is what separates a board that can answer a
     * threat from one that only looks like it can.
     *
     * <p>An ability that makes one mana of any colour counts once for each, so
     * the numbers are an upper bound rather than a sum that can be spent.
     */
    private static ManaCounts untappedProduction(Player player) {
        List<Byte> producible = new ArrayList<>();
        for (Card card : player.getCardsIn(ZoneType.Battlefield)) {
            if (card.isTapped()) {
                continue;
            }
            for (SpellAbility ability : card.getManaAbilities()) {
                var part = ability.getManaPart();
                if (part == null) {
                    continue;
                }
                for (byte symbol : MagicColor.WUBRGC) {
                    if (part.canProduce(ManaCounts.letter(symbol), ability)) {
                        producible.add(symbol);
                    }
                }
            }
        }
        return ManaCounts.tally(producible);
    }

    /**
     * Creatures this player lost from the battlefield this turn.
     *
     * <p>The engine keeps what left the battlefield, not what died, so a
     * bounced or exiled creature counts here too. It is a close reading of a
     * condition many triggers ask about; an exact count would need a hook where
     * the death is decided.
     */
    private static int creaturesLost(Game game, Player player) {
        int count = 0;
        for (Card card : game.getLeftBattlefieldThisTurn()) {
            if (card.isCreature() && card.getController() == player) {
                count++;
            }
        }
        return count;
    }
}
