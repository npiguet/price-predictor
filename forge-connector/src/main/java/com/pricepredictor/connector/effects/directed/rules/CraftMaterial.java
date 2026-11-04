package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.RemovePermanentOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CardCollection;
import forge.game.card.CardLists;
import forge.game.cost.CostExile;
import forge.game.cost.CostPart;
import forge.game.keyword.Keyword;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Optional;

/**
 * 702.167a, craft exiles the permanent and its materials as a cost and
 * returns the card transformed, on scenario {@code 702.167a-1}: A crafts
 * Braided Net with Short Sword into Braided Quipu.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it controls a permanent with craft whose cost's mana it has the
 * lands for, and exactly as many permanents it controls and cards in its
 * graveyard can be exiled as the craft's materials as the craft needs, one
 * of them a permanent. It binds that permanent.
 *
 * <p>Variant {@code no_material} ({@code remove_permanent}): that permanent
 * is gone, so the craft is one material short and cannot be activated.
 */
public final class CraftMaterial {

    public static final String RULE = "702.167a";
    static final String SCENARIO = "702.167a-1";
    static final String NO_MATERIAL = "no_material";

    private CraftMaterial() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, CraftMaterial::bind, NO_MATERIAL,
                new Application(ForkOperation.REMOVE_PERMANENT,
                        fork -> new RemovePermanentOperation(fork.card("material"))));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        for (Card host : active.getCardsIn(ZoneType.Battlefield)) {
            for (SpellAbility craft : host.getSpellAbilities()) {
                if (!craft.isKeyword(Keyword.CRAFT) || craft.getPayCosts() == null
                        || Boards.manaValue(craft.getPayCosts()) > Boards.untappedLands(active)) {
                    continue;
                }
                Optional<Card> material = soleBattlefieldMaterial(active, host, craft);
                if (material.isPresent()) {
                    return Optional.of(Binding.empty().with("material", material.get()));
                }
            }
        }
        return Optional.empty();
    }

    /**
     * A permanent among the craft's materials when the player has exactly as
     * many as the craft exiles, else empty.
     */
    private static Optional<Card> soleBattlefieldMaterial(Player player, Card host, SpellAbility craft) {
        for (CostPart part : craft.getPayCosts().getCostParts()) {
            if (!(part instanceof CostExile exile) || exile.payCostFromSource()) {
                continue;
            }
            Optional<Integer> amount = Boards.number(exile.getAmount());
            if (amount.isEmpty()) {
                return Optional.empty();
            }
            CardCollection pool = new CardCollection(player.getCardsIn(ZoneType.Battlefield));
            pool.addAll(player.getCardsIn(ZoneType.Graveyard));
            List<Card> materials = CardLists.getValidCards(pool, exile.getType(), player, host, craft).stream()
                    .filter(card -> card != host)
                    .toList();
            if (materials.size() != amount.get()) {
                return Optional.empty();
            }
            return materials.stream().filter(Card::isInPlay).findFirst();
        }
        return Optional.empty();
    }
}
