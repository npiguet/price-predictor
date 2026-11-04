package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.RemovePermanentOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CardLists;
import forge.game.cost.CostPart;
import forge.game.cost.CostSacrifice;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Optional;

/**
 * 118.11, a cost whose action a replacement effect modifies is still paid, on
 * scenario {@code 118.11-1}: A sacrifices Grizzly Bears to Goblin
 * Bombardment while Rest in Peace exiles it instead, and the ability still
 * resolves.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, a permanent on the battlefield replaces moves to a graveyard
 * ({@link GraveyardReplacement#replacesGraveyardMoves}, Rest in Peace), and
 * the active player controls a permanent with an activated ability whose cost
 * sacrifices one other permanent, for which it controls exactly one
 * candidate, a creature. It binds that creature and the ability's permanent.
 *
 * <p>Variant {@code nothing_to_sacrifice} ({@code remove_permanent}): the
 * creature is not on the battlefield, so nothing can be sacrificed and the
 * ability cannot be activated.
 */
public final class ReplacedSacrificeCost {

    public static final String RULE = "118.11";
    static final String SCENARIO = "118.11-1";
    static final String NOTHING_TO_SACRIFICE = "nothing_to_sacrifice";

    private ReplacedSacrificeCost() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, ReplacedSacrificeCost::bind, NOTHING_TO_SACRIFICE,
                new Application(ForkOperation.REMOVE_PERMANENT,
                        fork -> new RemovePermanentOperation(fork.card("creature"))));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)
                || live.getCardsIn(ZoneType.Battlefield).stream().noneMatch(GraveyardReplacement::replacesGraveyardMoves)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        for (Card permanent : active.getCardsIn(ZoneType.Battlefield)) {
            for (SpellAbility ability : permanent.getSpellAbilities()) {
                Optional<Card> only = soleSacrifice(ability, active);
                if (only.isPresent()) {
                    return Optional.of(Binding.empty().with("creature", only.get()).with("permanent", permanent));
                }
            }
        }
        return Optional.empty();
    }

    /**
     * The one creature the activated ability's cost could sacrifice, when its
     * cost sacrifices one permanent other than its own and the activator
     * controls exactly one that qualifies.
     */
    static Optional<Card> soleSacrifice(SpellAbility ability, Player activator) {
        if (!ability.isActivatedAbility() || ability.getPayCosts() == null) {
            return Optional.empty();
        }
        for (CostPart part : ability.getPayCosts().getCostParts()) {
            if (!(part instanceof CostSacrifice sacrifice) || sacrifice.payCostFromSource()
                    || !Integer.valueOf(1).equals(sacrifice.convertAmount())) {
                continue;
            }
            List<Card> candidates = CardLists.getValidCards(activator.getCardsIn(ZoneType.Battlefield),
                    sacrifice.getType().split(";"), activator, ability.getHostCard(), ability);
            if (candidates.size() == 1 && candidates.get(0).isCreature()) {
                return Optional.of(candidates.get(0));
            }
        }
        return Optional.empty();
    }
}
