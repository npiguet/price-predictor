package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.PlayabilityVerdict;
import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForcedPlay;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.AddPermanentOperation;
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
 * begins, the active player controls a permanent with an activated ability
 * whose cost sacrifices one other permanent, which the rules let it activate
 * now ({@link PlayabilityVerdict}), and a creature that cost could sacrifice,
 * whose move to a graveyard no permanent on the battlefield replaces
 * ({@link Boards#replacesGraveyardMoveOf}). It binds the active player, the
 * first such creature and the ability's permanent.
 *
 * <p>Variant {@code replaced_sacrifice} ({@code add_permanent}): Rest in
 * Peace is on the battlefield under the active player's control, so the
 * sacrificed creature is exiled instead. The fork forces the activation, its
 * cost sacrificing the bound creature (FR-010b), whose replaced move is the
 * point; Forge's AI rarely sacrifices a creature to an outlet. The
 * replacement's rewrite record is written as the cost is paid, with the
 * stack empty.
 */
public final class ReplacedSacrificeCost {

    public static final String RULE = "118.11";
    static final String SCENARIO = "118.11-1";
    static final String REPLACED_SACRIFICE = "replaced_sacrifice";
    static final String REST_IN_PEACE = "Rest in Peace";

    private ReplacedSacrificeCost() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, ReplacedSacrificeCost::bind, REPLACED_SACRIFICE,
                new Application(ForkOperation.ADD_PERMANENT,
                        fork -> AddPermanentOperation.of(REST_IN_PEACE, fork.player("player"), false))
                        .forcing(ForcedPlay.activate("player", "permanent",
                                new ForcedPlay.Line("sacrifice outlet", ReplacedSacrificeCost::sacrificesAnother))
                                .paying("creature")));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        for (Card permanent : active.getCardsIn(ZoneType.Battlefield)) {
            for (SpellAbility ability : permanent.getSpellAbilities()) {
                if (!sacrificesAnother(ability)) {
                    continue;
                }
                Optional<Card> creature = sacrificeable(ability, active).stream()
                        .filter(candidate -> live.getCardsIn(ZoneType.Battlefield).stream()
                                .noneMatch(replacer -> Boards.replacesGraveyardMoveOf(replacer, candidate)))
                        .findFirst();
                if (creature.isPresent() && PlayabilityVerdict.of(ability, active).playable()) {
                    return Optional.of(Binding.empty().with("player", active)
                            .with("creature", creature.get()).with("permanent", permanent));
                }
            }
        }
        return Optional.empty();
    }

    /** Whether the ability is an activated ability whose cost sacrifices one permanent other than its own. */
    static boolean sacrificesAnother(SpellAbility ability) {
        return sacrifice(ability).isPresent();
    }

    private static Optional<CostSacrifice> sacrifice(SpellAbility ability) {
        if (!ability.isActivatedAbility() || ability.getPayCosts() == null) {
            return Optional.empty();
        }
        for (CostPart part : ability.getPayCosts().getCostParts()) {
            if (part instanceof CostSacrifice sacrifice && !sacrifice.payCostFromSource()
                    && Integer.valueOf(1).equals(sacrifice.convertAmount())) {
                return Optional.of(sacrifice);
            }
        }
        return Optional.empty();
    }

    /** The creatures the activated ability's sacrifice cost could take from {@code activator}. */
    static List<Card> sacrificeable(SpellAbility ability, Player activator) {
        return sacrifice(ability).map(sacrifice -> CardLists.getValidCards(
                        activator.getCardsIn(ZoneType.Battlefield), sacrifice.getType().split(";"),
                        activator, ability.getHostCard(), ability).stream().filter(Card::isCreature).toList())
                .orElse(List.of());
    }
}
