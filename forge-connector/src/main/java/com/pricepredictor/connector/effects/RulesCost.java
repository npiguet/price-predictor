package com.pricepredictor.connector.effects;

import forge.card.mana.ManaCost;
import forge.game.card.Card;
import forge.game.card.CardPredicates;
import forge.game.cost.Cost;
import forge.game.cost.CostAdjustment;
import forge.game.cost.CostPartMana;
import forge.game.keyword.Emerge;
import forge.game.mana.ManaCostBeingPaid;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.Zone;
import forge.game.zone.ZoneType;

import java.util.Comparator;

/**
 * The mana a candidate costs now by the rules, and what its keywords let
 * pay of it, read without asking any player and without writing on the game
 * (root spec § 8.8; findings 8.1, 8.2).
 *
 * <p>The total cost (601.2f) is Forge's own: the {@code Cost} raised as its
 * payment raises it ({@code CostAdjustment.raise}: commander tax,
 * {@code RaiseCost} statics, a waterbend a static adds), its mana part
 * ({@code getManaCostFor}) reduced by {@code CostAdjustment.adjustByRules}
 * ({@code ReduceCost} statics, set costs), which leaves out every reduction
 * the payer chooses while paying. Of those, emerge's is in the total cost:
 * reduced by the largest mana value among the creatures it may sacrifice,
 * which may still tap for mana or convoke first (601.2g-h). Convoke,
 * improvise, waterbend and delve are {@link KeywordPayers}: mana paid by
 * other means, which {@code affordable} counts beside the player's mana and
 * {@link #mostReduced} takes off at most. Offering and assist pay nothing
 * here: an offering is an optional cost no candidate chooses, and an
 * assisting player's mana is that player's to give.
 *
 * <p>X stays an unpaid {X}: at a priority stop nothing is announced. The
 * host's cast-from zone is set to where it is for the duration, as Forge's
 * {@code ComputerUtilMana.calculateManaCost} sets it, so a reduction keyed
 * on where a spell is cast from applies.
 *
 * @param total  the total cost, emerge's reduction included
 * @param payers what may pay for it besides mana
 */
record RulesCost(ManaCostBeingPaid total, KeywordPayers payers) {

    /**
     * What {@code payer} would be charged for {@code candidate} now; null
     * where it has no costs.
     *
     * @throws RuntimeException where the cost cannot be computed on this board
     */
    static RulesCost of(SpellAbility candidate, Player payer) {
        Cost cost = candidate.getPayCosts();
        if (cost == null) {
            return null;
        }
        Card host = candidate.getHostCard();
        boolean castFromHere = candidate.isSpell() && host != null && !host.isInZone(ZoneType.Stack);
        Zone castFrom = castFromHere ? host.getCastFrom() : null;
        if (castFromHere) {
            host.setCastFrom(host.getZone());
        }
        try {
            Cost raised = CostAdjustment.raise(cost, candidate, false);
            ManaCostBeingPaid total = new ManaCostBeingPaid(manaOf(raised, candidate));
            CostAdjustment.adjustByRules(total, candidate, emergeSacrifice(candidate, payer));
            return new RulesCost(total, KeywordPayers.of(candidate, payer, raised));
        } finally {
            if (castFromHere) {
                host.setCastFrom(castFrom);
            }
        }
    }

    /** The printed cost, where the rules' cannot be read on this board. */
    static RulesCost printed(Cost cost) {
        return new RulesCost(new ManaCostBeingPaid(cost.getTotalMana()), KeywordPayers.NONE);
    }

    /**
     * The total cost less the most the keywords can pay of it
     * ({@link KeywordPayers#mostReduced}): {@code cost_after_adjustment}.
     */
    ManaCostBeingPaid mostReduced() {
        return payers.mostReduced(total);
    }

    private static ManaCost manaOf(Cost raised, SpellAbility candidate) {
        if (raised == null) {
            return ManaCost.NO_COST;
        }
        CostPartMana part = raised.getCostMana();
        return part == null ? ManaCost.ZERO : part.getManaCostFor(candidate);
    }

    /**
     * The creature an emerge cast sacrifices for the largest reduction, or
     * null where the candidate is no emerge cast or no creature qualifies.
     */
    private static Card emergeSacrifice(SpellAbility candidate, Player payer) {
        if (!candidate.isSpell() || !candidate.isEmerge()) {
            return null;
        }
        if (!(candidate.getKeyword() instanceof Emerge emerge)) {
            return null;
        }
        return payer.getCardsIn(ZoneType.Battlefield).stream()
                .filter(CardPredicates.restriction(emerge.getValidType(), payer, candidate.getHostCard(), candidate))
                .filter(CardPredicates.canBeSacrificedBy(candidate, false))
                .max(Comparator.comparingInt(Card::getCMC))
                .orElse(null);
    }
}
