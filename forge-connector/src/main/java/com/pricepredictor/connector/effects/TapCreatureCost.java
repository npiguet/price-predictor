package com.pricepredictor.connector.effects;

import forge.game.card.Card;
import forge.game.card.CardCollection;
import forge.game.card.CardLists;
import forge.game.card.CardPredicates;
import forge.game.cost.Cost;
import forge.game.cost.CostPart;
import forge.game.cost.CostTapType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;
import java.util.function.Predicate;

/**
 * A creature is counted once across a cost's "tap an untapped creature you
 * control" part and its mana part (finding 8.6, task E04c). 601.2g pays the
 * mana before 601.2h pays the rest, so a creature tapped for mana is no longer
 * untapped to pay the tap cost: Revelsong Horn ({1}, {T}, tap an untapped
 * creature you control) with Llanowar Elves the only creature and the only
 * mana cannot be paid.
 *
 * <p>The verdict asks {@link #manaPayable}: is there a choice of the creatures
 * to tap that leaves the mana payable. The random seat's draw asks
 * {@link #drawLeavingMana} for the creatures it taps, and holds them
 * ({@link #hold}) until the cost is paid, so the AI's mana payment, which
 * runs before the tap part pays and knows nothing of the draw, takes its mana
 * from the other sources: Forge refuses a held creature's mana abilities
 * ({@code SpellAbilityRestriction.canPlay} reads {@code isUsedToPay}).
 * Only a tap part whose type the draw can express is read; a part that
 * shares a creature type or has a total power is left as it was.
 */
public final class TapCreatureCost {

    /** Tap type properties the draw cannot express: a shared creature type, a total power. */
    private static final List<String> RELATIONAL = List.of("sharesCreatureTypeWith", "+withTotalPowerGE");
    /** Draws of a tap set before the first choice that leaves the mana payable is taken. */
    private static final int MAX_DRAWS = 16;
    /** Choices of which creatures to tap that are tried. */
    private static final int MAX_COMBINATIONS = 256;

    private TapCreatureCost() {
    }

    /**
     * Whether the creatures the tap part needs can be chosen so that the mana
     * stays payable, for the verdict. Without a single tap part, or with
     * creatures to spare that make no mana, the mana alone decides.
     */
    static boolean manaPayable(SpellAbility candidate, Player payer, RulesCost rules) {
        CostTapType part = tapPart(candidate);
        if (part == null || rules == null) {
            return ManaAffordability.canPay(candidate, payer, rules);
        }
        List<Card> valid = validCreatures(candidate, payer, part);
        int count = part.getAbilityAmount(candidate);
        List<Card> makingMana = valid.stream().filter(c -> makesMana(c, rules)).toList();
        int mustTap = count - (valid.size() - makingMana.size());
        if (count <= 0 || mustTap <= 0) {
            return ManaAffordability.canPay(candidate, payer, rules);
        }
        return firstChoice(makingMana, mustTap, tapped -> payableWithout(candidate, payer, rules, tapped)) != null;
    }

    /**
     * {@code count} distinct creatures of {@code valid}, for the activating player of {@code sa}, drawn uniformly, such
     * that the mana of {@code sa} is payable without them: the first such draw
     * of up to {@value #MAX_DRAWS}, else the first such choice there is, or
     * null when there is none. A cost with no mana part takes its first draw.
     */
    public static List<Card> drawLeavingMana(
            SpellAbility sa, CardCollection valid, int count, Random random) {
        Player payer = sa.getActivatingPlayer();
        if (!hasManaPart(sa)) {
            return RandomChoices.drawDistinct(valid, count, random);
        }
        RulesCost rules = chargedOrNull(sa, payer);
        for (int draw = 0; draw < MAX_DRAWS; draw++) {
            List<Card> picked = RandomChoices.drawDistinct(valid, count, random);
            if (payableWithout(sa, payer, rules, picked)) {
                return picked;
            }
        }
        return firstChoice(valid, count, tapped -> payableWithout(sa, payer, rules, tapped));
    }

    /** Marks {@code creatures} as used to pay, which refuses their mana abilities; {@link #release} undoes it. */
    public static void hold(Collection<Card> creatures) {
        creatures.forEach(c -> c.setUsedToPay(true));
    }

    /** Takes the mark {@link #hold} put on {@code creatures} off. */
    public static void release(Collection<Card> creatures) {
        creatures.forEach(c -> c.setUsedToPay(false));
    }

    /** Whether {@code sa} has a mana part to pay beside its tap part. */
    public static boolean hasManaPart(SpellAbility sa) {
        Cost cost = sa.getPayCosts();
        return cost != null && cost.hasManaCost();
    }

    private static boolean payableWithout(SpellAbility sa, Player payer, RulesCost rules, Collection<Card> tapped) {
        hold(tapped);
        try {
            return ManaAffordability.canPay(sa, payer, rules);
        } finally {
            release(tapped);
        }
    }

    /** The first {@code count} of {@code from}, as combinations in order, that {@code accepts}; null when none does. */
    private static List<Card> firstChoice(List<Card> from, int count, Predicate<List<Card>> accepts) {
        if (count > from.size()) {
            return null;
        }
        int[] at = new int[count];
        for (int i = 0; i < count; i++) {
            at[i] = i;
        }
        for (int tried = 0; tried < MAX_COMBINATIONS; tried++) {
            List<Card> choice = new ArrayList<>(count);
            for (int i : at) {
                choice.add(from.get(i));
            }
            if (accepts.test(choice)) {
                return choice;
            }
            if (!advance(at, from.size())) {
                return null;
            }
        }
        return null;
    }

    /** The next combination in lexicographic order, in place; false after the last. */
    private static boolean advance(int[] at, int size) {
        int i = at.length - 1;
        while (i >= 0 && at[i] == size - at.length + i) {
            i--;
        }
        if (i < 0) {
            return false;
        }
        at[i]++;
        for (int j = i + 1; j < at.length; j++) {
            at[j] = at[j - 1] + 1;
        }
        return true;
    }

    private static boolean makesMana(Card creature, RulesCost rules) {
        return !creature.getManaAbilities().isEmpty() || rules.payers().tappers().containsKey(creature);
    }

    /** The cost's one tap-creatures part whose type the draw can express, or null. */
    private static CostTapType tapPart(SpellAbility candidate) {
        if (!hasManaPart(candidate)) {
            return null;
        }
        List<CostPart> parts = candidate.getPayCosts().getCostParts().stream()
                .filter(p -> p instanceof CostTapType).toList();
        if (parts.size() != 1) {
            return null;
        }
        CostTapType part = (CostTapType) parts.get(0);
        return RELATIONAL.stream().anyMatch(part.getType()::contains) ? null : part;
    }

    /** The creatures the tap part may tap, as {@code CostTapType.canPay} reads them. */
    private static List<Card> validCreatures(SpellAbility sa, Player payer, CostTapType part) {
        CardCollection valid = CardLists.getValidCards(payer.getCardsIn(ZoneType.Battlefield),
                part.getType().split(";"), payer, sa.getHostCard(), sa);
        valid = CardLists.filter(valid, CardPredicates.CAN_TAP);
        if (!part.canTapSource) {
            valid.remove(sa.getHostCard());
        }
        return valid;
    }

    private static RulesCost chargedOrNull(SpellAbility sa, Player payer) {
        try {
            return new ChargedCost(sa, payer).get();
        } catch (RuntimeException e) {
            return RulesCost.printed(sa.getPayCosts());
        }
    }
}
