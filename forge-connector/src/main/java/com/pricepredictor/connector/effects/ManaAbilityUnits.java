package com.pricepredictor.connector.effects;

import forge.card.MagicColor;
import forge.card.mana.ManaAtom;
import forge.game.ability.AbilityUtils;
import forge.game.ability.ApiType;
import forge.game.cost.Cost;
import forge.game.cost.CostPayment;
import forge.game.player.Player;
import forge.game.spellability.AbilityManaPart;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * The mana one activation of a mana ability would make for a candidate, as
 * {@link ManaAffordability} counts it: none from an ability that cannot be
 * activated now, that has a mana cost of its own (a filter land) or that
 * makes special mana; mana of a colour chosen on activation as one unit of
 * any colour it may make for the candidate.
 */
final class ManaAbilityUnits {

    private ManaAbilityUnits() {
    }

    /** The mana one activation of {@code ability} would make for {@code candidate}, or none when it cannot be activated. */
    static List<ManaUnit> made(SpellAbility ability, SpellAbility candidate, Player payer) {
        if (!activatable(ability, candidate, payer)) {
            return List.of();
        }
        AbilityManaPart part = ability.getManaPart();
        boolean snow = part.getSourceCard() != null && part.getSourceCard().isSnow();
        if (choosesColour(ability, part)) {
            return ofChosenColour(ability, candidate, snow);
        }
        return ofNamedColours(ability, part, candidate, snow);
    }

    /** Whether the mana's colour is chosen on activation (any colour, a combination, a reflection). */
    private static boolean choosesColour(SpellAbility ability, AbilityManaPart part) {
        return part.isAnyMana() || part.isComboMana() || ability.getApi() == ApiType.ManaReflected;
    }

    /**
     * Whether {@code payer} can activate the mana ability now, its own costs
     * paid without mana, for mana that is not special.
     */
    private static boolean activatable(SpellAbility ability, SpellAbility candidate, Player payer) {
        AbilityManaPart part = ability.getManaPart();
        if (part == null || part.isSpecialMana()) {
            return false;
        }
        ability.setActivatingPlayer(payer);
        Cost own = ability.getPayCosts();
        if (own != null && own.hasManaCost()) {
            return false; // a filter: its mana comes from other sources
        }
        return ability.canPlay() && ability.checkRestrictions(payer) && payableFor(ability, own, candidate, payer);
    }

    /** Whether the mana ability's own costs can be paid and its mana spent on {@code candidate}. */
    private static boolean payableFor(SpellAbility ability, Cost own, SpellAbility candidate, Player payer) {
        return CostPayment.canPayAdditionalCosts(own, ability, false, payer)
                && ability.metConditions() && ability.getManaPart().meetsManaRestrictions(candidate);
    }

    /** Mana of a colour chosen on activation: each unit any colour it may make for the candidate. */
    private static List<ManaUnit> ofChosenColour(SpellAbility ability, SpellAbility candidate, boolean snow) {
        byte colors = 0;
        for (byte color : ManaAtom.MANATYPES) {
            colors |= ability.isManaAbilityFor(candidate, color) ? color : 0;
        }
        int amount = colors == 0 ? 0 : Math.max(0, ability.amountOfManaGenerated(false));
        return Collections.nCopies(amount, new ManaUnit(colors, snow, false));
    }

    /** Mana of the colours the ability names, {@code Amount} times over. */
    private static List<ManaUnit> ofNamedColours(
            SpellAbility ability, AbilityManaPart part, SpellAbility candidate, boolean snow) {
        int amount = ability.hasParam("Amount")
                ? AbilityUtils.calculateAmount(ability.getHostCard(), ability.getParam("Amount"), ability)
                : 1;
        List<ManaUnit> once = new ArrayList<>();
        for (String token : part.mana(ability).split(" ")) {
            byte color = colorOf(token);
            if (color != 0 && ability.isManaAbilityFor(candidate, color)) {
                once.add(new ManaUnit(color, snow, false));
            }
        }
        return Collections.nCopies(Math.max(amount, 0), once).stream().flatMap(List::stream).toList();
    }

    private static byte colorOf(String token) {
        return token.equals("C") ? (byte) ManaAtom.COLORLESS : MagicColor.fromName(token);
    }
}
