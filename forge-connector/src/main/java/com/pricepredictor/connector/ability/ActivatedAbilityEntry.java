package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.spellability.SpellAbility;

/**
 * Activated or planeswalker ability built from a SpellAbility.
 * Walks the sub-ability chain to collect all SpellDescription fragments.
 */
public record ActivatedAbilityEntry(AbilityType type, String descriptionText) implements Ability {

    public static ActivatedAbilityEntry of(SpellAbility sa) {
        if (sa.getParam("SpellDescription") == null
                || sa.getParam("SpellDescription").isEmpty()) {
            return null;
        }
        String desc = sa.getDescription();
        for (String subDesc : SpellAbilityUtils.collectParamInChain(sa.getSubAbility(), "SpellDescription")) {
            desc = desc + " " + subDesc;
        }
        AbilityType type = sa.isPwAbility() ? AbilityType.PLANESWALKER : AbilityType.ACTIVATED;
        String normalized = AbilityDescription.normalize(desc);
        if (normalized == null) return null;
        return new ActivatedAbilityEntry(type, type.formatDescription(normalized));
    }
}
