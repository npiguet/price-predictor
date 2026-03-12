package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.List;

/**
 * Spell effect ability. Factory walks the sub-ability chain to collect all
 * SpellDescription fragments, emitting one SpellEffect per fragment.
 */
public record SpellEffect(String descriptionText) implements Ability {

    @Override
    public AbilityType type() {
        return AbilityType.SPELL;
    }

    public static List<Ability> fromChain(SpellAbility sa) {
        List<Ability> abilities = new ArrayList<>();
        for (String spellDesc : SpellAbilityUtils.collectParamInChain(sa, "SpellDescription")) {
            String stripped = AbilityDescription.stripReminderText(spellDesc);
            if (stripped == null || stripped.isEmpty()) continue;
            abilities.add(new SpellEffect(AbilityDescription.applyCasing(stripped)));
        }
        return abilities;
    }
}
