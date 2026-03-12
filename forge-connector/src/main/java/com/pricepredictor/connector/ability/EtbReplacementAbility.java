package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.keyword.KeywordInterface;

import java.util.ArrayList;
import java.util.List;

/**
 * ETB replacement ability from etbCounter: or ETBReplacement: keywords.
 * One EtbReplacementAbility per replacement description.
 */
public record EtbReplacementAbility(String descriptionText) implements Ability {

    @Override
    public AbilityType type() {
        return AbilityType.REPLACEMENT;
    }

    public static List<Ability> fromKeyword(KeywordInterface ki) {
        List<Ability> abilities = new ArrayList<>();
        for (var replacement : ki.getReplacements()) {
            String normalized = AbilityDescription.normalize(replacement.getParam("Description"));
            if (normalized == null) continue;
            abilities.add(new EtbReplacementAbility(normalized));
        }
        return abilities;
    }
}
