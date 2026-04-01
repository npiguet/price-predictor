package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import com.pricepredictor.connector.NonBlankString;
import forge.game.keyword.KeywordInterface;

import java.util.ArrayList;
import java.util.List;

import static com.pricepredictor.connector.ability.SpellAbilityUtils.getParam;

/**
 * ETB replacement ability from etbCounter: or ETBReplacement: keywords.
 * One EtbReplacementAbility per replacement description.
 */
public record EtbReplacementAbility(NonBlankString descriptionText) implements Ability {

    @Override
    public AbilityType type() {
        return AbilityType.REPLACEMENT;
    }

    public static List<Ability> fromKeyword(KeywordInterface ki) {
        return ki.getReplacements().stream()
                .flatMap(replacement -> getParam(replacement, "Description")
                        .flatMap(AbilityDescription::normalize)
                        .stream()
                )
                .map(EtbReplacementAbility::new)
                .map(Ability.class::cast)
                .toList();
    }
}
