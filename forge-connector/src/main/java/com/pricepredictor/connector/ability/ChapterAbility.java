package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.keyword.KeywordInterface;

import java.util.ArrayList;
import java.util.List;

/**
 * Chapter ability from a Saga. One ChapterAbility per trigger, with uppercased Roman prefix.
 */
public record ChapterAbility(String descriptionText) implements Ability {

    @Override
    public AbilityType type() {
        return AbilityType.CHAPTER;
    }

    public static List<Ability> fromKeyword(KeywordInterface ki) {
        List<Ability> abilities = new ArrayList<>();
        for (var trigger : ki.getTriggers()) {
            if ("True".equals(trigger.getParam("Secondary"))) continue;
            String normalized = AbilityDescription.normalize(trigger.getParam("TriggerDescription"));
            if (normalized == null) continue;
            abilities.add(new ChapterAbility(AbilityType.CHAPTER.formatDescription(normalized)));
        }
        return abilities;
    }
}
