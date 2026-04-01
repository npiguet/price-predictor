package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.CardTraitBase;
import forge.game.keyword.KeywordInterface;

import java.util.Optional;

/**
 * Class level ability. Stores cost + description as the full text, exposes
 * innerDescription for class dedup matching, and ordinal = level number.
 */
public record ClassLevelAbility(
        String descriptionText,
        String innerDescription,
        int ordinal
) implements Ability {

    @Override
    public AbilityType type() {
        return AbilityType.LEVEL;
    }

    public static Optional<ClassLevelAbility> of(KeywordInterface ki) {
        int level = Integer.parseInt(KeywordFields.from(ki, 3).field(1));

        var it = ki.getAbilities().iterator();
        String cost = it.hasNext() ? it.next().getCostDescription() : null;
        if (cost == null || cost.isEmpty()) {
            return Optional.empty();
        }
        cost = cost.trim();
        if (cost.endsWith(":")) {
            cost = cost.substring(0, cost.length() - 1).trim();
        }

        String casedCost = AbilityDescription.applyCasing(cost);
        return findFirstDescription(ki.getTriggers(), "TriggerDescription")
                .or(() -> findFirstDescription(ki.getStaticAbilities(), "Description"))
                .or(() -> findFirstDescription(ki.getReplacements(), "Description"))
                .flatMap(AbilityDescription::normalize)
                .map(normalized -> new ClassLevelAbility(casedCost + ": " + normalized, normalized, level));
    }

    private static <T extends CardTraitBase> Optional<String> findFirstDescription(
            Iterable<T> traits, String param) {
        for (T trait : traits) {
            String d = trait.getParam(param);
            if (d != null && !d.isEmpty()) {
                return Optional.of(d);
            }
        }
        return Optional.empty();
    }
}
