package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.CardTraitBase;
import forge.game.keyword.KeywordInterface;

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

    public static ClassLevelAbility of(KeywordInterface ki) {
        String original = ki.getOriginal();
        int level = Integer.parseInt(original.split(":", 3)[1]);

        var it = ki.getAbilities().iterator();
        String cost = it.hasNext() ? it.next().getCostDescription() : null;
        if (cost == null || cost.isEmpty()) {
            return null;
        }
        cost = cost.trim();
        if (cost.endsWith(":")) {
            cost = cost.substring(0, cost.length() - 1).trim();
        }

        String rawDesc = findFirstDescription(ki.getTriggers(), "TriggerDescription");
        if (rawDesc == null) {
            rawDesc = findFirstDescription(ki.getStaticAbilities(), "Description");
        }
        if (rawDesc == null) {
            rawDesc = findFirstDescription(ki.getReplacements(), "Description");
        }
        if (rawDesc == null) {
            return null;
        }

        String normalized = AbilityDescription.normalize(rawDesc);
        if (normalized == null) return null;
        String casedCost = AbilityDescription.applyCasing(cost);
        return new ClassLevelAbility(casedCost + ": " + normalized, normalized, level);
    }

    private static <T extends CardTraitBase> String findFirstDescription(
            Iterable<T> traits, String param) {
        for (T trait : traits) {
            String d = trait.getParam(param);
            if (d != null && !d.isEmpty()) {
                return d;
            }
        }
        return null;
    }
}
