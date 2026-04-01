package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import com.pricepredictor.connector.NonBlankString;
import forge.game.CardTraitBase;
import forge.game.keyword.KeywordInterface;

import java.util.Optional;

/**
 * Class level ability. Stores cost + description as the full text, exposes
 * innerDescription for class dedup matching, and ordinal = level number.
 */
public record ClassLevelAbility(
        NonBlankString descriptionText,
        NonBlankString innerDescription,
        int ordinal
) implements Ability {

    @Override
    public AbilityType type() {
        return AbilityType.LEVEL;
    }

    public static Optional<ClassLevelAbility> of(KeywordInterface ki) {
        int level = Integer.parseInt(KeywordFields.from(ki, 3).field(1));

        var it = ki.getAbilities().iterator();
        String rawCost = it.hasNext() ? it.next().getCostDescription() : null;
        NonBlankString costNbs = NonBlankString.of(rawCost).orElse(null);
        if (costNbs == null) return Optional.empty();

        String cost = costNbs.value();
        if (cost.endsWith(":")) {
            cost = cost.substring(0, cost.length() - 1).trim();
        }

        NonBlankString casedCost = AbilityDescription.applyCasing(cost);
        return findFirstDescription(ki.getTriggers(), "TriggerDescription")
                .or(() -> findFirstDescription(ki.getStaticAbilities(), "Description"))
                .or(() -> findFirstDescription(ki.getReplacements(), "Description"))
                .flatMap(d -> AbilityDescription.normalize(d.value()))
                .map(normalized -> new ClassLevelAbility(
                        NonBlankString.require(casedCost + ": " + normalized), normalized, level));
    }

    private static <T extends CardTraitBase> Optional<NonBlankString> findFirstDescription(
            Iterable<T> traits, String param) {
        for (T trait : traits) {
            Optional<NonBlankString> d = NonBlankString.of(trait.getParam(param));
            if (d.isPresent()) return d;
        }
        return Optional.empty();
    }
}
