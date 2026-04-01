package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import com.pricepredictor.connector.NonBlankString;
import forge.game.cost.Cost;
import forge.game.keyword.KeywordInterface;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static com.pricepredictor.connector.AbilityDescription.applyCasing;

/**
 * AlternateAdditionalCost keyword — formats a disjunction of alternate payment options.
 *
 * <p>The keyword string format is {@code "AlternateAdditionalCost:cost1:cost2:..."}.
 * Each cost part is rendered via {@link Cost#toSimpleString()}; mana-only costs are
 * prefixed with "pay"; tap/sacrifice/other costs are used as-is. The resulting terms
 * are joined as a disjunction (e.g. "pay {G} or sacrifice a creature").
 */
public record AlternateAdditionalCostAbility(NonBlankString descriptionText) implements Ability {

    @Override
    public AbilityType type() {
        return AbilityType.ADDITIONAL_COST;
    }

    public static List<Ability> fromKeyword(KeywordInterface ki) {
        var terms = Stream.of(KeywordFields.parseAll(ki.getOriginal()).tailFields())
                .map(costPart -> {
                    Cost cost = new Cost(costPart, false);
                    String costText = cost.toSimpleString();
                    String term = costText.substring(0, 1).toLowerCase() + costText.substring(1);
                    return cost.isOnlyManaCost() ? "pay " + term : term;
                })
                .toList();
        String desc = AbilityDescription.joinDisjunction(terms);
        return List.of(new AlternateAdditionalCostAbility(applyCasing(desc)));
    }
}
