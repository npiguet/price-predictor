package com.pricepredictor.connector;

import lombok.experimental.UtilityClass;

import java.util.ArrayList;
import java.util.List;

/**
 * Serializes {@link CardAttributes} into Forge card script text format.
 */
@UtilityClass
class ForgeScriptSerializer {

    String serialize(CardAttributes card) {
        var sb = new StringBuilder();

        if (card.getName() != null && !card.getName().isEmpty()) {
            sb.append("Name:").append(card.getName()).append('\n');
        }

        if (card.getManaCost() != null && !card.getManaCost().isEmpty()) {
            sb.append("ManaCost:").append(card.getManaCost()).append('\n');
        }

        // Types line: supertypes + types + subtypes space-joined
        List<String> allTypes = new ArrayList<>();
        if (card.getSupertypes() != null) {
            allTypes.addAll(card.getSupertypes());
        }
        allTypes.addAll(card.getTypes());
        if (card.getSubtypes() != null) {
            allTypes.addAll(card.getSubtypes());
        }
        sb.append("Types:").append(String.join(" ", allTypes)).append('\n');

        if (card.getOracleText() != null && !card.getOracleText().isEmpty()) {
            sb.append("Oracle:").append(card.getOracleText()).append('\n');
        }

        if (card.getKeywords() != null) {
            for (String keyword : card.getKeywords()) {
                sb.append("K:").append(keyword).append('\n');
            }
        }

        if (card.getPower() != null && card.getToughness() != null) {
            sb.append("PT:").append(card.getPower()).append('/').append(card.getToughness()).append('\n');
        }

        if (card.getLoyalty() != null && !card.getLoyalty().isEmpty()) {
            sb.append("Loyalty:").append(card.getLoyalty()).append('\n');
        }

        return sb.toString();
    }
}
