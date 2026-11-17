package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import com.pricepredictor.connector.KeywordRendering;
import forge.game.keyword.KeywordInterface;

/**
 * ETB replacement ability from etbCounter: or ETBReplacement: keywords.
 * One EtbReplacementAbility per replacement description, each named as the
 * line of the replacement that printed it, so the line's script is that
 * replacement's chain rather than the keyword's text.
 */
public record EtbReplacementAbility(String descriptionText) implements Ability {

    @Override
    public AbilityType type() {
        return AbilityType.REPLACEMENT;
    }

    public static KeywordRendering fromKeyword(KeywordInterface ki) {
        KeywordRendering.Builder rendering = new KeywordRendering.Builder();
        for (var replacement : ki.getReplacements()) {
            String normalized = AbilityDescription.normalize(replacement.getParam("Description"));
            if (normalized == null) continue;
            Ability line = new EtbReplacementAbility(normalized);
            rendering.add(line);
            rendering.printed(replacement, line);
        }
        return rendering.build();
    }
}
