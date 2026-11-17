package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import com.pricepredictor.connector.KeywordRendering;
import forge.game.keyword.KeywordInterface;
import forge.game.replacement.ReplacementEffect;
import forge.game.spellability.SpellAbility;

/**
 * ETB replacement ability from etbCounter: or ETBReplacement: keywords.
 * One EtbReplacementAbility per replacement description, each named as the
 * line of the replacement that printed it, so the line's script is that
 * replacement's chain rather than the keyword's text. A replacement whose
 * effect has no description of its own (Sutured Ghoul's, where Forge writes
 * the card's name) prints the first Oracle line its chain states.
 */
public record EtbReplacementAbility(String descriptionText) implements Ability {

    @Override
    public AbilityType type() {
        return AbilityType.REPLACEMENT;
    }

    public static KeywordRendering fromKeyword(KeywordInterface ki, FaceOracle oracle) {
        KeywordRendering.Builder rendering = new KeywordRendering.Builder();
        for (var replacement : ki.getReplacements()) {
            String normalized = descriptionOf(replacement, oracle);
            if (normalized == null) continue;
            Ability line = new EtbReplacementAbility(normalized);
            rendering.add(line);
            rendering.printed(replacement, line);
        }
        return rendering.build();
    }

    /** The replacement's own description, else the Oracle line its chain states, or null. */
    private static String descriptionOf(ReplacementEffect replacement, FaceOracle oracle) {
        String own = AbilityDescription.normalize(replacement.getParam("Description"));
        if (own != null && !isOnlyTheName(own, oracle)) return own;
        SpellAbility effect = replacement.getOverridingAbility();
        if (effect == null) return own;
        String stated = oracle.firstLineIn(SpellAbilityUtils.chainText(effect));
        return stated != null ? AbilityDescription.normalize(stated) : own;
    }

    /** Whether a description is the card's name alone, what Forge writes for an undescribed effect. */
    private static boolean isOnlyTheName(String description, FaceOracle oracle) {
        return "CARDNAME".equalsIgnoreCase(description) || description.equalsIgnoreCase(oracle.name());
    }
}
