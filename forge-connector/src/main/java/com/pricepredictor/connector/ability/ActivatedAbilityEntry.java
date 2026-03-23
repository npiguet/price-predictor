package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Activated or planeswalker ability built from a SpellAbility.
 * The root description is taken from sa.getDescription() (cost + root SpellDescription only).
 * Sub-ability descriptions become child SpellEffect nodes via SpellEffect.fromChain().
 */
public record ActivatedAbilityEntry(AbilityType type, String descriptionText, List<Ability> subAbilities) implements Ability {

    public ActivatedAbilityEntry {
        Objects.requireNonNull(subAbilities);
        subAbilities = List.copyOf(subAbilities);
    }

    public static ActivatedAbilityEntry of(SpellAbility sa) {
        if (sa.getParam("SpellDescription") == null
                || sa.getParam("SpellDescription").isEmpty()) {
            return null;
        }
        String rootDesc = sa.getDescription();
        // For dice-roll activated abilities, getDescription() appends outcome lines after a
        // newline. Strip them here; they will be re-emitted as OPTION sub-abilities below.
        int nl = rootDesc.indexOf('\n');
        if (nl >= 0) rootDesc = rootDesc.substring(0, nl);

        AbilityType type = sa.isPwAbility() ? AbilityType.PLANESWALKER : AbilityType.ACTIVATED;
        String normalized = AbilityDescription.normalize(rootDesc);
        if (normalized == null) return null;

        List<Ability> children = new ArrayList<>(SpellEffect.fromChain(sa.getSubAbility()));
        children.addAll(diceOutcomesAsOptions(sa));
        return new ActivatedAbilityEntry(type, type.formatDescription(normalized), children);
    }

    /**
     * Expand {@code ResultSubAbilities} dice-outcome entries into OPTION sub-abilities.
     * Mirrors {@code RulesParser.diceOutcomesFromSA()} but uses OPTION type instead of SPELL,
     * so activated dice-roll results are formatted as {@code option[N]: range | description}.
     */
    private static List<Ability> diceOutcomesAsOptions(SpellAbility sa) {
        String resultSubAbilities = sa.getParam("ResultSubAbilities");
        if (resultSubAbilities == null || resultSubAbilities.isEmpty()) return List.of();

        List<Ability> result = new ArrayList<>();
        for (String entry : resultSubAbilities.split(",")) {
            String[] kv = entry.trim().split(":", 2);
            if (kv.length < 2) continue;
            String range = kv[0].trim();
            SpellAbility sub = sa.getAdditionalAbility(range);
            if (sub == null) continue;
            String rawDesc = sub.getParam("SpellDescription");
            if (rawDesc == null || rawDesc.isEmpty()) continue;
            String desc = AbilityDescription.replaceVert(rawDesc);
            String normalized = AbilityDescription.normalize(desc);
            if (normalized != null) {
                result.add(new TextAbility(AbilityType.OPTION, AbilityDescription.applyCasing(normalized)));
            }
        }
        return result;
    }
}
