package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.ability.ApiType;
import forge.game.keyword.KeywordInterface;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Chapter ability from a Saga. One ChapterAbility per trigger, with uppercased Roman prefix.
 */
public record ChapterAbility(String descriptionText, List<Ability> subAbilities) implements Ability {

    public ChapterAbility {
        Objects.requireNonNull(subAbilities);
        subAbilities = List.copyOf(subAbilities);
    }

    /** Convenience constructor: no sub-abilities. */
    public ChapterAbility(String descriptionText) {
        this(descriptionText, List.of());
    }

    @Override
    public AbilityType type() {
        return AbilityType.CHAPTER;
    }

    public static List<Ability> fromKeyword(KeywordInterface ki) {
        List<Ability> abilities = new ArrayList<>();
        for (var trigger : ki.getTriggers()) {
            if ("True".equals(trigger.getParam("Secondary"))) continue;
            String trigDesc = trigger.getParam("TriggerDescription");
            if (trigDesc == null) continue;

            // If TriggerDescription is just a chapter header (nothing after the em-dash),
            // resolve the execute SVar's SpellDescription to fill chapter content.
            String stripped = AbilityDescription.stripReminderText(trigDesc).trim();

            // If TriggerDescription contains the ABILITY placeholder and execute is a Charm,
            // replace with charm header and attach OPTION sub-abilities.
            if (stripped.contains("ABILITY")) {
                SpellAbility execute = trigger.getOverridingAbility();
                if (execute != null && execute.getApi() == ApiType.Charm) {
                    String charmHeader = CharmAbility.synthesizeCharmHeader(execute);
                    if (charmHeader == null) charmHeader = "choose one";
                    stripped = stripped.replace("ABILITY", charmHeader);
                    List<Ability> options = CharmAbility.optionsFrom(execute);
                    String normalized = AbilityDescription.normalize(stripped);
                    if (normalized == null) continue;
                    abilities.add(new ChapterAbility(AbilityType.CHAPTER.formatDescription(normalized), options));
                    continue;
                }
            }

            if (isHeaderOnly(stripped)) {
                SpellAbility execute = trigger.getOverridingAbility();
                if (execute != null) {
                    List<Ability> effectNodes = SpellEffect.fromChain(execute);
                    if (!effectNodes.isEmpty()) {
                        String effectText = collectChainText(effectNodes.get(0));
                        if (!effectText.isEmpty()) {
                            stripped = stripped + " " + effectText;
                        }
                    }
                }
            }

            String normalized = AbilityDescription.normalize(stripped);
            if (normalized == null) continue;
            abilities.add(new ChapterAbility(AbilityType.CHAPTER.formatDescription(normalized)));
        }
        return abilities;
    }

    /**
     * Returns true if the trigger description is only a chapter header (e.g. "I —" or "I, II —")
     * with no effect text after the em-dash.
     */
    private static boolean isHeaderOnly(String desc) {
        int dashIdx = desc.indexOf('\u2014'); // em-dash
        if (dashIdx < 0) return false;
        return desc.substring(dashIdx + 1).trim().isEmpty();
    }

    /** Collect and concatenate description text from an ability node and all its descendants. */
    private static String collectChainText(Ability node) {
        StringBuilder sb = new StringBuilder(node.descriptionText());
        for (Ability child : node.subAbilities()) {
            String ct = collectChainText(child);
            if (!ct.isEmpty()) sb.append(' ').append(ct);
        }
        return sb.toString();
    }
}
