package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.CardTraitBase;
import forge.game.ability.ApiType;
import forge.game.keyword.KeywordInterface;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.BiConsumer;

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
        return fromKeyword(ki, (trigger, line) -> { });
    }

    /**
     * The chapter lines, telling {@code rendered} which line each chapter's
     * trigger printed.
     *
     * <p>Forge generates one trigger per chapter, the chapters sharing an
     * ability grouped together, the first of a group primary and the rest
     * secondary ("I, II —" is one line from two triggers): a secondary trigger
     * belongs to the line its group's primary printed.
     */
    public static List<Ability> fromKeyword(KeywordInterface ki,
                                            BiConsumer<CardTraitBase, Ability> rendered) {
        List<Ability> abilities = new ArrayList<>();
        Ability groupLine = null;
        for (var trigger : ki.getTriggers()) {
            if (trigger.isSecondary()) {
                if (groupLine != null) rendered.accept(trigger, groupLine);
                continue;
            }
            groupLine = null;
            int before = abilities.size();
            addChapter(trigger, abilities);
            if (abilities.size() > before) {
                groupLine = abilities.get(abilities.size() - 1);
                rendered.accept(trigger, groupLine);
            }
        }
        return abilities;
    }

    private static void addChapter(Trigger trigger, List<Ability> abilities) {
        String trigDesc = trigger.getParam("TriggerDescription");
        if (trigDesc == null) return;

        // If TriggerDescription is just a chapter header (nothing after the em-dash),
        // resolve the execute SVar's SpellDescription to fill chapter content.
        String stripped = AbilityDescription.stripReminderText(trigDesc).trim();

        // If TriggerDescription contains the ABILITY placeholder and execute is a Charm,
        // replace with charm header and attach OPTION sub-abilities.
        if (stripped.contains("ABILITY")) {
            SpellAbility execute = trigger.getOverridingAbility();
            if (execute != null && execute.getApi() == ApiType.Charm) {
                // Owned by the trigger: two chapters naming one charm SVar
                // share the charm object, and only the trigger tells
                // chapter I's option lines from chapter II's.
                CharmAbility.Expanded expanded =
                        CharmAbility.expandPlaceholder(stripped, execute, trigger);
                if (expanded.description() == null) return;
                abilities.add(new ChapterAbility(
                        AbilityType.CHAPTER.formatDescription(expanded.description()),
                        expanded.options()));
                return;
            }
        }

        if (isHeaderOnly(stripped)) {
            SpellAbility execute = trigger.getOverridingAbility();
            if (execute != null) {
                String effectText = SpellAbilityUtils.chainText(execute);
                if (!effectText.isEmpty()) {
                    stripped = stripped + " " + effectText;
                }
            }
        }

        String normalized = AbilityDescription.normalize(stripped);
        if (normalized == null) return;
        abilities.add(new ChapterAbility(AbilityType.CHAPTER.formatDescription(normalized)));
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
}
