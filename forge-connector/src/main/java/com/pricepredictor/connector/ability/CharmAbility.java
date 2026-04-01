package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Charm ability with nested OPTION sub-abilities. If the charm has no description,
 * the factory returns the choices as top-level OPTION abilities instead.
 */
public record CharmAbility(String descriptionText, List<Ability> subAbilities) implements Ability {

    public CharmAbility {
        Objects.requireNonNull(subAbilities);
        subAbilities = List.copyOf(subAbilities);
    }

    @Override
    public AbilityType type() {
        return AbilityType.SPELL;
    }

    public static List<Ability> fromSpellAbility(SpellAbility sa) {
        String charmDesc = resolveCharmDescription(sa);
        List<Ability> choiceSubs = collectChoices(sa);

        List<Ability> result = new ArrayList<>();
        if (charmDesc != null && !charmDesc.isEmpty()) {
            result.add(new CharmAbility(AbilityDescription.applyCasing(charmDesc), choiceSubs));
        } else {
            // No charm description — add choices as top-level abilities
            result.addAll(choiceSubs);
        }
        return result;
    }

    /**
     * Resolve the header description for a Charm SA using a 4-source cascade:
     * SpellDescription → Pawprint total → AdditionalDescription → synthesized "choose N" header.
     *
     * <p>Each source represents a different card design pattern:
     * <ul>
     *   <li><b>SpellDescription</b>: most charms — explicit oracle text on the SA.</li>
     *   <li><b>Pawprint</b>: attraction-style charms — header built from paw-print count.</li>
     *   <li><b>AdditionalDescription</b>: charms that describe the choice mode inline.</li>
     *   <li><b>Synthesized</b>: charms where no text exists — generate "choose one/up to N/…"
     *       from {@code CharmNum} and {@code MinCharmNum} params.</li>
     * </ul>
     */
    private static String resolveCharmDescription(SpellAbility sa) {
        return resolveFromSpellDescription(sa)
                .or(() -> resolveFromPawprint(sa))
                .or(() -> resolveFromAdditionalDescription(sa))
                .or(() -> Optional.ofNullable(synthesizeCharmHeader(sa)))
                .orElse(null);
    }

    private static Optional<String> resolveFromSpellDescription(SpellAbility sa) {
        String desc = sa.getParam("SpellDescription");
        if (desc == null || desc.isEmpty()) return Optional.empty();
        desc = AbilityDescription.stripReminderText(desc);
        // Strip trailing em-dash (oracle/SpellDescription often ends with " —")
        desc = desc.replaceAll("\\s*\u2014\\s*$", "").trim();
        return desc.isEmpty() ? Optional.empty() : Optional.of(desc);
    }

    private static Optional<String> resolveFromPawprint(SpellAbility sa) {
        String total = sa.getParam("Pawprint");
        if (total == null) return Optional.empty();
        String desc = "Choose up to " + total + " {P} worth of modes.";
        if ("True".equals(sa.getParam("CanRepeatModes"))) {
            desc += " You may choose the same mode more than once.";
        }
        return Optional.of(desc);
    }

    private static Optional<String> resolveFromAdditionalDescription(SpellAbility sa) {
        String desc = sa.getParam("AdditionalDescription");
        if (desc == null || desc.isEmpty()) return Optional.empty();
        desc = AbilityDescription.stripReminderText(desc);
        return desc.isEmpty() ? Optional.empty() : Optional.of(desc);
    }

    /**
     * Collect charm choices as OPTION sub-abilities, applying PrecostDesc/ModeCost/Pawprint
     * prefix adornments so that the option text matches oracle.
     */
    private static List<Ability> collectChoices(SpellAbility sa) {
        List<Ability> choiceSubs = new ArrayList<>();
        var choices = sa.getAdditionalAbilityList("Choices");
        if (choices == null) return choiceSubs;
        for (var choice : choices) {
            String choiceDesc = extractChoiceDescription(choice);
            // Tiered charms (e.g. Vincent's Limit Break): the ability name lives in
            // PrecostDesc on the choice SA.  Prepend it so the option text matches oracle.
            String precostDesc = choice.getParam("PrecostDesc");
            if (precostDesc != null && !precostDesc.isEmpty()) {
                choiceDesc = choiceDesc != null && !choiceDesc.isEmpty()
                        ? precostDesc + " \u2014 " + choiceDesc
                        : precostDesc;
            }
            String modeCost = choice.getParam("ModeCost");
            if (modeCost != null && !modeCost.isEmpty()) {
                String costToken = "{" + modeCost.trim() + "}";
                choiceDesc = choiceDesc != null && !choiceDesc.isEmpty()
                        ? costToken + " \u2014 " + choiceDesc
                        : costToken;
            }
            String pawprint = choice.getParam("Pawprint");
            if (pawprint != null) {
                choiceDesc = "{P}".repeat(Integer.parseInt(pawprint))
                        + " \u2014 " + choiceDesc;
            }
            choiceSubs.add(new TextAbility(AbilityType.OPTION,
                    AbilityDescription.applyCasing(choiceDesc)));
        }
        return choiceSubs;
    }

    /**
     * Synthesize a "choose N —" header from CharmNum / MinCharmNum params.
     * Returns null if CharmNum is a variable/expression (can't determine statically).
     * Logic:
     *   MinCharmNum=0 → "choose up to N"
     *   MinCharmNum=1, CharmNum=2 → "choose one or both"
     *   MinCharmNum=1, CharmNum≥3 → "choose one or more"
     *   no MinCharmNum (exact) → "choose N"
     *   no CharmNum → "choose one" (default)
     * No trailing em-dash: the header stands on its own line.
     */
    static String synthesizeCharmHeader(SpellAbility sa) {
        String numStr = sa.getParam("CharmNum");
        String minNumStr = sa.getParam("MinCharmNum");
        int num = parseSimpleInt(numStr);    // -1 if null or non-integer
        int minNum = parseSimpleInt(minNumStr); // -1 if null or not set

        if (num == -1 && numStr != null) {
            // CharmNum is a variable/expression — can't synthesize
            return null;
        }
        if (num == -1) {
            // CharmNum absent → default choose one
            if (minNum == 0) return "Choose up to one";
            return "Choose one";
        }
        if (minNum == 0) {
            return "Choose up to " + numberWord(num);
        }
        if (minNum == 1) {
            if (num == 2) return "Choose one or both";
            if (num >= 3) return "Choose one or more";
        }
        // No MinCharmNum (or minNum == num): choose exactly N
        return "Choose " + numberWord(num);
    }

    private static int parseSimpleInt(String s) {
        if (s == null) return -1;
        try { return Integer.parseInt(s.trim()); }
        catch (NumberFormatException e) { return -1; }
    }

    private static String numberWord(int n) {
        return switch (n) {
            case 1 -> "one";
            case 2 -> "two";
            case 3 -> "three";
            case 4 -> "four";
            case 5 -> "five";
            default -> String.valueOf(n);
        };
    }

    /**
     * Return only the OPTION sub-abilities for a charm SA (no wrapper).
     * Used when the charm appears nested inside another ability (triggered charm,
     * ABILITY placeholder), where the parent description already provides context.
     *
     * <p>PrecostDesc, ModeCost, and Pawprint decorators are intentionally omitted here:
     * those adornments are part of the top-level charm presentation (e.g. "choose one —")
     * and are not repeated in oracle text when the choice list appears under a trigger.
     */
    public static List<Ability> optionsFrom(SpellAbility sa) {
        List<Ability> options = new ArrayList<>();
        var choices = sa.getAdditionalAbilityList("Choices");
        if (choices == null) return options;
        for (var choice : choices) {
            String desc = extractChoiceDescription(choice);
            if (desc == null || desc.isEmpty()) continue;
            options.add(new TextAbility(AbilityType.OPTION, AbilityDescription.applyCasing(desc)));
        }
        return options;
    }

    /**
     * Extract the raw choice description from a Charm choice SA.
     * Walks the sub-chain for {@code SpellDescription} and strips reminder text.
     * Returns null when no description is found.
     */
    private static String extractChoiceDescription(SpellAbility choice) {
        String desc = SpellAbilityUtils.findParamInChain(choice, "SpellDescription");
        if (desc != null) desc = AbilityDescription.stripReminderText(desc);
        return desc;
    }
}
