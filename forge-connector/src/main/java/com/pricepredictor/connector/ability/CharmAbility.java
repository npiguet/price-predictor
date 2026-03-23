package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

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
        String charmDesc = sa.getParam("SpellDescription");
        if (charmDesc != null && !charmDesc.isEmpty()) {
            charmDesc = AbilityDescription.stripReminderText(charmDesc);
            // Strip trailing em-dash (oracle/SpellDescription often ends with " —")
            charmDesc = charmDesc.replaceAll("\\s*\u2014\\s*$", "").trim();
        }
        if ((charmDesc == null || charmDesc.isEmpty()) && sa.hasParam("Pawprint")) {
            String total = sa.getParam("Pawprint");
            charmDesc = "Choose up to " + total + " {P} worth of modes.";
            if ("True".equals(sa.getParam("CanRepeatModes"))) {
                charmDesc += " You may choose the same mode more than once.";
            }
        }

        // Synthesize "choose N —" header from CharmNum/MinCharmNum when SpellDescription is absent.
        if (charmDesc == null || charmDesc.isEmpty()) {
            charmDesc = synthesizeCharmHeader(sa);
        }

        // Collect charm choices as sub-abilities
        List<Ability> choiceSubs = new ArrayList<>();
        var choices = sa.getAdditionalAbilityList("Choices");
        if (choices != null) {
            for (var choice : choices) {
                String choiceDesc = SpellAbilityUtils.findParamInChain(choice, "SpellDescription");
                if (choiceDesc != null) {
                    choiceDesc = AbilityDescription.stripReminderText(choiceDesc);
                }
                String pawprint = choice.getParam("Pawprint");
                if (pawprint != null) {
                    choiceDesc = "{P}".repeat(Integer.parseInt(pawprint))
                            + " \u2014 " + choiceDesc;
                }
                choiceSubs.add(new TextAbility(AbilityType.OPTION,
                        AbilityDescription.applyCasing(choiceDesc)));
            }
        }

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
     * Used when the parent description already exists (e.g. triggered charm).
     */
    static List<Ability> optionsFrom(SpellAbility sa) {
        List<Ability> options = new ArrayList<>();
        var choices = sa.getAdditionalAbilityList("Choices");
        if (choices == null) return options;
        for (var choice : choices) {
            String desc = SpellAbilityUtils.findParamInChain(choice, "SpellDescription");
            if (desc != null) desc = AbilityDescription.stripReminderText(desc);
            if (desc == null || desc.isEmpty()) continue;
            options.add(new TextAbility(AbilityType.OPTION, AbilityDescription.applyCasing(desc)));
        }
        return options;
    }
}
