package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.CardTraitBase;
import forge.game.spellability.AbilitySub;
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

        if (charmDesc == null || charmDesc.isEmpty()) {
            String additionalDesc = sa.getParam("AdditionalDescription");
            if (additionalDesc != null && !additionalDesc.isEmpty()) {
                charmDesc = AbilityDescription.stripReminderText(additionalDesc);
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
            for (int modeIndex = 0; modeIndex < choices.size(); modeIndex++) {
                String choiceDesc = optionText(choices.get(modeIndex));
                if (choiceDesc == null) continue;
                choiceSubs.add(new OptionAbility(
                        AbilityDescription.applyCasing(choiceDesc), modeIndex, sa, null));
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

    /**
     * A Forge cost string as the mana symbols a card prints.
     *
     * <p>Forge writes a cost space-separated — {@code ModeCost$ 3 W W} — and one
     * pair of braces around the whole of it produces {@code {3 W W}}, which the
     * tokenizer reads as a single symbol because braces delimit an atom. That
     * put a distinct token in the vocabulary for every multi-shard mode cost,
     * each unrelated to the {@code {3}}, {@code {W}} the rest of the corpus
     * uses. One pair of braces per shard is what a card prints and what every
     * other cost in the converted text already looks like.
     */
    static String manaCostSymbols(String cost) {
        StringBuilder out = new StringBuilder();
        for (String shard : cost.trim().split("\\s+")) {
            if (!shard.isEmpty()) {
                out.append('{').append(shard).append('}');
            }
        }
        return out.toString();
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
     *
     * @param owner the trait that executes the charm, so each option can be
     *              attributed to that trait's line rather than to whichever
     *              trigger last claimed a charm SVar two chapters share; null
     *              when one line stands for several triggers, so each option
     *              belongs to whichever of them executes the charm
     */
    public static List<Ability> optionsFrom(SpellAbility sa, CardTraitBase owner) {
        List<Ability> options = new ArrayList<>();
        var choices = sa.getAdditionalAbilityList("Choices");
        if (choices == null) return options;
        for (int modeIndex = 0; modeIndex < choices.size(); modeIndex++) {
            String desc = optionText(choices.get(modeIndex));
            if (desc == null) continue;
            options.add(new OptionAbility(
                    AbilityDescription.applyCasing(desc), modeIndex, sa, owner));
        }
        return options;
    }

    /**
     * The text of one mode's {@code option} line, or null when the mode has
     * none to print.
     *
     * <p>A mode's own description is prefixed by whatever the card prints ahead
     * of it: a tiered charm's ability name ({@code PrecostDesc}, e.g. Vincent's
     * Limit Break), its additional cost ({@code ModeCost}), and its pawprint
     * cost. Both a modal spell and a triggered charm render modes through here,
     * so the two cannot disagree on what a mode line says.
     */
    static String optionText(AbilitySub choice) {
        String desc = SpellAbilityUtils.findParamInChain(choice, "SpellDescription");
        if (desc != null) desc = AbilityDescription.stripReminderText(desc);
        desc = prefixed(choice.getParam("PrecostDesc"), desc);
        String modeCost = choice.getParam("ModeCost");
        if (modeCost != null && !modeCost.isEmpty()) {
            desc = prefixed(manaCostSymbols(modeCost), desc);
        }
        int pawprints = parseSimpleInt(choice.getParam("Pawprint"));
        if (pawprints > 0) {
            desc = prefixed("{P}".repeat(pawprints), desc);
        }
        return desc == null || desc.isEmpty() ? null : desc;
    }

    /**
     * {@code prefix — desc}, or whichever of the two is present on its own.
     * A null or empty prefix leaves {@code desc} as it was.
     */
    static String prefixed(String prefix, String desc) {
        if (prefix == null || prefix.isEmpty()) return desc;
        if (desc == null || desc.isEmpty()) return prefix;
        return prefix + " \u2014 " + desc;
    }

    /** A trait description with its charm spelled out: the header line and the mode lines. */
    public record Expanded(String description, List<Ability> options) {
    }

    /**
     * Spell out the {@code ABILITY} placeholder of a trait whose effect is a
     * charm.
     *
     * <p>Forge writes {@code "When CARDNAME enters, ABILITY"} and fills the
     * placeholder from the charm at runtime. Here it becomes the charm's
     * "choose N" header — "choose one" when the count is an expression — and
     * the modes become the trait's option lines.
     *
     * @param desc  the raw description holding the placeholder
     * @param owner the trait that executes the charm, or null (see
     *              {@link #optionsFrom(SpellAbility, CardTraitBase)})
     * @return the normalized description, null if nothing is left of it, and
     *         the option lines
     */
    public static Expanded expandPlaceholder(String desc, SpellAbility charm,
                                             CardTraitBase owner) {
        String header = synthesizeCharmHeader(charm);
        if (header == null) header = "choose one";
        return new Expanded(
                AbilityDescription.normalize(desc.replace("ABILITY", header)),
                optionsFrom(charm, owner));
    }
}
