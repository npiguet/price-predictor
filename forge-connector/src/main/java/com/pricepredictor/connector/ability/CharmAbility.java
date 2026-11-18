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
import java.util.regex.Pattern;

/**
 * Charm ability with nested OPTION sub-abilities, rendered as the kind of line
 * its root is: a {@code spell} line for a modal spell, an {@code activated}
 * (or {@code planeswalker}) line opening with its cost for an activated charm.
 * A charm that writes no description of its own prints its choice sentence
 * ({@link CharmHeader}), so no charm root is left unprinted.
 */
public record CharmAbility(AbilityType type, String descriptionText, List<Ability> subAbilities)
        implements Ability {

    private static final Pattern TRAILING_EM_DASH = Pattern.compile("\\s*\u2014\\s*$");
    private static final Pattern REPEATED_SPACES = Pattern.compile(" {2,}");

    public CharmAbility {
        Objects.requireNonNull(type);
        Objects.requireNonNull(subAbilities);
        subAbilities = List.copyOf(subAbilities);
    }

    public static List<Ability> fromSpellAbility(SpellAbility sa) {
        String own = spellDescription(sa);
        // Every charm prints its root: an activated charm's cost is on no other
        // line, and a root nobody prints lands in dropped_keys.
        String header = own != null && !own.isEmpty() ? own : CharmHeader.of(sa);
        List<Ability> choiceSubs = optionsFrom(sa, null);
        AbilityType type = typeOf(sa);
        String root = AbilityDescription.applyCasing(leadOf(sa) + header);
        return new ArrayList<>(List.of(
                new CharmAbility(type, type.formatDescription(root), choiceSubs)));
    }

    /** The kind of line the charm's root is. */
    private static AbilityType typeOf(SpellAbility sa) {
        if (!sa.isActivatedAbility()) return AbilityType.SPELL;
        return sa.isPwAbility() ? AbilityType.PLANESWALKER : AbilityType.ACTIVATED;
    }

    /**
     * What the card prints ahead of the charm's header: an activated charm's
     * cost and its colon ({@code {1}{G}, {T}: }), as Forge describes it, or a
     * modal spell's ability word ({@code PrecostDesc}, "Delirium —").
     */
    private static String leadOf(SpellAbility sa) {
        if (sa.isActivatedAbility()) {
            return REPEATED_SPACES.matcher(sa.getCostDescription()).replaceAll(" ");
        }
        String precost = sa.getParamOrDefault("PrecostDesc", "").trim();
        return precost.isEmpty() ? "" : precost + " ";
    }

    /** The charm's own {@code SpellDescription}, without reminder text or a trailing em-dash. */
    private static String spellDescription(SpellAbility sa) {
        String desc = sa.getParam("SpellDescription");
        if (desc == null || desc.isEmpty()) {
            return desc;
        }
        desc = AbilityDescription.stripReminderText(desc);
        // Strip trailing em-dash (oracle/SpellDescription often ends with " —")
        return TRAILING_EM_DASH.matcher(desc).replaceAll("").trim();
    }

    /**
     * A Forge cost string as the mana symbols a card prints.
     *
     * <p>Forge writes a cost space-separated — {@code ModeCost$ 3 W W} — and one
     * pair of braces around the whole of it produces {@code {3 W W}}, which the
     * tokenizer reads as a single symbol because braces delimit an atom. That
     * would put a distinct token in the vocabulary for every multi-shard mode cost,
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
     * choice sentence ({@link CharmHeader}), its {@code AdditionalDescription$}
     * included, and the modes become the trait's option lines.
     *
     * @param desc  the raw description holding the placeholder
     * @param owner the trait that executes the charm, or null (see
     *              {@link #optionsFrom(SpellAbility, CardTraitBase)})
     * @return the normalized description, null if nothing is left of it, and
     *         the option lines
     */
    public static Expanded expandPlaceholder(String desc, SpellAbility charm,
                                             CardTraitBase owner) {
        String header = CharmHeader.of(charm);
        return new Expanded(
                AbilityDescription.normalize(DescriptionText.fill(desc, "ABILITY", header)),
                optionsFrom(charm, owner));
    }
}
