package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import forge.game.spellability.SpellAbility;
import forge.util.FileSection;

import java.util.List;

/**
 * Utility methods for walking SpellAbility chains and reading their SVars.
 */
final class SpellAbilityUtils {

    private SpellAbilityUtils() {}

    static String findParamInChain(SpellAbility sa, String param) {
        for (SpellAbility s = sa; s != null; s = s.getSubAbility()) {
            String value = s.getParam(param);
            if (value != null && !value.isEmpty()) {
                return value;
            }
        }
        return null;
    }

    /**
     * The description {@code sa} declares for itself, its
     * {@code SpellDescription$} or else its {@code TriggerDescription$}, or
     * null when it declares neither.
     */
    static String declaredDescription(SpellAbility sa) {
        String desc = sa.getParam("SpellDescription");
        if (desc == null || desc.isEmpty()) {
            desc = sa.getParam("TriggerDescription");
        }
        return desc == null || desc.isEmpty() ? null : desc;
    }

    /**
     * {@code sa}'s {@code StackDescription$} when it is literal oracle text,
     * or null.
     *
     * <p>Forge also puts things there that are not text to render: the
     * {@code SpellDescription} keyword reference, the {@code None} no-display
     * sentinel, and templates ({@code REP } substitutions, {@code {p:}}
     * placeholders) it fills in only on the stack.
     */
    static String literalStackDescription(SpellAbility sa) {
        String stack = sa.getParam("StackDescription");
        if (stack == null || stack.isEmpty()
                || stack.equals("SpellDescription")
                || stack.equals("None")
                || stack.contains("REP ")
                || stack.contains("{p:")) {
            return null;
        }
        return stack;
    }

    /**
     * True when {@code sa} carries text of its own: a declared description or
     * a literal stack description.
     *
     * <p>The one test of "described" both the effect tree and a trigger's
     * transparent-execute check use, so a node the tree treats as transparent
     * is never one the trigger treats as described, or the other way round.
     */
    static boolean hasOwnDescription(SpellAbility sa) {
        return declaredDescription(sa) != null || literalStackDescription(sa) != null;
    }

    /**
     * The value of {@code key} in an SVar's {@code Key$ value | …} text, or
     * null when the text is null or has no such parameter.
     *
     * <p>Parsed the way Forge parses the SVar itself, so one key cannot match
     * inside another, as {@code Description$} does inside
     * {@code SpellDescription$} under a substring search.
     */
    static String svarParam(String svarText, String key) {
        if (svarText == null) return null;
        return FileSection.parseToMap(svarText, FileSection.DOLLAR_SIGN_KV_SEPARATOR).get(key);
    }

    /**
     * The text of the effect tree {@code sa}'s chain renders, every node's
     * description joined by spaces in tree order, or empty when it renders
     * none. Only the chain's first tree counts.
     *
     * <p>For an ability whose printed line is one paragraph — an activated
     * ability, a trigger or a chapter whose own description leaves the effect
     * out — so the chain's text is appended to that line instead of being
     * split into lines of its own.
     */
    static String chainText(SpellAbility sa) {
        List<Ability> chain = SpellEffect.fromChain(sa);
        return chain.isEmpty() ? "" : treeText(chain.get(0));
    }

    private static String treeText(Ability node) {
        StringBuilder sb = new StringBuilder(node.descriptionText());
        for (Ability child : node.subAbilities()) {
            String ct = treeText(child);
            if (!ct.isEmpty()) sb.append(' ').append(ct);
        }
        return sb.toString();
    }
}
