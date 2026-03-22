package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import com.pricepredictor.connector.ActionCounter;
import forge.game.spellability.SpellAbility;

import java.util.List;
import java.util.Objects;

/**
 * Spell effect ability. Factory walks the sub-ability chain recursively, building
 * a tree of SpellEffect nodes — one per SpellDescription fragment. Nodes without a
 * description are transparent: their children are promoted upward.
 */
public record SpellEffect(String descriptionText, List<Ability> subAbilities) implements Ability {

    public SpellEffect {
        Objects.requireNonNull(subAbilities);
        subAbilities = List.copyOf(subAbilities);
    }

    @Override
    public AbilityType type() {
        return AbilityType.SPELL;
    }

    /**
     * Format the entire SpellEffect sub-tree as a single {@code spell[N]} line.
     * All descriptions in the chain are concatenated with a space, matching how
     * MTG oracle text presents sequential sub-effects as one ability paragraph.
     * The internal tree structure (sub-abilities) is preserved; only the text
     * output is flattened.
     */
    @Override
    public String formatBlock(ActionCounter counter) {
        return type().getOutputPrefix() + "[" + counter.next() + "]: " + collectChainText(this);
    }

    /** Recursively collect and join descriptions from this node and all descendants. */
    private static String collectChainText(Ability node) {
        StringBuilder sb = new StringBuilder(node.descriptionText());
        for (Ability child : node.subAbilities()) {
            String childText = collectChainText(child);
            if (!childText.isEmpty()) {
                sb.append(' ').append(childText);
            }
        }
        return sb.toString();
    }

    /**
     * Recursively walks a SpellAbility chain, returning 0 or 1 tree nodes.
     * Nodes with no SpellDescription are transparent: their children are promoted.
     */
    public static List<Ability> fromChain(SpellAbility sa) {
        if (sa == null) return List.of();

        String rawDesc = sa.getParam("SpellDescription");
        String stripped = (rawDesc != null) ? AbilityDescription.stripReminderText(rawDesc) : null;
        boolean hasDesc = stripped != null && !stripped.isEmpty();

        List<Ability> children = fromChain(sa.getSubAbility());

        if (hasDesc) {
            return List.of(new SpellEffect(AbilityDescription.applyCasing(stripped), children));
        } else {
            return children;
        }
    }
}
