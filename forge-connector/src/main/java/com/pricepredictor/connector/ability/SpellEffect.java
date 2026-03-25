package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
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
     * Recursively walks a SpellAbility chain, returning 0 or 1 tree nodes.
     * Nodes with no SpellDescription are transparent: their children are promoted.
     * Also walks {@code RepeatSubAbility} (used by RepeatEach/Repeat) alongside the
     * regular {@code SubAbility} chain so those descriptions are not missed.
     */
    public static List<Ability> fromChain(SpellAbility sa) {
        if (sa == null) return List.of();

        String rawDesc = sa.getParam("SpellDescription");
        String stripped = (rawDesc != null) ? AbilityDescription.stripReminderText(rawDesc) : null;
        boolean hasDesc = stripped != null && !stripped.isEmpty();

        List<Ability> children = fromChain(sa.getSubAbility());
        SpellAbility repeatSub = sa.getAdditionalAbility("RepeatSubAbility");
        if (repeatSub != null) {
            List<Ability> repeatChildren = fromChain(repeatSub);
            if (!repeatChildren.isEmpty()) {
                children = new ArrayList<>(children);
                String parentCased = hasDesc ? AbilityDescription.applyCasing(stripped) : null;
                for (Ability rc : repeatChildren) {
                    // Skip repeat-sub children whose description duplicates the parent SA.
                    // Some cards (e.g. Hoarder's Greed) copy the root SpellDescription onto
                    // the RepeatSubAbility SVar for stack-display; including it would double-emit.
                    if (parentCased != null && parentCased.equals(rc.descriptionText())) continue;
                    children.add(rc);
                }
            }
        }

        if (hasDesc) {
            return List.of(new SpellEffect(AbilityDescription.applyCasing(stripped), children));
        } else {
            return children;
        }
    }
}
