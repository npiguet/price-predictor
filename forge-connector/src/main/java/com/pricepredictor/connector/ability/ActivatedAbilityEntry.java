package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.spellability.SpellAbility;

import java.util.List;
import java.util.Objects;

/**
 * Activated or planeswalker ability built from a SpellAbility.
 * The root description is taken from sa.getDescription() (cost + root SpellDescription only).
 * Sub-ability descriptions are appended to it, and dice outcomes become OPTION children.
 */
public record ActivatedAbilityEntry(AbilityType type, String descriptionText, List<Ability> subAbilities) implements Ability {

    public ActivatedAbilityEntry {
        Objects.requireNonNull(subAbilities);
        subAbilities = List.copyOf(subAbilities);
    }

    public static ActivatedAbilityEntry of(SpellAbility sa) {
        AbilityType type = sa.isPwAbility() ? AbilityType.PLANESWALKER : AbilityType.ACTIVATED;
        String spellDesc = sa.getParam("SpellDescription");
        boolean described = spellDesc != null && !spellDesc.isEmpty();

        // Concatenate sub-ability chain descriptions into the root line so that oracle
        // lines (one paragraph = one activated ability) are not over-split.
        String subText = SpellAbilityUtils.chainText(sa.getSubAbility());
        String rootDesc = sa.getDescription();
        if (!described) {
            // Some activated abilities have no SpellDescription on the root SA but carry their
            // description on a SubAbility$ SVar (e.g. Arachnus Spinner); with neither, there is
            // nothing to print. The root contributes only its cost.
            if (subText.isEmpty()) return null;
            if (rootDesc.isEmpty()) rootDesc = sa.getCostDescription();
        }
        // For dice-roll activated abilities, getDescription() appends outcome lines after a
        // newline. Strip them here; they are re-emitted as OPTION sub-abilities below.
        String normalized = AbilityDescription.normalize(AbilityDescription.firstLine(rootDesc));
        if (normalized == null && described) return null;

        String fullDesc = normalized == null ? subText
                : subText.isEmpty() ? normalized : normalized + " " + subText;
        return new ActivatedAbilityEntry(type, type.formatDescription(fullDesc),
                DiceOutcomes.of(sa, DiceOutcomes::optionLine));
    }
}
