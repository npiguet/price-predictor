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
 * Sub-ability descriptions become child SpellEffect nodes via SpellEffect.fromChain().
 */
public record ActivatedAbilityEntry(AbilityType type, String descriptionText, List<Ability> subAbilities) implements Ability {

    public ActivatedAbilityEntry {
        Objects.requireNonNull(subAbilities);
        subAbilities = List.copyOf(subAbilities);
    }

    public static ActivatedAbilityEntry of(SpellAbility sa) {
        AbilityType type = sa.isPwAbility() ? AbilityType.PLANESWALKER : AbilityType.ACTIVATED;

        if (sa.getParam("SpellDescription") == null
                || sa.getParam("SpellDescription").isEmpty()) {
            // Some activated abilities have no SpellDescription on the root SA but carry their
            // description on a SubAbility$ SVar (e.g. Arachnus Spinner).  Walk the sub-chain;
            // if it produces nodes, concatenate their text into the cost-based root description.
            String subText = SpellEffect.flattenChainText(sa.getSubAbility());
            if (subText.isEmpty()) return null;
            String rootDesc = sa.getDescription();
            if (rootDesc.isEmpty()) rootDesc = sa.getCostDescription();
            int nl = rootDesc.indexOf('\n');
            if (nl >= 0) rootDesc = rootDesc.substring(0, nl);
            String normalized = AbilityDescription.normalize(rootDesc);
            String fullDesc = (normalized != null && !normalized.isEmpty())
                    ? normalized + " " + subText : subText;
            if (fullDesc.isEmpty()) return null;
            return new ActivatedAbilityEntry(type, type.formatDescription(fullDesc),
                    SpellAbilityUtils.expandDiceOutcomes(sa, AbilityType.OPTION, false));
        }

        String rootDesc = sa.getDescription();
        // For dice-roll activated abilities, getDescription() appends outcome lines after a
        // newline. Strip them here; they will be re-emitted as OPTION sub-abilities below.
        int nl = rootDesc.indexOf('\n');
        if (nl >= 0) rootDesc = rootDesc.substring(0, nl);

        String normalized = AbilityDescription.normalize(rootDesc);
        if (normalized == null) return null;

        // Concatenate sub-ability chain descriptions into the root line so that oracle
        // lines (one paragraph = one activated ability) are not over-split.
        String subText = SpellEffect.flattenChainText(sa.getSubAbility());
        String fullDesc = subText.isEmpty() ? normalized : normalized + " " + subText;

        List<Ability> children = SpellAbilityUtils.expandDiceOutcomes(sa, AbilityType.OPTION, false);
        return new ActivatedAbilityEntry(type, type.formatDescription(fullDesc), children);
    }

}
