package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import com.pricepredictor.connector.NonBlankString;
import forge.game.spellability.SpellAbility;

import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Activated or planeswalker ability built from a SpellAbility.
 * The root description is taken from sa.getDescription() (cost + root SpellDescription only).
 * Sub-ability descriptions become child SpellEffect nodes via SpellEffect.fromChain().
 */
public record ActivatedAbilityEntry(AbilityType type, NonBlankString descriptionText, List<Ability> subAbilities) implements Ability {

    public ActivatedAbilityEntry {
        Objects.requireNonNull(subAbilities);
        subAbilities = List.copyOf(subAbilities);
    }

    public static Optional<ActivatedAbilityEntry> of(SpellAbility sa) {
        AbilityType type = sa.isPwAbility() ? AbilityType.PLANESWALKER : AbilityType.ACTIVATED;

        // Walk sub-chain unconditionally; used to pad the root description and detect
        // abilities that carry their description only on a SubAbility (e.g. Arachnus Spinner).
        String subText = SpellEffect.flattenChainText(sa.getSubAbility());

        String spellDesc = sa.getParam("SpellDescription");
        boolean hasSpellDesc = spellDesc != null && !spellDesc.isEmpty();

        // No root SpellDescription and no sub-chain text → nothing to emit.
        if (!hasSpellDesc && subText.isEmpty()) return Optional.empty();

        String rootDesc = sa.getDescription();
        // Fallback for abilities without SpellDescription: cost-based description.
        if (!hasSpellDesc && rootDesc.isEmpty()) rootDesc = sa.getCostDescription();
        // For dice-roll activated abilities, getDescription() appends outcome lines after a
        // newline. Strip them here; they will be re-emitted as OPTION sub-abilities below.
        int nl = rootDesc.indexOf('\n');
        if (nl >= 0) rootDesc = rootDesc.substring(0, nl);

        NonBlankString normalized = rootDesc.isEmpty() ? null : AbilityDescription.normalize(rootDesc).orElse(null);
        // When a root SpellDescription is present, normalize must succeed.
        if (hasSpellDesc && normalized == null) return Optional.empty();

        // Concatenate sub-ability chain descriptions into the root line so that oracle
        // lines (one paragraph = one activated ability) are not over-split.
        NonBlankString fullDesc = (normalized != null)
                ? (subText.isEmpty() ? normalized : NonBlankString.require(normalized + " " + subText))
                : NonBlankString.of(subText).orElse(null);
        if (fullDesc == null) return Optional.empty();

        List<Ability> children = SpellAbilityUtils.expandDiceOutcomes(sa, AbilityType.OPTION, false);
        return Optional.of(new ActivatedAbilityEntry(type, type.formatDescription(fullDesc), children));
    }

}
