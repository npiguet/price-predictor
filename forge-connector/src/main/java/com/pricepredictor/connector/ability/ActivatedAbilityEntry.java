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

        Optional<NonBlankString> spellDesc = NonBlankString.of(sa.getParam("SpellDescription"));

        // No root SpellDescription and no sub-chain text → nothing to emit.
        if (spellDesc.isEmpty() && subText.isEmpty()) return Optional.empty();

        // Resolve root description: prefer sa.getDescription() (includes cost prefix),
        // fall back to cost description alone when SpellDescription is absent.
        // Strip trailing newlines: dice-roll activated abilities append outcome lines after
        // a newline in getDescription(); those are re-emitted as OPTION children below.
        Optional<NonBlankString> rootNormalized = Optional.of(sa.getDescription())
                .map(d -> d.isEmpty() && spellDesc.isEmpty() ? sa.getCostDescription() : d)
                .map(d -> { int nl = d.indexOf('\n'); return nl >= 0 ? d.substring(0, nl) : d; })
                .flatMap(AbilityDescription::normalize);

        // When a root SpellDescription is present, normalization must succeed.
        if (spellDesc.isPresent() && rootNormalized.isEmpty()) return Optional.empty();

        // Concatenate sub-ability chain descriptions into the root line so that oracle
        // lines (one paragraph = one activated ability) are not over-split.
        Optional<NonBlankString> fullDesc = rootNormalized
                .map(n -> subText.isEmpty() ? n : NonBlankString.require(n + " " + subText))
                .or(() -> NonBlankString.of(subText));

        List<Ability> children = SpellAbilityUtils.expandDiceOutcomes(sa, AbilityType.OPTION, false);
        return fullDesc.map(d -> new ActivatedAbilityEntry(type, type.formatDescription(d), children));
    }

}
