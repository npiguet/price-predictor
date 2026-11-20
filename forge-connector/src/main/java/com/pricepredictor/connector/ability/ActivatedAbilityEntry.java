package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.spellability.SpellAbility;

import java.util.List;
import java.util.Objects;
import java.util.Optional;

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

    public static Optional<ActivatedAbilityEntry> of(SpellAbility sa) {
        AbilityType type = sa.isPwAbility() ? AbilityType.PLANESWALKER : AbilityType.ACTIVATED;
        boolean described = hasSpellDescription(sa);

        // Concatenate sub-ability chain descriptions into the root line so that oracle
        // lines (one paragraph = one activated ability) are not over-split.
        List<Ability> sub = SpellAbilityUtils.chainHead(sa.getSubAbility());
        if (!described && sub.isEmpty()) return Optional.empty();
        String rootDesc = rootDescription(sa, described);
        // For dice-roll activated abilities, getDescription() appends outcome lines after a
        // newline. Strip them here; they are re-emitted as OPTION sub-abilities below.
        String normalized = AbilityDescription.normalize(AbilityDescription.firstLine(rootDesc));
        if (normalized == null && described) return Optional.empty();
        // A cost Forge has no words for prints as its raw selector: no line, its key dropped.
        if (DescriptionText.hasRawSelector(normalized)) return Optional.empty();

        String fullDesc = RepeatedSentences.appendMissing(normalized == null ? "" : normalized, sub);
        return Optional.of(new ActivatedAbilityEntry(type, type.formatDescription(fullDesc),
                DiceOutcomes.of(sa, DiceOutcomes::optionLine)));
    }

    private static boolean hasSpellDescription(SpellAbility sa) {
        String spellDesc = sa.getParam("SpellDescription");
        return spellDesc != null && !spellDesc.isEmpty();
    }

    /**
     * Some activated abilities have no SpellDescription on the root SA but carry their
     * description on a SubAbility$ SVar (e.g. Arachnus Spinner), with neither there is
     * nothing to print; the root then contributes only its cost.
     */
    private static String rootDescription(SpellAbility sa, boolean described) {
        String rootDesc = sa.getDescription();
        return !described && rootDesc.isEmpty() ? sa.getCostDescription() : rootDesc;
    }
}
