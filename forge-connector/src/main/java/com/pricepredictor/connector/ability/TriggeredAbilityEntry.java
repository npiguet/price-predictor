package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.ability.ApiType;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;

import java.util.List;
import java.util.Objects;

/**
 * Ability wrapping a Forge Trigger. Type is TRIGGERED, or REPLACEMENT if the trigger is static.
 * The execute SVar chain (if any) becomes child SpellEffect nodes via SpellEffect.fromChain().
 */
public record TriggeredAbilityEntry(AbilityType type, String descriptionText, List<Ability> subAbilities) implements Ability {

    public TriggeredAbilityEntry {
        Objects.requireNonNull(subAbilities);
        subAbilities = List.copyOf(subAbilities);
    }

    public static TriggeredAbilityEntry of(Trigger trigger) {
        if (trigger.getKeyword() != null) return null;
        String normalized = AbilityDescription.normalize(trigger.getParam("TriggerDescription"));
        if (normalized == null) return null;
        AbilityType effectiveType = trigger.isStatic()
                ? AbilityType.REPLACEMENT : AbilityType.TRIGGERED;

        SpellAbility execute = trigger.getOverridingAbility();

        // If the raw TriggerDescription contains the "ABILITY" placeholder and the
        // execute SVar is a Charm (choose-one), expand it: replace the placeholder
        // with "choose one —" and attach the charm choices as OPTION sub-abilities.
        String rawDesc = trigger.getParam("TriggerDescription");
        if (rawDesc != null && rawDesc.contains("ABILITY")
                && execute != null && execute.getApi() == ApiType.Charm) {
            String expanded = rawDesc.replace("ABILITY", "choose one \u2014");
            normalized = AbilityDescription.normalize(expanded);
            List<Ability> options = CharmAbility.optionsFrom(execute);
            return new TriggeredAbilityEntry(effectiveType, normalized, options);
        }

        List<Ability> children = SpellEffect.fromChain(execute);
        return new TriggeredAbilityEntry(effectiveType, normalized, children);
    }
}
