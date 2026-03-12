package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.trigger.Trigger;

/**
 * Ability wrapping a Forge Trigger. Type is TRIGGERED, or REPLACEMENT if the trigger is static.
 */
public record TriggeredAbilityEntry(AbilityType type, String descriptionText) implements Ability {

    public static TriggeredAbilityEntry of(Trigger trigger) {
        if (trigger.getKeyword() != null) return null;
        String normalized = AbilityDescription.normalize(trigger.getParam("TriggerDescription"));
        if (normalized == null) return null;
        AbilityType effectiveType = trigger.isStatic()
                ? AbilityType.REPLACEMENT : AbilityType.TRIGGERED;
        return new TriggeredAbilityEntry(effectiveType, normalized);
    }
}
