package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.ability.ApiType;
import forge.game.keyword.Keyword;
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
        if (trigger.getKeyword() != null) {
            return handleKeywordTrigger(trigger);
        }
        String rawDesc = trigger.getParam("TriggerDescription");
        String normalized = AbilityDescription.normalize(rawDesc);
        if (normalized == null) return null;
        AbilityType effectiveType = trigger.isStatic()
                ? AbilityType.REPLACEMENT : AbilityType.TRIGGERED;

        SpellAbility execute = trigger.getOverridingAbility();

        // Resolve the ABILITY placeholder (charm, dice-roll).
        SpellAbilityUtils.AbilityPlaceholderResult resolved =
                SpellAbilityUtils.resolveAbilityPlaceholder(rawDesc, execute, null);
        if (resolved != null) {
            return new TriggeredAbilityEntry(effectiveType, resolved.expandedDescription(), resolved.options());
        }

        // Do NOT walk the execute SA chain for SpellEffect descriptions: TriggerDescription
        // is the authoritative oracle text for triggered abilities.  Walking the chain would
        // re-emit SVars that are already covered by ETBReplacement / replacement keywords,
        // causing duplicates (e.g. sigarda's_splendor's NoteNum SVar).
        // Dice-outcome options are the one exception — they represent distinct result lines.
        List<Ability> children = execute != null
                ? SpellAbilityUtils.expandDiceOutcomes(execute, AbilityType.OPTION, true)
                : List.of();

        normalized = appendTransparentChainText(execute, normalized);
        return new TriggeredAbilityEntry(effectiveType, normalized, children);
    }

    /**
     * Handles keyword-associated triggers. Most keyword triggers merely restate the keyword
     * (already output by StandardKeyword) and are skipped. The Tribute exception emits the
     * "if tribute wasn't paid" ability with TriggerDescription as self-contained oracle text.
     * Returns null for all non-Tribute keyword triggers.
     */
    private static TriggeredAbilityEntry handleKeywordTrigger(Trigger trigger) {
        if (trigger.getKeyword().getKeyword() != Keyword.TRIBUTE) return null;
        // For Tribute, TriggerDescription is the full oracle text (built directly from
        // TrigNotTribute's SpellDescription), so the execute chain would duplicate it.
        // Return with empty children — the description is self-contained.
        String normalized = AbilityDescription.normalize(trigger.getParam("TriggerDescription"));
        if (normalized == null) return null;
        AbilityType effectiveType = trigger.isStatic() ? AbilityType.REPLACEMENT : AbilityType.TRIGGERED;
        return new TriggeredAbilityEntry(effectiveType, normalized, List.of());
    }

    /**
     * Appends extra oracle text from the execute SA chain when the SA is transparent
     * (no description of its own), not an ImmediateTrigger, and not an ETBReplacement.
     * Sub-ability SpellDescriptions in that case contain oracle text not in TriggerDescription.
     * Returns {@code normalized} unchanged if no extra text is found.
     */
    private static String appendTransparentChainText(SpellAbility execute, String normalized) {
        if (execute != null
                && isTransparentSA(execute)
                && execute.getApi() != ApiType.ImmediateTrigger
                && !"ETBReplacement".equals(execute.getParam("Mode"))) {
            String extra = SpellEffect.flattenChainText(execute);
            if (!extra.isEmpty()) {
                return normalized + " " + extra;
            }
        }
        return normalized;
    }

    /**
     * Returns true when a SpellAbility has no description of its own (transparent).
     * Transparent SAs are safe to walk: extra oracle text comes only from sub-abilities.
     */
    private static boolean isTransparentSA(SpellAbility sa) {
        String spellDesc = sa.getParam("SpellDescription");
        if (spellDesc != null && !spellDesc.isEmpty()) return false;
        String trigDesc = sa.getParam("TriggerDescription");
        if (trigDesc != null && !trigDesc.isEmpty()) return false;
        String stackDesc = sa.getParam("StackDescription");
        if (stackDesc != null && !stackDesc.isEmpty() && !stackDesc.equals("None")) return false;
        return true;
    }

}
