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
 * The line of a Forge Trigger, built from its {@code TriggerDescription$}. Type
 * is TRIGGERED, or REPLACEMENT if the trigger is static.
 *
 * <p>The execute chain contributes text only where the description leaves it
 * out: a charm's modes become option sub-lines (and its {@code ABILITY}
 * placeholder, if any, its header), a dice placeholder is expanded into
 * outcome sub-lines, and a transparent execute's sub-ability descriptions are appended
 * to the line itself rather than becoming children.
 */
public record TriggeredAbilityEntry(AbilityType type, String descriptionText, List<Ability> subAbilities) implements Ability {

    public TriggeredAbilityEntry {
        Objects.requireNonNull(subAbilities);
        subAbilities = List.copyOf(subAbilities);
    }

    public static TriggeredAbilityEntry of(Trigger trigger) {
        // Skip keyword-associated triggers except for Tribute's "if tribute wasn't paid" ability.
        // Most keyword triggers merely restate the keyword (already output by StandardKeyword),
        // but Tribute's TrigNotTribute trigger is a distinct oracle ability.
        if (trigger.getKeyword() != null
                && trigger.getKeyword().getKeyword() != Keyword.TRIBUTE) return null;
        String rawDesc = trigger.getParam("TriggerDescription");
        String normalized = AbilityDescription.normalize(rawDesc);
        if (normalized == null) return null;
        AbilityType type = AbilityType.forTrigger(trigger);
        if (trigger.getKeyword() != null) {
            // For Tribute, TriggerDescription is the full oracle text (built directly from
            // TrigNotTribute's SpellDescription), so the execute chain would duplicate it.
            // Return with empty children — the description is self-contained.
            return new TriggeredAbilityEntry(type, normalized, List.of());
        }

        SpellAbility execute = trigger.getOverridingAbility();
        boolean placeholder = rawDesc.contains("ABILITY");
        SpellAbility charm = execute == null ? null : charmOf(execute);
        if (charm != null) return charmEntry(type, trigger, charm, normalized);

        // Do NOT walk the execute SA chain for SpellEffect descriptions: TriggerDescription
        // is the authoritative oracle text for triggered abilities.  Walking the chain would
        // re-emit SVars that are already covered by ETBReplacement / replacement keywords,
        // causing duplicates (e.g. sigarda's_splendor's NoteNum SVar).
        // Dice-outcome options are the one exception — they represent distinct result lines.
        List<Ability> children = execute != null
                ? DiceOutcomes.of(execute, DiceOutcomes::optionLine) : List.of();

        // If ABILITY placeholder remains and dice outcomes were found, replace with dice-roll description.
        if (placeholder && !children.isEmpty()) {
            String diceDesc = DiceOutcomes.rollDescription(execute);
            if (diceDesc != null) {
                normalized = AbilityDescription.normalize(
                        rawDesc.replace("ABILITY", AbilityDescription.replaceVert(diceDesc)));
            }
        }

        // Walk the execute chain for trailing sub-SpellDescriptions.
        // Only when the execute SA itself is transparent (has no description of its own):
        // in that case, sub-ability SpellDescriptions contain oracle text not in TriggerDescription.
        // Guard: also skip ImmediateTrigger (processed separately) and ETBReplacement (registered
        // as replacement effects) to avoid duplicates (sigarda's_splendor concern).
        if (execute != null
                && !SpellAbilityUtils.hasOwnDescription(execute)
                && execute.getApi() != ApiType.ImmediateTrigger
                && !"ETBReplacement".equals(execute.getParam("Mode"))) {
            String extra = SpellAbilityUtils.chainText(execute);
            normalized = DescriptionText.appendMissing(normalized, extra);
        }
        return new TriggeredAbilityEntry(type, normalized, children);
    }

    /**
     * The charm a trigger executes, directly or through an ImmediateTrigger
     * anywhere in its SubAbility chain (Caesar, Hylda: direct; Cemetery
     * Desecrator: ChangeZone → SubAbility → ImmediateTrigger), or null.
     */
    private static SpellAbility charmOf(SpellAbility execute) {
        return execute.getApi() == ApiType.Charm ? execute : immediateTriggerCharm(execute);
    }

    /**
     * The line of a trigger executing {@code charm}: its description, then one
     * option line per mode. An {@code ABILITY} placeholder becomes the charm's
     * header; a description without one (The Fantastic Four's "choose one that
     * hasn't been chosen this turn") already states it and is printed as written.
     */
    private static TriggeredAbilityEntry charmEntry(AbilityType type, Trigger trigger,
                                                    SpellAbility charm, String normalized) {
        String rawDesc = trigger.getParam("TriggerDescription");
        if (!rawDesc.contains("ABILITY")) {
            return new TriggeredAbilityEntry(type, normalized, CharmAbility.optionsFrom(charm, trigger));
        }
        CharmAbility.Expanded expanded = CharmAbility.expandPlaceholder(rawDesc, charm, trigger);
        return new TriggeredAbilityEntry(type, expanded.description(), expanded.options());
    }

    /**
     * Walk the SubAbility chain of {@code sa} to find an ImmediateTrigger whose Execute is a Charm.
     * Returns that Charm, or null if not found.
     */
    private static SpellAbility immediateTriggerCharm(SpellAbility sa) {
        for (SpellAbility cur = sa; cur != null; cur = cur.getSubAbility()) {
            if (cur.getApi() == ApiType.ImmediateTrigger) {
                SpellAbility exec = cur.getAdditionalAbility("Execute");
                if (exec != null && exec.getApi() == ApiType.Charm) return exec;
            }
        }
        return null;
    }
}
