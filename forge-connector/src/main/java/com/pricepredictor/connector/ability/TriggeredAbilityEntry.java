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
        if (!printsOwnLine(trigger)) return null;
        String rawDesc = trigger.getParam("TriggerDescription");
        String normalized = AbilityDescription.normalize(rawDesc);
        if (normalized == null) return null;
        AbilityType type = AbilityType.forTrigger(trigger, normalized);
        if (trigger.getKeyword() != null) {
            // For Tribute, TriggerDescription is the full oracle text (built directly from
            // TrigNotTribute's SpellDescription), so the execute chain would duplicate it.
            // Return with empty children: the description is self-contained.
            return new TriggeredAbilityEntry(type, normalized, List.of());
        }

        SpellAbility execute = trigger.getOverridingAbility();
        SpellAbility charm = execute == null ? null : charmOf(execute);
        if (charm != null) return charmEntry(type, trigger, charm, normalized);
        return executeEntry(type, rawDesc, normalized, execute);
    }

    /**
     * Whether the trigger prints a line of its own. Keyword-associated
     * triggers do not, except for Tribute's "if tribute was not paid" ability:
     * most keyword triggers merely restate the keyword (already output by
     * StandardKeyword), but Tribute's TrigNotTribute trigger is a distinct
     * oracle ability.
     */
    private static boolean printsOwnLine(Trigger trigger) {
        return trigger.getKeyword() == null
                || trigger.getKeyword().getKeyword() == Keyword.TRIBUTE;
    }

    /**
     * The line of a trigger whose execute chain is no charm: the description,
     * with the execute chain's text only where the description leaves it out.
     *
     * <p>The chain's SpellEffect descriptions are not walked:
     * TriggerDescription is the authoritative oracle text, and walking the
     * chain would re-emit SVars already covered by ETBReplacement /
     * replacement keywords, causing duplicates (sigarda's_splendor's NoteNum
     * SVar). Dice-outcome options are the one exception: they are distinct
     * result lines.
     */
    private static TriggeredAbilityEntry executeEntry(AbilityType type, String rawDesc,
                                                      String normalized, SpellAbility execute) {
        List<Ability> children = execute != null
                ? DiceOutcomes.of(execute, DiceOutcomes::optionLine) : List.of();
        String text = DescriptionMarkers.hasAbilityPlaceholder(rawDesc) && !children.isEmpty()
                ? withDiceRoll(rawDesc, normalized, execute) : normalized;
        if (execute != null && isTransparent(execute)) {
            text = DescriptionText.appendMissing(text, SpellAbilityUtils.chainText(execute));
        }
        return new TriggeredAbilityEntry(type, text, children);
    }

    /**
     * The description with its ABILITY placeholder replaced by the dice-roll
     * description of {@code execute}, or {@code normalized} when it has none.
     */
    private static String withDiceRoll(String rawDesc, String normalized, SpellAbility execute) {
        String diceDesc = DiceOutcomes.rollDescription(execute);
        if (diceDesc == null) return normalized;
        return AbilityDescription.normalize(DescriptionText.fill(
                rawDesc, DescriptionMarkers.ABILITY_PLACEHOLDER,
                AbilityDescription.replaceVert(diceDesc)));
    }

    /**
     * Whether the sub-ability descriptions of {@code execute}'s chain hold
     * oracle text the TriggerDescription lacks: only when the execute itself
     * has no description of its own. ImmediateTrigger (processed separately)
     * and ETBReplacement (registered as replacement effects) are skipped to
     * avoid duplicates (sigarda's_splendor concern).
     */
    private static boolean isTransparent(SpellAbility execute) {
        return !SpellAbilityUtils.hasOwnDescription(execute)
                && execute.getApi() != ApiType.ImmediateTrigger
                && !"ETBReplacement".equals(execute.getParam("Mode"));
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
        if (!DescriptionMarkers.hasAbilityPlaceholder(rawDesc)) {
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
