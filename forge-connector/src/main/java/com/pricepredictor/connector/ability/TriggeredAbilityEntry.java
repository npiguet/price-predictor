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
     * (carries no description of its own), subject to two additional guards:
     * <ul>
     *   <li>{@code ImmediateTrigger} SAs fire immediately on enter-the-battlefield and
     *       are also registered as top-level triggers via {@code createTraits()}. Walking
     *       their chain here would duplicate the TriggeredAbilityEntry already emitted
     *       by {@code RulesParser.collectTriggers()}.</li>
     *   <li>{@link ForgeParams#ETB_REPLACEMENT_MODE ETBReplacement} SAs are replacement
     *       effects handled by the replacement-effects loop in
     *       {@code RulesParser.collectStaticsAndReplacements()}. They must not be walked
     *       here to avoid double-emission.</li>
     * </ul>
     * Returns {@code normalized} unchanged if no extra text is found.
     */
    private static String appendTransparentChainText(SpellAbility execute, String normalized) {
        if (execute != null
                && SaDescription.resolve(execute).isEmpty()
                && execute.getApi() != ApiType.ImmediateTrigger
                && !ForgeParams.ETB_REPLACEMENT_MODE.equals(execute.getParam("Mode"))) {
            String extra = SpellEffect.flattenChainText(execute);
            if (!extra.isEmpty()) {
                return normalized + " " + extra;
            }
        }
        return normalized;
    }

}
