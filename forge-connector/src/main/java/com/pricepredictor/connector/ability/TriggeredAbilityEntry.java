package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.ability.ApiType;
import forge.game.keyword.Keyword;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;

import java.util.ArrayList;
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
        // Skip keyword-associated triggers except for Tribute's "if tribute wasn't paid" ability.
        // Most keyword triggers merely restate the keyword (already output by StandardKeyword),
        // but Tribute's TrigNotTribute trigger is a distinct oracle ability.
        if (trigger.getKeyword() != null) {
            if (trigger.getKeyword().getKeyword() != Keyword.TRIBUTE) return null;
            // For Tribute, TriggerDescription is the full oracle text (built directly from
            // TrigNotTribute's SpellDescription), so the execute chain would duplicate it.
            // Return with empty children — the description is self-contained.
            String normalized = AbilityDescription.normalize(trigger.getParam("TriggerDescription"));
            if (normalized == null) return null;
            AbilityType effectiveType = trigger.isStatic() ? AbilityType.REPLACEMENT : AbilityType.TRIGGERED;
            return new TriggeredAbilityEntry(effectiveType, normalized, List.of());
        }
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
            String charmHeader = CharmAbility.synthesizeCharmHeader(execute);
            if (charmHeader == null) charmHeader = "choose one";
            String expanded = rawDesc.replace("ABILITY", charmHeader);
            normalized = AbilityDescription.normalize(expanded);
            List<Ability> options = CharmAbility.optionsFrom(execute);
            return new TriggeredAbilityEntry(effectiveType, normalized, options);
        }

        // Do NOT walk the execute SA chain for SpellEffect descriptions: TriggerDescription
        // is the authoritative oracle text for triggered abilities.  Walking the chain would
        // re-emit SVars that are already covered by ETBReplacement / replacement keywords,
        // causing duplicates (e.g. sigarda's_splendor's NoteNum SVar).
        // Dice-outcome options are the one exception — they represent distinct result lines.
        List<Ability> children = execute != null ? diceOutcomesAsOptions(execute) : List.of();
        return new TriggeredAbilityEntry(effectiveType, normalized, children);
    }

    /**
     * Expand {@code ResultSubAbilities} dice-outcome entries into OPTION sub-abilities.
     * Mirrors {@code ActivatedAbilityEntry.diceOutcomesAsOptions()}.
     */
    private static List<Ability> diceOutcomesAsOptions(SpellAbility sa) {
        // Walk the sub-ability chain to find a SA with ResultSubAbilities.
        // Some cards (e.g. journey_to_the_lost_city) have the dice SA as a sub-ability.
        for (SpellAbility cur = sa; cur != null; cur = cur.getSubAbility()) {
            String resultSubAbilities = cur.getParam("ResultSubAbilities");
            if (resultSubAbilities == null || resultSubAbilities.isEmpty()) continue;
            List<Ability> result = new ArrayList<>();
            for (String entry : resultSubAbilities.split(",")) {
                String[] kv = entry.trim().split(":", 2);
                if (kv.length < 2) continue;
                String range = kv[0].trim();
                SpellAbility sub = cur.getAdditionalAbility(range);
                if (sub == null) continue;
                String rawDesc = sub.getParam("SpellDescription");
                if (rawDesc == null || rawDesc.isEmpty()) continue;
                String desc = AbilityDescription.replaceVert(rawDesc);
                String normalized = AbilityDescription.normalize(desc);
                if (normalized != null) {
                    result.add(new TextAbility(AbilityType.OPTION, AbilityDescription.applyCasing(normalized)));
                }
            }
            return result;
        }
        return List.of();
    }
}
