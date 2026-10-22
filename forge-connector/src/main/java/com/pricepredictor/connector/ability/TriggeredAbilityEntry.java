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
            List<Ability> options = CharmAbility.optionsFrom(execute, trigger);
            return new TriggeredAbilityEntry(effectiveType, normalized, options);
        }

        // Handle ImmediateTrigger → Charm anywhere in the execute SubAbility chain
        // (e.g. Caesar, Hylda: direct; Cemetery Desecrator: ChangeZone → SubAbility → ImmediateTrigger).
        if (rawDesc != null && rawDesc.contains("ABILITY") && execute != null) {
            SpellAbility immTrig = findImmediateTriggerWithCharm(execute);
            if (immTrig != null) {
                SpellAbility nestedCharm = immTrig.getAdditionalAbility("Execute");
                String charmHeader = CharmAbility.synthesizeCharmHeader(nestedCharm);
                if (charmHeader == null) charmHeader = "choose one";
                String expanded = rawDesc.replace("ABILITY", charmHeader);
                normalized = AbilityDescription.normalize(expanded);
                List<Ability> options = CharmAbility.optionsFrom(nestedCharm, trigger);
                return new TriggeredAbilityEntry(effectiveType, normalized, options);
            }
        }

        // Do NOT walk the execute SA chain for SpellEffect descriptions: TriggerDescription
        // is the authoritative oracle text for triggered abilities.  Walking the chain would
        // re-emit SVars that are already covered by ETBReplacement / replacement keywords,
        // causing duplicates (e.g. sigarda's_splendor's NoteNum SVar).
        // Dice-outcome options are the one exception — they represent distinct result lines.
        List<Ability> children = execute != null
                ? DiceOutcomes.of(execute, DiceOutcomes::optionLine) : List.of();

        // If ABILITY placeholder remains and dice outcomes were found, replace with dice-roll description.
        if (rawDesc != null && rawDesc.contains("ABILITY") && !children.isEmpty()) {
            String diceDesc = DiceOutcomes.rollDescription(execute);
            if (diceDesc != null) {
                normalized = AbilityDescription.normalize(
                        rawDesc.replace("ABILITY", AbilityDescription.replaceVert(diceDesc)));
            }
        }

        // Sub-pattern 4a: walk execute chain for trailing sub-SpellDescriptions.
        // Only when the execute SA itself is transparent (has no description of its own):
        // in that case, sub-ability SpellDescriptions contain oracle text not in TriggerDescription.
        // Guard: also skip ImmediateTrigger (processed separately) and ETBReplacement (registered
        // as replacement effects) to avoid duplicates (sigarda's_splendor concern).
        if (execute != null
                && isTransparentSA(execute)
                && execute.getApi() != ApiType.ImmediateTrigger
                && !"ETBReplacement".equals(execute.getParam("Mode"))) {
            List<Ability> executeChain = SpellEffect.fromChain(execute);
            if (!executeChain.isEmpty()) {
                String extra = collectChainText(executeChain.get(0));
                if (!extra.isEmpty()) {
                    normalized = normalized + " " + extra;
                }
            }
        }
        return new TriggeredAbilityEntry(effectiveType, normalized, children);
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

    /** Recursively collect and join descriptions from an ability node and all its descendants. */
    private static String collectChainText(Ability node) {
        StringBuilder sb = new StringBuilder(node.descriptionText());
        for (Ability child : node.subAbilities()) {
            String ct = collectChainText(child);
            if (!ct.isEmpty()) sb.append(' ').append(ct);
        }
        return sb.toString();
    }

    /**
     * Walk the SubAbility chain of {@code sa} to find an ImmediateTrigger whose Execute is a Charm.
     * Returns that ImmediateTrigger SA, or null if not found.
     */
    private static SpellAbility findImmediateTriggerWithCharm(SpellAbility sa) {
        for (SpellAbility cur = sa; cur != null; cur = cur.getSubAbility()) {
            if (cur.getApi() == ApiType.ImmediateTrigger) {
                SpellAbility exec = cur.getAdditionalAbility("Execute");
                if (exec != null && exec.getApi() == ApiType.Charm) return cur;
            }
        }
        return null;
    }
}
