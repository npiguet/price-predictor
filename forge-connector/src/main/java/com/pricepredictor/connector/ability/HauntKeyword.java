package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.card.Card;
import forge.game.keyword.KeywordInterface;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Haunt keyword ability — expands the haunt keyword into its constituent lines:
 * <ol>
 *   <li>For non-creature spells: the primary spell effect (from the haunt SVar).</li>
 *   <li>The "haunt" keyword line (reminder text stripped).</li>
 *   <li>The "When … haunts dies, [effect]" trigger (ABILITY placeholder replaced).</li>
 * </ol>
 */
public final class HauntKeyword {

    private HauntKeyword() {}

    public static List<Ability> of(KeywordInterface ki, Card card) {
        List<Ability> result = new ArrayList<>();

        // Get spell effect description from the SA Forge built for non-creature haunt.
        // For creature haunt ki.getAbilities() is empty, so we fall back to the SVar.
        String effectDesc = null;
        for (SpellAbility sa : ki.getAbilities()) {
            String d = sa.getParam("SpellDescription");
            if (d != null && !d.isEmpty()) { effectDesc = d; break; }
        }
        // Fallback: extract SpellDescription from the SVar content string
        if (effectDesc == null) {
            String original = ki.getOriginal(); // "Haunt:SvarName"
            if (original.contains(":")) {
                String svarName = original.substring(original.indexOf(':') + 1);
                effectDesc = SpellAbilityUtils.extractParam(card.getSVar(svarName), "SpellDescription");
            }
        }

        boolean isCreature = card.isCreature();

        // 1. For non-creature spells: emit the primary spell effect first
        if (!isCreature && effectDesc != null) {
            String normalized = AbilityDescription.normalize(effectDesc);
            if (normalized != null) {
                result.add(new SpellEffect(AbilityDescription.applyCasing(normalized), List.of()));
            }
        }

        // 2. Process triggers — deduplicate by description
        Set<String> emitted = new HashSet<>();
        for (Trigger t : ki.getTriggers()) {
            String tDesc = t.getParam("TriggerDescription");
            if (tDesc == null || "Blank".equals(tDesc)) continue;

            // Replace ABILITY placeholder: charm, dice-roll, or direct haunt-effect substitution.
            if (tDesc.contains("ABILITY")) {
                SpellAbility overrideSa = t.getOverridingAbility();
                SpellAbilityUtils.AbilityPlaceholderResult resolved =
                        SpellAbilityUtils.resolveAbilityPlaceholder(tDesc, overrideSa, effectDesc);
                if (resolved != null) {
                    String normalized2 = resolved.expandedDescription();
                    if (normalized2 == null || normalized2.isEmpty()) continue;
                    if (!emitted.add(normalized2)) continue;
                    AbilityType type2 = t.isStatic() ? AbilityType.REPLACEMENT : AbilityType.TRIGGERED;
                    result.add(new TextAbility(type2, AbilityDescription.applyCasing(normalized2), 0, resolved.options()));
                    continue;
                }
            }

            String normalized = AbilityDescription.normalize(tDesc);
            if (normalized == null || normalized.isEmpty()) continue;
            if (!emitted.add(normalized)) continue; // skip duplicates

            AbilityType type = t.isStatic() ? AbilityType.REPLACEMENT : AbilityType.TRIGGERED;
            result.add(new TextAbility(type, AbilityDescription.applyCasing(normalized)));
        }

        return result;
    }

}
