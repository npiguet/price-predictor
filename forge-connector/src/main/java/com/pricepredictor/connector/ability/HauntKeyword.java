package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.ability.ApiType;
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
                effectDesc = SpellAbilityUtils.svarParam(card.getSVar(svarName), "SpellDescription");
            }
        }

        boolean isCreature = card.isCreature();

        // 1. For non-creature spells: emit the primary spell effect first
        if (!isCreature && effectDesc != null) {
            String normalized = AbilityDescription.normalize(effectDesc);
            if (normalized != null) {
                result.add(new SpellEffect(normalized, List.of()));
            }
        }

        // 2. Process triggers — deduplicate by description
        Set<String> emitted = new HashSet<>();
        for (Trigger t : ki.getTriggers()) {
            String tDesc = t.getParam("TriggerDescription");
            if (tDesc == null || "Blank".equals(tDesc)) continue;

            // Replace ABILITY placeholder with the actual effect description
            if (tDesc.contains("ABILITY")) {
                SpellAbility overrideSa = t.getOverridingAbility();
                if (overrideSa != null && overrideSa.getApi() == ApiType.Charm) {
                    // Haunt SVar is a Charm — synthesize "choose one —" header.
                    // No owner: the triggers are deduplicated by description, so
                    // the surviving line's options must belong to every trigger
                    // that shares it (Orzhov Pontiff).
                    CharmAbility.Expanded expanded =
                            CharmAbility.expandPlaceholder(tDesc, overrideSa, null);
                    String normalized2 = expanded.description();
                    if (normalized2 == null) continue;
                    if (!emitted.add(normalized2)) continue;
                    result.add(new TextAbility(AbilityType.forTrigger(t),
                            normalized2, 0, expanded.options()));
                    continue;
                } else if (effectDesc != null) {
                    tDesc = tDesc.replace("ABILITY", effectDesc);
                }
            }

            String normalized = AbilityDescription.normalize(tDesc);
            if (normalized == null) continue;
            if (!emitted.add(normalized)) continue; // skip duplicates

            result.add(new TextAbility(AbilityType.forTrigger(t), normalized));
        }

        return result;
    }
}
