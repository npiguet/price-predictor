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
                effectDesc = parseSpellDesc(card.getSVar(svarName));
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

            // Replace ABILITY placeholder with the actual effect description
            if (tDesc.contains("ABILITY")) {
                SpellAbility overrideSa = t.getOverridingAbility();
                if (overrideSa != null && overrideSa.getApi() == ApiType.Charm) {
                    // Haunt SVar is a Charm — synthesize "choose one —" header
                    String charmHeader = CharmAbility.synthesizeCharmHeader(overrideSa);
                    if (charmHeader == null) charmHeader = "choose one";
                    tDesc = tDesc.replace("ABILITY", charmHeader);
                    String normalized2 = AbilityDescription.normalize(tDesc);
                    if (normalized2 == null || normalized2.isEmpty()) continue;
                    if (!emitted.add(normalized2)) continue;
                    AbilityType type2 = t.isStatic() ? AbilityType.REPLACEMENT : AbilityType.TRIGGERED;
                    List<Ability> options = CharmAbility.optionsFrom(overrideSa);
                    result.add(new TextAbility(type2, AbilityDescription.applyCasing(normalized2), 0, options));
                    continue;
                } else if (effectDesc != null) {
                    tDesc = tDesc.replace("ABILITY", effectDesc);
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

    /** Extract SpellDescription value from a raw SVar string like "DB$ Foo | SpellDescription$ Bar | …". */
    static String parseSpellDesc(String svarContent) {
        if (svarContent == null || svarContent.isEmpty()) return null;
        for (String part : svarContent.split(" \\| ")) {
            String trimmed = part.trim();
            if (trimmed.startsWith("SpellDescription$")) {
                return trimmed.substring("SpellDescription$".length()).trim();
            }
        }
        return null;
    }
}
