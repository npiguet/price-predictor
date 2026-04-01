package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import com.pricepredictor.connector.NonBlankString;
import forge.game.card.Card;
import forge.game.keyword.KeywordInterface;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
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
        // Fallback: extract SpellDescription from the SVar content string.
        KeywordFields hauntFields = KeywordFields.from(ki, 2); // "Haunt:SvarName"
        String effectDesc = ki.getAbilities().stream()
                .map(sa -> sa.getParam("SpellDescription"))
                .filter(d -> d != null && !d.isEmpty())
                .findFirst()
                .or(() -> hauntFields.hasField(1)
                        ? SpellAbilityUtils.extractParam(card.getSVar(hauntFields.field(1)), "SpellDescription")
                                .map(NonBlankString::value)
                        : Optional.empty())
                .orElse(null);

        boolean isCreature = card.isCreature();

        // 1. For non-creature spells: emit the primary spell effect first
        if (!isCreature && effectDesc != null) {
            AbilityDescription.normalize(effectDesc)
                    .ifPresent(n -> result.add(new SpellEffect(n, List.of())));
        }

        // 2. Process triggers — deduplicate by description
        Set<NonBlankString> emitted = new HashSet<>();
        for (Trigger t : ki.getTriggers()) {
            String tDesc = t.getParam("TriggerDescription");
            if (tDesc == null || ForgeParams.BLANK_DESC.equals(tDesc)) continue;

            // Replace ABILITY placeholder: charm, dice-roll, or direct haunt-effect substitution.
            if (tDesc.contains("ABILITY")) {
                SpellAbility overrideSa = t.getOverridingAbility();
                Optional<SpellAbilityUtils.AbilityPlaceholderResult> resolved =
                        SpellAbilityUtils.resolveAbilityPlaceholder(tDesc, overrideSa, effectDesc);
                if (resolved.isPresent()) {
                    SpellAbilityUtils.AbilityPlaceholderResult r = resolved.get();
                    NonBlankString normalized2 = r.expandedDescription();
                    if (!emitted.add(normalized2)) continue;
                    AbilityType type2 = t.isStatic() ? AbilityType.REPLACEMENT : AbilityType.TRIGGERED;
                    result.add(new TextAbility(type2, normalized2, 0, r.options()));
                    continue;
                }
            }

            NonBlankString normalized = AbilityDescription.normalize(tDesc).orElse(null);
            if (normalized == null) continue;
            if (!emitted.add(normalized)) continue; // skip duplicates

            AbilityType type = t.isStatic() ? AbilityType.REPLACEMENT : AbilityType.TRIGGERED;
            result.add(new TextAbility(type, normalized));
        }

        return result;
    }

}
