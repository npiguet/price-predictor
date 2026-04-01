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
        Optional<NonBlankString> effectDesc = ki.getAbilities().stream()
                .flatMap(sa -> SpellAbilityUtils.getParam(sa, "SpellDescription").stream())
                .findFirst()
                .or(() -> hauntFields.hasField(1)
                        ? SpellAbilityUtils.extractParam(card.getSVar(hauntFields.field(1)), "SpellDescription")
                        : Optional.empty());

        boolean isCreature = card.isCreature();

        // 1. For non-creature spells: emit the primary spell effect first
        if (!isCreature) {
            effectDesc.flatMap(AbilityDescription::normalize)
                    .ifPresent(n -> result.add(new SpellEffect(n, List.of())));
        }

        // 2. Process triggers — deduplicate by description
        Set<NonBlankString> emitted = new HashSet<>();
        for (Trigger t : ki.getTriggers()) {
            String tDesc = t.getParam("TriggerDescription");
            if (tDesc == null || ForgeParams.BLANK_DESC.equals(tDesc)) {
                continue;
            }

            // Replace ABILITY placeholder: charm, dice-roll, or direct haunt-effect substitution.
            if (tDesc.contains("ABILITY")) {
                SpellAbilityUtils.resolveAbilityPlaceholder(tDesc, t.getOverridingAbility(), effectDesc.orElse(null))
                        .ifPresent(r -> addTriggerAbility(result, emitted, t, r.expandedDescription(), r.options()));
                continue;
            }

            AbilityDescription.normalize(tDesc)
                    .ifPresent(n -> addTriggerAbility(result, emitted, t, n, List.of()));
        }

        return result;
    }

    private static void addTriggerAbility(List<Ability> result, Set<NonBlankString> emitted,
                                           Trigger t, NonBlankString desc, List<Ability> options) {
        if (!emitted.add(desc)) return;
        AbilityType type = t.isStatic() ? AbilityType.REPLACEMENT : AbilityType.TRIGGERED;
        result.add(new TextAbility(type, desc, 0, options));
    }

}
