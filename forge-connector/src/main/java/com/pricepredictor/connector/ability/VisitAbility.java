package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.ability.ApiType;
import forge.game.keyword.KeywordInterface;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;

import java.util.ArrayList;
import java.util.List;

/**
 * Attraction Visit keyword — emits one TRIGGERED ability per trigger.
 *
 * <p>For Charm-based visits the TriggerDescription is used as the header and
 * the Charm options become OPTION sub-abilities. For plain visits the header
 * is derived from the execute SA's SpellDescription to avoid the doubled
 * "Visit — Visit — " prefix that Forge places in TriggerDescription.
 */
public final class VisitAbility {

    private VisitAbility() {}

    public static List<Ability> fromKeyword(KeywordInterface ki) {
        List<Ability> abilities = new ArrayList<>();
        for (Trigger t : ki.getTriggers()) {
            SpellAbility overriding = t.getOverridingAbility();
            String header;
            List<Ability> children;

            if (overriding != null && overriding.getApi() == ApiType.Charm) {
                // Charm-based visit: header from TriggerDescription, children from Charm choices.
                String tDesc = t.getParam("TriggerDescription");
                if (tDesc == null || "Blank".equals(tDesc)) continue;
                int nl = tDesc.indexOf('\n');
                if (nl >= 0) tDesc = tDesc.substring(0, nl);
                header = AbilityDescription.normalize(tDesc);
                children = CharmAbility.optionsFrom(overriding);
            } else {
                // Plain visit: prefer SpellDescription over TriggerDescription to avoid doubled prefix.
                String spellDesc = overriding != null ? overriding.getParam("SpellDescription") : null;
                if (spellDesc == null || spellDesc.isEmpty()) {
                    // Fallback: use TriggerDescription (may have doubled prefix, but better than nothing).
                    String tDesc = t.getParam("TriggerDescription");
                    if (tDesc == null || "Blank".equals(tDesc)) continue;
                    int nl = tDesc.indexOf('\n');
                    if (nl >= 0) tDesc = tDesc.substring(0, nl);
                    header = AbilityDescription.normalize(tDesc);
                } else {
                    // Ensure exactly one "Visit — " prefix.
                    if (!spellDesc.startsWith("Visit — ")) {
                        spellDesc = "Visit — " + spellDesc;
                    }
                    header = AbilityDescription.normalize(spellDesc);
                }
                children = List.of();
            }

            if (header == null || header.isEmpty()) continue;
            abilities.add(new TriggeredAbilityEntry(AbilityType.TRIGGERED,
                    AbilityDescription.applyCasing(header), children));
        }
        return abilities;
    }
}
