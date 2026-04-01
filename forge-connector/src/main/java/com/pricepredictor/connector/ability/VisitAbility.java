package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import com.pricepredictor.connector.NonBlankString;
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
            NonBlankString header;
            List<Ability> children;

            if (overriding != null && overriding.getApi() == ApiType.Charm) {
                // Charm-based visit: header from TriggerDescription, children from Charm choices.
                String tDesc = t.getParam("TriggerDescription");
                if (tDesc == null || ForgeParams.BLANK_DESC.equals(tDesc)) continue;
                int nl = tDesc.indexOf('\n');
                if (nl >= 0) tDesc = tDesc.substring(0, nl);
                header = AbilityDescription.normalize(tDesc).orElse(null);
                children = CharmAbility.optionsFrom(overriding);
            } else {
                // Plain visit: prefer SpellDescription over TriggerDescription.
                // Forge places a doubled "Visit — Visit — " prefix in TriggerDescription for plain
                // visits (the keyword name is prepended by both the keyword handler and the trigger
                // script). SpellDescription is written without this duplication, so it is the
                // correct source for oracle text.
                String spellDesc = overriding != null ? overriding.getParam("SpellDescription") : null;
                if (spellDesc == null || spellDesc.isEmpty()) {
                    // Fallback: use TriggerDescription (may have doubled prefix, but better than nothing).
                    String tDesc = t.getParam("TriggerDescription");
                    if (tDesc == null || ForgeParams.BLANK_DESC.equals(tDesc)) continue;
                    int nl = tDesc.indexOf('\n');
                    if (nl >= 0) tDesc = tDesc.substring(0, nl);
                    header = AbilityDescription.normalize(tDesc).orElse(null);
                } else {
                    // Ensure exactly one "Visit — " prefix.
                    if (!spellDesc.startsWith("Visit — ")) {
                        spellDesc = "Visit — " + spellDesc;
                    }
                    header = AbilityDescription.normalize(spellDesc).orElse(null);
                }
                children = List.of();
            }

            if (header == null) continue;
            abilities.add(new TriggeredAbilityEntry(AbilityType.TRIGGERED, header, children));
        }
        return abilities;
    }
}
