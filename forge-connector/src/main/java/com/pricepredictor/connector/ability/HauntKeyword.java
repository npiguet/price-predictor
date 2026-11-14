package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.CardTraitBase;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.keyword.KeywordInterface;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;

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
        return of(ki, card, (trait, line) -> { });
    }

    /**
     * The lines, telling {@code rendered} which line each generated trait
     * printed: the spell Forge builds for a non-creature card prints the spell
     * effect; a trigger prints the line of its description, a duplicate the
     * line its first copy printed; and the two description-less triggers that
     * end a haunting belong to the haunt line, beside the trigger that haunts.
     */
    public static List<Ability> of(KeywordInterface ki, Card card,
                                   BiConsumer<CardTraitBase, Ability> rendered) {
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
            String svarName = new KeywordSpec(ki.getOriginal()).rest(1); // "Haunt:SvarName"
            if (svarName != null) {
                effectDesc = SpellAbilityUtils.svarParam(card.getSVar(svarName), "SpellDescription");
            }
        }

        boolean isCreature = card.isCreature();

        // 1. For non-creature spells: emit the primary spell effect first
        if (!isCreature && effectDesc != null) {
            String normalized = AbilityDescription.normalize(effectDesc);
            if (normalized != null) {
                SpellEffect spell = new SpellEffect(normalized, List.of());
                result.add(spell);
                for (SpellAbility sa : ki.getAbilities()) {
                    rendered.accept(sa, spell);
                }
            }
        }

        // 2. Process triggers — deduplicate by description
        Map<String, Ability> emitted = new HashMap<>();
        List<Trigger> blank = new ArrayList<>();
        Ability hauntLine = null;
        for (Trigger t : ki.getTriggers()) {
            String tDesc = t.getParam("TriggerDescription");
            if (tDesc == null || "Blank".equals(tDesc)) {
                blank.add(t);
                continue;
            }

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
                    String charmDesc = expanded.description();
                    if (charmDesc == null) continue;
                    Ability charm = emitted.get(charmDesc);
                    if (charm == null) {
                        charm = new TextAbility(AbilityType.forTrigger(t),
                                charmDesc, 0, expanded.options());
                        emitted.put(charmDesc, charm);
                        result.add(charm);
                    }
                    rendered.accept(t, charm);
                    continue;
                } else if (effectDesc != null) {
                    tDesc = tDesc.replace("ABILITY", effectDesc);
                }
            }

            String normalized = AbilityDescription.normalize(tDesc);
            if (normalized == null) continue;
            Ability line = emitted.get(normalized);
            if (line == null) {
                line = new TextAbility(AbilityType.forTrigger(t), normalized);
                emitted.put(normalized, line);
                result.add(line);
            }
            rendered.accept(t, line);
            if (tDesc.startsWith("Haunt")) hauntLine = line;
        }
        for (Trigger t : blank) {
            rendered.accept(t, hauntLine);
        }

        return result;
    }
}
