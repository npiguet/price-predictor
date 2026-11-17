package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import com.pricepredictor.connector.KeywordRendering;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.keyword.KeywordInterface;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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

    /**
     * The lines, each with the generated traits that printed it: the spell
     * Forge builds for a non-creature card prints the spell effect; a trigger
     * prints the line of its description, a duplicate the line its first copy
     * printed; and the two description-less triggers that end a haunting
     * belong to the haunt line, beside the trigger that haunts.
     */
    public static KeywordRendering of(KeywordInterface ki, Card card) {
        KeywordRendering.Builder rendering = new KeywordRendering.Builder();
        String effectDesc = effectDescription(ki, card);

        // 1. For non-creature spells: emit the primary spell effect first
        if (!card.isCreature() && effectDesc != null) {
            addSpellEffect(ki, effectDesc, rendering);
        }

        // 2. Process triggers — deduplicate by description
        TriggerLines triggers = new TriggerLines(effectDesc, rendering);
        for (Trigger t : ki.getTriggers()) {
            triggers.add(t);
        }
        triggers.closeBlank();
        return rendering.build();
    }

    /**
     * The spell effect's description: from the SA Forge built for a
     * non-creature haunt or, for a creature, whose {@code ki.getAbilities()}
     * is empty, from the SVar the keyword names. Null when neither has one.
     */
    private static String effectDescription(KeywordInterface ki, Card card) {
        for (SpellAbility sa : ki.getAbilities()) {
            String d = sa.getParam("SpellDescription");
            if (d != null && !d.isEmpty()) return d;
        }
        String svarName = new KeywordSpec(ki.getOriginal()).rest(1); // "Haunt:SvarName"
        if (svarName == null) return null;
        return SpellAbilityUtils.svarParam(card.getSVar(svarName), "SpellDescription");
    }

    private static void addSpellEffect(KeywordInterface ki, String effectDesc,
                                       KeywordRendering.Builder rendering) {
        String normalized = AbilityDescription.normalize(effectDesc);
        if (normalized == null) return;
        SpellEffect spell = new SpellEffect(normalized, List.of());
        rendering.add(spell);
        for (SpellAbility sa : ki.getAbilities()) {
            rendering.printed(sa, spell);
        }
    }

    /** The trigger lines, one per distinct description, as the triggers arrive. */
    private static final class TriggerLines {
        private final String effectDesc;
        private final KeywordRendering.Builder rendering;
        private final Map<String, Ability> emitted = new HashMap<>();
        private final List<Trigger> blank = new ArrayList<>();
        private Ability hauntLine;

        TriggerLines(String effectDesc, KeywordRendering.Builder rendering) {
            this.effectDesc = effectDesc;
            this.rendering = rendering;
        }

        void add(Trigger t) {
            String tDesc = t.getParam("TriggerDescription");
            if (tDesc == null || "Blank".equals(tDesc)) {
                blank.add(t);
                return;
            }
            SpellAbility charm = placeholderCharm(t, tDesc);
            if (charm != null) {
                addCharm(t, tDesc, charm);
                return;
            }
            // Replace ABILITY placeholder with the actual effect description
            if (effectDesc != null) tDesc = tDesc.replace("ABILITY", effectDesc);
            String normalized = AbilityDescription.normalize(tDesc);
            if (normalized == null) return;
            Ability line = emitted.computeIfAbsent(normalized,
                    text -> append(new TextAbility(AbilityType.forTrigger(t), text)));
            rendering.printed(t, line);
            if (tDesc.startsWith("Haunt")) hauntLine = line;
        }

        /** The charm {@code t}'s ABILITY placeholder names, or null. */
        private static SpellAbility placeholderCharm(Trigger t, String tDesc) {
            if (!tDesc.contains("ABILITY")) return null;
            SpellAbility overrideSa = t.getOverridingAbility();
            return overrideSa != null && overrideSa.getApi() == ApiType.Charm ? overrideSa : null;
        }

        /**
         * The haunt SVar is a Charm: a synthesized "choose one —" header. No
         * owner: the triggers are deduplicated by description, so the
         * surviving line's options must belong to every trigger that shares
         * it (Orzhov Pontiff).
         */
        private void addCharm(Trigger t, String tDesc, SpellAbility overrideSa) {
            CharmAbility.Expanded expanded =
                    CharmAbility.expandPlaceholder(tDesc, overrideSa, null);
            String charmDesc = expanded.description();
            if (charmDesc == null) return;
            Ability charm = emitted.computeIfAbsent(charmDesc, text -> append(
                    new TextAbility(AbilityType.forTrigger(t), text, 0, expanded.options())));
            rendering.printed(t, charm);
        }

        private Ability append(Ability line) {
            rendering.add(line);
            return line;
        }

        /** The description-less triggers belong to the haunt line. */
        void closeBlank() {
            for (Trigger t : blank) {
                rendering.printed(t, hauntLine);
            }
        }
    }
}
