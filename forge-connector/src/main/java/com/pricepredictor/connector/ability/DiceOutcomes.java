package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/**
 * The outcome lines of a die roll ({@code ResultSubAbilities$}), one per result
 * range.
 *
 * <p>The roll can sit anywhere below the ability that prints it: on the root,
 * further down its {@code SubAbility$} chain (Journey to the Lost City), or
 * under a {@code RepeatSubAbility$} (Delina, Wild Mage; Strategy, Schmategy).
 * Spells, activated and triggered abilities all find it the same way, so one
 * walk serves all three; what differs is the line an outcome becomes, which the
 * caller supplies.
 */
public final class DiceOutcomes {

    private DiceOutcomes() {}

    /**
     * One ability per outcome of the first roll found below {@code sa}.
     *
     * @param factory builds the line from the outcome's normalized, cased text
     *                ({@code "1—9 | tap all…"})
     */
    public static List<Ability> of(SpellAbility sa, Function<String, Ability> factory) {
        SpellAbility roll = findRoll(sa);
        return roll == null ? List.of() : outcomes(roll, factory);
    }

    /**
     * A dice outcome as an {@code option[N]: range | effect} line: what an
     * activated or triggered ability's outcomes become, where a spell's are
     * {@code spell} lines.
     */
    public static Ability optionLine(String text) {
        return new TextAbility(AbilityType.OPTION, text);
    }

    /** The roll's own description ({@code "Roll a d20."}), or null if there is no roll. */
    static String rollDescription(SpellAbility sa) {
        SpellAbility roll = findRoll(sa);
        return roll == null ? null : roll.getParam("SpellDescription");
    }

    /** The first ability with result ranges, walking the sub-ability and repeat chains. */
    private static SpellAbility findRoll(SpellAbility sa) {
        for (SpellAbility cur = sa; cur != null; cur = cur.getSubAbility()) {
            String results = cur.getParam("ResultSubAbilities");
            if (results != null && !results.isEmpty()) return cur;
            SpellAbility repeatSub = cur.getAdditionalAbility("RepeatSubAbility");
            if (repeatSub != null) {
                SpellAbility nested = findRoll(repeatSub);
                if (nested != null) return nested;
            }
        }
        return null;
    }

    /**
     * Each {@code "range:SvarName"} entry's SVar description, VERT placeholders
     * replaced so "1—9 VERT Tap all…" becomes "1—9 | Tap all…".
     */
    private static List<Ability> outcomes(SpellAbility roll, Function<String, Ability> factory) {
        List<Ability> result = new ArrayList<>();
        for (String entry : roll.getParam("ResultSubAbilities").split(",")) {
            String[] kv = entry.trim().split(":", 2);
            if (kv.length < 2) continue;
            SpellAbility sub = roll.getAdditionalAbility(kv[0].trim());
            if (sub == null) continue;
            String rawDesc = sub.getParam("SpellDescription");
            if (rawDesc == null || rawDesc.isEmpty()) continue;
            String normalized = AbilityDescription.normalize(AbilityDescription.replaceVert(rawDesc));
            if (normalized != null) {
                result.add(factory.apply(AbilityDescription.applyCasing(normalized)));
            }
        }
        return result;
    }
}
