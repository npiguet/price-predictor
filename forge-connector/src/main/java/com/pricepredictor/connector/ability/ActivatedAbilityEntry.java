package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityDescription;
import com.pricepredictor.connector.AbilityType;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Activated or planeswalker ability built from a SpellAbility.
 * The root description is taken from sa.getDescription() (cost + root SpellDescription only).
 * Sub-ability descriptions become child SpellEffect nodes via SpellEffect.fromChain().
 */
public record ActivatedAbilityEntry(AbilityType type, String descriptionText, List<Ability> subAbilities) implements Ability {

    public ActivatedAbilityEntry {
        Objects.requireNonNull(subAbilities);
        subAbilities = List.copyOf(subAbilities);
    }

    public static ActivatedAbilityEntry of(SpellAbility sa) {
        AbilityType type = sa.isPwAbility() ? AbilityType.PLANESWALKER : AbilityType.ACTIVATED;

        if (sa.getParam("SpellDescription") == null
                || sa.getParam("SpellDescription").isEmpty()) {
            // Some activated abilities have no SpellDescription on the root SA but carry their
            // description on a SubAbility$ SVar (e.g. Arachnus Spinner).  Walk the sub-chain;
            // if it produces nodes, concatenate their text into the cost-based root description.
            List<Ability> subChain = SpellEffect.fromChain(sa.getSubAbility());
            if (subChain.isEmpty()) return null;
            String rootDesc = sa.getDescription();
            if (rootDesc.isEmpty()) rootDesc = sa.getCostDescription();
            int nl = rootDesc.indexOf('\n');
            if (nl >= 0) rootDesc = rootDesc.substring(0, nl);
            String normalized = AbilityDescription.normalize(rootDesc);
            String subText = collectChainText(subChain.get(0));
            String fullDesc = (normalized != null && !normalized.isEmpty())
                    ? normalized + " " + subText : subText;
            if (fullDesc.isEmpty()) return null;
            return new ActivatedAbilityEntry(type, type.formatDescription(fullDesc),
                    diceOutcomesAsOptions(sa));
        }

        String rootDesc = sa.getDescription();
        // For dice-roll activated abilities, getDescription() appends outcome lines after a
        // newline. Strip them here; they will be re-emitted as OPTION sub-abilities below.
        int nl = rootDesc.indexOf('\n');
        if (nl >= 0) rootDesc = rootDesc.substring(0, nl);

        String normalized = AbilityDescription.normalize(rootDesc);
        if (normalized == null) return null;

        // Concatenate sub-ability chain descriptions into the root line so that oracle
        // lines (one paragraph = one activated ability) are not over-split.
        List<Ability> subEffects = SpellEffect.fromChain(sa.getSubAbility());
        String subText = subEffects.isEmpty() ? "" : collectChainText(subEffects.get(0));
        String fullDesc = subText.isEmpty() ? normalized : normalized + " " + subText;

        List<Ability> children = diceOutcomesAsOptions(sa);
        return new ActivatedAbilityEntry(type, type.formatDescription(fullDesc), children);
    }

    /** Recursively collect and join descriptions from an ability node and all its descendants. */
    private static String collectChainText(Ability node) {
        StringBuilder sb = new StringBuilder(node.descriptionText());
        for (Ability child : node.subAbilities()) {
            String ct = collectChainText(child);
            if (!ct.isEmpty()) sb.append(' ').append(ct);
        }
        return sb.toString();
    }

    /**
     * Expand {@code ResultSubAbilities} dice-outcome entries into OPTION sub-abilities.
     * Mirrors {@code RulesParser.diceOutcomesFromSA()} but uses OPTION type instead of SPELL,
     * so activated dice-roll results are formatted as {@code option[N]: range | description}.
     */
    private static List<Ability> diceOutcomesAsOptions(SpellAbility sa) {
        // Walk the sub-ability chain to find a SA with ResultSubAbilities.
        for (SpellAbility cur = sa; cur != null; cur = cur.getSubAbility()) {
            String resultSubAbilities = cur.getParam("ResultSubAbilities");
            if (resultSubAbilities == null || resultSubAbilities.isEmpty()) continue;
            List<Ability> result = new ArrayList<>();
            for (String entry : resultSubAbilities.split(",")) {
                String[] kv = entry.trim().split(":", 2);
                if (kv.length < 2) continue;
                String range = kv[0].trim();
                SpellAbility sub = cur.getAdditionalAbility(range);
                if (sub == null) continue;
                String rawDesc = sub.getParam("SpellDescription");
                if (rawDesc == null || rawDesc.isEmpty()) continue;
                String desc = AbilityDescription.replaceVert(rawDesc);
                String normalized = AbilityDescription.normalize(desc);
                if (normalized != null) {
                    result.add(new TextAbility(AbilityType.OPTION, AbilityDescription.applyCasing(normalized)));
                }
            }
            return result;
        }
        return List.of();
    }
}
