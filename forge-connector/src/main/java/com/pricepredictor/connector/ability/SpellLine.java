package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityType;
import forge.game.spellability.SpellAbility;

import java.util.List;
import java.util.Objects;
import java.util.StringJoiner;

/**
 * The one line a spell prints: its additional cost's sentence, then the
 * sentences of its sub-ability chain in chain order, as an activated or a
 * triggered ability prints its chain on one line. A die roll's outcomes are
 * {@code option} lines below it.
 *
 * <p>One line per spell is what lets a record acting through the spell read
 * its whole text: a join reads the one line a key names, so a spell spread
 * over a cost line and several effect lines was read by whichever came first.
 *
 * @param costLength the length of the cost sentence the description opens
 *                   with, or 0 when the spell has no additional cost
 */
public record SpellLine(String descriptionText, int costLength, List<Ability> subAbilities)
        implements Ability {

    public SpellLine {
        Objects.requireNonNull(subAbilities);
        subAbilities = List.copyOf(subAbilities);
    }

    @Override
    public AbilityType type() {
        return AbilityType.SPELL;
    }

    /**
     * The lines {@code sa} prints: its spell line, or, for a spell whose
     * chain states nothing (a permanent's), its additional cost alone, on a
     * cost line of its own.
     */
    public static List<Ability> linesOf(SpellAbility sa) {
        String effect = chainSentences(sa);
        if (effect.isEmpty()) return costLineOf(sa);
        String cost = SpellAdditionalCost.sentenceOf(sa);
        List<Ability> outcomes = DiceOutcomes.of(sa, DiceOutcomes::optionLine);
        return List.of(cost == null
                ? new SpellLine(effect, 0, outcomes)
                : new SpellLine(cost + " " + effect, cost.length(), outcomes));
    }

    private static List<Ability> costLineOf(SpellAbility sa) {
        SpellAdditionalCost cost = SpellAdditionalCost.of(sa);
        return cost == null ? List.of() : List.of(cost);
    }

    /** Every sentence of {@code sa}'s chain, in chain order, joined by spaces. */
    private static String chainSentences(SpellAbility sa) {
        StringJoiner joined = new StringJoiner(" ");
        for (Ability node : SpellEffect.fromChain(sa)) {
            joined.add(SpellAbilityUtils.treeText(node));
        }
        return joined.toString();
    }
}
