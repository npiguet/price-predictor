package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityType;
import forge.game.spellability.SpellAbility;

import java.util.List;
import java.util.Objects;

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
        return spellLines(sa, SpellEffect.fromChain(sa));
    }

    /**
     * The lines {@code sa} prints with {@code effect} as its effect's text,
     * the text {@link UndescribedSpellText} finds for a spell whose chain
     * states nothing. A null or empty effect prints the additional cost alone.
     */
    public static List<Ability> linesOf(SpellAbility sa, String effect) {
        return spellLines(sa, effect == null || effect.isEmpty()
                ? List.of() : List.of(new SpellEffect(effect, List.of())));
    }

    /**
     * The spell line of the cost sentence then the sentences of
     * {@code effect}'s nodes, each sentence once ({@link RepeatedSentences}):
     * a chain restating the additional cost does not print it twice.
     */
    private static List<Ability> spellLines(SpellAbility sa, List<Ability> effect) {
        if (effect.isEmpty()) return costLineOf(sa);
        String cost = SpellAdditionalCost.sentenceOf(sa);
        String text = RepeatedSentences.appendMissing(cost == null ? "" : cost, effect);
        List<Ability> outcomes = DiceOutcomes.of(sa, DiceOutcomes::optionLine);
        return List.of(new SpellLine(text, cost == null ? 0 : cost.length(), outcomes));
    }

    /** Whether no ability of {@code sa}'s chain states a sentence. */
    public static boolean statesNothing(SpellAbility sa) {
        return SpellEffect.fromChain(sa).isEmpty();
    }

    private static List<Ability> costLineOf(SpellAbility sa) {
        return SpellAdditionalCost.of(sa).<List<Ability>>map(List::of).orElseGet(List::of);
    }
}
