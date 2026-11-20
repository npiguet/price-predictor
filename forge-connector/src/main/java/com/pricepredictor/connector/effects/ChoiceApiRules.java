package com.pricepredictor.connector.effects;

import com.google.common.base.Strings;
import com.pricepredictor.connector.effects.ApiEvents.Rule;
import forge.game.card.Card;

import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static com.pricepredictor.connector.effects.ApiEvents.put;

/**
 * The {@link ApiEvents} rules for choices: every Choose* API.
 */
final class ChoiceApiRules {

    private ChoiceApiRules() {
    }

    /**
     * Every Choose* API, read back off what it left on its host.
     *
     * <p>Two keys are NOT spelled like their effect classes ({@code
     * ApiType}): {@code NameCard} declares {@code ChooseCardNameEffect.class},
     * {@code GenericChoice} declares {@code ChooseGenericEffect.class}. A rule
     * keyed by the CLASS stem matches nothing and fails silently.
     *
     * <p>{@code GenericChoice}, {@code ChooseDirection}, {@code ChooseEvenOdd}
     * and {@code NameCard} are wired like the rest, so their {@link #choice}
     * read branches are required, not optional: a rule that matches here and
     * reads nothing back would be worse than one left out of this table
     * entirely -- it looks wired.
     *
     * <p>{@code ChooseSector} is deliberately NOT wired here, though {@code
     * ApiType} has it and {@code Card.getChosenSector()} is a real accessor.
     * This corpus is collected from sealed and draft pools only, which never
     * include Unfinity's silver-bordered attraction cards, so the clause this
     * would read can never actually run here. Wiring it needs a {@link
     * #choice} branch and a fixture in {@code
     * everyChoiceMadeApiHasAReadableChoice}, which fails the moment the rule
     * and the fixtures disagree.
     */
    static void choiceRules(Map<String, Rule<?>> rules) {
        choiceRule(rules, "ChooseColor", "color");
        choiceRule(rules, "ChooseType", "type");
        choiceRule(rules, "ChooseNumber", "number");
        choiceRule(rules, "NameCard", "card_name");
        choiceRule(rules, "ChoosePlayer", "player");
        choiceRule(rules, "ChooseCard", "cards");
        choiceRule(rules, "ChooseSource", "cards");
        choiceRule(rules, "ChooseDirection", "direction");
        choiceRule(rules, "ChooseEvenOdd", "even_odd");
        choiceRule(rules, "GenericChoice", "mode");
    }

    /** The rule of a Choose* API: the choice it leaves in its own accessor, of {@code kind}. */
    private static void choiceRule(Map<String, Rule<?>> rules, String api, String kind) {
        put(rules, Rule.one(EventType.CHOICE_MADE, null, (sa, host, memo) -> choice(host, kind)),
                api);
    }

    /**
     * The choice a Choose* clause left on its host, as a kind and a value:
     * read from the one accessor the clause's API populates ({@link
     * #choiceRules}), never from another the host may still hold from an
     * earlier choice — a colour chosen as it entered outranks nothing a
     * later {@code ChooseNumber} left.
     *
     * <p><b>{@code getChosenColor()} and {@code getNamedCard()} never return
     * {@code null}</b> -- an unset colour reads back as {@code ""} ({@code
     * Card.getChosenColor}), the same way an unset named card reads back as
     * {@code ""} rather than {@code null}. {@code hasChosenColor()} is what
     * {@code Card} itself defines for this distinction, so this reads that
     * instead; the named-card branch is guarded by {@link Strings#emptyToNull}.
     *
     * <p>{@code ChooseDirectionEffect.resolve()} always calls {@code
     * setChosenDirection}, and {@code ChooseEvenOddEffect.resolve()} always
     * calls {@code setChosenEvenOdd}. {@code GenericChoice} leaves a reading
     * only for the {@code SetChosenMode$ True} scripts (Tarkir's Sieges,
     * Fallout's Hoover Dam, 15 real cards); the rest resolve a chosen
     * sub-ability directly and leave no scalar on the host to describe which
     * one.
     *
     * <p>There is no {@code getChosenSector()} reading, for the reason there
     * is no {@code ChooseSector} rule ({@link #choiceRules}).
     *
     * @return the event, or null where the API's accessor holds nothing: a
     *         {@code choice_made} naming a kind it cannot read would train
     *         the head on a distinction the corpus cannot make
     */
    private static EffectEvent choice(Card host, String kind) {
        if (host == null) {
            return null;
        }
        for (ChoiceReading reading : CHOICES) {
            if (!reading.kind().equals(kind)) {
                continue;
            }
            String value = reading.read().apply(host);
            return value == null ? null : new EffectEvent(EventType.CHOICE_MADE)
                    .param("choice_kind", kind)
                    .param("value", value);
        }
        return null;
    }

    /**
     * One accessor a choice can be left in, and its reading: the value, or
     * null where the accessor holds nothing.
     */
    private record ChoiceReading(String kind, Function<Card, String> read) {
    }

    /**
     * The accessors {@link #choice} reads, one per kind.
     *
     * <p>Read only when an emitter runs, never while {@link ApiEvents#RULES} is built,
     * so being declared below it is safe.
     */
    private static final List<ChoiceReading> CHOICES = List.of(
            new ChoiceReading("color",
                    host -> host.hasChosenColor() ? host.getChosenColor() : null),
            new ChoiceReading("type", host -> Strings.emptyToNull(host.getChosenType())),
            new ChoiceReading("number", host -> host.getChosenNumber() == null
                    ? null : String.valueOf(host.getChosenNumber())),
            new ChoiceReading("card_name", host -> Strings.emptyToNull(host.getNamedCard())),
            new ChoiceReading("player", host -> Refs.ofNullable(host.getChosenPlayer())),
            new ChoiceReading("cards",
                    host -> host.getChosenCards() == null || host.getChosenCards().isEmpty()
                            ? null : String.valueOf(host.getChosenCards().size())),
            new ChoiceReading("direction", host -> host.getChosenDirection() == null
                    ? null : host.getChosenDirection().name()),
            new ChoiceReading("even_odd", host -> host.getChosenEvenOdd() == null
                    ? null : host.getChosenEvenOdd().name()),
            new ChoiceReading("mode", host -> Strings.emptyToNull(host.getChosenMode())));
}
