package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.ReplaceCardOperation;
import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.trigger.Trigger;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Optional;

/**
 * 701.12h, exchanging text boxes gives each permanent the other's rules text,
 * on scenario {@code 701.12h-1}: Exchange of Words enters, targets Grizzly
 * Bears and Typhoid Rats, and the Bears gains the Rats' deathtouch.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it holds a card it has the mana for whose ability exchanges text
 * boxes (Exchange of Words), and the battlefield holds exactly two creatures,
 * one with no rules text and one with some, so the exchange has no other
 * pair to choose. It binds the creature with text.
 *
 * <p>Variant {@code both_vanilla} ({@code replace_card}): Hill Giant is in
 * that creature's place, so both text boxes are empty and the exchange
 * changes nothing.
 */
public final class TextBoxExchange {

    public static final String RULE = "701.12h";
    static final String SCENARIO = "701.12h-1";
    static final String BOTH_VANILLA = "both_vanilla";
    static final String GIANT = "Hill Giant";

    private TextBoxExchange() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, TextBoxExchange::bind, BOTH_VANILLA,
                new Application(ForkOperation.REPLACE_CARD,
                        fork -> new ReplaceCardOperation(fork.card("creature"), GIANT)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        boolean exchanger = active.getCardsIn(ZoneType.Hand).stream()
                .filter(card -> Boards.affordable(active, card))
                .flatMap(card -> card.getTriggers().stream())
                .anyMatch(TextBoxExchange::exchangesText);
        List<Card> creatures = live.getCardsIn(ZoneType.Battlefield).stream().filter(Card::isCreature).toList();
        if (!exchanger || creatures.size() != 2) {
            return Optional.empty();
        }
        List<Card> withText = creatures.stream().filter(TextBoxExchange::hasRulesText).toList();
        return withText.size() == 1
                ? Optional.of(Binding.empty().with("creature", withText.get(0)))
                : Optional.empty();
    }

    /** Whether the trigger's ability exchanges text boxes. */
    static boolean exchangesText(Trigger trigger) {
        return ApiType.ExchangeTextBox.name().equals(Boards.api(Boards.executed(trigger)));
    }

    /** Whether the card's text box holds any rules text. */
    static boolean hasRulesText(Card card) {
        return !card.getOracleText().isBlank();
    }
}
