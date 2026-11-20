package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.effects.ApiEvents.Rule;
import forge.game.card.Card;
import forge.game.spellability.SpellAbility;

import java.util.Map;

import static com.pricepredictor.connector.effects.ApiEvents.put;

/**
 * The {@link ApiEvents} rule for text changes.
 */
final class TextApiRules {

    private TextApiRules() {
    }

    /**
     * {@code ChangeText} alone: {@code ExchangeTextBox}'s own contribution to
     * text_change is a {@code note()} call from {@code TextBoxExchangeEffect}
     * itself (see that class), not a rule here -- its two early returns
     * ({@code checkValidDuration}, {@code tgtCards.size() < 2}) are not
     * cheaply replicable from outside without duplicating engine logic.
     */
    static void textRules(Map<String, Rule<?>> rules) {
        put(rules, Rule.one(EventType.TEXT_CHANGE, null, (sa, host, memo) -> changeTextEvent(sa)),
                "ChangeText");
    }

    /**
     * text_change: the card(s) whose text changed, or {@code null} when the
     * clause names no word to replace at all -- {@code
     * ChangeTextEffect.resolve()}'s target loop still fires {@code
     * GameEventCardStatsChanged} for every target even then, so the params
     * themselves, read directly with no post-resolution state needed, are the
     * only signal that anything textual actually changed.
     */
    private static EffectEvent changeTextEvent(SpellAbility sa) {
        if (!sa.hasParam("ChangeColorWord") && !sa.hasParam("ChangeTypeWord")) {
            return null;
        }
        EffectEvent event = new EffectEvent(EventType.TEXT_CHANGE);
        for (Card c : ClauseTargets.cardsFromTargets(sa)) {
            event.subject(Refs.of(c));
        }
        return event.subjects().isEmpty() ? null : event;
    }
}
