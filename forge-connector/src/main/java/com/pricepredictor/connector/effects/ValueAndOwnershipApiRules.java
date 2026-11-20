package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.effects.ApiEvents.Rule;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.ApiEvents.amount;
import static com.pricepredictor.connector.effects.ApiEvents.playerOf;
import static com.pricepredictor.connector.effects.ApiEvents.put;

/**
 * The {@link ApiEvents} rules for X and ownership: a changed X, a card
 * changing owner.
 */
final class ValueAndOwnershipApiRules {

    private ValueAndOwnershipApiRules() {
    }

    static void valueAndOwnershipRules(Map<String, Rule<?>> rules) {
        put(rules, Rule.shaped(EventType.X_CHANGED, null, (event, sa, host, memo) ->
                        event.param("value", amount(sa, host, "Value", "0"))),
                "ChangeX");
        put(rules, Rule.shaped(EventType.OWNERSHIP_CHANGE,
                        (sa, host) -> ClauseTargets.affectedCards(sa), ValueAndOwnershipApiRules::ownershipChange),
                "GainOwnership");
    }

    /**
     * The cards the clause hands over, read before it runs, as {@code
     * OwnershipGainEffect.resolve} reads them ({@code getTargetCards}), and
     * their new owner.
     */
    private static EffectEvent ownershipChange(
            EffectEvent event, SpellAbility sa, Card host, List<Card> cards) {
        if (cards != null) {
            cards.forEach(card -> event.subject(Refs.of(card)));
        }
        return event.param("owner", ownerOf(sa));
    }

    /**
     * The new owner {@code GainOwnership} promises: a named player where the
     * clause names one, the activating player otherwise.
     *
     * <p>{@code OwnershipGainEffect.resolve} reads {@code DefinedPlayer} first
     * and falls back to {@code sa.getActivatingPlayer()} only when that
     * resolves to nobody. A purely parameter-based rule that always read the
     * activating player would name the wrong owner on a real card: Tempest
     * Efreet's second clause is {@code DB$ GainOwnership | Defined$
     * CorrectedSelf | DefinedPlayer$ Targeted}, which hands the card to the
     * opponent it targeted, not to whoever sacrificed the Efreet; Bronze
     * Tablet's is {@code DefinedPlayer$ Remembered}, naming a player chosen
     * three clauses earlier. Both are real printed cards, not hypotheticals.
     */
    private static String ownerOf(SpellAbility sa) {
        if (sa.hasParam("DefinedPlayer")) {
            List<Player> named = ClauseTargets.definedPlayers(sa, sa.getParam("DefinedPlayer"));
            if (!named.isEmpty()) {
                return Refs.of(named.get(0));
            }
        }
        return playerOf(sa);
    }
}
