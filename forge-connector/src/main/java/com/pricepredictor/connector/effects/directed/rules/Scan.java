package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.ForcedPlay;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.card.CardState;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;
import forge.game.zone.ZoneType;

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Predicate;

/**
 * The scans a precondition makes over a hand and a card's triggers, in the
 * words the scenarios use: a card the player holds and can pay for, a card
 * that does something as it enters, a spell of the card that searches a
 * library. Read only.
 */
final class Scan {

    /** A spell of the card that searches a library ({@link #searches}), the line its forced cast plays. */
    static final ForcedPlay.Line LIBRARY_SEARCH_SPELL = ForcedPlay.Line.spell("library search", Scan::searches);

    private Scan() {
    }

    /**
     * The first card in {@code player}'s hand, in hand order, that {@code wanted}
     * accepts and the player can pay one of its casts for
     * ({@link Mana#affordable(Player, Card)}); empty when there is none.
     * {@code wanted} is asked first, the cost only of a card it accepts.
     */
    static Optional<Card> handCard(Player player, Predicate<Card> wanted) {
        return player.getCardsIn(ZoneType.Hand).stream()
                .filter(card -> wanted.test(card) && Mana.affordable(player, card))
                .findFirst();
    }

    /**
     * The scripts of the abilities the given triggers (such as a card state's)
     * execute whose API is {@code api} ({@code ApiType} name): triggers that
     * fire as the card itself enters ({@link Triggers#entersItself}), with the
     * conditions a script adds to {@code ValidCard$ Card.Self} such as
     * {@code +wasCastByYou}.
     */
    static List<Map<String, String>> entersTrigger(Collection<Trigger> triggers, String api) {
        return triggers.stream()
                .filter(Triggers::entersItself)
                .map(Scripts::executed)
                .filter(script -> api.equals(Scripts.api(script)))
                .toList();
    }

    /**
     * Like {@link #entersTrigger(Collection, String)}, but only a trigger whose
     * {@code ValidCard} is exactly {@code Card.Self}: one with no condition
     * added.
     */
    static List<Map<String, String>> unconditionedEntersTrigger(Card card, String api) {
        return card.getTriggers().stream()
                .filter(trigger -> Triggers.entersItself(trigger) && "Card.Self".equals(trigger.getParam("ValidCard")))
                .map(Scripts::executed)
                .filter(script -> api.equals(Scripts.api(script)))
                .toList();
    }

    /** Whether the spell searches a library: it moves a card of a stated type out of one, its contents hidden. */
    static boolean searches(SpellAbility spell) {
        return spell.getApi() == ApiType.ChangeZone && Scripts.lists(spell.getParam("Origin"), "Library")
                && spell.hasParam("ChangeType");
    }

    /**
     * The cards in {@code player}'s hand it has the mana for whose spell
     * searches its library ({@link #LIBRARY_SEARCH_SPELL}) and that it can
     * cast at this boundary ({@link Boards#castableAtItsSpeed}: a sorcery only
     * as the active player).
     */
    static List<Card> searchersInHand(Player player) {
        return castableInHand(player, LIBRARY_SEARCH_SPELL).stream()
                .filter(card -> Boards.castableAtItsSpeed(player, card))
                .toList();
    }

    /**
     * The spells of the cards in {@code player}'s hand that are {@code line}
     * and that it has the mana for ({@link Mana#affordable(Player, SpellAbility)}),
     * in hand order. A variant forcing the cast forces the same line, so a
     * split card's matching half is the one cast.
     */
    static List<SpellAbility> castableSpellsInHand(Player player, ForcedPlay.Line line) {
        return player.getCardsIn(ZoneType.Hand).stream()
                .flatMap(card -> card.getSpells().stream())
                .filter(spell -> line.matches(spell) && Mana.affordable(player, spell))
                .toList();
    }

    /** The cards of {@link #castableSpellsInHand}, each once, in hand order. */
    static List<Card> castableInHand(Player player, ForcedPlay.Line line) {
        return castableSpellsInHand(player, line).stream().map(SpellAbility::getHostCard).distinct().toList();
    }

    /**
     * The card's own cast of a spell that acts with {@code api}, an
     * {@code ApiType} name: the spell has it at its root, or the permanent it
     * makes has a trigger executing it (a creature that explores, connives or
     * discovers as it enters). Named by string: the registry builds it where
     * Forge is not initialised.
     */
    static ForcedPlay.Line acting(String api) {
        return ForcedPlay.Line.spell(api, spell -> acts(spell, api));
    }

    private static boolean acts(SpellAbility spell, String api) {
        if (spell.getApi() != null && spell.getApi().name().equals(api)) {
            return true;
        }
        CardState state = spell.getCardState();
        return state != null && !entersTrigger(state.getTriggers(), api).isEmpty();
    }

    /** The first card {@code player} holds with a castable spell of {@code line} ({@link #castableInHand}). */
    static Optional<Card> acting(Player player, ForcedPlay.Line line) {
        return castableInHand(player, line).stream().findFirst();
    }
}
