package com.pricepredictor.connector.effects.directed.rules;

import forge.game.Game;
import forge.game.ability.AbilityFactory;
import forge.game.card.Card;
import forge.game.cost.Cost;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;

/**
 * What the board preconditions read off a live game, in the words the
 * scenarios use: the phase a boundary begins, the player whose turn it is,
 * the creatures a player controls. Read only: a precondition changes nothing
 * in the game it is asked about.
 */
final class Boards {

    private Boards() {
    }

    /** Whether the boundary the live game is at begins one of {@code phases}. */
    static boolean at(Game live, PhaseType... phases) {
        return Set.of(phases).contains(live.getPhaseHandler().getPhase());
    }

    /** Whether the boundary begins a main phase, where a sorcery-speed decision is next. */
    static boolean atMainPhase(Game live) {
        return at(live, PhaseType.MAIN1, PhaseType.MAIN2);
    }

    /** The player whose turn it is. */
    static Player active(Game live) {
        return live.getPhaseHandler().getPlayerTurn();
    }

    /** The creatures {@code player} controls that are untapped. */
    static List<Card> untappedCreatures(Player player) {
        return player.getCreaturesInPlay().stream().filter(card -> !card.isTapped()).toList();
    }

    /** How many untapped lands {@code player} controls: a cheap bound on the mana it can spend. */
    static int untappedLands(Player player) {
        return (int) player.getLandsInPlay().stream().filter(land -> !land.isTapped()).count();
    }

    /** The highest life total among the game's players other than {@code player}. */
    static int mostLifeBesides(Player player) {
        return player.getGame().getPlayers().stream()
                .filter(other -> other != player)
                .mapToInt(Player::getLife)
                .max()
                .orElseThrow(() -> new IllegalStateException("no other player"));
    }

    /** Whether {@code player} has as many untapped lands as {@code card}'s mana value: a cheap bound on casting it. */
    static boolean affordable(Player player, Card card) {
        return card.getCMC() <= untappedLands(player);
    }

    /**
     * The untapped permanents {@code player} controls that can make mana now.
     * A summoning-sick creature's tap ability makes none.
     */
    static List<Card> manaSources(Player player) {
        return player.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> !card.isTapped() && !card.getManaAbilities().isEmpty())
                .filter(card -> !(card.isCreature() && card.isSick()))
                .toList();
    }

    /**
     * The one untapped permanent {@code player} controls that can make mana
     * now, or empty when it controls none or several: the last mana it has
     * for a cost.
     */
    static Optional<Card> soleManaSource(Player player) {
        List<Card> sources = manaSources(player);
        return sources.size() == 1 ? Optional.of(sources.get(0)) : Optional.empty();
    }

    /** The mana value of a cost's mana part, 0 for none. */
    static int manaValue(Cost cost) {
        return cost == null || cost.getTotalMana() == null ? 0 : cost.getTotalMana().getCMC();
    }

    /**
     * The cards in {@code player}'s hand it has the mana to cast
     * ({@link #affordable}) with a spell {@code spell} accepts.
     */
    static List<Card> castableInHand(Player player, Predicate<SpellAbility> spell) {
        return player.getCardsIn(ZoneType.Hand).stream()
                .filter(card -> affordable(player, card))
                .filter(card -> card.getSpells().stream().anyMatch(spell))
                .toList();
    }

    /** Whether a comma-separated script value lists {@code item}. */
    static boolean lists(String value, String item) {
        return value != null && List.of(value.split(",")).contains(item);
    }

    /** The parameters of the ability script {@code host} holds as {@code svar}, empty when it holds none. */
    static Map<String, String> script(Card host, String svar) {
        if (svar == null || !host.hasSVar(svar)) {
            return Map.of();
        }
        return AbilityFactory.getMapParams(host.getSVar(svar));
    }

    /** The parameters of the ability {@code trigger} executes, empty when it names none. */
    static Map<String, String> executed(Trigger trigger) {
        return trigger.hasParam("Execute") ? script(trigger.getHostCard(), trigger.getParam("Execute")) : Map.of();
    }

    /** The API a script's parameters name, whether it is a spell, an ability or a sub-ability. */
    static String api(Map<String, String> script) {
        for (String kind : List.of("SP", "AB", "DB")) {
            if (script.containsKey(kind)) {
                return script.get(kind);
            }
        }
        return "";
    }

    /** {@code value} as a number, or empty when it is an X, a count or absent. */
    static Optional<Integer> number(String value) {
        if (value == null) {
            return Optional.empty();
        }
        try {
            return Optional.of(Integer.parseInt(value.trim()));
        } catch (NumberFormatException e) {
            return Optional.empty();
        }
    }
}
