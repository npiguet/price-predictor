package com.pricepredictor.connector.effects.directed.rules;

import forge.game.Game;
import forge.game.ability.AbilityFactory;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.cost.Cost;
import forge.game.keyword.Keyword;
import forge.game.keyword.KeywordInterface;
import forge.game.keyword.KeywordWithCostAndAmount;
import forge.game.keyword.KeywordWithCostInterface;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;
import forge.game.trigger.TriggerType;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
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

    /**
     * A land {@code player} needs every one of for a cost of {@code value}
     * mana: when the permanents it can make mana with now ({@link #manaSources})
     * are all lands and number exactly {@code value}, the first of them; else
     * empty. Without it the cost is one mana short.
     */
    static Optional<Card> lastLandFor(Player player, int value) {
        List<Card> sources = manaSources(player);
        if (value <= 0 || sources.size() != value || !sources.stream().allMatch(Card::isLand)) {
            return Optional.empty();
        }
        return Optional.of(sources.get(0));
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

    /** The costs of the card's instances of {@code keyword} (Transmute, Unearth, Suspend), in the card's order. */
    static List<Cost> keywordCosts(Card card, Keyword keyword) {
        List<Cost> costs = new ArrayList<>();
        for (KeywordInterface instance : card.getKeywords(keyword)) {
            if (instance instanceof KeywordWithCostInterface withCost) {
                costs.add(withCost.getCost());
            } else if (instance instanceof KeywordWithCostAndAmount withCost) {
                costs.add(withCost.getCost());
            }
        }
        return costs;
    }

    /**
     * Whether {@code player} has as many untapped lands as the mana value of
     * one of the card's {@code keyword} costs: a cheap bound on paying it.
     */
    static boolean canPayKeyword(Player player, Card card, Keyword keyword) {
        return keywordCosts(card, keyword).stream().anyMatch(cost -> manaValue(cost) <= untappedLands(player));
    }

    /**
     * Whether one of the card's spells, or the ability one of its triggers
     * executes, has {@code api} at its root: a card that explores, connives or
     * discovers as it resolves or as it enters.
     */
    static boolean acts(Card card, ApiType api) {
        return card.getSpells().stream().anyMatch(spell -> spell.getApi() == api)
                || card.getTriggers().stream().anyMatch(trigger -> api.name().equals(api(executed(trigger))));
    }

    /** Whether {@code player} holds a card it has the lands for ({@link #affordable}) that {@link #acts} with {@code api}. */
    static boolean holdsActing(Player player, ApiType api) {
        return player.getCardsIn(ZoneType.Hand).stream()
                .anyMatch(card -> affordable(player, card) && acts(card, api));
    }

    /** Whether a permanent on the battlefield has a trigger of {@code mode} (Explores, Connives, Discover). */
    static boolean watched(Game live, TriggerType mode) {
        return live.getCardsIn(ZoneType.Battlefield).stream()
                .flatMap(card -> card.getTriggers().stream())
                .anyMatch(trigger -> trigger.getMode() == mode);
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
