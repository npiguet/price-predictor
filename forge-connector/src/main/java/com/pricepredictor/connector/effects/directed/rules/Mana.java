package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.PlayabilityVerdict;
import com.pricepredictor.connector.effects.directed.ForcedPlay;
import forge.game.card.Card;
import forge.game.cost.Cost;
import forge.game.keyword.Keyword;
import forge.game.keyword.KeywordInterface;
import forge.game.keyword.KeywordWithCostAndAmount;
import forge.game.keyword.KeywordWithCostInterface;
import forge.game.player.Player;
import forge.game.spellability.AlternativeCost;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * What a player can pay with, read off a live game: the mana its permanents
 * make, the land drop it has left, the costs of a card's keywords and whether
 * a cost can be paid now. Read only.
 */
final class Mana {

    private Mana() {
    }

    /**
     * Whether {@code player} cannot add a land to its mana this turn: it holds
     * no card it could play as a land, or it has played as many lands this
     * turn as it may. A "one land short" variant forked at a main phase
     * before the turn's land drop is undone by that drop.
     */
    static boolean noLandToPlay(Player player) {
        boolean dropLeft = player.getMaxLandPlaysInfinite()
                || player.getLandsPlayedThisTurn() < player.getMaxLandPlays();
        return !dropLeft || player.getCardsIn(ZoneType.Hand).stream().noneMatch(Card::hasPlayableLandFace);
    }

    /**
     * Whether {@code player} can pay one of {@code card}'s own casts
     * ({@link ForcedPlay.Line#SPELL}) now, colours included
     * ({@link #affordable(Player, SpellAbility)}).
     */
    static boolean affordable(Player player, Card card) {
        return card.getSpells().stream().anyMatch(spell -> ForcedPlay.Line.SPELL.matches(spell)
                && affordable(player, spell));
    }

    /**
     * Whether {@code player} can pay {@code ability}'s cost now, by the rules
     * the collector's playability reads ({@link PlayabilityVerdict#affordable}):
     * the mana its sources make, colours included, and the additional costs.
     * Asked of a copy activated by {@code player}, so the live ability stays
     * as it is; a cost that cannot be read on this board is not affordable.
     */
    static boolean affordable(Player player, SpellAbility ability) {
        try {
            return PlayabilityVerdict.affordable(asPlayedBy(ability, player), player);
        } catch (RuntimeException e) {
            return false;
        }
    }

    /** A copy of {@code ability} as {@code player} would play it: what a precondition asks questions of. */
    static SpellAbility asPlayedBy(SpellAbility ability, Player player) {
        return ability.copy(ability.getHostCard(), player, true);
    }

    /**
     * How much mana one activation of the permanent's most productive mana
     * ability makes: two for a bounce land, two for Sol Ring, one for a basic
     * land.
     */
    static int manaUnits(Card source) {
        return source.getManaAbilities().stream()
                .mapToInt(ability -> ability.amountOfManaGenerated(true))
                .max().orElse(0);
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
     * A land {@code player} needs every one of for a cost of {@code value}
     * mana: when the permanents it can make mana with now ({@link #manaSources})
     * are all lands and number exactly {@code value}, the first of them; else
     * empty. Without it the cost is one mana short.
     */
    static Optional<Card> lastLandFor(Player player, int value) {
        List<Card> sources = manaSources(player);
        return value > 0 && allLandsCounting(sources, value) ? Optional.of(sources.get(0)) : Optional.empty();
    }

    private static boolean allLandsCounting(List<Card> sources, int value) {
        return sources.size() == value && sources.stream().allMatch(Card::isLand);
    }

    /** The mana value of a cost's mana part, 0 for none. */
    static int manaValue(Cost cost) {
        return cost == null || cost.getTotalMana() == null ? 0 : cost.getTotalMana().getCMC();
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
     * Whether {@code player} can pay one of the card's abilities of
     * {@code keyword} now ({@link #affordable(Player, SpellAbility)}): the
     * activated ability of Transmute, Unearth or Aura swap, or the alternative
     * cost of the spell that Bestow or Web-slinging makes. Keywords that pay
     * by tapping creatures (Improvise, Convoke) are not asked here.
     */
    static boolean canPayKeyword(Player player, Card card, Keyword keyword) {
        return card.getAllSpellAbilities().stream()
                .filter(ability -> viaKeyword(ability, keyword))
                .anyMatch(ability -> affordable(player, ability));
    }

    private static boolean viaKeyword(SpellAbility ability, Keyword keyword) {
        return ability.isKeyword(keyword)
                || keyword == Keyword.BESTOW && ability.isBestow()
                || keyword == Keyword.WEB_SLINGING && ability.isAlternativeCost(AlternativeCost.WebSlinging);
    }
}
