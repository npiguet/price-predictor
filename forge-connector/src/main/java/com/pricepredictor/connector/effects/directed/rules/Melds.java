package com.pricepredictor.connector.effects.directed.rules;

import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.card.CardCollection;
import forge.game.card.CardLists;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * The meld pairs a player could meld this turn, for the three meld rules
 * (701.42a-c): a permanent it owns and controls whose trigger or activated
 * ability melds it (Graf Rats, Hanweir Battlements), beside a permanent it
 * owns and controls named as that ability's partner (Midnight Scavengers),
 * read at the first main phase of its turn, before the beginning of combat
 * or the end step a meld trigger waits for.
 */
final class Melds {

    /**
     * One meld pair.
     *
     * @param host      the permanent whose ability melds it
     * @param partner   the permanent named as its partner
     * @param activated whether the meld is an activated ability its
     *                  controller has the lands for, rather than a trigger
     */
    record Pair(Card host, Card partner, boolean activated) {
    }

    private Melds() {
    }

    /** The active player's meld pairs at the start of its first main phase; none at any other boundary. */
    static List<Pair> ofActivePlayer(Game live) {
        if (!Boards.at(live, PhaseType.MAIN1)) {
            return List.of();
        }
        Player active = Boards.active(live);
        List<Pair> pairs = new ArrayList<>();
        for (Card host : ownedAndControlled(active)) {
            for (Trigger trigger : host.getTriggers()) {
                Map<String, String> meld = Boards.executed(trigger);
                if (ApiType.Meld.name().equals(Boards.api(meld))) {
                    partners(active, host, meld).forEach(partner -> pairs.add(new Pair(host, partner, false)));
                }
            }
            for (SpellAbility ability : host.getSpellAbilities()) {
                if (ability.getApi() == ApiType.Meld && ability.isActivatedAbility()
                        && Boards.manaValue(ability.getPayCosts()) <= landsBesides(active, host)) {
                    partners(active, host, ability.getMapParams())
                            .forEach(partner -> pairs.add(new Pair(host, partner, true)));
                }
            }
        }
        return pairs;
    }

    /** The permanents {@code player} owns and controls named as {@code host}'s partner by a meld script. */
    private static List<Card> partners(Player player, Card host, Map<String, String> meld) {
        if (!host.getName().equals(meld.get("Primary")) || meld.get("Secondary") == null) {
            return List.of();
        }
        CardCollection named = CardLists.filter(ownedAndControlled(player),
                card -> card != host && card.getName().equals(meld.get("Secondary")));
        return CardLists.getType(named, meld.getOrDefault("SecondaryType", "Creature"));
    }

    private static CardCollection ownedAndControlled(Player player) {
        return CardLists.filter(player.getCardsIn(ZoneType.Battlefield), card -> card.getOwner() == player);
    }

    /** The untapped lands {@code player} controls other than {@code host}, which may tap itself for the cost. */
    private static int landsBesides(Player player, Card host) {
        int lands = Boards.untappedLands(player);
        return host.isLand() && !host.isTapped() ? lands - 1 : lands;
    }
}
