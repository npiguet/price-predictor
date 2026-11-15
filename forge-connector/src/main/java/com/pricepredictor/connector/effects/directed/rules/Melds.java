package com.pricepredictor.connector.effects.directed.rules;

import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.card.CardCollection;
import forge.game.card.CardLists;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.trigger.Trigger;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * The meld hosts a player could meld this turn, for the meld rules: a
 * permanent it owns and controls whose trigger melds it (Graf Rats), and the
 * partner that trigger names (Midnight Scavengers), read at the first main
 * phase of its turn, before the beginning of combat or the end step a meld
 * trigger waits for.
 */
final class Melds {

    private Melds() {
    }

    /**
     * A meld host whose trigger would meld it with a partner its controller
     * does not have, beside a stand-in: a creature it owns and controls that
     * the partner can take the place of (not the host, not a token, not
     * named as the partner).
     *
     * @param partner the partner's name, as the host's meld script names it
     */
    record Unpaired(Card host, Card standIn, String partner) {
    }

    /**
     * The active player's meld hosts at the start of its first main phase
     * whose meld is a trigger (Graf Rats': no choice melds it) and whose
     * partner it neither owns nor controls on the battlefield, each with a
     * stand-in the partner put in its place survives beside; none at any
     * other boundary.
     */
    static List<Unpaired> unpairedOfActivePlayer(Game live) {
        if (!Boards.at(live, PhaseType.MAIN1)) {
            return List.of();
        }
        Player active = Boards.active(live);
        List<Unpaired> found = new ArrayList<>();
        for (Card host : ownedAndControlled(active)) {
            for (Trigger trigger : host.getTriggers()) {
                Map<String, String> meld = Boards.executed(trigger);
                if (!ApiType.Meld.name().equals(Boards.api(meld)) || !partners(active, host, meld).isEmpty()
                        || !host.getName().equals(meld.get("Primary")) || meld.get("Secondary") == null) {
                    continue;
                }
                String partner = meld.get("Secondary");
                CardLists.getType(ownedAndControlled(active), meld.getOrDefault("SecondaryType", "Creature"))
                        .stream()
                        .filter(card -> card != host && !card.isToken() && !card.getName().equals(partner))
                        .filter(card -> Boards.replacementSurvives(card, partner))
                        .findFirst()
                        .ifPresent(standIn -> found.add(new Unpaired(host, standIn, partner)));
            }
        }
        return found;
    }

    /** The partner the host's meld trigger names, if it has one. */
    static Optional<String> partnerName(Card host) {
        for (Trigger trigger : host.getTriggers()) {
            Map<String, String> meld = Boards.executed(trigger);
            if (ApiType.Meld.name().equals(Boards.api(meld)) && meld.get("Secondary") != null) {
                return Optional.of(meld.get("Secondary"));
            }
        }
        return Optional.empty();
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
}
