package com.pricepredictor.connector.effects.directed.rules;

import forge.game.CardTraitBase;
import forge.game.Game;
import forge.game.ability.AbilityUtils;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.card.CardCollection;
import forge.game.card.CardLists;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.trigger.Trigger;
import forge.game.trigger.TriggerType;
import forge.game.zone.ZoneType;
import forge.util.Expressions;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * The meld hosts a player could meld this turn, for the meld rules: a
 * permanent it owns and controls whose trigger melds it (Graf Rats), and the
 * partner that trigger names (Midnight Scavengers), read at the first main
 * phase of its turn, before the beginning of combat or the end step a meld
 * trigger waits for. A trigger whose phase has passed by then (Titania,
 * Voice of Gaea's upkeep) or whose SVar check fails is no meld the fork makes.
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

    /** A script's check of an SVar against a comparison ({@code GE1} when unstated), as {@code CardTraitBase} reads it. */
    private record SVarCheck(String svar, String compare) {
    }

    private static final List<SVarCheck> SVAR_CHECKS = List.of(
            new SVarCheck("CheckSVar", "SVarCompare"), new SVarCheck("CheckSecondSVar", "SecondSVarCompare"));

    /**
     * The active player's meld hosts at the start of its first main phase
     * whose meld is a trigger (Graf Rats': no choice melds it) that can still
     * fire this turn ({@link #firesLaterThisTurn}) and whose partner it
     * neither owns nor controls on the battlefield, each with a stand-in the
     * partner put in its place survives beside; none at any other boundary.
     */
    static List<Unpaired> unpairedOfActivePlayer(Game live) {
        if (!Boards.at(live, PhaseType.MAIN1)) {
            return List.of();
        }
        Player active = Boards.active(live);
        List<Unpaired> found = new ArrayList<>();
        for (Card host : ownedAndControlled(active)) {
            for (Trigger trigger : host.getTriggers()) {
                unpaired(active, host, trigger).ifPresent(found::add);
            }
        }
        return found;
    }

    /** The host's trigger as an {@link Unpaired} meld, or empty when it melds nothing the fork could pair. */
    private static Optional<Unpaired> unpaired(Player active, Card host, Trigger trigger) {
        Map<String, String> meld = Scripts.executed(trigger);
        if (!meldsHost(host, meld) || !firesLaterThisTurn(trigger)) {
            return Optional.empty();
        }
        if (!partners(active, host, meld).isEmpty()) {
            return Optional.empty();
        }
        String partner = meld.get("Secondary");
        return CardLists.getType(ownedAndControlled(active), meld.getOrDefault("SecondaryType", "Creature"))
                .stream()
                .filter(card -> card != host && !card.isToken() && !card.getName().equals(partner))
                .filter(card -> Boards.replacementSurvives(card, partner))
                .findFirst()
                .map(standIn -> new Unpaired(host, standIn, partner));
    }

    /** Whether the meld script melds {@code host}, as its primary, with a partner it names. */
    private static boolean meldsHost(Card host, Map<String, String> meld) {
        return ApiType.Meld.name().equals(Scripts.api(meld)) && host.getName().equals(meld.get("Primary"))
                && meld.get("Secondary") != null;
    }

    /**
     * Whether the meld trigger can still fire in the turn the fork plays: a
     * phase trigger's phase is still to begin (Titania, Voice of Gaea's
     * upkeep has passed at the first main phase; Graf Rats' beginning of
     * combat has not), and its SVar checks hold on this board (Titania's four
     * land cards in the graveyard), which the fork's edit, putting the
     * partner in a stand-in's place, does not change. A trigger of another
     * mode waits for no phase.
     */
    static boolean firesLaterThisTurn(Trigger trigger) {
        return phaseAhead(trigger) && svarsHold(trigger);
    }

    private static boolean phaseAhead(Trigger trigger) {
        if (trigger.getMode() != TriggerType.Phase) {
            return true;
        }
        String phases = trigger.getParam("Phase");
        PhaseType now = trigger.getHostCard().getGame().getPhaseHandler().getPhase();
        return phases != null && PhaseType.parseRange(phases).stream().anyMatch(phase -> phase.isAfter(now));
    }

    /** Whether the trait's {@code CheckSVar} and {@code CheckSecondSVar} comparisons hold, as Forge reads them. */
    static boolean svarsHold(CardTraitBase trait) {
        return SVAR_CHECKS.stream().allMatch(check -> svarHolds(trait, check));
    }

    private static boolean svarHolds(CardTraitBase trait, SVarCheck check) {
        if (!trait.hasParam(check.svar())) {
            return true;
        }
        Card host = trait.getHostCard();
        int value = AbilityUtils.calculateAmount(host, trait.getParam(check.svar()), trait);
        String comparator = trait.getParamOrDefault(check.compare(), "GE1");
        int operand = AbilityUtils.calculateAmount(host, comparator.substring(2), trait);
        return Expressions.compare(value, comparator.substring(0, 2), operand);
    }

    /** The partner the host's meld trigger names, if it has one. */
    static Optional<String> partnerName(Card host) {
        for (Trigger trigger : host.getTriggers()) {
            Map<String, String> meld = Scripts.executed(trigger);
            if (ApiType.Meld.name().equals(Scripts.api(meld)) && meld.get("Secondary") != null) {
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
