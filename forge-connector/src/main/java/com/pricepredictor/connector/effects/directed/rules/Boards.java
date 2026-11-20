package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.operations.ReplaceCardOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.keyword.Keyword;
import forge.game.keyword.KeywordsChange;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.trigger.TriggerType;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Set;

/**
 * What the board preconditions read off a live game, in the words the
 * scenarios use: the phase a boundary begins, the player whose turn it is,
 * the creatures a player controls. Read only: a precondition changes nothing
 * in the game it is asked about. The scripts, mana, combat and hand scans
 * are {@link Scripts}, {@link Mana}, {@link Combats} and {@link Scan}.
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

    /**
     * Whether {@code holder} can cast {@code card} at this boundary as far as
     * timing goes: the active player at a main phase can cast anything, any
     * other player only an instant or a card with flash. A sorcery in the
     * non-active player's hand cannot be cast this turn, so a fork waiting
     * for it would be spent on nothing.
     */
    static boolean castableAtItsSpeed(Player holder, Card card) {
        return holder == active(holder.getGame()) || card.isInstant() || card.hasKeyword(Keyword.FLASH);
    }

    /**
     * Whether {@code replace_card} puts {@code cardName} in {@code permanent}'s
     * place, refusing none of what the permanent holds (a token, a merged or
     * melded permanent, a face-down one: {@link ReplaceCardOperation#carries}),
     * and the new card survives the fork's first state-based check, carrying
     * the permanent's marked damage, deathtouch damage, counters, boosts and
     * resolved effects: the edit's own readings
     * ({@link ReplaceCardOperation#replacementDies}), so a precondition binds
     * no board the edit would refuse.
     */
    static boolean replacementSurvives(Card permanent, String cardName) {
        return ReplaceCardOperation.carries(permanent) && !ReplaceCardOperation.replacementDies(permanent, cardName);
    }

    /**
     * Whether every instance of {@code keyword} the card has is its own text's:
     * none is granted by a static ability (an Equipment, an Aura, an anthem)
     * or by a resolved effect. {@code replace_card} relinks the attachments
     * and carries the resolved effects, and the statics apply to the new card
     * anew, so a granted keyword would stay on the card put in its place.
     */
    static boolean ownKeywordOnly(Card card, Keyword keyword) {
        for (KeywordsChange change : card.getChangedCardKeywords().values()) {
            if (change.getKeywords().stream().anyMatch(granted -> granted.getKeyword() == keyword)) {
                return false;
            }
        }
        return true;
    }

    /** Forge's spelling of trample over planeswalkers, a trample instance with a detail. */
    static final String TRAMPLE_OVER_PLANESWALKERS = "Trample:Planeswalker";

    /** Whether the card has trample over planeswalkers ({@link #TRAMPLE_OVER_PLANESWALKERS}). */
    static boolean tramplesOverPlaneswalkers(Card card) {
        return card.hasKeyword(TRAMPLE_OVER_PLANESWALKERS);
    }

    /** The creatures {@code player} controls that are untapped. */
    static List<Card> untappedCreatures(Player player) {
        return player.getCreaturesInPlay().stream().filter(card -> !card.isTapped()).toList();
    }

    /** The highest life total among the game's players other than {@code player}. */
    static int mostLifeBesides(Player player) {
        return player.getGame().getPlayers().stream()
                .filter(other -> other != player)
                .mapToInt(Player::getLife)
                .max()
                .orElseThrow(() -> new IllegalStateException("no other player"));
    }

    /**
     * Whether a permanent on the battlefield has a trigger of {@code mode}
     * (Explores, Connives, Discover) that {@code active}'s {@code actor}
     * sets off ({@link Triggers#watchedFor}).
     */
    static boolean watched(Game live, TriggerType mode, Player active, Card actor) {
        return Triggers.watchedFor(live.getCardsIn(ZoneType.Battlefield), mode, active, actor);
    }
}
