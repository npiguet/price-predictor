package com.pricepredictor.connector.effects.directed.rules;

import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;

import java.util.List;
import java.util.Set;

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
}
