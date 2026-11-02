package com.pricepredictor.connector.effects;

import forge.game.Game;

import java.util.Random;

/**
 * Makes a fork's seats play as the live game's do (D4).
 *
 * <p>A game copy builds each seat's controller afresh from its lobby player,
 * and a seat that needs attaching to its game before it plays as itself, the
 * random seat, would otherwise play the fork as the plain AI. The match
 * runner knows which seats those are; the recording only knows when a fork
 * starts playing on.
 */
@FunctionalInterface
public interface ForkSeats {

    /** No seat needs attaching: both are the plain AI. */
    ForkSeats NONE = (fork, candidates, fruitless, random) -> { };

    /**
     * Attach the fork's seats to {@code fork}.
     *
     * @param candidates the fork's own priority stops, which a random seat
     *                   draws its plays from
     * @param fruitless  the fork's own free activations that changed nothing
     * @param random     the seat's draws, seeded from the game's fork seed
     *                   rather than the match's source, so a fork leaves
     *                   later games' seats as they would have been
     */
    void attach(Game fork, PriorityCandidates candidates, FruitlessActivations fruitless, Random random);
}
