package com.pricepredictor.connector;

import forge.ai.LobbyPlayerAi;
import forge.game.Game;
import forge.game.player.Player;

import java.util.Random;

/**
 * The lobby seat that installs a {@link RandomSeatController} (FR-024).
 *
 * <p>{@code LobbyPlayerAi.createControllerFor} is private, but
 * {@code createIngamePlayer} is public and not final and
 * {@code Player.setFirstController} is public, so a subclass is the whole
 * installation and the engine is untouched. Built with no {@code AIOption},
 * like the ordinary seats {@link GamePlayer#registeredPlayers} builds, and for
 * the same reason: an option would let the AI enter simulation mode, whose
 * hypothetical resolutions the static effect-record listeners cannot tell
 * from real ones.
 *
 * <p>A fork clones the match's registered players through
 * {@code GameCopier.clonePlayer}, which keeps any {@code LobbyPlayerAi}
 * subclass — so this creates a controller for the fork's player too. That
 * controller is never attached to a live game and defers every override to
 * the AI, which is what keeps a fork from acting at random.
 */
public final class RandomSeatLobbyPlayer extends LobbyPlayerAi {

    private final double probability;
    private final Random seeds;

    /**
     * @param probability {@code P}, the chance of a random draw at each
     *                    decision point (FR-022)
     * @param seeds       the match's seeded source; each game's controller
     *                    draws its own seed from it, so the games of one
     *                    match do not repeat one another's draws
     */
    public RandomSeatLobbyPlayer(String name, double probability, Random seeds) {
        super(name, null);
        this.probability = probability;
        this.seeds = seeds;
    }

    @Override
    public Player createIngamePlayer(Game game, final int id) {
        Player seat = new Player(getName(), game, id);
        seat.setFirstController(new RandomSeatController(
                game, seat, this, probability, new Random(seeds.nextLong())));
        return seat;
    }
}
