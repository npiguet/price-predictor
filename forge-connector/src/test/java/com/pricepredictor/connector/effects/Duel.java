package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.event.GameEventPlayerPriority;
import forge.game.phase.PhaseHandler;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.player.PlayerView;
import forge.game.spellability.SpellAbility;

/**
 * A started two-player game in one step of the first seat's turn: the seat
 * that casts, the seat it plays against, and the engine's own moves a stack
 * test makes on that game.
 *
 * <p>Every state-based check here is the loop a priority runs ({@link
 * #stateBasedLoop}), so on the patched checkout the engine signals its begin
 * and its end as it does before every priority of a worker's game.
 *
 * @param game     the game, past its start: state-based actions run and the stack resolves
 * @param caster   the first seat, whose turn it is
 * @param opponent the second seat
 */
record Duel(Game game, Player caster, Player opponent) {

    /** A duel in {@code step} of the first seat's turn. */
    static Duel at(PhaseType step) {
        Game game = TestBoard.startedTwoPlayerGame(step);
        return new Duel(game, game.getPlayers().get(0), game.getPlayers().get(1));
    }

    /** A duel in the first seat's first main phase. */
    static Duel inMainPhase() {
        return at(PhaseType.MAIN1);
    }

    /**
     * The state-based loop the next priority runs: the check, then the
     * triggers it set off put on the stack, and again while any was, between
     * the begin and end signals of the loop hook where one is installed.
     *
     * <p>Advancing to the step the game is already in advances nothing and
     * runs exactly that loop ({@code PhaseHandler.checkStateBasedEffects}),
     * the one public way to it that asks no player for an action.
     */
    void stateBasedLoop() {
        PhaseHandler phases = game.getPhaseHandler();
        phases.devAdvanceToPhase(phases.getPhase());
    }

    /**
     * What {@code PhaseHandler.mainLoopStep} does before a player acts: the
     * priority event, then the state-based loop.
     */
    void priority() {
        PhaseHandler phases = game.getPhaseHandler();
        PlayerView turn = PlayerView.get(phases.getPlayerTurn());
        game.fireEvent(new GameEventPlayerPriority(turn, phases.getPhase(), turn));
        stateBasedLoop();
    }

    /** Resolve the top of the stack and run the priority after it. */
    void resolve() {
        game.getStack().resolveStack();
        priority();
    }

    /**
     * Put a spell or ability on the stack, resolve it, and run the
     * state-based loop the next priority runs.
     */
    void castAndResolve(SpellAbility spell) {
        game.getStack().add(spell);
        game.getStack().resolveStack();
        stateBasedLoop();
    }
}
