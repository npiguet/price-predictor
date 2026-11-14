package com.pricepredictor.connector.effects;

/**
 * A fork that declines for an expected reason: the live object it maps has
 * no counterpart in the copy, its line has no legal choices there, a board
 * edit cannot apply to the card it is bound to. Thrown out of a fork's body
 * ({@code Forks.ForkRun}) and counted, by reason, in {@link ForkRefusals}
 * rather than as a failure; carries no stack.
 */
public final class ForkRefusal extends RuntimeException {

    /** The live object a fork maps has no counterpart in the copy. */
    public static final String NO_COUNTERPART = "no counterpart";
    /** The forced line has no legal choices on the copy. */
    public static final String NO_CHOICES = "no choices";
    /**
     * The fork's stack holds an ability that restarts the game (FR-010d),
     * which no fork resolves: Forge's restart resets the JVM-wide trigger id
     * counter the live game's triggers take their ids from.
     */
    public static final String GAME_RESTART = "game restart";

    /** @param reason why the fork declined, as the refusal tally names it */
    public ForkRefusal(String reason) {
        super(reason, null, false, false);
    }

    /** Why the fork declined, as the refusal tally names it. */
    public String reason() {
        return getMessage();
    }
}
