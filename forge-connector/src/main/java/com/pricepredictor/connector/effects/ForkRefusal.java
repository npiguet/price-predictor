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
    /**
     * A directed fork's forced play (FR-010b) has no legal line, source,
     * target or attack on the board: found on the live board before the fork
     * is taken where it can be, else on the edited copy or as the play is made.
     */
    public static final String FORCED_NO_LINE = "forced play has no legal line";
    /** A directed fork's forced play was not made by the end of its step, or of the turn. */
    public static final String FORCED_NOT_MADE = "forced play not made";
    /**
     * A directed fork's forced play could not be paid for: the AI's own check
     * ({@code ComputerUtilCost.canPayCost}) said no at the first priority
     * where the rules let the line be played, before any payment was tried;
     * the seat's mana payment or a named cost card failed; or a "you may pay"
     * answered yes could not be paid.
     */
    public static final String FORCED_COST_NOT_PAID = "forced play cost not paid";

    /** @param reason why the fork declined, as the refusal tally names it */
    public ForkRefusal(String reason) {
        super(reason, null, false, false);
    }

    /** Why the fork declined, as the refusal tally names it. */
    public String reason() {
        return getMessage();
    }
}
