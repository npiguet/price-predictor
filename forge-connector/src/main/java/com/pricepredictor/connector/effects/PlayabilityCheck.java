package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.WireNamed;

/**
 * The checks a candidate can fail, as a decision record's {@code failed_checks}
 * names them (contract {@code record-schema-delta.md}, FR-032).
 *
 * <p>Declared in the contract's order, which is the order a candidate lists
 * its failed checks in. {@code PlayabilityCheckContractTest} pins the wire
 * names against the contract.
 */
public enum PlayabilityCheck implements WireNamed {
    /** A split-second spell is on the stack. */
    SPLIT_SECOND("split_second"),
    /** Sorcery timing, Forge's {@code canCastTiming}. */
    CASTING_SPEED("casting_speed"),
    /** The turn a restriction requires ({@code PlayerTurn$}, {@code OpponentTurn$}). */
    WHOSE_TURN("whose_turn"),
    /** The phase or step a restriction names ({@code ActivationPhases$}). */
    PHASE("phase"),
    /** A combat restriction: first combat only, after blockers, sneak. */
    COMBAT("combat"),
    /** Activations per turn or per game, a loyalty ability's included. */
    ACTIVATION_LIMIT("activation_limit"),
    /** A land play once the turn's land plays have reached the player's limit (rule 305.2). */
    LAND_PLAY_LIMIT("land_play_limit"),
    /** A script's own condition on the board ({@code Activation$}, {@code IsPresent$}, ...). */
    SCRIPT_CONDITION("script_condition"),
    /** The zone the card must be in. */
    ZONE("zone"),
    /** A static that forbids the play. */
    STATIC_PROHIBITION("static_prohibition"),
    /** The cost cannot be paid. */
    COST("cost"),
    /** A check Forge evaluates together with others and does not name. */
    SCRIPT_RESTRICTION("script_restriction");

    private final String wire;

    PlayabilityCheck(String wire) {
        this.wire = wire;
    }

    @Override
    public String wireName() {
        return wire;
    }
}
