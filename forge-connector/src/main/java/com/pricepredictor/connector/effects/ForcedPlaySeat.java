package com.pricepredictor.connector.effects;

/**
 * A seat controller that consults a directed fork's forced play before its
 * own choices (FR-010b): the random seat, whose fork copy keeps playing at
 * random everywhere the play does not decide. A plain AI seat is replaced by a
 * {@link ForcingSeatController} instead.
 */
public interface ForcedPlaySeat {

    /** Consult {@code run} first at every decision it may make. */
    void force(ForcedPlayRun run);
}
