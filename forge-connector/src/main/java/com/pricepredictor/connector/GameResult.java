package com.pricepredictor.connector;

import java.util.Objects;

/**
 * Who won one game of a match and who was on the play in it: one character
 * of a match row's {@code games} sequence and the matching one of its
 * {@code play} sequence.
 *
 * <p>Held as a pair rather than as two parallel strings so a game cannot
 * have a winner without a starter, and the sequences cannot disagree in
 * length.
 */
public record GameResult(Side winner, Side starter) {

    public GameResult {
        Objects.requireNonNull(winner, "winner");
        Objects.requireNonNull(starter, "starter");
    }
}
