package com.pricepredictor.connector;

import java.util.List;

/**
 * Value object capturing the outcome of one sealed best-of-3 match.
 *
 * @param deckA  All 40 cards in deck A (including basic lands, duplicates repeat).
 * @param deckB  All 40 cards in deck B.
 * @param winsA  Games won by deck A (0-2).
 * @param winsB  Games won by deck B (0-2).
 */
public record MatchResult(List<String> deckA, List<String> deckB, int winsA, int winsB) {

    public MatchResult {
        if (winsA + winsB != 2 && winsA + winsB != 3) {
            throw new IllegalArgumentException(
                    "winsA + winsB must be 2 or 3, got " + winsA + " + " + winsB);
        }
        if (winsA < 0 || winsA > 2 || winsB < 0 || winsB > 2) {
            throw new IllegalArgumentException(
                    "wins must be in range [0,2], got winsA=" + winsA + " winsB=" + winsB);
        }
    }
}
