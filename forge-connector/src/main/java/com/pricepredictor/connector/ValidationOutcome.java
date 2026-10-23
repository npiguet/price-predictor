package com.pricepredictor.connector;

/**
 * One line of a validation outcomes file, {@code winsA;winsB}: the games each
 * deck of the matching input line won.
 *
 * <p>The outcomes are paired with the matches by position, so every match
 * writes exactly one, played or not; see {@link #NONE}.
 *
 * @param winsA games deck A won
 * @param winsB games deck B won
 */
record ValidationOutcome(int winsA, int winsB) implements CorpusRow {

    /**
     * The filler a match that could not be played writes, {@code 0;0}: it
     * keeps every later outcome on its own match and counts no games for
     * either side.
     */
    static final ValidationOutcome NONE = new ValidationOutcome(0, 0);

    ValidationOutcome {
        if (winsA < 0 || winsB < 0) {
            throw new IllegalArgumentException(
                    "win counts must be non-negative, got " + winsA + ";" + winsB);
        }
    }

    /** The outcome of a played match. */
    static ValidationOutcome of(GamePlayer.PlayedMatch played) {
        return new ValidationOutcome(played.winsA(), played.winsB());
    }

    @Override
    public String toLine() {
        return winsA + CardNameList.FIELD_SEPARATOR + winsB;
    }
}
