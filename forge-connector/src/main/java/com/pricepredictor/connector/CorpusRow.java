package com.pricepredictor.connector;

/**
 * One row of an append-only corpus file: whatever it is, it is written as
 * the single line {@link #toLine()} renders.
 *
 * <p>The line is the contract with the Python reader, so it carries no
 * line separator; {@link CorpusWriter} adds that.
 */
public interface CorpusRow {

    /** The row as one corpus line, without its line separator. */
    String toLine();
}
