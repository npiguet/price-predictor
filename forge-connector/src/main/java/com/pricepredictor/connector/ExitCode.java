package com.pricepredictor.connector;

/**
 * The process exit codes every main ends with.
 *
 * <p>The Python supervisors and connectors only ever test for zero versus
 * non-zero ({@code forge_jvm.py}, {@code evaluation_connector.py},
 * {@code pool_connector.py}, ...), so the non-zero codes are for the operator
 * reading a log: they tell "the worker was started wrong" apart from "the
 * work itself failed". A supervisor restarts the slot either way.
 */
enum ExitCode {

    /** The work finished, or the worker stopped because its input closed. */
    OK(0),

    /**
     * The work threw, or a worker loop gave up after too many failed
     * attempts in a row.
     */
    FAILURE(1),

    /**
     * The process was started wrong: a missing or malformed {@code -D}
     * property or argument, or an input it was pointed at that cannot be
     * used. Restarting it unchanged will fail the same way.
     */
    BAD_CONFIG(2),

    /**
     * A file the validation worker reads or writes positionally cannot be:
     * the matches file, or the outcomes file whose line count is where a
     * restart resumes. The same code as {@link #BAD_CONFIG}, because the
     * supervisor tells exit codes apart only from zero; it is named on its
     * own so the call site says which failure it is.
     */
    IO_FAILURE(2);

    private final int status;

    ExitCode(int status) {
        this.status = status;
    }

    /** The value passed to {@link System#exit}. */
    int status() {
        return status;
    }
}
