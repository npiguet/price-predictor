package com.pricepredictor.connector;

import java.time.Instant;
import java.time.format.DateTimeFormatter;

/** Line assembly shared by the corpus files that open with the same five fields. */
final class CorpusLine {

    private CorpusLine() {
    }

    /**
     * {@code timestamp;run_id;set_code;method_A;method_B} followed by
     * {@code rest}, joined by {@link CardNameList#FIELD_SEPARATOR}.
     */
    static String of(
            Instant timestamp, String runId, String setCode, String methodA, String methodB,
            String... rest) {
        String leading = String.join(CardNameList.FIELD_SEPARATOR,
                DateTimeFormatter.ISO_INSTANT.format(timestamp), runId, setCode, methodA, methodB);
        return rest.length == 0
                ? leading
                : leading + CardNameList.FIELD_SEPARATOR
                        + String.join(CardNameList.FIELD_SEPARATOR, rest);
    }
}
