package com.pricepredictor.connector;

import java.time.Instant;
import java.time.format.DateTimeFormatter;
import java.util.Objects;

/**
 * The five fields both sealed corpus files open with,
 * {@code timestamp;run_id;set_code;method_A;method_B}.
 *
 * <p>One value for both rows, because {@code cards-played.txt} is joined to
 * {@code match-outcomes.txt} by these very fields (plus a game's position
 * within its match): a cards-played row carries its parent match's header
 * verbatim, timestamp included, and a value shared rather than copied field
 * by field cannot drift between the two.
 *
 * @param timestamp When the match finished (UTC); every game of one match shares it.
 * @param runId     UUID identifying the supervisor invocation that produced the match.
 * @param setCode   MTG set code both decks were drawn from.
 * @param methodA   How deck A was built: a {@link BuildMethod} tag, or the per-deck
 *                  {@code LABEL} field when deck A came from a generated-decks file.
 * @param methodB   How deck B was built, spelled as {@code methodA} is.
 */
public record MatchHeader(
        Instant timestamp, String runId, String setCode, String methodA, String methodB) {

    public MatchHeader {
        Objects.requireNonNull(timestamp, "timestamp");
    }

    /**
     * The header's five fields followed by {@code rest}, joined by
     * {@link CardNameList#FIELD_SEPARATOR}: a whole corpus line.
     */
    String line(String... rest) {
        String leading = String.join(CardNameList.FIELD_SEPARATOR,
                DateTimeFormatter.ISO_INSTANT.format(timestamp), runId, setCode, methodA, methodB);
        return rest.length == 0
                ? leading
                : leading + CardNameList.FIELD_SEPARATOR
                        + String.join(CardNameList.FIELD_SEPARATOR, rest);
    }
}
