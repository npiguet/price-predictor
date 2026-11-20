package com.pricepredictor.connector.effects;

/**
 * A record's rendered {@code payload} and what its {@link RecordFacts} read
 * of it, held together from where it is built to the write.
 *
 * @param json         the rendered payload
 * @param namesOutcome whether an event it holds is the game's outcome
 *                     ({@code player_lost}, {@code player_won})
 * @param namedScript  the script file of the first line it names under
 *                     {@code ability} (a {@code decision} record's candidate),
 *                     or null where it names none
 */
record RecordPayload(String json, boolean namesOutcome, String namedScript) {

    /** The payload of a record that names no line and has none yet: {@code {}}. */
    static final RecordPayload EMPTY = withoutEvents("{}");

    /** A payload that holds no event and names no line. */
    static RecordPayload withoutEvents(String json) {
        return new RecordPayload(json, false, null);
    }
}
