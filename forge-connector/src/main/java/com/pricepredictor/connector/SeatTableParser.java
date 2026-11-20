package com.pricepredictor.connector;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Reads a seat table's lines into {@link SeatTableIndex.SeatEntry} values; the
 * format is described on {@link SeatTableIndex}.
 */
final class SeatTableParser {

    private SeatTableParser() {
    }

    /** Every non-blank line of {@code seatsFile}, parsed, in file order. */
    static List<SeatTableIndex.SeatEntry> parse(Path seatsFile) throws IOException {
        List<SeatTableIndex.SeatEntry> seats = new ArrayList<>();
        for (String line : CorpusFiles.nonBlankLines(seatsFile)) {
            seats.add(parseLine(line));
        }
        return seats;
    }

    private static SeatTableIndex.SeatEntry parseLine(String line) {
        String[] parts = line.split(CardNameList.FIELD_SEPARATOR, 5);
        if (parts.length < 5) {
            throw new IllegalArgumentException(
                    "seat line needs 5 ';'-separated fields, got " + parts.length + ": " + line);
        }
        SeatTableIndex.SeatKind kind;
        try {
            kind = WireNamed.fromWireName(SeatTableIndex.SeatKind.class, parts[3]);
        } catch (IllegalArgumentException e) {
            // Worded for whoever wrote the table, not in the generic lookup's
            // terms: the operator fixes a seat line, not an enum.
            throw new IllegalArgumentException(
                    "seat kind must be 'deck' or 'pool', got '" + parts[3] + "': " + line, e);
        }
        return new SeatTableIndex.SeatEntry(
                parts[0], parts[1], parts[2], kind, CardNameList.parse(parts[4]));
    }
}
