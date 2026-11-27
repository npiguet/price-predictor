package com.pricepredictor.connector;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Live records with the ids a fork would shift replaced by places that do not
 * depend on it: a fork's records take ids from the same sequence as the live
 * game's, which is no change to the live game.
 */
final class LiveRecordIds {

    private static final Pattern RECORD_ID = Pattern.compile("\"record_id\":\"([^\"]+)\"");

    /** A member whose value is an id of this run's sequence. */
    private static final Pattern ID_MEMBER = Pattern.compile("\"(\\w+)\":\"(run\\.0-l1\\.\\d+)\"");

    private static final Pattern LINK_ID =
            Pattern.compile("run\\.0-l1\\.\\d+\\.link\\.\\d+|(?<=\")D\\d+(?=\")");

    private LiveRecordIds() {
    }

    /**
     * The live records with each record id that names one of them (its own,
     * a link's, a mirror's) replaced by the record's place among them, then
     * each link id by the place of its first appearance: a fork's cost and
     * effect halves take link ids from the same sequence, and so does a
     * delayed trigger (D&lt;trigger id&gt;), from the counter every game of
     * the JVM shares.
     */
    static List<String> renumbered(List<String> written) {
        Map<String, String> place = placesOf(written);
        List<String> live = written.stream().map(line -> replaceRecordIds(line, place)).toList();
        Map<String, String> links = new HashMap<>();
        return live.stream().map(line -> replaceLinkIds(line, links)).toList();
    }

    private static Map<String, String> placesOf(List<String> written) {
        Map<String, String> place = new HashMap<>();
        for (String line : written) {
            Matcher own = RECORD_ID.matcher(line);
            assertTrue(own.find(), "every record has an id");
            place.put(own.group(1), "#" + place.size());
        }
        return place;
    }

    private static String replaceRecordIds(String line, Map<String, String> place) {
        Matcher id = ID_MEMBER.matcher(line);
        StringBuilder out = new StringBuilder();
        while (id.find()) {
            String to = id.group(1).equals("game_id") ? id.group(2)
                    : place.getOrDefault(id.group(2), "fork:" + id.group(2));
            id.appendReplacement(out, Matcher.quoteReplacement(
                    "\"" + id.group(1) + "\":\"" + to + "\""));
        }
        id.appendTail(out);
        return out.toString();
    }

    private static String replaceLinkIds(String line, Map<String, String> links) {
        Matcher link = LINK_ID.matcher(line);
        StringBuilder out = new StringBuilder();
        while (link.find()) {
            String to = links.computeIfAbsent(link.group(), l -> "link#" + links.size());
            link.appendReplacement(out, Matcher.quoteReplacement(to));
        }
        link.appendTail(out);
        return out.toString();
    }
}
