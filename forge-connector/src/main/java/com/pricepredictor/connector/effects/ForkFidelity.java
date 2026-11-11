package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.json.JsonParts;
import forge.game.Game;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * A fork's copy compared with the game it came from on everything a record's
 * state says, before anything in the copy is changed.
 *
 * <p>The score check compares one number, the AI's rating of the position,
 * so a difference the rating does not weigh passes it, and the fork's records
 * then describe the copy's board and its consequences rather than the game's.
 * This compares the board snapshot a fork record would carry — the copy's
 * rendered as a fork renders it, reading the live game's clause notes —
 * with the live game's: the global block field by field, each player and each
 * entity by id and field, and the links as a multiset. The copier keeps every
 * card's and player's id, so a faithful copy renders the same text.
 *
 * <p>A report, never a verdict: {@link #report} prints what differs beside
 * the fork and keeps it, as collection-time checks do.
 */
final class ForkFidelity {

    /** At most this many differences printed for one fork. */
    static final int MAX_REPORTED = 20;
    /** At most this many characters of a value printed. */
    static final int MAX_VALUE = 600;

    private ForkFidelity() {
    }

    /** Every difference between {@code live}'s board snapshot and {@code fork}'s, as readable lines. */
    static List<String> differences(Game live, Game fork) {
        Snapshot real = new SnapshotBuilder(live).ofBoard();
        // The builder the fork's own records read through.
        Snapshot copy = SnapshotBuilder.forFork(fork, live).ofBoard();
        List<String> out = new ArrayList<>();
        compareFields("global", JsonParts.fields(real.global()), JsonParts.fields(copy.global()), out);
        compareById("player", real.players(), copy.players(), out);
        compareById("entity", real.entities(), copy.entities(), out);
        compareMultiset("link", real.links(), copy.links(), out);
        return out;
    }

    /**
     * Print {@code fork}'s differences from {@code live}, and what the copier
     * left without a counterpart, if either is anything. Never throws: an
     * audit that failed is reported as one, and the fork goes on.
     *
     * @param noCounterpart the copier's per-copy tally of references it could
     *                      not map, by kind ({@code GameCopier.getNoCounterpartTally()}),
     *                      only the kinds with drops present
     */
    static void report(Game live, Game fork, Map<?, Integer> noCounterpart) {
        List<String> found;
        try {
            found = differences(live, fork);
        } catch (RuntimeException e) {
            System.err.println("Effect records: fork fidelity audit failed (" + e + ")");
            return;
        }
        String line = line(live, found.size(), noCounterpart);
        if (line == null) {
            return;
        }
        StringBuilder out = new StringBuilder(line);
        for (String difference : found.subList(0, Math.min(MAX_REPORTED, found.size()))) {
            out.append("\n  ").append(difference);
        }
        System.err.println(out);
    }

    /**
     * The report's first line, or null when there is nothing to report: the
     * differences' count, where the fork was taken, and the kinds the copier
     * dropped references of, with how many.
     */
    static String line(Game live, int differences, Map<?, Integer> noCounterpart) {
        Map<String, Integer> dropped = new TreeMap<>();
        if (noCounterpart != null) {
            noCounterpart.forEach((kind, count) -> {
                if (count != null && count > 0) {
                    dropped.put(String.valueOf(kind), count);
                }
            });
        }
        if (differences == 0 && dropped.isEmpty()) {
            return null;
        }
        StringBuilder out = new StringBuilder("Effect records: fork fidelity: ")
                .append(differences).append(" difference(s), phase ")
                .append(live.getPhaseHandler().getPhase())
                .append(", turn ").append(live.getPhaseHandler().getTurn())
                .append(", caller ").append(ScoreMismatchReport.caller());
        if (!dropped.isEmpty()) {
            out.append(", no counterpart ").append(dropped);
        }
        return out.toString();
    }

    private static void compareById(String kind, String realArray, String copyArray, List<String> out) {
        Map<String, Map<String, String>> real = byId(realArray);
        Map<String, Map<String, String>> copy = byId(copyArray);
        Set<String> ids = new TreeSet<>(real.keySet());
        ids.addAll(copy.keySet());
        for (String id : ids) {
            Map<String, String> left = real.get(id);
            Map<String, String> right = copy.get(id);
            String label = kind + " " + id + name(left != null ? left : right);
            if (right == null) {
                out.add(label + ": only in live");
            } else if (left == null) {
                out.add(label + ": only in fork");
            } else {
                compareFields(label, left, right, out);
            }
        }
    }

    private static Map<String, Map<String, String>> byId(String array) {
        Map<String, Map<String, String>> byId = new LinkedHashMap<>();
        for (String element : JsonParts.elements(array)) {
            Map<String, String> fields = JsonParts.fields(element);
            byId.put(unquote(fields.getOrDefault("id", "?")), fields);
        }
        return byId;
    }

    private static void compareFields(String label, Map<String, String> real, Map<String, String> copy,
                                      List<String> out) {
        Set<String> names = new java.util.LinkedHashSet<>(real.keySet());
        names.addAll(copy.keySet());
        for (String name : names) {
            String left = real.get(name);
            String right = copy.get(name);
            if (left == null ? right != null : !left.equals(right)) {
                out.add(label + "." + name + ": live=" + shorten(left) + " fork=" + shorten(right));
            }
        }
    }

    private static void compareMultiset(String kind, String realArray, String copyArray, List<String> out) {
        Map<String, Integer> left = new HashMap<>();
        for (String element : JsonParts.elements(realArray)) {
            left.merge(element, 1, Integer::sum);
        }
        for (String element : JsonParts.elements(copyArray)) {
            left.merge(element, -1, Integer::sum);
        }
        left.forEach((element, count) -> {
            if (count > 0) {
                out.add(kind + " only in live: " + shorten(element));
            } else if (count < 0) {
                out.add(kind + " only in fork: " + shorten(element));
            }
        });
    }

    private static String name(Map<String, String> fields) {
        String name = fields.get("name");
        return name == null ? "" : " (" + unquote(name) + ")";
    }

    private static String unquote(String text) {
        return text.length() >= 2 && text.startsWith("\"") && text.endsWith("\"")
                ? text.substring(1, text.length() - 1) : text;
    }

    private static String shorten(String value) {
        if (value == null) {
            return "(absent)";
        }
        return value.length() <= MAX_VALUE ? value : value.substring(0, MAX_VALUE) + "...";
    }
}
