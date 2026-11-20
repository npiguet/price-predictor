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
 * card's and player's id, so a faithful copy renders the same text, up to the
 * order of a list of keywords or of ability lines: those compare as
 * multisets wherever they appear, since their order never matters and a
 * repeat (bushido, flanking, rampage) does.
 *
 * <p>A report, never a verdict: {@link #report} prints what differs beside
 * the fork and keeps it, as collection-time checks do.
 */
final class ForkFidelity {

    /** At most this many differences printed for one fork. */
    static final int MAX_REPORTED = 20;
    /** At most this many characters of a value printed. */
    static final int MAX_VALUE = 600;

    /**
     * The members holding a list of keywords or of ability lines, whose order
     * never matters: an entity's {@code granted_temporary.keywords} and
     * {@code abilities}, its {@code lines} and each line's {@code keys}, the
     * global block's {@code emblems} and each floating effect's
     * {@code source_lines}.
     */
    private static final Set<String> UNORDERED =
            Set.of("keywords", "abilities", "lines", "keys", "emblems", "source_lines");

    /** One kind's two rendered JSON arrays: the live game's and the fork's. */
    record Section(String kind, String live, String fork) {
    }

    /** One entity's, player's or the global block's members on each side, under its report label. */
    private record Fields(String label, Map<String, String> live, Map<String, String> fork) {
    }

    /** A JSON value and the name of the member holding it, null for a list element. */
    private record Member(String name, String value) {
    }

    private ForkFidelity() {
    }

    /** Every difference between {@code live}'s board snapshot and {@code fork}'s, as readable lines. */
    static List<String> differences(Game live, Game fork) {
        Snapshot real = new SnapshotBuilder(live).ofBoard();
        // The builder the fork's own records read through.
        Snapshot copy = SnapshotBuilder.forFork(fork, live).ofBoard();
        List<String> out = new ArrayList<>();
        compareFields(
                new Fields("global", JsonParts.fields(real.global()), JsonParts.fields(copy.global())), out);
        out.addAll(differencesById(new Section("player", real.players(), copy.players())));
        out.addAll(differencesById(new Section("entity", real.entitiesJson(), copy.entitiesJson())));
        compareMultiset(new Section("link", real.links(), copy.links()), out);
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
     * @param examples      a few of what the copier left out, by kind
     *                      ({@code GameCopier.getNoCounterpartExamples()}),
     *                      printed after the differences
     */
    static void report(Game live, Game fork, Map<?, Integer> noCounterpart,
                       Map<?, ? extends List<String>> examples) {
        List<String> found;
        try {
            found = differences(live, fork);
        } catch (RuntimeException | StackOverflowError e) {
            System.err.println("Effect records: fork fidelity audit failed (" + e + ")");
            return;
        }
        String line = line(live, found.size(), noCounterpart);
        if (line == null) {
            return;
        }
        StringBuilder out = new StringBuilder(line);
        appendIndented(out, found.subList(0, Math.min(MAX_REPORTED, found.size())));
        appendIndented(out, exampleLines(examples));
        System.err.println(out);
    }

    private static void appendIndented(StringBuilder out, List<String> lines) {
        for (String line : lines) {
            out.append("\n  ").append(line);
        }
    }

    /**
     * One line per example of what the copier left out, kinds in name order,
     * each shortened: {@code no counterpart COMMAND: forge...PumpEffect$1: ...}.
     */
    static List<String> exampleLines(Map<?, ? extends List<String>> examples) {
        List<String> lines = new ArrayList<>();
        if (examples == null) {
            return lines;
        }
        Map<String, List<String>> byKind = new TreeMap<>();
        examples.forEach((kind, list) -> {
            if (list != null) {
                byKind.put(String.valueOf(kind), list);
            }
        });
        byKind.forEach((kind, list) -> {
            for (String example : list) {
                lines.add("no counterpart " + kind + ": " + shorten(example));
            }
        });
        return lines;
    }

    /**
     * The report's first line, or null when there is nothing to report: the
     * differences' count, where the fork was taken, and the kinds the copier
     * dropped references of, with how many. A copy with no difference, only
     * drops, reads {@code Effect records: fork copy dropped: ...}, so a count
     * of {@code fork fidelity:} lines counts the copies that differ.
     */
    static String line(Game live, int differences, Map<?, Integer> noCounterpart) {
        Map<String, Integer> dropped = droppedKinds(noCounterpart);
        if (differences == 0 && dropped.isEmpty()) {
            return null;
        }
        StringBuilder out = new StringBuilder(differences == 0
                ? "Effect records: fork copy dropped: phase "
                : "Effect records: fork fidelity: " + differences + " difference(s), phase ")
                .append(live.getPhaseHandler().getPhase())
                .append(", turn ").append(live.getPhaseHandler().getTurn())
                .append(", caller ").append(ScoreMismatchReport.caller());
        if (!dropped.isEmpty()) {
            out.append(", no counterpart ").append(dropped);
        }
        return out.toString();
    }

    /** The kinds with at least one drop, by name, with their counts. */
    private static Map<String, Integer> droppedKinds(Map<?, Integer> noCounterpart) {
        Map<String, Integer> dropped = new TreeMap<>();
        if (noCounterpart != null) {
            noCounterpart.forEach((kind, count) -> {
                if (count != null && count > 0) {
                    dropped.put(String.valueOf(kind), count);
                }
            });
        }
        return dropped;
    }

    /**
     * Every difference between two arrays of players or entities, matched by
     * id, field by field: {@code kind id (name).field: live=... fork=...}, or
     * an id only one side has.
     */
    static List<String> differencesById(Section section) {
        List<String> out = new ArrayList<>();
        Map<String, List<Map<String, String>>> real = byId(section.live());
        Map<String, List<Map<String, String>>> copy = byId(section.fork());
        Set<String> ids = new TreeSet<>(real.keySet());
        ids.addAll(copy.keySet());
        for (String id : ids) {
            pairDifferences(section.kind() + " " + id,
                    real.getOrDefault(id, List.of()), copy.getOrDefault(id, List.of()), out);
        }
        return out;
    }

    /** The entities sharing one id (506.4c's placeholders) pair up in order. */
    private static void pairDifferences(String idLabel, List<Map<String, String>> lefts,
                                        List<Map<String, String>> rights, List<String> out) {
        for (int i = 0; i < Math.max(lefts.size(), rights.size()); i++) {
            Map<String, String> left = i < lefts.size() ? lefts.get(i) : null;
            Map<String, String> right = i < rights.size() ? rights.get(i) : null;
            String label = idLabel + name(left != null ? left : right);
            if (right == null) {
                out.add(label + ": only in live");
            } else if (left == null) {
                out.add(label + ": only in fork");
            } else {
                compareFields(new Fields(label, left, right), out);
            }
        }
    }

    /** The array's elements by id, those sharing an id kept apart in array order. */
    private static Map<String, List<Map<String, String>>> byId(String array) {
        Map<String, List<Map<String, String>>> byId = new LinkedHashMap<>();
        for (String element : JsonParts.elements(array)) {
            Map<String, String> fields = JsonParts.fields(element);
            byId.computeIfAbsent(unquote(fields.getOrDefault("id", "?")), id -> new ArrayList<>()).add(fields);
        }
        return byId;
    }

    private static void compareFields(Fields fields, List<String> out) {
        Set<String> names = new java.util.LinkedHashSet<>(fields.live().keySet());
        names.addAll(fields.fork().keySet());
        for (String name : names) {
            String left = fields.live().get(name);
            String right = fields.fork().get(name);
            if (left == null ? right != null
                    : !canonical(new Member(name, left)).equals(canonical(new Member(name, right)))) {
                out.add(fields.label() + "." + name + ": live=" + shorten(left) + " fork=" + shorten(right));
            }
        }
    }

    /**
     * {@code value}, a member named {@code name}, written so that two values
     * differing only in the order of a list of keywords or of ability lines
     * read the same: the elements of every such list, at any depth, sorted
     * (repeats kept, as some keywords stack), every other list in its order.
     */
    private static String canonical(Member member) {
        String text = member.value().strip();
        if (text.startsWith("{")) {
            return canonicalObject(text);
        }
        if (text.startsWith("[")) {
            return canonicalArray(text, member.name() != null && UNORDERED.contains(member.name()));
        }
        return text;
    }

    private static String canonicalObject(String text) {
        List<String> members = new ArrayList<>();
        JsonParts.fields(text).forEach((member, inner) ->
                members.add("\"" + member + "\":" + canonical(new Member(member, inner))));
        return "{" + String.join(",", members) + "}";
    }

    private static String canonicalArray(String text, boolean sorted) {
        List<String> elements = new ArrayList<>();
        for (String element : JsonParts.elements(text)) {
            elements.add(canonical(new Member(null, element)));
        }
        if (sorted) {
            elements.sort(null);
        }
        return "[" + String.join(",", elements) + "]";
    }

    private static void compareMultiset(Section section, List<String> out) {
        Map<String, Integer> left = new HashMap<>();
        for (String element : JsonParts.elements(section.live())) {
            left.merge(element, 1, Integer::sum);
        }
        for (String element : JsonParts.elements(section.fork())) {
            left.merge(element, -1, Integer::sum);
        }
        left.forEach((element, count) -> {
            if (count != 0) {
                out.add(section.kind() + " only in " + (count > 0 ? "live" : "fork") + ": " + shorten(element));
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
