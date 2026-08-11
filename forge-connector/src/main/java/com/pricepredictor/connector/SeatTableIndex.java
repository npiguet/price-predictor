package com.pricepredictor.connector;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * In-memory index of a seat table for draft game evaluation.
 *
 * <p>File format: one seat per line,
 * {@code draft_id;set_code;label;kind;Card1|Card2|...}. {@code kind} is
 * {@code deck} when the cards are a finished deck to play as given, or
 * {@code pool} when they are a seat's drafted pool, from which Forge's own
 * sealed builder produces the deck.
 *
 * <p>Seats are grouped by {@code draft_id} into pods. A pairing is two distinct
 * seats of one pod, so pool and set quality are held constant within every
 * comparison. Mirror pairs — both seats carrying the same label — are rejected
 * unless the caller asks for them.
 *
 * <p>Written once by {@code python -m draft play-draft-games}; this class is the
 * seat-table counterpart of {@link GeneratedDecksIndex}.
 */
public class SeatTableIndex {

    public static final String KIND_DECK = "deck";
    public static final String KIND_POOL = "pool";

    /**
     * One parsed seat line.
     *
     * @param draftId   Pod identifier; compared only for equality
     * @param setCode   MTG set code the pod drafted
     * @param label     Agent label; becomes {@code method_A} / {@code method_B}
     * @param kind      {@link #KIND_DECK} or {@link #KIND_POOL}
     * @param cardNames A finished deck, or a drafted pool
     */
    public record SeatEntry(
            String draftId,
            String setCode,
            String label,
            String kind,
            List<String> cardNames) {

        public boolean isPool() {
            return KIND_POOL.equals(kind);
        }
    }

    /** Two distinct seats of one pod, selected to play. */
    public record Pairing(SeatEntry a, SeatEntry b) {}

    /** Pods that can yield a legal pairing, in file order. */
    private final List<List<SeatEntry>> eligiblePods;
    private final int seatCount;
    private final Random random;

    public SeatTableIndex(List<SeatEntry> seats, boolean includeMirrors) {
        this(seats, includeMirrors, new Random());
    }

    SeatTableIndex(List<SeatEntry> seats, boolean includeMirrors, Random random) {
        this.random = random;
        this.seatCount = seats.size();

        Map<String, List<SeatEntry>> byPod = new LinkedHashMap<>();
        for (SeatEntry seat : seats) {
            byPod.computeIfAbsent(seat.draftId(), k -> new ArrayList<>()).add(seat);
        }

        this.eligiblePods = new ArrayList<>();
        for (List<SeatEntry> pod : byPod.values()) {
            if (canYieldPairing(pod, includeMirrors)) {
                this.eligiblePods.add(pod);
            }
        }
    }

    /** Load a seat table from disk. */
    public static SeatTableIndex load(Path seatsFile, boolean includeMirrors) throws IOException {
        List<SeatEntry> seats = new ArrayList<>();
        for (String line : Files.readAllLines(seatsFile)) {
            String trimmed = line.trim();
            if (trimmed.isEmpty()) continue;
            seats.add(parseLine(trimmed));
        }
        return new SeatTableIndex(seats, includeMirrors);
    }

    /** Total seats loaded, including those in pods that cannot yield a pairing. */
    public int seatCount() {
        return seatCount;
    }

    /** Pods that can yield a pairing under the mirror setting given at construction. */
    public int eligiblePodCount() {
        return eligiblePods.size();
    }

    /**
     * Draw a pod uniformly at random, then two distinct seats of it uniformly at
     * random, rejecting mirror pairs when they are excluded.
     *
     * <p>Pods that cannot yield a legal pairing are excluded at construction, so
     * the rejection loop always terminates.
     *
     * @return a pairing, or null when no pod can yield one
     */
    public Pairing randomPairing(boolean includeMirrors) {
        if (eligiblePods.isEmpty()) {
            return null;
        }
        List<SeatEntry> pod = eligiblePods.get(random.nextInt(eligiblePods.size()));
        while (true) {
            int i = random.nextInt(pod.size());
            int j = random.nextInt(pod.size());
            if (i == j) continue;
            SeatEntry a = pod.get(i);
            SeatEntry b = pod.get(j);
            if (includeMirrors || !a.label().equals(b.label())) {
                return new Pairing(a, b);
            }
        }
    }

    /** True when this pod holds two seats that could legally be paired. */
    private static boolean canYieldPairing(List<SeatEntry> pod, boolean includeMirrors) {
        if (pod.size() < 2) {
            return false;
        }
        if (includeMirrors) {
            return true;
        }
        String first = pod.get(0).label();
        for (SeatEntry seat : pod) {
            if (!seat.label().equals(first)) {
                return true;
            }
        }
        return false;  // every seat shares one label; excluded so the retry terminates
    }

    // ── Static helpers (package-visible for testing) ────────────────

    static SeatEntry parseLine(String line) {
        String[] parts = line.split(";", 5);
        if (parts.length < 5) {
            throw new IllegalArgumentException(
                    "seat line needs 5 ';'-separated fields, got " + parts.length + ": " + line);
        }
        List<String> cards = parts[4].isEmpty()
                ? List.of()
                : Arrays.asList(parts[4].split("\\|", -1));
        return new SeatEntry(parts[0], parts[1], parts[2], parts[3], cards);
    }
}
