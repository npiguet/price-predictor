package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.json.Json;

import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLongArray;

/**
 * What each directed rule did in this worker JVM's lifetime (FR-010a): how
 * often its board appeared, drew a fork, found the game's budget spent, took
 * a fork, kept it, and how often its precondition threw.
 *
 * <p>A JVM lifetime, not a run: the supervisor recycles workers, and a worker
 * can count only what it saw. The worker prints the tally as running totals
 * in its status line's {@code directed} member ({@code WorkerStatus}), and
 * the supervisor sums each lifetime's last line. One {@link #lifetime()} per
 * JVM; a test builds its own.
 */
public final class DirectedForkTally {

    /** What is counted per rule, in the order the tally's JSON writes the counts. */
    enum Count {
        /** A boundary where the rule had not yet forked this game and its precondition held. */
        MATCHED,
        /** Of those, the draws against the rule's probability that came up a fork. */
        DRAWN,
        /** Of those, the ones the game's directed-fork budget was already spent for. */
        BUDGET_BLOCKED,
        /** Forks taken, the budget spent, kept or discarded. */
        TAKEN,
        /** Of those, the forks kept. */
        KEPT,
        /** Precondition calls that threw, each read as not met. */
        PRECONDITION_THREW;

        /** The count's name in the tally's JSON. */
        String wireName() {
            return name().toLowerCase(Locale.ROOT);
        }
    }

    private static final DirectedForkTally LIFETIME = new DirectedForkTally();

    private final Map<String, AtomicLongArray> counts = new ConcurrentHashMap<>();

    /** The tally of this JVM's lifetime. */
    static DirectedForkTally lifetime() {
        return LIFETIME;
    }

    /** The lifetime tally as JSON, {@code {}} while no rule has counted anything. */
    public static String lifetimeJson() {
        return LIFETIME.json();
    }

    /** {@code rule}'s count of {@code count} so far. */
    long get(String rule, Count count) {
        AtomicLongArray counted = counts.get(rule);
        return counted == null ? 0 : counted.get(count.ordinal());
    }

    /** Count one more {@code count} for {@code rule}; @return the new count. */
    long add(String rule, Count count) {
        return counts.computeIfAbsent(rule, name -> new AtomicLongArray(Count.values().length))
                .incrementAndGet(count.ordinal());
    }

    /**
     * Count a throw of a registration's precondition, which read as not met,
     * under each of its {@code rules}. The first of each registration is
     * printed with its stack, naming its rules joined by {@code +}; later
     * ones are counted only.
     */
    void preconditionThrew(List<String> rules, RuntimeException failure) {
        boolean first = false;
        for (String rule : rules) {
            first |= add(rule, Count.PRECONDITION_THREW) == 1;
        }
        if (first) {
            System.err.println("Effect records: directed rule " + String.join("+", rules)
                    + "'s precondition threw and"
                    + " reads as not met (later throws are counted in the directed fork tally,"
                    + " not printed): " + HookFailures.withStack(failure));
        }
    }

    /**
     * A JSON object from each rule that has counted anything, in name order,
     * to its counts by {@link Count#wireName()}, every count present; {@code {}}
     * while no rule has counted anything.
     */
    String json() {
        Json.ObjectWriter rules = Json.object();
        new TreeMap<>(counts).forEach((rule, counted) -> rules.raw(rule, countsOf(counted)));
        return rules.render();
    }

    private static String countsOf(AtomicLongArray counted) {
        Json.ObjectWriter object = Json.object();
        for (Count count : Count.values()) {
            object.value(count.wireName(), counted.get(count.ordinal()));
        }
        return object.render();
    }
}
