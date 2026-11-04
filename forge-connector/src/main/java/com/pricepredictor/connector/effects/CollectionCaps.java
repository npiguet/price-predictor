package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.WorkerProperties;
import com.pricepredictor.connector.WorkerProperties.OnMalformed;

import java.util.List;
import java.util.Optional;

/**
 * The caps and budgets every collecting supervisor shares (FR-028).
 *
 * @param playabilityRate stage 1 of decision sampling (FR-033b): the share of
 *                enumerated decision candidates evaluated
 * @param zoneCandidateRate the share of zone-bucket candidates written, both
 *                stages together ({@link DecisionSample#ZONE})
 * @param plainCandidateRate the share of plain-bucket candidates written,
 *                both stages together ({@link DecisionSample#PLAIN})
 * @param aimedAt the rules a {@code collect-coverage} or {@code
 *                collect-variants} run aims at (FR-009a), stamped on every
 *                record it writes as {@code aimed_at}, possibly empty; absent
 *                on a {@code match-outcomes} run, whose records carry none
 * @param countDroppedEvents whether each game's events that reached no record
 *                are written beside the shard for the supervisor to sum
 *                (FR-041, {@link DroppedEvents})
 * @param directedForkRules the rules directed forks are taken for, in the
 *                order that breaks a tie between two rules matching one
 *                boundary (FR-010a); empty takes none
 * @param directedForksPerGame the directed forks one game may take, kept or
 *                discarded, beside the other fork budgets (FR-012)
 * @param directedForkProbability the chance a boundary where an enabled
 *                rule's precondition holds forks, in (0, 1]; the supervisor
 *                refuses any other value before a worker starts (FR-014)
 */
public record CollectionCaps(
        int manaCap,
        double playabilityRate,
        double zoneCandidateRate,
        double plainCandidateRate,
        int interventionsPerGame,
        int probesPerGame,
        List<String> probeKeywords,
        double randomSeatShare,
        double randomSeatProbability,
        Optional<List<String>> aimedAt,
        boolean countDroppedEvents,
        List<String> directedForkRules,
        int directedForksPerGame,
        double directedForkProbability) {

    public CollectionCaps {
        directedForkRules = List.copyOf(directedForkRules);
    }

    public static CollectionCaps defaults() {
        return new CollectionCaps(
                1, 0.01, 0.0001, 0.001, 2, 2, List.of(), 0.0, 0.0, Optional.empty(), false,
                List.of(), 2, 0.25);
    }

    public static final String MANA_CAP = "effect.mana.cap";
    public static final String PLAYABILITY_RATE = "effect.playability.rate";
    public static final String ZONE_CANDIDATE_RATE = "effect.zone.candidate.rate";
    public static final String PLAIN_CANDIDATE_RATE = "effect.plain.candidate.rate";
    public static final String INTERVENTIONS_PER_GAME = "effect.interventions.per.game";
    public static final String PROBES_PER_GAME = "effect.probes.per.game";
    public static final String PROBE_KEYWORDS = "effect.probe.keywords";
    public static final String RANDOM_SEAT_SHARE = "effect.random.seat.share";
    public static final String RANDOM_SEAT_PROBABILITY = "effect.random.seat.probability";
    public static final String AIMED_AT = "effect.aimed.at";
    public static final String COUNT_DROPPED_EVENTS = "effect.count.dropped.events";
    public static final String DIRECTED_FORK_RULES = "effect.directed.fork.rules";
    public static final String DIRECTED_FORKS_PER_GAME = "effect.directed.forks.per.game";
    public static final String DIRECTED_FORK_PROBABILITY = "effect.directed.fork.probability";

    /**
     * Read from the {@code effect.*} system properties the supervisor sets.
     *
     * <p>Each falls back to its default rather than failing, because a
     * worker started without instrumentation sets none of them and one
     * started by an older supervisor may set only some. A malformed value
     * keeps the default too, and says so once on stderr.
     */
    public static CollectionCaps fromSystemProperties() {
        CollectionCaps defaults = defaults();
        OnMalformed policy = OnMalformed.WARN_AND_DEFAULT;
        return new CollectionCaps(
                WorkerProperties.intProperty(MANA_CAP, defaults.manaCap(), policy),
                WorkerProperties.doubleProperty(
                        PLAYABILITY_RATE, defaults.playabilityRate(), policy),
                WorkerProperties.doubleProperty(
                        ZONE_CANDIDATE_RATE, defaults.zoneCandidateRate(), policy),
                WorkerProperties.doubleProperty(
                        PLAIN_CANDIDATE_RATE, defaults.plainCandidateRate(), policy),
                WorkerProperties.intProperty(
                        INTERVENTIONS_PER_GAME, defaults.interventionsPerGame(), policy),
                WorkerProperties.intProperty(
                        PROBES_PER_GAME, defaults.probesPerGame(), policy),
                WorkerProperties.listProperty(PROBE_KEYWORDS),
                // The random seat's two knobs (FR-021). A share of matches
                // and a per-decision P: both are drawn in the JVM, so the
                // supervisor can only name them. The Python side refuses a
                // share without a P before any worker starts (FR-026); a
                // worker reads what it is given and defaults the rest.
                WorkerProperties.doubleProperty(
                        RANDOM_SEAT_SHARE, defaults.randomSeatShare(), policy),
                WorkerProperties.doubleProperty(
                        RANDOM_SEAT_PROBABILITY, defaults.randomSeatProbability(), policy),
                // Present, even blank, on every coverage and variant run and
                // on no other: its presence is what marks the run's records
                // aimed (FR-009a), its value the rules they were aimed at.
                WorkerProperties.presentListProperty(AIMED_AT),
                WorkerProperties.flag(COUNT_DROPPED_EVENTS),
                // The directed forks (FR-012). Unset, as on a worker an older
                // supervisor started, no rule is named and none is taken.
                WorkerProperties.listProperty(DIRECTED_FORK_RULES),
                WorkerProperties.intProperty(
                        DIRECTED_FORKS_PER_GAME, defaults.directedForksPerGame(), policy),
                WorkerProperties.doubleProperty(
                        DIRECTED_FORK_PROBABILITY, defaults.directedForkProbability(), policy));
    }

    /**
     * Whether any directed fork may be taken: a rule is named and the budget
     * allows one.
     */
    public boolean directedForksEnabled() {
        return !directedForkRules.isEmpty() && directedForksPerGame > 0;
    }

    /** Whether any match of this run may seat a random player at all. */
    public boolean randomSeatEnabled() {
        return randomSeatShare > 0.0;
    }

    /**
     * Whether any probe fork should be taken at all.
     *
     * <p>With no keywords named, none is — whatever the build state. The
     * flag is the runtime switch, separate from the build decision gate 2
     * drives.
     */
    public boolean probesEnabled() {
        return !probeKeywords.isEmpty();
    }
}
