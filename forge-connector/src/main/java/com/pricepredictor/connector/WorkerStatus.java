package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.AttributionMode;
import com.pricepredictor.connector.effects.DirectedForkTally;
import com.pricepredictor.connector.effects.ForkRefusals;
import com.pricepredictor.connector.effects.HookFailures;
import com.pricepredictor.connector.effects.PatchHooks;
import com.pricepredictor.connector.json.Json;

import java.util.List;

/**
 * The worker's status line: {@link #LINE}, then one JSON object, the only
 * line of a worker's log the supervisor reads counts from
 * ({@code contracts/cli.md} § Match worker stop protocol).
 *
 * <p>Printed at start-up and after each match whose counts changed, every
 * count a running total of this JVM, so the last line a worker printed holds
 * its lifetime's counts. The object's members:
 * <ul>
 *   <li>{@code lifetime}: the JVM lifetime's token, the one its shard is
 *       named by, which tells one lifetime's lines from the next.</li>
 *   <li>{@code hooks}: the attribution {@code mode} and the {@code gaps} of
 *       the hook inventory ({@link PatchHooks#gaps()}, empty when whole).</li>
 *   <li>{@code pinned_preferences}: the four Forge preferences the engine
 *       runs with ({@link PinnedForgePreferences#inEffect()}).</li>
 *   <li>{@code hook_failures}: the connector's counts ({@link HookFailures#counts})
 *       and Forge's own per slot ({@link PatchHooks#forgeFailureCounts}), a
 *       listener's failure in both.</li>
 *   <li>{@code fork_refusals}: the forks that declined for an expected
 *       reason, by kind and reason ({@link ForkRefusals}), no failures.</li>
 *   <li>{@code directed}: the directed fork tally ({@link DirectedForkTally}).</li>
 * </ul>
 *
 * <p>Every character outside ASCII is written as a {@code \}{@code uXXXX}
 * escape, so the line reads the same whatever encoding the JVM writes its
 * stderr in.
 */
final class WorkerStatus {

    /** How the status line starts; one JSON object follows. */
    static final String LINE = "Effect records: status: ";

    private final String lifetime;
    private final String hooks;
    private final String pinnedPreferences;
    private Counts printed;

    WorkerStatus(String lifetime, AttributionMode mode, List<String> gaps,
            PinnedForgePreferences preferences) {
        this.lifetime = lifetime;
        this.hooks = Json.object().value("mode", mode).raw("gaps", Json.stringArray(gaps)).render();
        this.pinnedPreferences = Json.object()
                .value("mulligan_rule", preferences.mulliganRule())
                .value("filtered_hands", preferences.filteredHands())
                .value("performance_mode", preferences.performanceMode())
                .value("ui_language", preferences.uiLanguage())
                .render();
    }

    /** The status of this JVM, once Forge is initialised: its mode, gaps and preferences. */
    static WorkerStatus ofThisJvm(String lifetime) {
        return new WorkerStatus(lifetime, AttributionMode.detect(), PatchHooks.gaps(),
                PinnedForgePreferences.inEffect());
    }

    /** Print the line, whatever the counts: the worker's start-up. */
    void atStartup() {
        print(Counts.now());
    }

    /**
     * Print the line when a count changed since the last one printed: after
     * a match however it ended, and as the worker stops.
     */
    void afterMatch() {
        Counts counts = Counts.now();
        if (!counts.equals(printed)) {
            print(counts);
        }
    }

    private void print(Counts counts) {
        System.err.println(line(counts));
        printed = counts;
    }

    String line(Counts counts) {
        return LINE + asciiOnly(Json.object()
                .value("lifetime", lifetime)
                .raw("hooks", hooks)
                .raw("pinned_preferences", pinnedPreferences)
                .raw("hook_failures", counts.hookFailures())
                .raw("fork_refusals", counts.forkRefusals())
                .raw("directed", counts.directed())
                .render());
    }

    /** {@code json} with every character past ASCII escaped, which only a string can hold. */
    private static String asciiOnly(String json) {
        StringBuilder out = new StringBuilder(json.length());
        json.chars().forEach(c -> out.append(c < 0x7f ? Character.toString(c)
                : String.format("\\u%04x", c)));
        return out.toString();
    }

    /** The counts a line carries, each rendered JSON, compared to tell a change. */
    record Counts(String hookFailures, String forkRefusals, String directed) {

        static Counts now() {
            return new Counts(
                    Json.object()
                            .value("connector", HookFailures.counts())
                            .value("forge", PatchHooks.forgeFailureCounts())
                            .render(),
                    Json.value(ForkRefusals.counts()),
                    DirectedForkTally.lifetimeJson());
        }
    }
}
