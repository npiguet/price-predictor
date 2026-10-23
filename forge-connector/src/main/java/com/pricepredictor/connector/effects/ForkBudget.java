package com.pricepredictor.connector.effects;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * How many forks one game may still take, of each kind.
 *
 * <p>Both kinds spend their budget when the fork is <b>taken</b>, before the
 * score check, so a discarded fork still counts: otherwise a systematically
 * failing copy would retry until the game ended.
 */
final class ForkBudget {

    /** At most this many forks may target one real resolution (FR-040). */
    static final int MAX_FORKS_PER_RESOLUTION = 2;

    private final int interventionsPerGame;
    private final int probesPerGame;
    private final List<String> probeKeywords;

    private int interventionsUsed;
    private int probesUsed;
    private final Map<Integer, Integer> forksPerResolution = new HashMap<>();

    ForkBudget(CollectionCaps caps) {
        this.interventionsPerGame = caps.interventionsPerGame();
        this.probesPerGame = caps.probesPerGame();
        this.probeKeywords = List.copyOf(caps.probeKeywords());
    }

    /**
     * Whether another interventional fork may be taken.
     *
     * <p>Two budgets, both binding: the per-game one from
     * {@code --interventions-per-game}, and a fixed cap of
     * {@value #MAX_FORKS_PER_RESOLUTION} forks per real resolution that no flag
     * can raise. The second exists because forking the same resolution twice
     * already gives both counterfactuals worth having; a third would cost a
     * game's simulation for a third variation on one board.
     */
    boolean mayIntervene(int realResolutionId) {
        if (interventionsUsed >= interventionsPerGame) {
            return false;
        }
        return forksPerResolution.getOrDefault(realResolutionId, 0)
                < MAX_FORKS_PER_RESOLUTION;
    }

    /** Spend one intervention on {@code realResolutionId}, against both budgets. */
    void spendIntervention(int realResolutionId) {
        interventionsUsed++;
        forksPerResolution.merge(realResolutionId, 1, Integer::sum);
    }

    /**
     * Whether another probe fork may be taken.
     *
     * <p>With {@code --probe-keywords} empty, never — whatever the build state.
     * The flag is the runtime switch; whether the probe code exists at all is
     * the build decision gate 2 drives, and the two are deliberately separate.
     * A worker started with no {@code effect.probe.keywords} property reads
     * the list as empty; every collecting supervisor sets it, naming all eight
     * damage-step keywords unless the run narrows them.
     */
    boolean mayProbe(String keyword) {
        // Normalized before the check: the flag names keywords the way the
        // corpus spells them (first_strike) and a caller holding a card's
        // keyword has Forge's (First Strike). Compared directly, every probe
        // of a keyword spelled differently would silently decline here.
        if (keyword == null || probeKeywords.isEmpty()
                || !probeKeywords.contains(
                        ProbeScheduler.normalizeKeyword(keyword))) {
            return false;
        }
        return probesUsed < probesPerGame;
    }

    /** Spend one probe. */
    void spendProbe() {
        probesUsed++;
    }
}
