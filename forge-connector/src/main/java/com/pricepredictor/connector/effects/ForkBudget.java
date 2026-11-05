package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.effects.directed.KeywordEdit;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * How many forks one game may still take, of each kind.
 *
 * <p>Every kind spends its budget when the fork is <b>taken</b>, before the
 * score check, so a discarded fork still counts: otherwise a systematically
 * failing copy would retry until the game ended. Directed forks (FR-012)
 * have a budget of their own beside the other two, so a run that enables
 * them takes no intervention or probe fewer.
 */
final class ForkBudget {

    /** At most this many forks may target one real resolution (FR-040). */
    static final int MAX_FORKS_PER_RESOLUTION = 2;

    /**
     * At most this many protection forks a game (FR-058), on a run taking
     * interventions: a budget of their own rather than a share of
     * {@code --interventions-per-game}, which the forced resolutions spend
     * from the game's first boundary on, before any creature is out.
     */
    static final int MAX_PROTECTION_FORKS_PER_GAME = 1;

    private final int interventionsPerGame;
    private final int probesPerGame;
    private final List<String> probeKeywords;
    private final int directedPerGame;

    private int interventionsUsed;
    private int probesUsed;
    private int directedUsed;
    private int protectionUsed;
    private final Map<Integer, Integer> forksPerResolution = new HashMap<>();

    ForkBudget(CollectionCaps caps) {
        this.interventionsPerGame = caps.interventionsPerGame();
        this.probesPerGame = caps.probesPerGame();
        this.probeKeywords = List.copyOf(caps.probeKeywords());
        this.directedPerGame = caps.directedForksPerGame();
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
                        KeywordEdit.normalize(keyword))) {
            return false;
        }
        return probesUsed < probesPerGame;
    }

    /** Spend one probe. */
    void spendProbe() {
        probesUsed++;
    }

    /** Whether another directed fork may be taken this game ({@code --directed-forks-per-game}). */
    boolean mayDirect() {
        return directedUsed < directedPerGame;
    }

    /** Spend one directed fork, kept or discarded. */
    void spendDirected() {
        directedUsed++;
    }

    /**
     * Whether a protection fork may be taken: never on a run taking no
     * interventions, and at most {@value #MAX_PROTECTION_FORKS_PER_GAME} a game.
     */
    boolean mayProtect() {
        return interventionsPerGame > 0 && protectionUsed < MAX_PROTECTION_FORKS_PER_GAME;
    }

    /** Spend one protection fork, kept or discarded. */
    void spendProtection() {
        protectionUsed++;
    }
}
