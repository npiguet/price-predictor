package com.pricepredictor.connector.effects;

import forge.ai.simulation.GameCopier;
import forge.ai.simulation.GameStateEvaluator;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Stage three: records for what observation cannot reach.
 *
 * <p>Two kinds of fork, and the flags tell them apart on their own.
 *
 * <p>An <b>interventional resolution</b> forces an ability Forge's AI never
 * chose to use. Some abilities are never cast in any game the AI plays — it
 * judges them bad, or cannot afford them, or never draws them into a board
 * where they do anything — and no amount of observation will produce a record
 * for them. The fork resolves the ability with chosen targets and modes,
 * routing an unaffordable candidate through the play-without-paying path, and
 * writes the <b>effect half only</b>: there was no real activation, so there is
 * no cost half to link to.
 *
 * <p>A <b>damage-step probe</b> re-runs one combat with a keyword stripped, to
 * isolate what that keyword did. It is built only for keywords gate 2 routed to
 * it, and taken only when {@code --probe-keywords} names one — the flag is the
 * runtime switch, separate from the build decision, so a checkout that has the
 * machinery still takes no fork unless asked.
 *
 * <p><b>Every fork is score-checked against the live game before any
 * perturbation.</b> A copy that already disagrees with the game it came from is
 * measuring the copier rather than the ability, and a discarded fork still
 * counts against its budget — otherwise a systematically-failing copy would
 * retry forever.
 */
public final class ForkCollector {

    /** At most this many forks may target one real resolution (FR-040). */
    public static final int MAX_FORKS_PER_RESOLUTION = 2;

    private final Game game;
    private final RecordShardWriter writer;
    private final String gameId;
    private final AttributionMode mode;
    private final int interventionsPerGame;
    private final int probesPerGame;
    private final List<String> probeKeywords;
    private final Random random;

    private int interventionsUsed;
    private int probesUsed;
    private final Map<Integer, Integer> forksPerResolution = new HashMap<>();
    private int discarded;
    private long recordsWritten;

    public ForkCollector(
            Game game, RecordShardWriter writer, String gameId,
            PatchedCollectors.CollectionCaps caps, long seed) {
        this.game = game;
        this.writer = writer;
        this.gameId = gameId;
        this.mode = AttributionMode.detect();
        this.interventionsPerGame = caps.interventionsPerGame();
        this.probesPerGame = caps.probesPerGame();
        this.probeKeywords = List.copyOf(caps.probeKeywords());
        // Seeded so both branches of a probe see the same shuffles: a
        // difference between them has to be the keyword, not the draw.
        this.random = new Random(seed);
    }

    public long recordsWritten() {
        return recordsWritten;
    }

    /** Forks created and thrown away by the score check. */
    public int discardedForks() {
        return discarded;
    }

    // ── budgets ─────────────────────────────────────────────────────────

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
    public boolean mayIntervene(int realResolutionId) {
        if (interventionsUsed >= interventionsPerGame) {
            return false;
        }
        return forksPerResolution.getOrDefault(realResolutionId, 0)
                < MAX_FORKS_PER_RESOLUTION;
    }

    /**
     * Whether another probe fork may be taken.
     *
     * <p>With {@code --probe-keywords} unset, never — whatever the build state.
     * The flag is the runtime switch; whether the probe code exists at all is
     * the build decision gate 2 drives, and the two are deliberately separate
     * so a checkout that has the machinery does not start using it by default.
     */
    public boolean mayProbe(String keyword) {
        if (probeKeywords.isEmpty() || !probeKeywords.contains(keyword)) {
            return false;
        }
        return probesUsed < probesPerGame;
    }

    // ── the score check ─────────────────────────────────────────────────

    /**
     * Verify a fresh copy scores the same as the game it came from.
     *
     * <p>Run <b>before any perturbation</b>: a copy that already disagrees is
     * measuring the copier, and a record from it would be noise dressed as a
     * counterfactual. A mismatch logs a warning, discards the fork, and
     * <b>still counts against the budget</b> — otherwise a systematically
     * failing copy would retry until the game ended.
     */
    public boolean scoreCheck(Game fork, Player perspective) {
        try {
            GameStateEvaluator evaluator = new GameStateEvaluator();
            int original = evaluator.getScoreForGameState(game, perspective).value;
            Player forkPerspective = matchingPlayer(fork, perspective);
            if (forkPerspective == null) {
                return false;
            }
            int copied = evaluator
                    .getScoreForGameState(fork, forkPerspective).value;
            if (original != copied) {
                System.err.println(
                        "Effect records: discarding a fork whose score "
                                + copied + " disagrees with the live game's "
                                + original + "; it would measure the copier "
                                + "rather than the ability");
                return false;
            }
            return true;
        } catch (RuntimeException e) {
            System.err.println(
                    "Effect records: fork score check failed (" + e.getMessage()
                            + "); discarding");
            return false;
        }
    }

    private static Player matchingPlayer(Game fork, Player original) {
        for (Player candidate : fork.getPlayers()) {
            if (candidate.getId() == original.getId()) {
                return candidate;
            }
        }
        return null;
    }

    // ── interventional resolutions ──────────────────────────────────────

    /**
     * Fork the game, force {@code ability} to resolve, and record the effect.
     *
     * @param realResolutionId the real resolution this fork varies, for the
     *                         per-resolution cap
     * @return whether a record was written
     */
    public boolean intervene(SpellAbility ability, int realResolutionId) {
        if (!mayIntervene(realResolutionId)) {
            return false;
        }
        // Counted before the score check, so a discarded fork still spends its
        // budget.
        interventionsUsed++;
        forksPerResolution.merge(realResolutionId, 1, Integer::sum);

        Game fork;
        try {
            fork = new GameCopier(game).makeCopy();
        } catch (RuntimeException e) {
            discarded++;
            return false;
        }
        Player perspective = ability.getActivatingPlayer() != null
                ? ability.getActivatingPlayer()
                : game.getPhaseHandler().getPlayerTurn();
        if (perspective == null || !scoreCheck(fork, perspective)) {
            discarded++;
            return false;
        }

        // The fork's own state, not the live game's: the intervention changed
        // the board, and recording the original would describe a situation the
        // resolution never saw.
        SnapshotBuilder snapshots = new SnapshotBuilder(fork, new int[]{
                SnapshotBuilder.TIER_REFERENCED,
                SnapshotBuilder.TIER_CORE,
                SnapshotBuilder.TIER_UNREFERENCED_STACK,
                // Tier 4 arrives with stage three: unreferenced hand and
                // graveyard, which a forced resolution needs because the
                // intervention chose from cards nobody was going to play.
                SnapshotBuilder.TIER_UNREFERENCED_HAND_GRAVEYARD,
        });
        emit(new EffectRecord(
                writer.nextRecordId(), writer.runId(),
                RecordShardWriter.timestamp(), gameId,
                EffectRecord.KIND_RESOLUTION, mode)
                .moment(EffectRecord.MOMENT_RESOLUTION)
                .interventional(true)
                .fork(true)
                // No link_id: the effect half is written alone, because there
                // was no real activation to pair it with.
                .actor(SnapshotBuilder.playerId(perspective))
                .ability(keysOf(ability))
                .state(snapshots.toJson(ability, referencedOf(ability)))
                .payload(EffectRecord.eventsPayload(List.of())));
        return true;
    }

    // ── damage-step probes ──────────────────────────────────────────────

    /**
     * Record a probe branch as an ordinary combat record.
     *
     * <p>{@code fork = true}, {@code interventional = false} — which is what
     * distinguishes a probe from an intervention by flags alone — and
     * {@code mirror_of} naming the real record it varies.
     *
     * <p>The real-versus-fork <b>difference is never computed here</b> and never
     * becomes a training target: it is a diagnostic about the model, computed at
     * evaluation time, and training on it would teach the model to reproduce its
     * own errors.
     */
    public boolean recordProbeBranch(
            String keyword, String mirrorOfRecordId, String snapshotJson,
            String payloadJson, String actorPlayerId) {
        if (!mayProbe(keyword)) {
            return false;
        }
        probesUsed++;
        emit(new EffectRecord(
                writer.nextRecordId(), writer.runId(),
                RecordShardWriter.timestamp(), gameId,
                EffectRecord.KIND_COMBAT, mode)
                .fork(true)
                .interventional(false)
                .mirrorOf(mirrorOfRecordId)
                .actor(actorPlayerId)
                .state(snapshotJson)
                .payload(payloadJson));
        return true;
    }

    /**
     * The seeded source both probe branches draw from.
     *
     * <p>Forge's own restore of the random source is not in a {@code finally},
     * so a branch that throws would leave the live game with the probe's
     * sequence. Callers must install this and restore the original in a
     * {@code finally} of their own.
     */
    public Random seededRandom() {
        return random;
    }

    // ── plumbing ────────────────────────────────────────────────────────

    private static List<ProvenanceKey> keysOf(SpellAbility ability) {
        ProvenanceKey key = ProvenanceKey.of(ability);
        return key == null ? List.of() : List.of(key);
    }

    private static List<Card> referencedOf(SpellAbility ability) {
        List<Card> referenced = new ArrayList<>();
        if (ability != null && ability.getHostCard() != null) {
            referenced.add(ability.getHostCard());
        }
        return referenced;
    }

    private void emit(EffectRecord record) {
        writer.write(record.toJson());
        recordsWritten++;
    }
}
