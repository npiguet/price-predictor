package com.pricepredictor.connector.effects;

import forge.game.Game;

import java.lang.reflect.InvocationHandler;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.StringJoiner;

/**
 * The four record kinds the engine patch unlocks.
 *
 * <p>{@code rewrite}, {@code trigger}, {@code playability} and {@code mana}
 * records all need a hook at a point the bus never publishes: what a replacement
 * received before it rewrote it, a trigger condition that evaluated to false, a
 * legality verdict the AI computed and discarded, a mana ability that resolved
 * on the inline path. Each is installed reflectively and simply does not run on
 * an unpatched checkout.
 *
 * <p>Two caps live here rather than in the supervisor, because both are
 * per-worker-process quantities the Python side cannot see: {@code --mana-cap}
 * counts resolution records per unique mana-ability text in this JVM, and
 * {@code --playability-rate} samples the {@code decision}-subkind logging points
 * (the {@code attackers} and {@code blockers} ones are always logged, because
 * they fire once per combat rather than once per candidate).
 *
 * <p>{@code continuous} records take no cap at all: coalescing per stable board
 * is itself the cap. A static that applies on every recompute would otherwise
 * write a record per priority pass, and none of them would say anything the
 * first did not.
 */
public final class PatchedCollectors implements AutoCloseable {

    private final Game game;
    private final RecordShardWriter writer;
    private final SnapshotBuilder snapshots;
    private final AttributionMode mode;
    private final String gameId;
    private final CollectionCaps caps;

    /** Per-unique-mana-text counts, for {@code --mana-cap}. */
    private final Map<String, Integer> manaRecords = new HashMap<>();
    /** Boards a continuous static has already been recorded on, for coalescing. */
    private final Set<String> coalescedBoards = new LinkedHashSet<>();
    private final List<String> installed = new ArrayList<>();
    private final java.util.Random sampler;

    private long recordsWritten;

    /** The caps and budgets every collecting supervisor shares (FR-028). */
    public record CollectionCaps(
            int manaCap,
            double playabilityRate,
            int interventionsPerGame,
            int probesPerGame,
            List<String> probeKeywords) {

        public static CollectionCaps defaults() {
            return new CollectionCaps(2000, 0.1, 2, 2, List.of());
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

    public PatchedCollectors(
            Game game, RecordShardWriter writer, String gameId,
            CollectionCaps caps, long seed) {
        this.game = game;
        this.writer = writer;
        this.gameId = gameId;
        this.caps = caps;
        this.snapshots = new SnapshotBuilder(
                game, new int[]{
                        SnapshotBuilder.TIER_REFERENCED,
                        SnapshotBuilder.TIER_CORE,
                        // Tier 3 arrives with the patch: unreferenced stack
                        // contents, which a rewrite or trigger record needs to
                        // describe what else was waiting to resolve.
                        SnapshotBuilder.TIER_UNREFERENCED_STACK,
                });
        this.mode = AttributionMode.detect();
        this.sampler = new java.util.Random(seed);
    }

    public long recordsWritten() {
        return recordsWritten;
    }

    /** Hooks that were found and installed; empty on a stock checkout. */
    public List<String> installedHooks() {
        return List.copyOf(installed);
    }

    /**
     * Install every hook the checkout offers.
     *
     * @return how many were found
     */
    public int install() {
        if (mode == AttributionMode.DEGRADED) {
            return 0;
        }
        if (PatchHooks.install(
                PatchHooks.REPLACEMENT_HANDLER, "setEffectRecordListener",
                rewriteHandler())) {
            installed.add("replacement");
        }
        if (PatchHooks.install(
                PatchHooks.TRIGGER_HANDLER, "setEffectRecordTriggerListener",
                triggerFireHandler())) {
            installed.add("trigger-fire");
        }
        if (PatchHooks.install(
                PatchHooks.AI_CONTROLLER, "setEffectRecordPlayabilityListener",
                playabilityHandler())) {
            installed.add("playability");
        }
        return installed.size();
    }

    @Override
    public void close() {
        PatchHooks.uninstall(
                PatchHooks.REPLACEMENT_HANDLER, "setEffectRecordListener");
        PatchHooks.uninstall(
                PatchHooks.TRIGGER_HANDLER, "setEffectRecordTriggerListener");
        PatchHooks.uninstall(
                PatchHooks.AI_CONTROLLER, "setEffectRecordPlayabilityListener");
        installed.clear();
    }

    // ── rewrite records ─────────────────────────────────────────────────

    /**
     * One record per replacement, carrying the event it received and the one it
     * produced.
     *
     * <p>Two replacements stacked on one event each get their own record, and
     * the second's incoming is the first's outgoing — which only holds because
     * the hook deep-copies the parameter map before the call.
     */
    private InvocationHandler rewriteHandler() {
        return (proxy, method, args) -> {
            if (!"onReplacement".equals(method.getName()) || args == null
                    || args.length < 3) {
                return null;
            }
            EffectEvent incoming = describeParams(args[1]);
            EffectEvent outgoing = describeParams(args[2]);
            emit(new EffectRecord(
                    writer.nextRecordId(), writer.runId(),
                    RecordShardWriter.timestamp(), gameId,
                    EffectRecord.KIND_REWRITE, mode)
                    .actor(activePlayerId())
                    .state(snapshots.toJson(null, List.of()))
                    .payload("{\"incoming\":" + incoming.toJson()
                            + ",\"outgoing\":" + outgoing.toJson() + "}"));
            return null;
        };
    }

    // ── trigger-fire records ────────────────────────────────────────────

    /**
     * One record per evaluated trigger condition, fired or not.
     *
     * <p>Non-fired negatives are what teach the condition; without them the
     * model would learn the base rate of a trigger type firing and nothing
     * about when. Drawn at roughly 1:1 by sampling the negatives, since a turn
     * evaluates far more conditions than it fires.
     */
    private InvocationHandler triggerFireHandler() {
        return (proxy, method, args) -> {
            if (!"onConditionEvaluated".equals(method.getName()) || args == null
                    || args.length < 3) {
                return null;
            }
            boolean fired = Boolean.TRUE.equals(args[2]);
            if (!fired && sampler.nextDouble() > NEGATIVE_SAMPLE_RATE) {
                return null;
            }
            EffectEvent event = describeParams(args[1]);
            emit(new EffectRecord(
                    writer.nextRecordId(), writer.runId(),
                    RecordShardWriter.timestamp(), gameId,
                    EffectRecord.KIND_TRIGGER, mode)
                    .actor(activePlayerId())
                    .state(snapshots.toJson(null, List.of()))
                    .payload("{\"event\":" + event.toJson()
                            + ",\"fired\":" + fired + "}"));
            return null;
        };
    }

    /**
     * Share of non-fired evaluations kept.
     *
     * <p>Tuned to land near 1:1 against the fired ones: a turn evaluates every
     * registered trigger against every event and fires a handful.
     */
    private static final double NEGATIVE_SAMPLE_RATE = 0.02;

    // ── playability records ─────────────────────────────────────────────

    /**
     * Rules-level verdicts only.
     *
     * <p>The AI's policy judgments never reach a record: they are its opinion
     * rather than the game's rules, and training on them would teach the model
     * Forge's play style instead of Magic.
     *
     * <p>Sampled at {@code --playability-rate} because the AI evaluates every
     * candidate at every priority, which would otherwise swamp the corpus. The
     * {@code attackers} and {@code blockers} subkinds are always logged: they
     * fire once per combat, not once per candidate.
     */
    private InvocationHandler playabilityHandler() {
        return (proxy, method, args) -> {
            if (!"onCandidate".equals(method.getName()) || args == null
                    || args.length < 4) {
                return null;
            }
            if (sampler.nextDouble() > caps.playabilityRate()) {
                return null;
            }
            // Snapshot defensively: a verdict can be abandoned mid-evaluation,
            // and the legality check mutates the checked ability's targets, so
            // nothing here may keep a reference to the ability object.
            boolean canPlay = Boolean.TRUE.equals(args[1]);
            boolean affordable = Boolean.TRUE.equals(args[2]);
            boolean hasLegalTarget = Boolean.TRUE.equals(args[3]);
            emit(new EffectRecord(
                    writer.nextRecordId(), writer.runId(),
                    RecordShardWriter.timestamp(), gameId,
                    EffectRecord.KIND_PLAYABILITY, mode)
                    .subkind("decision")
                    .actor(activePlayerId())
                    .state(snapshots.toJson(null, List.of()))
                    .payload("{\"candidates\":[{"
                            + "\"ability\":[]"
                            + ",\"verdict\":{\"can_play\":" + canPlay
                            + ",\"affordable\":" + affordable
                            + ",\"has_legal_target\":" + hasLegalTarget + "}"
                            + ",\"legal_targets\":[]"
                            + ",\"cost_after_adjustment\":{}"
                            + ",\"responsible_static\":[]}]}"));
            return null;
        };
    }

    // ── mana records ────────────────────────────────────────────────────

    /**
     * Whether this mana ability may still be recorded (FR-037).
     *
     * <p>Counted per unique mana-ability text, per worker process: a basic
     * land's tap ability resolves thousands of times a run and would otherwise
     * be most of the corpus.
     */
    public boolean allowManaRecord(String abilityText) {
        int seen = manaRecords.getOrDefault(abilityText, 0);
        if (seen >= caps.manaCap()) {
            return false;
        }
        manaRecords.put(abilityText, seen + 1);
        return true;
    }

    // ── continuous records ──────────────────────────────────────────────

    /**
     * Whether a continuous record for this static on this board is new.
     *
     * <p>Coalescing per {@code (game, static, board hash)} is the cap: a static
     * applies on every recompute, and a second record for the same board would
     * repeat the first exactly.
     */
    public boolean allowContinuousRecord(String staticKey, String boardHash) {
        return coalescedBoards.add(staticKey + "@" + boardHash);
    }

    // ── plumbing ────────────────────────────────────────────────────────

    /**
     * A patched hook's parameter map as an event.
     *
     * <p>The map's contents are Forge's own {@code AbilityKey} enum, which this
     * side does not link against, so the description is by string. That loses
     * type information the Python reader does not use.
     */
    private EffectEvent describeParams(Object params) {
        EffectEvent event = new EffectEvent(EffectEvent.ZONE_CHANGE);
        if (params instanceof Map<?, ?> map) {
            StringJoiner keys = new StringJoiner(",");
            for (Object key : map.keySet()) {
                keys.add(String.valueOf(key));
            }
            event.param("cause", keys.toString());
        }
        return event;
    }

    private String activePlayerId() {
        var phase = game.getPhaseHandler();
        if (phase == null || phase.getPlayerTurn() == null) {
            return null;
        }
        return SnapshotBuilder.playerId(phase.getPlayerTurn());
    }

    private void emit(EffectRecord record) {
        writer.write(record.toJson());
        recordsWritten++;
    }
}
