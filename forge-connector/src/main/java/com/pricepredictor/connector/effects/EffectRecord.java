package com.pricepredictor.connector.effects;

import java.util.List;
import java.util.Objects;

/**
 * One record line, assembled field by field.
 *
 * <p>The envelope is frozen before stage-one collection begins, so this builder
 * writes exactly the keys the schema names and nothing else. Later stages widen
 * the corpus — new kinds become reachable, new snapshot tiers appear — but never
 * redefine a field, which is what lets a stage-one corpus stay trainable
 * alongside stage-four records.
 *
 * <p>The envelope only: what goes under {@code payload} is built by
 * {@link RecordPayloads}, one factory per payload shape.
 */
public final class EffectRecord {

    private final String recordId;
    private final String runId;
    private final String timestamp;
    private final String gameId;
    private final RecordKind kind;
    private final Moment moment;
    private final PlayabilitySubkind subkind;
    private final AttributionMode mode;
    private String linkId;
    private String mirrorOf;
    private boolean interventional;
    private boolean fork;
    /**
     * Whether {@code actor_player} is the match's random seat (FR-030).
     *
     * <p>Collection metadata, like the three flags above it: it says how the
     * decision behind this record was made, never what the board was, and the
     * reader keeps it out of the model. Written on every record, so a shard
     * that carries it on one record carries it on all — that presence is what
     * names a gen-2 shard to a reader (FR-033).
     */
    private boolean randomSeat;
    /**
     * Real decision or what-if query, on the two legality subkinds only
     * (FR-031): {@code false} is a declaration the engine acted on, {@code
     * true} an answer the AI computed while weighing a combat and discarded.
     * The collector reports legality at the declaration itself and so writes
     * only {@code false}; {@code true} is read from shards collected inside
     * the AI. Null everywhere else, and absent from the JSON then.
     */
    private Boolean whatIf;
    private String actorPlayer;
    private List<ProvenanceKey> ability;
    private UnresolvedReason abilityUnresolved;
    private String stateJson = "{}";
    private String payloadJson = "{}";

    /**
     * A record of one of the four kinds that take neither a moment nor a
     * subkind.
     *
     * <p>The other two are made by {@link #resolution} and {@link #playability}:
     * a resolution record needs its moment and a playability record its
     * subkind, and taking them at construction is what makes a record missing
     * one impossible to build rather than merely wrong.
     */
    public EffectRecord(
            String recordId, String runId, String timestamp, String gameId,
            RecordKind kind, AttributionMode mode) {
        this(recordId, runId, timestamp, gameId, kind, null, null, mode);
        if (kind == RecordKind.RESOLUTION || kind == RecordKind.PLAYABILITY) {
            throw new IllegalArgumentException(
                    "a " + kind.wireName() + " record is made with its moment or subkind");
        }
    }

    private EffectRecord(
            String recordId, String runId, String timestamp, String gameId,
            RecordKind kind, Moment moment, PlayabilitySubkind subkind,
            AttributionMode mode) {
        this.recordId = recordId;
        this.runId = runId;
        this.timestamp = timestamp;
        this.gameId = gameId;
        this.kind = Objects.requireNonNull(kind, "kind");
        this.moment = moment;
        this.subkind = subkind;
        this.mode = Objects.requireNonNull(mode, "mode");
    }

    /** A {@code resolution} record: one half of a resolution pair. */
    public static EffectRecord resolution(
            String recordId, String runId, String timestamp, String gameId,
            Moment moment, AttributionMode mode) {
        return new EffectRecord(recordId, runId, timestamp, gameId, RecordKind.RESOLUTION,
                Objects.requireNonNull(moment, "moment"), null, mode);
    }

    /** A {@code playability} record: one rules verdict. */
    public static EffectRecord playability(
            String recordId, String runId, String timestamp, String gameId,
            PlayabilitySubkind subkind, AttributionMode mode) {
        return new EffectRecord(recordId, runId, timestamp, gameId, RecordKind.PLAYABILITY,
                null, Objects.requireNonNull(subkind, "subkind"), mode);
    }

    /**
     * Joins a cost half to its effect halves; absent where there is no partner.
     *
     * <p>One effect half ordinarily, and one per chosen mode of a modal
     * resolution on a patched worker (FR-029d), all carrying the cost half's
     * id — the link widened from a pair to a fan without changing its type
     * or spelling.
     */
    public EffectRecord linkId(String value) {
        this.linkId = value;
        return this;
    }

    public EffectRecord mirrorOf(String value) {
        this.mirrorOf = value;
        return this;
    }

    public EffectRecord interventional(boolean value) {
        this.interventional = value;
        return this;
    }

    public EffectRecord fork(boolean value) {
        this.fork = value;
        return this;
    }

    public EffectRecord randomSeat(boolean value) {
        this.randomSeat = value;
        return this;
    }

    /** Only the {@code attackers} and {@code blockers} subkinds set this. */
    public EffectRecord whatIf(Boolean value) {
        this.whatIf = value;
        return this;
    }

    public EffectRecord actor(String playerId) {
        this.actorPlayer = playerId;
        return this;
    }

    /**
     * The player this record is about, for the collector that stamps
     * {@link #randomSeat} at the moment it writes: the actor is set where
     * the record is built, which for a held record is long before the write.
     */
    public String actorPlayer() {
        return actorPlayer;
    }

    /**
     * The acting line, or the reason there is none — both set from one answer.
     *
     * <p>The two fields are written together and only together, because they
     * are two halves of one statement and a record that got them out of step
     * would be unreadable: a reason beside a named line is a contradiction, and
     * an empty list with no reason is the ambiguity this field exists to
     * remove. An empty {@code ability} alone conflates "the Monarch has no
     * printed line in any tree, correctly" with "the resolver regressed".
     *
     * <p>A null answer is treated as {@code unknown_kind} rather than as no
     * lookup at all: a caller reaching this method did look, so the record owes
     * a reason.
     */
    public EffectRecord ability(ProvenanceKey.Resolved resolved) {
        if (resolved != null && resolved.key() != null) {
            this.ability = List.of(resolved.key());
            this.abilityUnresolved = null;
            return this;
        }
        this.ability = List.of();
        UnresolvedReason reason = resolved == null ? null : resolved.reason();
        this.abilityUnresolved = reason == null ? UnresolvedReason.UNKNOWN_KIND : reason;
        return this;
    }

    public EffectRecord state(String json) {
        this.stateJson = json;
        return this;
    }

    public EffectRecord payload(String json) {
        this.payloadJson = json;
        return this;
    }

    public String recordId() {
        return recordId;
    }

    /**
     * The rendered parts of what this record says, for {@link RecordDedup}.
     *
     * <p>A record is a duplicate of another when what it <b>says</b> repeats —
     * the same kind of record about the same line, the same board, the same
     * answer — and its id and timestamp always differ. Reading the rendered
     * blocks is how a coalescer asks that question without reparsing the line
     * it is about to write.
     */
    String stateJson() {
        return stateJson;
    }

    String payloadJson() {
        return payloadJson;
    }

    /**
     * The kind, with its moment or subkind where it has one: what keeps one
     * kind's record from coalescing with another's that renders the same
     * blocks.
     */
    String kindTag() {
        String tag = kind.wireName();
        if (moment != null) {
            tag += "/" + moment.wireName();
        }
        if (subkind != null) {
            tag += "/" + subkind.wireName();
        }
        return tag;
    }

    /** Why {@link #abilityJson} is empty, as written; null where it names a line. */
    String abilityUnresolvedWire() {
        return abilityUnresolved == null ? null : abilityUnresolved.wireName();
    }

    public String toJson() {
        Json.ObjectWriter json = Json.object()
                .value("record_id", recordId)
                .value("run_id", runId)
                .value("timestamp", timestamp)
                .value("game_id", gameId)
                .value("kind", kind)
                .value("moment", moment)
                .value("subkind", subkind)
                .value("link_id", linkId)
                .value("mirror_of", mirrorOf)
                // Frozen schema fields no collector sets: a reader still
                // expects them on every record.
                .value("variant_of", null)
                .value("mode", mode)
                .value("interventional", interventional)
                .value("fork", fork)
                .value("synthetic", false)
                // Right after synthetic on every record, and what_if right
                // after it on the two subkinds alone: the order is part of the
                // contract (record-schema-delta.md), and the reader names a
                // shard's generation by the first field's presence.
                .value("random_seat", randomSeat);
        if (whatIf != null) {
            json.value("what_if", whatIf);
        }
        return json
                .value("actor_player", actorPlayer)
                .raw("ability", abilityJson())
                .value("ability_unresolved", abilityUnresolved)
                .raw("state", stateJson)
                .raw("payload", payloadJson)
                .render();
    }

    String abilityJson() {
        return ability == null ? "null" : ProvenanceKey.toJsonArray(ability);
    }

    /**
     * {@link RecordPayloads#rewritePayload}, kept here for {@code RewriteChannel}
     * until it calls {@link RecordPayloads} directly.
     */
    public static String rewritePayload(
            EffectEvent incoming, EffectEvent outgoing, String result,
            List<ProvenanceKey> replacedBy) {
        return RecordPayloads.rewritePayload(incoming, outgoing, result, replacedBy);
    }

    /**
     * {@link RecordPayloads#rewriteResult}, kept here for {@code RewriteChannel}
     * until it calls {@link RecordPayloads} directly.
     */
    public static String rewriteResult(Object result) {
        return RecordPayloads.rewriteResult(result);
    }
}
