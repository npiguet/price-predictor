package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.WireNamed;
import com.pricepredictor.connector.effects.directed.DirectedTag;
import com.pricepredictor.connector.json.Json;
import java.util.List;
import java.util.Map;
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

    /**
     * The record generation this connector writes (FR-018): the first envelope
     * member after {@code record_id} on every record, and what names a gen-3
     * shard to a reader. Gen-2 and gen-1 records carry no such member.
     */
    public static final int GENERATION = 3;

    private final RecordHeader header;
    private final RecordKind kind;
    private final Moment moment;
    /** A {@link PlayabilitySubkind} or a {@link TurnActionSubkind}; null on every other kind. */
    private final WireNamed subkind;
    private final AttributionMode mode;
    private String linkId;
    private String mirrorOf;
    private boolean interventional;
    /**
     * The kind of fork that wrote this record, written as {@code fork_kind};
     * null on a live record, which writes no {@code fork_kind}. {@code fork}
     * is written from it: a record is a fork's exactly when it names its kind.
     */
    private ForkKind forkKind;
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
    /**
     * The rules a collection run aimed at (FR-009a): the run's
     * {@code --directed-fork-rules} on every record of a coverage or variant
     * run, possibly empty, and the one rule a directed fork forked for. Null on
     * every other record, and absent from the JSON then. Collection metadata.
     */
    private List<String> aimedAt;
    /** What a directed fork changed; null on every record but a directed fork's. */
    private DirectedTag directed;
    /**
     * What the copier left out of a fork's copy, by kind, to how many; empty
     * on a live record and on a fork whose copy left nothing out, and absent
     * from the JSON then.
     */
    private Map<String, Integer> copyDropped = Map.of();
    private String actorPlayer;
    private List<ProvenanceKey> ability;
    private UnresolvedReason abilityUnresolved;
    private RecordState state = RecordState.EMPTY;
    private RecordPayload payload = RecordPayload.EMPTY;

    /**
     * A record of one of the four kinds that take neither a moment nor a
     * subkind.
     *
     * <p>The other three are made by {@link #resolution}, {@link #playability}
     * and {@link #turnAction}: a resolution record needs its moment and the
     * other two their subkind, and taking them at construction is what makes
     * a record missing one impossible to build rather than merely wrong.
     */
    public EffectRecord(RecordHeader header, RecordKind kind, AttributionMode mode) {
        this(header, kind, null, null, mode);
        if (kind.isQualified()) {
            throw new IllegalArgumentException(
                    "a " + kind.wireName() + " record is made with its moment or subkind");
        }
    }

    private EffectRecord(
            RecordHeader header, RecordKind kind, Moment moment, WireNamed subkind,
            AttributionMode mode) {
        this.header = header;
        this.kind = Objects.requireNonNull(kind, "kind");
        this.moment = moment;
        this.subkind = subkind;
        this.mode = Objects.requireNonNull(mode, "mode");
    }

    /** A {@code resolution} record: one half of a resolution pair. */
    public static EffectRecord resolution(
            RecordHeader header, Moment moment, AttributionMode mode) {
        return new EffectRecord(header, RecordKind.RESOLUTION,
                Objects.requireNonNull(moment, "moment"), null, mode);
    }

    /** A {@code playability} record: one rules verdict. */
    public static EffectRecord playability(
            RecordHeader header, PlayabilitySubkind subkind, AttributionMode mode) {
        return new EffectRecord(header, RecordKind.PLAYABILITY,
                null, Objects.requireNonNull(subkind, "subkind"), mode);
    }

    /** A {@code turn_action} record: the events of one turn-based action (FR-036). */
    public static EffectRecord turnAction(
            RecordHeader header, TurnActionSubkind subkind, AttributionMode mode) {
        return new EffectRecord(header, RecordKind.TURN_ACTION,
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

    /** A record {@code kind} of fork wrote; null for a live record. */
    public EffectRecord fork(ForkKind kind) {
        this.forkKind = kind;
        return this;
    }

    public EffectRecord randomSeat(boolean value) {
        this.randomSeat = value;
        return this;
    }

    /**
     * Only the {@code blockers} subkind sets this: a gen-2 {@code attackers}
     * record carried it too, and no what-if attackers record exists (FR-037).
     */
    public EffectRecord whatIf(Boolean value) {
        this.whatIf = value;
        return this;
    }

    /** The rules this record's run or fork aimed at; null leaves the member out. */
    public EffectRecord aimedAt(List<String> rules) {
        this.aimedAt = rules == null ? null : List.copyOf(rules);
        return this;
    }

    /** Whether this record already names the rules it was collected for. */
    boolean isAimed() {
        return aimedAt != null;
    }

    /** A directed fork's variant and operation; null leaves the member out. */
    public EffectRecord directed(DirectedTag tag) {
        this.directed = tag;
        return this;
    }

    /**
     * What the copier left out of this fork's copy, by kind as
     * {@code Forks.dropped} spells it; empty leaves the member out.
     */
    public EffectRecord copyDropped(Map<String, Integer> dropped) {
        this.copyDropped = dropped == null ? Map.of() : new java.util.TreeMap<>(dropped);
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

    /**
     * This record seeks no acting line (the game-end record): {@code ability}
     * is the empty array and there is no reason, since nothing failed to
     * resolve. A record that does seek a line goes through {@link #ability}.
     */
    public EffectRecord seeksNoLine() {
        this.ability = List.of();
        this.abilityUnresolved = null;
        return this;
    }

    EffectRecord state(RecordState value) {
        this.state = value;
        return this;
    }

    EffectRecord payload(RecordPayload value) {
        this.payload = value;
        return this;
    }

    public String recordId() {
        return header.recordId();
    }

    /** Whether this is an effect half under {@code link}: one of the halves its item's resolution wrote. */
    boolean isEffectHalfOf(String link) {
        return moment == Moment.RESOLUTION && link != null && link.equals(linkId);
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
        return state.json();
    }

    String payloadJson() {
        return payload.json();
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
        if (directed != null && !signsAsADirectedFork()) {
            throw new IllegalStateException(
                    "a directed fork is a non-interventional fork naming the rule it aimed at");
        }
        Json.ObjectWriter json = Json.object()
                .value("record_id", header.recordId())
                .value("generation", GENERATION)
                .value("run_id", header.runId())
                .value("timestamp", header.timestamp())
                .value("game_id", header.gameId())
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
                .value("fork", forkKind != null);
        if (forkKind != null) {
            // Beside the fork flags it says more about; absent on a live record.
            json.value("fork_kind", forkKind);
        }
        json.value("synthetic", false)
                // Right after synthetic on every record, and what_if right
                // after it where set (blockers alone): the order is part of the
                // contract (record-schema-delta.md), and the reader names a
                // shard's generation by the first field's presence.
                .value("random_seat", randomSeat);
        writeOptionalMembers(json);
        return json
                .value("actor_player", actorPlayer)
                .raw("ability", abilityJson())
                .value("ability_unresolved", abilityUnresolved)
                .raw("state", state.json())
                .raw("payload", payload.json())
                .render();
    }

    /** The line as the shard takes it: rendered, beside the facts its cap reads. */
    ShardLine toLine() {
        return new ShardLine(toJson(), facts());
    }

    /**
     * What the shard writer's cap reads of this record: its game, the turn
     * its state was read on, its acting script, and whether it is a live
     * record holding the game's outcome: one whose own events end the game.
     * A rewrite that replaced a loss or a win holds none, the game going on;
     * a fork's record holding them holds the fork's outcome, not the game's.
     */
    RecordFacts facts() {
        return new RecordFacts(
                header.gameId(), state.turn(), actingScript(),
                forkKind == null && payload.namesOutcome());
    }

    /** The acting line's script file, else the line the payload names; null for neither. */
    private String actingScript() {
        if (ability == null || ability.isEmpty()) {
            return payload.namedScript();
        }
        return ability.get(0).scriptFile();
    }

    /** Whether the stamps are a directed fork's: a non-interventional fork naming its rule. */
    private boolean signsAsADirectedFork() {
        return aimedAt != null && forkKind != null && !interventional;
    }

    /** The members written only where set, in their contract order. */
    private void writeOptionalMembers(Json.ObjectWriter json) {
        if (whatIf != null) {
            json.value("what_if", whatIf);
        }
        if (aimedAt != null) {
            json.raw("aimed_at", Json.stringArray(aimedAt));
        }
        if (directed != null) {
            json.raw("directed", directed.toJson());
        }
        if (!copyDropped.isEmpty()) {
            json.value("copy_dropped", copyDropped);
        }
    }

    String abilityJson() {
        return ability == null ? "null" : ProvenanceKey.toJsonArray(ability);
    }
}
