package com.pricepredictor.connector.effects;

import forge.game.spellability.SpellAbility;

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
 * <p>{@link #rewritePayload} is where that widening happens, and it obeys the
 * rule rather than bending it: {@code incoming} keeps exactly
 * the meaning it had, {@code outgoing} gains {@code null} the way a field gains
 * a sentinel, and {@code result} and {@code replaced_by} are new keys. No
 * already-collected record is reinterpreted — an old line that carries no
 * {@code result} reads as "written before this contract", which is a third
 * state and not one of the five results.
 */
public final class EffectRecord {

    private final String recordId;
    private final String runId;
    private final String timestamp;
    private final String gameId;
    private final RecordKind kind;
    private final Moment moment;
    private final PlayabilitySubkind subkind;
    private final String mode;
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
        this.mode = mode.wireName();
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
                .value("kind", kind.wireName())
                .value("moment", moment == null ? null : moment.wireName())
                .value("subkind", subkind == null ? null : subkind.wireName())
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
                .value("ability_unresolved", abilityUnresolvedWire())
                .raw("state", stateJson)
                .raw("payload", payloadJson)
                .render();
    }

    String abilityJson() {
        return ability == null ? "null" : ProvenanceKey.toJsonArray(ability);
    }

    /** ``{"events": [...]}`` — a resolution effect half. */
    public static String eventsPayload(List<EffectEvent> events) {
        return Json.object().raw("events", eventArray(events)).render();
    }

    /**
     * ``{<combat fields>, "events": [...]}`` — an observed damage step, its
     * combat as {@link CombatShape#fields} hands it over.
     */
    static String combatPayload(Map<String, Object> combatFields, List<EffectEvent> events) {
        return Json.object()
                .values(combatFields)
                .raw("events", eventArray(events))
                .render();
    }

    /**
     * A damage-step probe's combat payload, which also names what it
     * perturbed: the keyword it stripped and the creature it stripped it from.
     */
    static String probePayload(
            Map<String, Object> combatFields, List<EffectEvent> events,
            String keyword, String carrier) {
        return Json.object()
                .values(combatFields)
                .raw("events", eventArray(events))
                .raw("probed_keyword", Json.string(keyword))
                .raw("probed_entity", Json.string(carrier))
                .render();
    }

    private static String eventArray(List<EffectEvent> events) {
        return Json.array(events, EffectEvent::toJson);
    }

    /**
     * ``{"incoming": ..., "outgoing": ...|null, "result": ..., "replaced_by": []}``
     *
     * <p>The pair alone cannot express what a replacement does. Forge hardly
     * ever edits an event's parameter map: {@code ReplacementHandler} substitutes
     * by running the {@code ReplaceWith$} ability and returning a
     * {@code ReplacementResult}, and its {@code Prevented}, {@code Skipped} and
     * {@code NotReplaced} branches return without touching the map at all. So
     * "enters tapped", "if it would die, exile it instead" and "prevent that
     * damage" would all arrive here as two byte-identical halves.
     *
     * <p>{@code outgoing} is {@code null} — never a copy of {@code incoming} —
     * when the map was not edited in place. A record whose two halves read the
     * same says nothing; null says "not rewritten" outright, and the corpus
     * validator judges an identity pair at zero rather than letting it parse
     * quietly.
     *
     * @param incoming    the event as the replacement received it
     * @param outgoing    the rewritten event, or null where nothing was rewritten
     * @param result      the wire spelling from {@link #rewriteResult}, or null
     *                    where the engine returned no answer
     * @param replacedBy  the ability that stood in for the event; empty when
     *                    none ran, and always a list rather than null
     */
    public static String rewritePayload(
            EffectEvent incoming, EffectEvent outgoing, String result,
            List<ProvenanceKey> replacedBy) {
        return Json.object()
                .raw("incoming", incoming.toJson())
                .raw("outgoing", outgoing == null ? "null" : outgoing.toJson())
                .value("result", result)
                .raw("replaced_by",
                        ProvenanceKey.toJsonArray(replacedBy == null ? List.of() : replacedBy))
                .render();
    }

    /**
     * Forge's {@code ReplacementResult} in the spelling the corpus reads.
     *
     * <p>Lower snake case of the enum member, so {@code NotReplaced} becomes
     * {@code not_replaced}. Converted rather than looked up in a table of the
     * five known members on purpose: a sixth member added to Forge would take
     * the same rule and reach the Python reader, which refuses a value its own
     * closed enum does not have. That is a loud failure on the next shard, where
     * a table would have quietly written null and turned a new outcome into the
     * same "written before this contract" state an old shard reports.
     *
     * <p>Null in, null out. The engine has no answer to report in exactly one
     * case — the replacement threw, and the hook notifies anyway rather than
     * hiding it — and this side sees the same null when it is running against a
     * Forge jar whose listener predates the two extra arguments.
     */
    public static String rewriteResult(Object result) {
        if (result == null) {
            return null;
        }
        String name = String.valueOf(result);
        StringBuilder out = new StringBuilder(name.length() + 2);
        for (int i = 0; i < name.length(); i++) {
            char c = name.charAt(i);
            if (Character.isUpperCase(c)) {
                if (i > 0) {
                    out.append('_');
                }
                out.append(Character.toLowerCase(c));
            } else {
                out.append(c);
            }
        }
        return out.toString();
    }

    /**
     * ``{"costs": {...}, "outcome": "..."}`` — a resolution cost half.
     *
     * <p>The two halves become knowable at different moments: what was paid is
     * only readable while the cast event is being published, and whether the
     * spell resolved, fizzled or was countered is only readable later. Holding
     * the {@link PaidCosts} read at the cast is what lets one record carry both
     * without keeping a reference to a {@link SpellAbility} whose paid lists
     * the next activation will overwrite.
     */
    static String costPayload(PaidCosts costs, ActivationOutcome outcome) {
        return Json.object()
                .raw("costs", costs.toJson())
                .value("outcome", outcome.wireName())
                .render();
    }

}
