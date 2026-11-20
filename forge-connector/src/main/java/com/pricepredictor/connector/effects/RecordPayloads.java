package com.pricepredictor.connector.effects;

import com.google.common.base.CaseFormat;
import com.pricepredictor.connector.json.Json;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.List;

/**
 * What goes under a record's {@code payload}, one factory per shape.
 *
 * <p>The envelope around it is {@link EffectRecord}'s, and frozen; the payloads
 * are where the corpus widens. {@link #rewritePayload} is where that widening
 * happens, and it obeys the rule rather than bending it: {@code incoming} keeps
 * exactly the meaning it had, {@code outgoing} gains {@code null} the way a
 * field gains a sentinel, and {@code result} and {@code replaced_by} are new
 * keys. No already-collected record is reinterpreted — an old line that
 * carries no {@code result} reads as "written before this contract", which is
 * a third state and not one of the five results.
 */
final class RecordPayloads {

    private RecordPayloads() {
    }

    /** ``{"events": [...]}`` — a turn-based action's record. */
    static RecordPayload eventsPayload(List<EffectEvent> events) {
        List<EffectEvent> happened = happened(events);
        return holding(Json.object().raw("events", eventArray(happened)).render(), happened);
    }

    /**
     * ``{"events": [...], "conditions_met": [{segment, met}], "outcome": ...}``
     * — a resolution effect half: its events, whether each conditional clause
     * that resolved in it met its condition (FR-034), in resolution order,
     * empty where none did, and what became of the stack item (FR-042b).
     *
     * @param outcome the stack item's outcome, or null for a half that
     *                resolves no stack item — a mana ability's, the game-end
     *                rescue record — which writes {@code "outcome": null}
     */
    static RecordPayload resolutionPayload(
            List<EffectEvent> events, List<ConditionsMet> conditions,
            ActivationOutcome outcome) {
        List<EffectEvent> happened = happened(events);
        return holding(Json.object()
                .raw("events", eventArray(happened))
                .raw("conditions_met", ConditionsMet.toJsonArray(conditions))
                .value("outcome", outcome)
                .render(), happened);
    }

    /**
     * ``{<combat fields>, "events": [...]}`` — an observed damage step, its
     * combat as {@link CombatShape#writeTo} writes it, each damage event
     * naming its dealer ({@link EventAttribution#withCombatSources}).
     */
    static RecordPayload combatPayload(CombatShape combat, List<EffectEvent> events) {
        List<EffectEvent> happened = happened(EventAttribution.withCombatSources(events, combat));
        return holding(combat.writeTo(Json.object())
                .raw("events", eventArray(happened))
                .render(), happened);
    }

    /**
     * A damage-step probe's combat payload, which also names what it
     * perturbed: the keyword it stripped and the creature it stripped it from.
     * Its damage events name their dealers as the observed record's do
     * (FR-035), so the two read alike.
     */
    static RecordPayload probePayload(
            CombatShape combat, List<EffectEvent> events,
            String keyword, String carrier) {
        List<EffectEvent> happened = happened(EventAttribution.withCombatSources(events, combat));
        return holding(combat.writeTo(Json.object())
                .raw("events", eventArray(happened))
                .raw("probed_keyword", Json.string(keyword))
                .raw("probed_entity", Json.string(carrier))
                .render(), happened);
    }

    /** The events that happened: one filed before it was told apart is left out if it did not. */
    private static List<EffectEvent> happened(List<EffectEvent> events) {
        return events.stream().filter(EffectEvent::happened).toList();
    }

    private static String eventArray(List<EffectEvent> happened) {
        return Json.array(happened, EffectEvent::toJson);
    }

    /** A payload rendered from {@code events}, naming the game's outcome where one of them is. */
    private static RecordPayload holding(String json, List<EffectEvent> events) {
        return new RecordPayload(json, events.stream().anyMatch(RecordPayloads::isOutcome), null);
    }

    private static boolean isOutcome(EffectEvent event) {
        return event.type().isGameOutcome();
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
    static RecordPayload rewritePayload(
            EffectEvent incoming, EffectEvent outgoing, String result,
            List<ProvenanceKey> replacedBy) {
        return rewritePayload(incoming, outgoing, result, replacedBy, false, null, false, List.of());
    }

    /**
     * The same, with what a gen-3 rewrite record adds after
     * {@code replaced_by}, each only where it says something:
     * {@code incoming_unknown: true} where Forge could not copy the event as
     * it arrived ({@code incoming} is then the event as it left, and
     * {@code outgoing} null); and, for a damage replacement that set an
     * ability aside, {@code buffered} (that ability's keys),
     * {@code buffered_ran} (whether Forge ran it) and
     * {@code buffered_events} (the events of its run this record took).
     *
     * @param buffered the buffered ability's keys, or null when none was set aside
     */
    static RecordPayload rewritePayload(
            EffectEvent incoming, EffectEvent outgoing, String result,
            List<ProvenanceKey> replacedBy, boolean incomingUnknown,
            List<ProvenanceKey> buffered, boolean bufferedRan, List<EffectEvent> bufferedEvents) {
        var payload = Json.object()
                .raw("incoming", incoming.toJson())
                .raw("outgoing", outgoing == null ? "null" : outgoing.toJson())
                .value("result", result)
                .raw("replaced_by",
                        ProvenanceKey.toJsonArray(replacedBy == null ? List.of() : replacedBy));
        if (incomingUnknown) {
            payload.value("incoming_unknown", true);
        }
        List<EffectEvent> named = new ArrayList<>();
        named.add(incoming);
        if (outgoing != null) {
            named.add(outgoing);
        }
        if (buffered != null) {
            List<EffectEvent> ran = happened(bufferedEvents);
            payload.raw("buffered", ProvenanceKey.toJsonArray(buffered))
                    .value("buffered_ran", bufferedRan)
                    .raw("buffered_events", eventArray(ran));
            named.addAll(ran);
        }
        return holding(payload.render(), named);
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
    static String rewriteResult(Object result) {
        if (result == null) {
            return null;
        }
        return CaseFormat.UPPER_CAMEL.to(CaseFormat.LOWER_UNDERSCORE, String.valueOf(result));
    }

    /**
     * ``{"events": [...], "costs": {...}}`` — a resolution cost half
     * (FR-042a): what casting did, a spell card's {@code zone_change} to the
     * stack ({@code []} for an activated ability, which moves no card), and
     * what was paid. No {@code outcome}: what became of the item is its
     * effect half's.
     */
    static RecordPayload costPayload(List<EffectEvent> events, PaidCosts costs) {
        List<EffectEvent> happened = happened(events);
        return holding(Json.object()
                .raw("events", eventArray(happened))
                .raw("costs", costs.toJson())
                .render(), happened);
    }
}
