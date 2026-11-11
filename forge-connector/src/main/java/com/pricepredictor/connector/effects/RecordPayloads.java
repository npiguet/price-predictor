package com.pricepredictor.connector.effects;

import com.google.common.base.CaseFormat;
import com.pricepredictor.connector.json.Json;
import forge.game.spellability.SpellAbility;

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
    static String eventsPayload(List<EffectEvent> events) {
        return Json.object().raw("events", eventArray(events)).render();
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
    static String resolutionPayload(
            List<EffectEvent> events, List<ConditionsMet> conditions,
            ActivationOutcome outcome) {
        return Json.object()
                .raw("events", eventArray(events))
                .raw("conditions_met", ConditionsMet.toJsonArray(conditions))
                .value("outcome", outcome)
                .render();
    }

    /**
     * ``{<combat fields>, "events": [...]}`` — an observed damage step, its
     * combat as {@link CombatShape#writeTo} writes it, each damage event
     * naming its dealer ({@link EventAttribution#withCombatSources}).
     */
    static String combatPayload(CombatShape combat, List<EffectEvent> events) {
        return combat.writeTo(Json.object())
                .raw("events", eventArray(EventAttribution.withCombatSources(events, combat)))
                .render();
    }

    /**
     * A damage-step probe's combat payload, which also names what it
     * perturbed: the keyword it stripped and the creature it stripped it from.
     * Its damage events name their dealers as the observed record's do
     * (FR-035), so the two read alike.
     */
    static String probePayload(
            CombatShape combat, List<EffectEvent> events,
            String keyword, String carrier) {
        return combat.writeTo(Json.object())
                .raw("events", eventArray(EventAttribution.withCombatSources(events, combat)))
                .raw("probed_keyword", Json.string(keyword))
                .raw("probed_entity", Json.string(carrier))
                .render();
    }

    /** The events that happened, rendered: one filed before it was told apart is left out if it did not. */
    private static String eventArray(List<EffectEvent> events) {
        return Json.array(events.stream().filter(EffectEvent::happened).toList(),
                EffectEvent::toJson);
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
    static String rewritePayload(
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
    static String costPayload(List<EffectEvent> events, PaidCosts costs) {
        return Json.object()
                .raw("events", eventArray(events))
                .raw("costs", costs.toJson())
                .render();
    }
}
