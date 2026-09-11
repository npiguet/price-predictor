package com.pricepredictor.connector.effects;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.StringJoiner;

/**
 * One attributed outcome, in the schema's event shape.
 *
 * <p>{@code {type, subjects, params, duration, attributed_to}}. The type
 * vocabulary is owned by the Python side's {@code event_schema.py}; this class
 * carries the names as strings so a new type needs no change here, and the
 * completeness test on the other side is what keeps the two in step.
 *
 * <p>Types name **observable outcomes**, not the script API that produced them:
 * two effects that both move a card to the graveyard emit {@code zone_change},
 * and the model learns one field group for it.
 */
public final class EffectEvent {

    // The canonical type names this collector emits. The full vocabulary lives
    // in effects/domain/event_schema.py; these are the ones stage one reaches.
    public static final String ZONE_CHANGE = "zone_change";
    public static final String DESTROYED = "destroyed";
    public static final String SACRIFICED = "sacrificed";
    public static final String REGENERATED = "regenerated";
    public static final String PHASED = "phased";
    public static final String TOKEN_CREATED = "token_created";
    public static final String PERMANENT_COPIED = "permanent_copied";
    public static final String LIFE_CHANGE = "life_change";
    public static final String DAMAGE_DEALT = "damage_dealt";
    public static final String DAMAGE_PREVENTED = "damage_prevented";
    public static final String POISON_CHANGE = "poison_change";
    public static final String ENERGY_CHANGE = "energy_change";
    public static final String RADIATION_CHANGE = "radiation_change";
    public static final String COUNTER_CHANGE = "counter_change";
    public static final String PT_CHANGE = "pt_change";
    public static final String TAPPED = "tapped";
    public static final String UNTAPPED = "untapped";
    public static final String ATTACHED = "attached";
    public static final String UNATTACHED = "unattached";
    public static final String CONTROL_CHANGE = "control_change";
    public static final String FACE_CHANGE = "face_change";
    /** The generic outcome, for an engine mode the vocabulary has no member for. */
    public static final String STATE_FLAG_CHANGE = "state_flag_change";
    public static final String KEYWORD_CHANGE = "keyword_change";
    public static final String TYPE_CHANGE = "type_change";
    public static final String COLOR_CHANGE = "color_change";
    public static final String CARD_DRAWN = "card_drawn";
    public static final String CARD_DISCARDED = "card_discarded";
    public static final String CARD_MILLED = "card_milled";
    public static final String CARD_LOOKED_AT = "card_looked_at";
    public static final String CARD_REVEALED = "card_revealed";
    public static final String LIBRARY_REORDERED = "library_reordered";
    public static final String LIBRARY_SHUFFLED = "library_shuffled";
    public static final String MANA_PRODUCED = "mana_produced";
    public static final String MANA_LOST = "mana_lost";
    public static final String SPELL_CAST = "spell_cast";
    public static final String SPELL_COUNTERED = "spell_countered";
    public static final String SPELL_COPIED = "spell_copied";
    public static final String ATTACKERS_DECLARED = "attackers_declared";
    public static final String BLOCKERS_DECLARED = "blockers_declared";
    public static final String BECAME_BLOCKED = "became_blocked";
    public static final String COMBAT_ENDED = "combat_ended";
    public static final String COIN_FLIPPED = "coin_flipped";
    public static final String CLASH_RESOLVED = "clash_resolved";
    public static final String VOTE_TAKEN = "vote_taken";
    public static final String PILES_MADE = "piles_made";
    public static final String DUNGEON_VENTURED = "dungeon_ventured";
    public static final String DICE_ROLLED = "dice_rolled";
    public static final String DAY_NIGHT_CHANGED = "day_night_changed";
    public static final String PLAYER_WON = "player_won";
    public static final String PLAYER_LOST = "player_lost";
    public static final String SPEED_CHANGED = "speed_changed";
    public static final String TURN_ADDED = "turn_added";
    public static final String TURN_SKIPPED = "turn_skipped";
    public static final String PHASE_ADDED = "phase_added";
    public static final String PHASE_SKIPPED = "phase_skipped";
    public static final String X_CHANGED = "x_changed";
    public static final String OWNERSHIP_CHANGE = "ownership_change";
    public static final String CARD_MADE = "card_made";
    public static final String RESTRICTION_CHANGE = "restriction_change";
    public static final String CHOICE_MADE = "choice_made";
    public static final String DAMAGE_HEALED = "damage_healed";
    /**
     * What an {@code Animate}/{@code AnimateAll} clause was scripted to grant.
     *
     * <p>Carries the <b>scripted</b> text of the clause, not what the layer
     * system finally applied -- that is not knowable at the clause. The
     * applied result is recorded separately, in a {@code continuous} record's
     * contributions, so a reader of this event alone should not mistake it
     * for the applied result; the pair is complete only across both record
     * kinds.
     */
    public static final String ABILITY_CHANGE = "ability_change";
    /**
     * What an {@code Effect} clause was scripted to add as a continuous
     * effect.
     *
     * <p>Same limitation as {@link #ABILITY_CHANGE}: this is the scripted
     * {@code StaticAbilities$} text, not the applied result, which lives in a
     * {@code continuous} record's contributions instead.
     */
    public static final String CONTINUOUS_EFFECT_CREATED = "continuous_effect_created";
    public static final String TARGETS_CHANGED = "targets_changed";
    public static final String DELAYED_TRIGGER_CREATED = "delayed_trigger_created";
    /**
     * A strict subset of the {@code rewrite} record for the same replacement --
     * see the comment above {@code EventType.REPLACEMENT_APPLIED} in
     * {@code event_schema.py} for why the two must not be read as independent
     * evidence of the same replacement.
     */
    public static final String REPLACEMENT_APPLIED = "replacement_applied";
    public static final String MONARCH_CHANGED = "monarch_changed";
    public static final String RING_TEMPTS = "ring_tempts";
    public static final String REMOVED_FROM_COMBAT = "removed_from_combat";
    public static final String INITIATIVE_TAKEN = "initiative_taken";
    public static final String TEXT_CHANGE = "text_change";
    public static final String TURN_ENDED = "turn_ended";

    private final String type;
    private final List<String> subjects = new ArrayList<>();
    private final Map<String, Object> params = new LinkedHashMap<>();
    private String duration;
    private String attributedTo;

    public EffectEvent(String type) {
        this.type = type;
    }

    public EffectEvent subject(String id) {
        if (id != null) {
            subjects.add(id);
        }
        return this;
    }

    public EffectEvent param(String key, Object value) {
        if (value != null) {
            params.put(key, value);
        }
        return this;
    }

    /**
     * The object the engine names as having caused this outcome, as a ref.
     *
     * <p><b>One channel, one spelling.</b> Every collector writes the cause
     * through this method and nowhere else, so the trait-derived side (a
     * replacement's or a trigger's run-parameter map) and the bus-derived side
     * (the event the bus delivered) put the same kind of value under the same
     * key. The Python schema keeps {@code cause} in {@code PROVENANCE_PARAMS}
     * rather than in one type's row, so every event type accepts it and no
     * per-type table has to be widened for a new one.
     *
     * <p>The value is an entity or player ref — {@code "E24"}, {@code "P1"} —
     * the same identity {@code state.entities} carries, never a description and
     * never a list of the run-parameter keys the map happened to hold.
     *
     * <p>Why it is not decoration: {@code subjects} says who an outcome
     * happened to and the normalized params say how much, which leaves two
     * attackers each dealing 1 damage to the same player serializing
     * byte-identically. Those are distinct events that must not be deduplicated,
     * and the cause is what tells them apart.
     *
     * <p>A null ref writes nothing rather than a placeholder. Forge genuinely
     * names no cause for a phase change, an untap, a block or a declared
     * attack, and an honest absence is what a reader can act on.
     */
    public EffectEvent cause(String ref) {
        return param("cause", ref);
    }

    /** ``end_of_turn``, ``permanent``, … for a continuous outcome. */
    public EffectEvent duration(String value) {
        this.duration = value;
        return this;
    }

    /**
     * The sub-ability link this outcome came from, or null for the root line.
     *
     * <p>An event attributed to a link the sidecar does not map falls back to
     * the root line rather than being dropped, so attribution is never lossy.
     */
    public EffectEvent attributedTo(String link) {
        this.attributedTo = link;
        return this;
    }

    public String type() {
        return type;
    }

    public List<String> subjects() {
        return subjects;
    }

    /** The normalized params a reader can check a specific key of, rather than {@link #toJson()}. */
    public Map<String, Object> params() {
        return params;
    }

    public String toJson() {
        StringJoiner subjectJson = new StringJoiner(",", "[", "]");
        for (String subject : subjects) {
            subjectJson.add(Json.string(subject));
        }
        return "{\"type\":" + Json.string(type)
                + ",\"subjects\":" + subjectJson
                + ",\"params\":" + renderParams()
                + ",\"duration\":" + Json.string(duration)
                + ",\"attributed_to\":" + Json.string(attributedTo) + "}";
    }

    private String renderParams() {
        StringJoiner joiner = new StringJoiner(",", "{", "}");
        for (Map.Entry<String, Object> entry : params.entrySet()) {
            joiner.add(Json.string(entry.getKey()) + ":" + renderValue(entry.getValue()));
        }
        return joiner.toString();
    }

    private static String renderValue(Object value) {
        if (value instanceof Boolean || value instanceof Number) {
            return String.valueOf(value);
        }
        if (value instanceof Map<?, ?> map) {
            StringJoiner joiner = new StringJoiner(",", "{", "}");
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                joiner.add(
                        Json.string(String.valueOf(entry.getKey())) + ":"
                                + renderValue(entry.getValue()));
            }
            return joiner.toString();
        }
        if (value instanceof Iterable<?> items) {
            StringJoiner joiner = new StringJoiner(",", "[", "]");
            for (Object item : items) {
                joiner.add(renderValue(item));
            }
            return joiner.toString();
        }
        return Json.string(String.valueOf(value));
    }
}
