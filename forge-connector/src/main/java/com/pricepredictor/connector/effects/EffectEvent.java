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
    public static final String LIFE_CHANGE = "life_change";
    public static final String DAMAGE_DEALT = "damage_dealt";
    public static final String DAMAGE_PREVENTED = "damage_prevented";
    public static final String POISON_CHANGE = "poison_change";
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
    public static final String LIBRARY_REORDERED = "library_reordered";
    public static final String LIBRARY_SHUFFLED = "library_shuffled";
    public static final String MANA_PRODUCED = "mana_produced";
    public static final String SPELL_CAST = "spell_cast";
    public static final String SPELL_COUNTERED = "spell_countered";
    public static final String ATTACKERS_DECLARED = "attackers_declared";
    public static final String BLOCKERS_DECLARED = "blockers_declared";
    public static final String COMBAT_ENDED = "combat_ended";
    public static final String COIN_FLIPPED = "coin_flipped";
    public static final String DICE_ROLLED = "dice_rolled";
    public static final String DAY_NIGHT_CHANGED = "day_night_changed";
    public static final String SPEED_CHANGED = "speed_changed";

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
