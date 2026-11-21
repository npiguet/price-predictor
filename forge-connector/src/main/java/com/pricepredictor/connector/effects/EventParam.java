package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.WireNamed;

/**
 * The keys an event's {@code params} are written under: the one place the
 * connector spells them. {@code EVENT_PARAMS} in {@code events.py} is the
 * Python half, each event type's row of these names; a key the schema does not
 * list fails every record that carries it, and {@code
 * test_event_schema_completeness.py} reads this table to check each one.
 *
 * <p>An enum rather than a constants class, so that {@link EffectEvent#param}
 * takes only a name from the table and a new bare string key cannot compile.
 * The order is alphabetical; it is not the order a params map is written in,
 * which is the order of the {@code param} calls.
 */
public enum EventParam implements WireNamed {
    ABILITIES("abilities"),
    ABILITY("ability"),
    AMOUNT("amount"),
    ATTACHED_TO("attached_to"),
    BLOCKED("blocked"),
    BLOCKERS("blockers"),
    CAUSE("cause"),
    CHARACTERISTICS("characteristics"),
    CHOICE_KIND("choice_kind"),
    COLORS("colors"),
    COLORS_REMOVED("colors_removed"),
    COMBAT("combat"),
    CONTROLLER("controller"),
    COUNT("count"),
    COUNTER_TYPE("counter_type"),
    DAMAGE_TYPE("damage_type"),
    DEFENDER("defender"),
    DELTA("delta"),
    DETACHED_FROM("detached_from"),
    FLAG("flag"),
    FROM_ZONE("from_zone"),
    HIDDEN("hidden"),
    KEYWORDS("keywords"),
    LAYERS("layers"),
    LIBRARY_POSITION("library_position"),
    MANA_BY_COLOR("mana_by_color"),
    MANA_COST("mana_cost"),
    MANA_VALUE("mana_value"),
    MODE("mode"),
    NAME("name"),
    OUT("out"),
    OWNER("owner"),
    PHASE("phase"),
    POWER_DELTA("power_delta"),
    REASON("reason"),
    REGENERABLE("regenerable"),
    REMOVED("removed"),
    RESULTS("results"),
    SET_PT("set_pt"),
    SIDES("sides"),
    SOURCE("source"),
    TARGETS("targets"),
    TO("to"),
    TO_STATE("to_state"),
    TO_ZONE("to_zone"),
    TOKEN_SCRIPT_ID("token_script_id"),
    TOUGHNESS_DELTA("toughness_delta"),
    TYPES_ADDED("types_added"),
    TYPES_REMOVED("types_removed"),
    VALUE("value");

    private final String wireName;

    EventParam(String wireName) {
        this.wireName = wireName;
    }

    /** The key as the Python side reads it. */
    @Override
    public String wireName() {
        return wireName;
    }
}
