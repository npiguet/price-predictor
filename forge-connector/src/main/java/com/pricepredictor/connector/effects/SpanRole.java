package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.WireNamed;

/**
 * The role a prose span plays in its line, as a sidecar's {@code role_spans}
 * spell it.
 *
 * <p>The encoder adds one role embedding per token, and its {@code ROLES} in
 * {@code provenance.py} and {@code ROLE_ORDER} in {@code ability_encoder.py}
 * are the other half of this vocabulary.
 */
public enum SpanRole implements WireNamed {
    COST("cost"),
    EFFECT("effect"),
    TRIGGER_CONDITION("trigger-condition"),
    TARGET_SPEC("target-spec");

    private final String wireName;

    SpanRole(String wireName) {
        this.wireName = wireName;
    }

    /** The {@code role} a span carries. */
    @Override
    public String wireName() {
        return wireName;
    }
}
