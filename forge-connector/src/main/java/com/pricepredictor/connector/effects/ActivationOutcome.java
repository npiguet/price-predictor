package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.WireNamed;

/**
 * What became of a cast, as a cost half's {@code outcome} spells it.
 *
 * <p>The reader's {@code ResolutionOutcome} in {@code records.py} is the other
 * half of this vocabulary. Only {@link #RESOLVED} and {@link #PARTIALLY_FIZZLED}
 * reach a resolution, so only they can carry a {@code link_id}.
 */
public enum ActivationOutcome implements WireNamed {
    RESOLVED("resolved"),
    FIZZLED("fizzled"),
    PARTIALLY_FIZZLED("partially_fizzled"),
    DECLINED("declined"),
    COUNTERED("countered");

    private final String wireName;

    ActivationOutcome(String wireName) {
        this.wireName = wireName;
    }

    /** The {@code outcome} a cost payload carries. */
    @Override
    public String wireName() {
        return wireName;
    }
}
