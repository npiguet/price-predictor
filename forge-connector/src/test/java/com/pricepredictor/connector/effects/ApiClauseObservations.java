package com.pricepredictor.connector.effects;

import forge.game.spellability.SpellAbility;

/**
 * What the {@code ApiEvents*Test} classes share: a clause observed through
 * the real rules.
 */
final class ApiClauseObservations {

    private ApiClauseObservations() {
    }

    /**
     * A clause's rule and what its memo takes before the clause resolves,
     * through the real rules; {@code finish()} builds the events after it.
     */
    static ApiEvents.ClauseObservation<?> observe(SpellAbility sa) {
        return ApiEvents.observe(sa, ApiEvents::ruleFor);
    }
}
