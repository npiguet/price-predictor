package com.pricepredictor.connector.effects;

import forge.game.spellability.SpellAbility;

/**
 * One stack item's bracket in the live collector, from its cast until it
 * resolves or leaves the stack without resolving.
 *
 * <p>Built at the cast. A cast whose stack top holds no ability builds none,
 * and nothing is filed for it.
 */
final class OpenResolution {
    final SpellAbility ability;
    final ProvenanceKey.Resolved keys;
    final String actor;
    /** The board at the cast, paired at the resolution with the refs read then. */
    final Snapshot state;
    /** This resolution's chosen modes, split as they resolve; its own and no other's. */
    final ModeSplit modes = new ModeSplit();
    /**
     * What the bus filed for this item: the consequences of its cast — what
     * its card's move onto the stack set off, the state-based check after it,
     * until the next action — and what it does while it resolves.
     */
    final EventBuffer events = new EventBuffer();

    /**
     * Whether the controller turned something down while this bracket was open.
     *
     * <p>Scoped to the bracket rather than matched against the declined
     * ability's id, because Forge resolves a copy of the ability it was handed
     * and an id comparison would silently never match — the same reason
     * {@link EventAttribution} counts a clause upward from the pointer instead
     * of downward from the root. What keeps the flag from over-claiming is the
     * second half of the test in {@link BusBracketCollector}'s
     * {@code wasDeclined}: a resolution that produced any outcome at all is
     * {@code resolved} however many optional clauses inside it were declined.
     */
    boolean declined;

    OpenResolution(
            SpellAbility ability, ProvenanceKey.Resolved keys, String actor, Snapshot state) {
        this.ability = ability;
        this.keys = keys;
        this.actor = actor;
        this.state = state;
    }

    /** Whether this is the bracket of the ability the engine names by id. */
    boolean is(int abilityId) {
        return ability.getId() == abilityId;
    }

    /**
     * Whether a clause is on this resolution's card; see
     * {@link BusBracketCollector#beginMode}.
     */
    boolean hosts(SpellAbility clause) {
        return clause != null && clause.getHostCard() == ability.getHostCard();
    }
}
