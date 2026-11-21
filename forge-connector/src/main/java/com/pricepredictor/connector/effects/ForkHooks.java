package com.pricepredictor.connector.effects;

import forge.game.spellability.SpellAbility;

/**
 * What a damage-step probe hears of the hooks while it runs: the events
 * its fork's clauses promise and the outcomes its effects report.
 *
 * <p>The hooks are JVM statics that the live game's handlers own, and
 * each handler drops a fork's callbacks on game identity — rightly, for
 * its own records. A probe needs the events the live record takes from
 * the hooks rather than the bus, the clauses' and the outcomes' (damage
 * prevented), or its record would lack what its live twin holds. So the
 * live handlers hand those to whatever probe is running on their thread
 * ({@link Forks#forkHooks}), and {@code DamageStepProbe} installs one for
 * exactly the span of its damage step ({@link Forks#listeningForHooks}). A
 * forced resolution needs more, and has it all: its fork-bound recording
 * is handed every hook's callbacks ({@link ForcedResolution}).
 */
interface ForkHooks {

    /** Hears nothing: what {@link Forks#forkHooks} hands out while no probe listens. */
    ForkHooks NONE = new ForkHooks() {
    };

    /**
     * The event a clause of some other game promised by its effect API's
     * own parameters, told once the clause ran ({@code ApiEvents}): a
     * turn added, a choice made, a continuous effect created.
     */
    default void clauseEvent(SpellAbility clause, EffectEvent event) {
    }

    /**
     * What an effect of some other game computed and reported through
     * the outcome hook: a coin flipped, a token made, damage prevented.
     *
     * @param ability the ability the effect named, or null for one with
     *                none (combat damage a shield prevented)
     */
    default void outcome(SpellAbility ability, EffectEvent event) {
    }
}
