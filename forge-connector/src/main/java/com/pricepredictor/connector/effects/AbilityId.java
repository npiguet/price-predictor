package com.pricepredictor.connector.effects;

import forge.game.spellability.SpellAbility;

/**
 * The id Forge keys a stack item by: the ability's id, which {@code
 * MagicStack} finds an item by and which the resolution and the cost half
 * both carry.
 */
record AbilityId(int value) {

    /** The id of {@code ability}. */
    static AbilityId of(SpellAbility ability) {
        return new AbilityId(ability.getId());
    }
}
