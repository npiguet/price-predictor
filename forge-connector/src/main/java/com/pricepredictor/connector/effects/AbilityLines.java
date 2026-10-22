package com.pricepredictor.connector.effects;

import forge.game.spellability.SpellAbility;

/**
 * What every copy of one card's ability has in common, for a collector that
 * counts abilities rather than ability objects.
 */
final class AbilityLines {

    private AbilityLines() {
    }

    /**
     * The ability's API and its host's name.
     *
     * <p>Not the instance's id: every Mountain is its own object with the
     * same text, so a memory keyed by id would count each copy apart. The
     * mana cap counts by this, so it caps the whole class rather than each
     * copy, and the intervention scheduler forces each line once per game by
     * it, so a card redrawn is not forced again.
     */
    static String key(SpellAbility ability) {
        Object api = ability.getApi();
        return (api == null ? "?" : api.toString()) + "|" + ability.getHostCard().getName();
    }
}
