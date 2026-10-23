package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.WireNamed;

/**
 * Which question a {@code playability} record answers.
 *
 * <p>Every playability record has one and no other kind does, which is why a
 * record is given its subkind when it is made, by
 * {@link EffectRecord#playability}. {@code PlayabilitySubkind} in
 * {@code records.py} is the other half of this vocabulary.
 */
public enum PlayabilitySubkind implements WireNamed {
    /** A candidate the AI weighed at priority, with the verdict on it. */
    DECISION("decision"),
    /** Who may attack, declared at the attackers step. */
    ATTACKERS("attackers"),
    /** Who may block an attacker, declared at the blockers step. */
    BLOCKERS("blockers");

    private final String wireName;

    PlayabilitySubkind(String wireName) {
        this.wireName = wireName;
    }

    /** The {@code subkind} a record carries. */
    @Override
    public String wireName() {
        return wireName;
    }
}
