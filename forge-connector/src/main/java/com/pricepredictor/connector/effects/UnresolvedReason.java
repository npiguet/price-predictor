package com.pricepredictor.connector.effects;

/**
 * Why a record that names an acting line carries no key for it, as its
 * {@code ability_unresolved} spells it.
 *
 * <p>A closed vocabulary because the point of the field is to separate "no
 * printed line exists" from "the resolver failed". Python's
 * {@code UNRESOLVED_REASONS} in {@code records.py} rejects any record whose
 * reason is not in its set, and the corpus it guards cannot be rebuilt.
 */
public enum UnresolvedReason {

    /** The trait names no state, so there is nothing to index it against. */
    NO_CARD_STATE("no_card_state"),

    /** The trait is none of the five kinds a printed line can be. */
    UNKNOWN_KIND("unknown_kind"),

    /** The acting card is engine-built and has no script file in any tree. */
    ENGINE_EFFECT("engine_effect"),

    /**
     * The trait came from a keyword the card does not print, and the permanent
     * that handed it out is no longer reachable.
     *
     * <p>Truthful rather than alarming: a keyword an anthem or a pump spell
     * granted names no printed line <em>on the recipient</em>, so there is
     * nothing to key even when nothing is broken. The static-granted case does
     * key — through the grantor — so what is left here is the spell-granted
     * one, where Forge keeps no back-reference to the ability that granted it.
     */
    GRANTED_KEYWORD("granted_keyword"),

    /**
     * The trait was built at runtime from another card's script and lives on a
     * card other than the one whose text it came from.
     *
     * <p>An aura's or an equipment's granted trigger, an {@code Animate}'s
     * granted ability, and an effect card's own traits are all parsed from an
     * SVar of the donor. Forge points them at the donor's {@code CardState},
     * which is enough to name the donor's script file, but the SVar is not a
     * member of any of that state's trait slices and nothing records which
     * printed line named the SVar — so the card is known and the line is not.
     * Distinct from {@link #UNINDEXABLE} because it is a property of how Forge
     * grants traits, not evidence of a resolver that regressed.
     */
    GRANTED_TRAIT("granted_trait"),

    /**
     * A keyable kind, printed on the state it names, whose position in that
     * state could not be established.
     *
     * <p>The alarm. Every shape that legitimately has no printed line has a
     * reason of its own, so this one means the resolver regressed.
     */
    UNINDEXABLE("unindexable");

    private final String wireName;

    UnresolvedReason(String wireName) {
        this.wireName = wireName;
    }

    /** The {@code ability_unresolved} a record carries. */
    public String wireName() {
        return wireName;
    }
}
