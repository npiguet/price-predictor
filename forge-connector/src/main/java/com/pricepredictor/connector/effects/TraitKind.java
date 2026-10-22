package com.pricepredictor.connector.effects;

import forge.game.CardTraitBase;
import forge.game.card.CardState;
import forge.game.replacement.ReplacementEffect;
import forge.game.spellability.SpellAbility;
import forge.game.staticability.StaticAbility;
import forge.game.trigger.Trigger;

import java.util.List;

/**
 * The five kinds of printed trait a provenance key can name, as its
 * {@code trait_kind} spells them.
 *
 * <p>Written into every record's {@code ability} and every sidecar's keys, and
 * the join between the two is by these spellings, so the converter and the
 * collectors take them from here rather than each writing its own.
 */
public enum TraitKind {
    KEYWORD("keyword"),
    SPELL("spell"),
    TRIGGER("trigger"),
    STATIC("static"),
    REPLACEMENT("replacement");

    private final String wireName;

    TraitKind(String wireName) {
        this.wireName = wireName;
    }

    /** The {@code trait_kind} a key carries. */
    public String wireName() {
        return wireName;
    }

    /**
     * The kind a runtime trait object belongs to, or null if it is none of them.
     *
     * <p>Never {@link #KEYWORD}: a keyword-derived trait is a trigger, a spell
     * or a static like any other, and only {@link CardTraitBase#getKeyword()}
     * says where it came from.
     */
    public static TraitKind of(CardTraitBase trait) {
        if (trait instanceof Trigger) return TRIGGER;
        if (trait instanceof ReplacementEffect) return REPLACEMENT;
        if (trait instanceof StaticAbility) return STATIC;
        if (trait instanceof SpellAbility) return SPELL;
        return null;
    }

    /**
     * The state's trait list for this kind, in declaration order.
     *
     * <p>Printed traits occupy the head of every slice — a state builds each
     * list from its own parsed traits and only then appends the keyword-derived
     * and runtime-granted ones — which is why an index into a slice is stable
     * across processes for a scripted line.
     */
    Iterable<? extends CardTraitBase> slice(CardState state) {
        return switch (this) {
            case SPELL -> state.getSpellAbilities();
            case TRIGGER -> state.getTriggers();
            case STATIC -> state.getStaticAbilities();
            case REPLACEMENT -> state.getReplacementEffects();
            // A keyword's traits are appended to the other kinds' slices; it
            // indexes by printed keyword ordinal and has no slice of its own.
            case KEYWORD -> List.of();
        };
    }
}
