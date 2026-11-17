package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.WireNamed;

import forge.game.CardTraitBase;
import forge.game.card.CardState;
import forge.game.keyword.KeywordInterface;
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
public enum TraitKind implements WireNamed {
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
    @Override
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

    /**
     * A keyword instance's list of the traits of this kind it built, in the
     * order it built them: what a key's {@code built} index counts in. A
     * {@link #SPELL} is an activated or a spell ability
     * ({@code KeywordInterface.getAbilities()}).
     */
    Iterable<? extends CardTraitBase> builtBy(KeywordInterface keyword) {
        return switch (this) {
            case SPELL -> keyword.getAbilities();
            case TRIGGER -> keyword.getTriggers();
            case STATIC -> keyword.getStaticAbilities();
            case REPLACEMENT -> keyword.getReplacements();
            case KEYWORD -> List.of();
        };
    }

    /**
     * {@link #slice} as it would be without one static's changes: the lines
     * it granted gone, the lines it took away back in their printed places.
     */
    Iterable<? extends CardTraitBase> sliceWithout(CardState state, long staticId) {
        return switch (this) {
            case SPELL -> state.getSpellAbilitiesWithout(staticId);
            case TRIGGER -> state.getTriggersWithout(staticId);
            case STATIC -> state.getStaticAbilitiesWithout(staticId);
            case REPLACEMENT -> state.getReplacementEffectsWithout(staticId);
            case KEYWORD -> List.of();
        };
    }
}
