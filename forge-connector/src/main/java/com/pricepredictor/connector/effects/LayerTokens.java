package com.pricepredictor.connector.effects;

import com.google.common.base.CaseFormat;
import forge.card.CardChangedType;
import forge.card.CardTypeView;
import forge.card.ColorSet;
import forge.card.ICardChangedType;
import forge.card.RemoveType;
import forge.card.StateChangedType;
import forge.card.WordChangedType;
import forge.game.CardTraitBase;
import forge.game.card.CardTraitChanges;
import forge.game.keyword.KeywordInterface;
import forge.game.keyword.KeywordsChange;

import java.util.List;

/** What one layer-table entry does, as the tokens of the contribution it is filed under. */
final class LayerTokens {

    private LayerTokens() {
    }

    /** What one keyword-layer entry does, as tokens. */
    static void keywordTokens(KeywordsChange change, Contribution into) {
        // "Loses all abilities" (RemoveAllAbilities$): the keyword line
        // is set outright, to what the static grants after the marker.
        if (change.isRemoveAllKeywords()) {
            into.keyword(LayerToken.SETS_THE_LINE);
        }
        for (KeywordInterface keyword : change.getKeywords()) {
            into.keyword(LayerToken.added(keyword.getOriginal()));
        }
        // The same channel's other half, and the only path to
        // keywords_lost, which is a head field: without it a static that
        // takes a keyword away records "loses flying" as "does nothing".
        for (String removed : change.getRemoveKeywords()) {
            into.keyword(LayerToken.removed(removed));
        }
    }

    /** The abilities {@code changes} grants, by their lines' keys. */
    static void grantedTokens(CardTraitChanges changes, Contribution into) {
        for (Iterable<? extends CardTraitBase> granted : List.of(
                changes.getAbilities(), changes.getTriggers(),
                changes.getReplacements(), changes.getStaticAbilities())) {
            granted.forEach(trait -> into.ability(LayerToken.added(keyOf(trait))));
        }
    }

    /** The abilities a static takes away, each as {@code -} and its line's key. */
    static void lostTokens(Iterable<CardTraitBase> lost, Contribution into) {
        for (CardTraitBase trait : lost) {
            into.ability(LayerToken.removed(keyOf(trait)));
        }
    }

    /** A trait's line key ({@code ProvenanceKey.describe}), {@code runtime} where it keys to none. */
    private static String keyOf(CardTraitBase trait) {
        ProvenanceKey key = ProvenanceResolver.of(trait);
        return key == null ? "runtime" : key.describe();
    }

    /**
     * What one type-layer entry does, as tokens.
     *
     * <p>Read off the change object rather than diffed out of the applied type
     * line, for the reason the P/T and keyword channels are: two statics that
     * both grant {@code Creature} are indistinguishable in the result, and the
     * record has to say which one this is.
     *
     * <p>A bare token is a type the static adds, {@code -x} one it removes, and
     * a lone {@code =} marks the rest as a type line the static sets outright
     * rather than adds to. Removing every type of a kind has no name to give, so
     * it gets {@code -all-creature-types} and its siblings.
     */
    static void typeTokens(ICardChangedType change, Contribution into) {
        if (change instanceof CardChangedType typed) {
            changedTypeTokens(typed, into);
        } else if (change instanceof StateChangedType state) {
            into.type(LayerToken.SETS_THE_LINE);
            addTypes(into, LayerToken.Sign.ADDS, state.type());
        } else if (change instanceof WordChangedType word) {
            // A text change swaps one subtype word for another, which is a
            // removal and an addition in the same entry.
            into.type(LayerToken.removed(CardSpelling.typeName(word.oldWord())));
            into.type(LayerToken.added(CardSpelling.typeName(word.newWord())));
        }
    }

    private static void changedTypeTokens(CardChangedType typed, Contribution into) {
        addTypes(into, LayerToken.Sign.ADDS, typed.addType());
        addTypes(into, LayerToken.Sign.REMOVES, typed.removeType());
        if (typed.addAllCreatureTypes()) {
            into.type(LayerToken.added("all-creature-types"));
        }
        for (RemoveType removed : typed.remove()) {
            into.type(everyTypeOf(removed));
        }
    }

    private static void addTypes(Contribution into, LayerToken.Sign sign, CardTypeView types) {
        if (types == null) {
            return;
        }
        for (String name : TypeLine.names(types)) {
            into.type(new LayerToken(sign, name));
        }
    }

    /** Removing every type of a kind: {@code CreatureTypes} as {@code -all-creature-types}. */
    private static LayerToken everyTypeOf(RemoveType removed) {
        return LayerToken.removed("all-" + CaseFormat.UPPER_CAMEL.to(CaseFormat.LOWER_HYPHEN, removed.name()));
    }

    /**
     * What one colour-layer entry does, as tokens.
     *
     * <p>The letters are the snapshot's, so a static's share of an entity's
     * colours reads the same as the entity's own. A lone {@code =} carries the
     * same meaning it does for types: the static sets the colour rather than
     * adding to it, and a {@code =} with nothing after it turns the entity
     * colourless.
     *
     * <p>Reflective on both components, because the record they belong to is
     * private in the stock Forge this side compiles against; the patch makes it
     * public.
     */
    static void colorTokens(Object change, Contribution into) {
        // Both components or neither: a half-read entry would report a colour
        // change as a replacement, which is the one reading that is never safe
        // to guess at.
        if (!(PatchHooks.COLOR_CHANGE_ADDITIONAL.read(change) instanceof Boolean additional)
                || !(PatchHooks.COLOR_CHANGE_COLORS.read(change) instanceof ColorSet colors)) {
            return;
        }
        List<String> letters = CardSpelling.colorLetters(colors);
        if (!additional) {
            into.color(LayerToken.SETS_THE_LINE);
            if (letters.isEmpty()) {
                // Setting the colour to nothing is what makes a permanent
                // colourless, which the vocabulary spells C rather than as an
                // empty list.
                into.color(LayerToken.added("C"));
            }
        }
        for (String letter : letters) {
            into.color(LayerToken.added(letter));
        }
    }

    /**
     * What one name-layer entry does: the whole contribution is the name
     * itself, unlike a type or colour token, and a card either has a static
     * overwriting its name or it does not -- {@code newName} is null on
     * every entry that only ever set {@code addNonLegendaryCreatureNames}.
     *
     * <p>Reflective for the reason {@link #colorTokens} is: the record is
     * private in stock Forge.
     */
    static void nameTokens(Object change, Contribution into) {
        if (PatchHooks.NAME_CHANGE_NAME.read(change) instanceof String name) {
            into.name(name);
        }
    }
}
