package com.pricepredictor.connector.effects;

import com.google.common.collect.Table;
import forge.card.CardTypeView;
import forge.card.MagicColor;
import forge.game.CardTraitBase;
import forge.game.card.Card;
import forge.game.card.CardState;
import forge.game.card.CardTraitChanges;
import forge.game.card.ICardTraitChanges;
import forge.game.keyword.Keyword;
import forge.game.replacement.ReplacementEffect;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * One kind's slice of a card state as the sidecar numbers it, and where a
 * trait stands in it.
 *
 * <p>The slice is built once, through Forge's own assembly by reflection
 * ({@link ForgeFields#traitsThrough}), when the object is made, and every
 * index search on it reads that one build. An instance lives for the keying
 * of one trait in its state ({@link ProvenanceResolver}'s
 * {@code keyInState} and {@code printedSpellOf} each make one and drop it on
 * return) and is never kept past it: a card's trait changes and types move
 * between records, and a slice read before they moved numbers the wrong
 * lines.
 */
final class PrintedSlice {

    private final CardState state;
    private final TraitKind kind;

    /** The slice as the sidecar numbers it, or null when Forge's assembly cannot be read. */
    private final Iterable<? extends CardTraitBase> indexed;

    private PrintedSlice(CardState state, TraitKind kind) {
        this.state = state;
        this.kind = kind;
        this.indexed = indexedSlice(state, kind);
    }

    /** The {@code kind} slice of {@code state}, built now. */
    static PrintedSlice of(CardState state, TraitKind kind) {
        return new PrintedSlice(state, kind);
    }

    /**
     * Index of {@code trait} within the slice, counted as the sidecar counts
     * it: with every line a trait change took away back in its printed place
     * ({@link #indexedSlice}).
     *
     * <p>Two cases need that. A printed line a static took away ("loses all
     * abilities") is in no live slice, yet a {@code continuous} record's
     * board, read without that static, lists it. And a static that takes away
     * only some lines (Blood Sun: lands lose all but their mana abilities)
     * moves every line it leaves up the live slice, where a live index would
     * name the line printed before it. A type change does both at once: Blood
     * Moon clears a nonbasic land's lines and adds a Mountain's mana ability
     * where they stood, and Urborg lists a Forest's Swamp mana ability before
     * its own, so such a trait is counted past every printed line instead.
     *
     * <p>When Forge's assembly cannot be read, the live slice, then the slice
     * without each static that removes lines from the card.
     */
    int indexOf(CardTraitBase trait) {
        if (indexed != null) return indexIn(indexed, trait);
        int index = indexIn(kind.slice(state), trait);
        if (index >= 0 || state.getCard() == null) return index;
        return indexWithoutARemoval(trait);
    }

    /** The trait's index in the live slice without the first static that removes lines and lists it, or -1. */
    private int indexWithoutARemoval(CardTraitBase trait) {
        for (Table.Cell<Long, Long, ICardTraitChanges> cell
                : state.getCard().getChangedCardTraits().cellSet()) {
            int index = removesLines(cell.getValue())
                    ? indexIn(kind.sliceWithout(state, cell.getColumnKey()), trait) : -1;
            if (index >= 0) return index;
        }
        return -1;
    }

    /**
     * The position of the one same-shape member of the slice, or -1.
     *
     * <p>The fingerprint is {@code getOriginalMapParams()}, the script's own
     * parameters as parsed. {@code SpellAbility.copy} preserves it because
     * {@code putParam} writes only the live {@code mapParams}, so a stack copy,
     * an alternative-cost copy and a copied spell all still fingerprint back to
     * the line they came from even though none of them carries a back-reference.
     *
     * <p>Refuses to answer when two members match ({@link SoleMatch}).
     */
    int indexLike(CardTraitBase trait) {
        Iterable<? extends CardTraitBase> slice = toCount();
        Map<String, String> want = trait.getOriginalMapParams();
        if (want == null || want.isEmpty()) return indexIfUnique(trait);
        return SoleMatch.indexOf(slice, candidate -> candidate.getClass() == trait.getClass()
                && want.equals(candidate.getOriginalMapParams()));
    }

    /**
     * The position of the slice's only member of this trait's class, or -1.
     *
     * <p>The fallback for a trait with no fingerprint at all. A permanent spell
     * is built in Java rather than parsed from a line -- {@code SpellPermanent}
     * is constructed with an empty parameter map -- so a cost variant of one
     * carries nothing to recognise it by: an Adventure half, a kicker, a
     * {@code MayFlashCost} cast all copy the spell and set no back-reference,
     * and the copy's fingerprint is as empty as the original's.
     *
     * <p>What is left is arithmetic. A card state has exactly one permanent
     * spell, and a single candidate of the right class cannot be the wrong one.
     * Two candidates and this refuses, for the reason {@link #indexLike}
     * refuses.
     */
    private int indexIfUnique(CardTraitBase trait) {
        return SoleMatch.indexOf(toCount(), candidate -> candidate.getClass() == trait.getClass());
    }

    /** The indexed slice, or the live slice when Forge's assembly cannot be read. */
    private Iterable<? extends CardTraitBase> toCount() {
        return indexed != null ? indexed : kind.slice(state);
    }

    /** The position of {@code trait} itself in {@code slice}, or -1. */
    static int indexIn(Iterable<? extends CardTraitBase> slice, CardTraitBase trait) {
        int index = 0;
        for (CardTraitBase candidate : slice) {
            if (candidate == trait) return index;
            index++;
        }
        return -1;
    }

    /**
     * Whether a trait is one a type change attached at runtime: a basic land
     * type's mana ability on a land not printed with that type (Blood Moon,
     * Spreading Seas, Urborg, an {@code Animate} that sets a land type), a
     * loyalty replacement on a permanent made a planeswalker. Such a trait is
     * missing from the printed assembly ({@link #printedChanges}), so it is
     * keyed past every printed line, and its card slot flags it granted. A
     * line a layer-6 change granted is in that assembly and is not one of
     * these.
     */
    static boolean addedByTypeChange(CardTraitBase trait) {
        if (trait == null || trait.getKeyword() != null) return false;
        CardState state = trait.getCardState();
        TraitKind kind = TraitKind.of(trait);
        List<? extends CardTraitBase> assembled = state == null || kind == null ? null
                : printedAssembly(state, kind);
        return assembled != null && indexIn(assembled, trait) < 0;
    }

    /**
     * The kind's printed assembly ({@link #printedChanges}) as Forge builds
     * it; null when that is the live list or Forge's assembly cannot be read.
     */
    private static List<? extends CardTraitBase> printedAssembly(CardState state, TraitKind kind) {
        List<ICardTraitChanges> printed = printedChanges(state, kind);
        return printed == null ? null : ForgeFields.traitsThrough(state, kind, printed);
    }

    /**
     * The kind's slice of {@code state} as the sidecar numbers it: the
     * printed assembly ({@link #printedChanges}), then every live trait it
     * lacks — the ones a type change added — so a printed line stands at the
     * index the sidecar gives it and a runtime one past them all; the live
     * slice when no change takes a line or changes a type; null when Forge's
     * assembly cannot be read ({@link ForgeFields#traitsThrough}).
     */
    private static Iterable<? extends CardTraitBase> indexedSlice(CardState state, TraitKind kind) {
        List<ICardTraitChanges> printed = printedChanges(state, kind);
        if (printed == null) return kind.slice(state);
        List<? extends CardTraitBase> assembled = ForgeFields.traitsThrough(state, kind, printed);
        return assembled == null ? null : withLiveTail(assembled, kind.slice(state));
    }

    /** {@code assembled}, then every trait of {@code live} it lacks, in live order. */
    private static List<CardTraitBase> withLiveTail(List<? extends CardTraitBase> assembled,
                                                    Iterable<? extends CardTraitBase> live) {
        List<CardTraitBase> indexed = new ArrayList<>(assembled);
        Set<CardTraitBase> listed = Collections.newSetFromMap(new IdentityHashMap<>());
        listed.addAll(assembled);
        for (CardTraitBase each : live) {
            if (listed.add(each)) indexed.add(each);
        }
        return indexed;
    }

    /**
     * The trait changes that assemble the printed list of a kind (the
     * printed assembly): the card's own, but with every line a change took
     * away left in place, and layer 4's land traits built from the state's
     * printed type rather than its live one ({@link PrintedLandTraits}).
     * Lines a change added stay, appended after the printed ones. Null when
     * that list is the live one: no change takes a line and no type changed.
     */
    private static List<ICardTraitChanges> printedChanges(CardState state, TraitKind kind) {
        Card card = state.getCard();
        if (card == null || kind == TraitKind.KEYWORD) return null;
        boolean differs = card.getChangedCardTypes().iterator().hasNext();
        List<?> layerFour = state.getLandTraitChanges();
        List<ICardTraitChanges> printed = new ArrayList<>();
        for (ICardTraitChanges changes : card.getChangedCardTraitsList(state)) {
            boolean removing = removesLines(changes);
            differs |= removing;
            printed.add(removing ? keepingLines((CardTraitChanges) changes)
                    : printedLandOr(changes, state, layerFour));
        }
        return differs ? printed : null;
    }

    /** Whether a trait change takes lines away from its card. */
    private static boolean removesLines(ICardTraitChanges changes) {
        return changes instanceof CardTraitChanges taking && taking.remove() != null;
    }

    /** {@code taking} with its lines added and none taken away. */
    private static ICardTraitChanges keepingLines(CardTraitChanges taking) {
        return new CardTraitChanges(taking.abilities(), taking.triggers(),
                taking.replacements(), taking.staticAbilities(), null);
    }

    /** Layer 4's land traits as printed when {@code changes} is one of them, else {@code changes}. */
    private static ICardTraitChanges printedLandOr(ICardTraitChanges changes, CardState state,
                                                   List<?> layerFour) {
        return layerFour.contains(changes) ? new PrintedLandTraits(state) : changes;
    }

    /**
     * Layer 4's land traits as the state's printed type builds them, which is
     * how the converter, reading the card with no game around it, numbered
     * them: each printed basic land type's mana ability, in Forge's colour
     * order, and the replacement a printed planeswalker, battle or saga
     * carries. Unlike Forge's own, it clears nothing: a type change that
     * removes every land type (Blood Moon) takes the printed lines from the
     * live list, not from their places.
     */
    private record PrintedLandTraits(CardState state) implements ICardTraitChanges {

        @Override
        public List<SpellAbility> applySpellAbility(List<SpellAbility> list) {
            CardTypeView type = state.getType();
            if (!type.isLand()) return list;
            for (MagicColor.Color colour : MagicColor.Color.values()) {
                String landType = colour.getBasicLandType();
                if (landType != null && type.hasSubtype(landType)) {
                    list.add(state.getLandManaForColor(colour));
                }
            }
            return list;
        }

        @Override
        public List<ReplacementEffect> applyReplacementEffect(List<ReplacementEffect> list) {
            CardTypeView type = state.getType();
            if (type.isPlaneswalker()) list.add(state.getLoyaltyRep());
            if (type.isBattle()) list.add(state.getDefenseRep());
            if (type.isSaga() && !state.hasKeyword(Keyword.READ_AHEAD)) list.add(state.getSagaRep());
            return list;
        }

        @Override
        public ICardTraitChanges copy(Card host, boolean lki) {
            return this;
        }
    }
}
