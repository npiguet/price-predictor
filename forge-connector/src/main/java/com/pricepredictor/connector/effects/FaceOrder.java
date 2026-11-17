package com.pricepredictor.connector.effects;

import forge.StaticData;
import forge.card.CardRules;
import forge.card.CardSplitType;
import forge.card.CardStateName;
import forge.card.ICardFace;
import forge.game.card.Card;
import forge.game.card.CardState;
import forge.game.keyword.KeywordInterface;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * The two ordinals a provenance key carries that the converter and the
 * collectors must compute identically: which face a state is, and which
 * printed keyword a keyword-derived trait came from.
 *
 * <p>Both sides take them from here rather than each counting for itself —
 * the converter emits faces in {@link #faceOrder}, the collectors number them
 * with {@link #faceIndex}, and both call {@link #keywordIndex} — because a key
 * is joined by value across two JVMs, and an ordinal counted
 * differently on each side names a different line on each side while staying
 * in range on both — a silent wrong join rather than a loud failure.
 */
public final class FaceOrder {

    /** The states {@link #faceOrder} adds no face for after the first: Original leads it. */
    private static final Set<CardStateName> NO_FACE_OF_ITS_OWN = EnumSet.of(
            CardStateName.Original, CardStateName.FaceDown, CardStateName.EmptyRoom);

    private FaceOrder() {
    }

    /**
     * The face ordinal the converter emitted a state under, or null for a
     * state it emitted no face for: no state, one the card lacks, a face-down
     * or empty-room state.
     *
     * <p>Shared by the converter and the collectors so the two cannot drift:
     * the converter renders a face for each state of {@link #faceOrder} that
     * has a printed part, in that order. Null rather than 0, because 0 names
     * the front face: a key with it would join the wrong line.
     */
    public static Integer faceIndex(Card card, CardStateName state) {
        if (state == null) return null;
        int index = faceOrder(card).indexOf(state);
        return index < 0 ? null : index;
    }

    /**
     * The state names in the order the converter emits faces.
     *
     * <p>Over the states a printing has, not every state the game card holds:
     * Forge adds a face-down state the first time a card turns face down,
     * and an empty-room state to a Room, and either would push every later
     * face one ordinal on (B6).
     */
    public static List<CardStateName> faceOrder(Card card) {
        if (card == null) return List.of(CardStateName.Original);
        if (card.getStates().contains(CardStateName.LeftSplit)) {
            return List.of(CardStateName.LeftSplit, CardStateName.RightSplit);
        }
        List<CardStateName> order = new ArrayList<>();
        order.add(CardStateName.Original);
        for (CardStateName name : card.getStates()) {
            if (!NO_FACE_OF_ITS_OWN.contains(name) && !order.contains(name)) {
                order.add(name);
            }
        }
        return List.copyOf(order);
    }

    /**
     * The printed face of {@code rules} the converter renders under
     * {@code state}, or null for a state no face is printed under.
     *
     * <p>Shared by the converter, which renders each face of
     * {@link #faceOrder} from it, and the collectors, which number a token or
     * spell copy's keywords against it ({@link KeywordPrinters}). A meld
     * half's Meld state prints its partner's melded face ({@link #otherPart}).
     */
    public static ICardFace printedFace(CardRules rules, CardStateName state) {
        CardSplitType splitType = rules.getSplitType();
        CardStateName mainState = splitType == CardSplitType.Split
                ? CardStateName.LeftSplit : CardStateName.Original;
        if (state == mainState) {
            return rules.getMainPart();
        }
        if (splitType == CardSplitType.Specialize) {
            return rules.getSpecializeParts().get(state);
        }
        return state == splitType.getChangedStateName() ? otherPart(rules) : null;
    }

    /**
     * The second face of {@code rules}, or for a meld half whose script does
     * not hold the melded face, its meld partner's second face: the melded
     * card, which Forge's {@code CardFactory} reads into the half's Meld state
     * from the partner's rules in the card database (the one
     * {@code CopiedFaces} supplies copied faces from), and which the
     * collectors key as face 1 of the half's own file.
     */
    private static ICardFace otherPart(CardRules rules) {
        if (rules.getOtherPart() != null || rules.getMeldWith().isEmpty()) {
            return rules.getOtherPart();
        }
        return StaticData.instance().getCommonCards()
                .getRulesOrElseUnsupported(rules.getMeldWith()).getOtherPart();
    }

    /**
     * {@code keywords} sorted by original text: the order the converter prints a
     * face's keyword lines in, and the one {@link #keywordIndex} numbers them
     * in. The sort is stable, so identical keyword instances stay adjacent.
     */
    public static List<KeywordInterface> keywordOrder(Iterable<KeywordInterface> keywords) {
        List<KeywordInterface> ordered = new ArrayList<>();
        keywords.forEach(ordered::add);
        ordered.sort(Comparator.comparing(FaceOrder::original));
        return ordered;
    }

    private static String original(KeywordInterface keyword) {
        return keyword.getOriginal() == null ? "" : keyword.getOriginal();
    }

    /**
     * The printed-keyword ordinal a keyword-derived trait keys to.
     *
     * <p>Shared by the converter and the collectors so the two cannot drift,
     * the same discipline {@link #faceIndex} already enforces for faces.
     *
     * <p>Not the position in {@code getKeywords()}:
     * {@code forge.game.keyword.KeywordCollection} is a
     * {@code MultimapBuilder.hashKeys()} over the {@code Keyword} enum, whose
     * order is the script's insertion order only while Guava is the
     * {@code -android} flavour Forge pins, and is an identity-hash order
     * (different in the convert JVM and the collect JVM) under the
     * {@code -jre} one. Sorting the state's own keywords by their printed
     * text, as {@link #keywordOrder} does, is stable in both processes and
     * under either flavour.
     *
     * <p>Restricted to intrinsics, so a <em>granted</em> keyword answers -1 and
     * keys to nothing. That is correct: a keyword an anthem handed out names no
     * printed line on the card that received it, and the alternative — the
     * position it happens to occupy in the recipient's list — is a wrong key
     * dressed as an answer.
     */
    public static int keywordIndex(CardState state, KeywordInterface keyword) {
        if (state == null || keyword == null) return -1;
        String original = keyword.getOriginal();
        if (original == null || original.isEmpty()) return -1;
        List<String> originals = new ArrayList<>();
        for (KeywordInterface intrinsic : state.getIntrinsicKeywords()) {
            if (intrinsic.getOriginal() != null) originals.add(intrinsic.getOriginal());
        }
        Collections.sort(originals);
        return originals.indexOf(original);
    }
}
