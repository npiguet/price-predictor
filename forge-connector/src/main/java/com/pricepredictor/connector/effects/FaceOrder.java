package com.pricepredictor.connector.effects;

import forge.card.CardStateName;
import forge.game.card.Card;
import forge.game.card.CardState;
import forge.game.keyword.KeywordInterface;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

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

    private FaceOrder() {
    }

    /**
     * The face ordinal the converter emitted a state under.
     *
     * <p>Shared by the converter and the collectors so the two cannot drift:
     * the converter renders a face for each state of {@link #faceOrder} that
     * has a printed part, in that order.
     */
    public static int faceIndex(Card card, CardStateName state) {
        if (state == null) return 0;
        List<CardStateName> order = faceOrder(card);
        int index = order.indexOf(state);
        return index < 0 ? 0 : index;
    }

    /** The state names in the order the converter emits faces. */
    public static List<CardStateName> faceOrder(Card card) {
        if (card == null) return List.of(CardStateName.Original);
        if (card.getStates().contains(CardStateName.LeftSplit)) {
            return List.of(CardStateName.LeftSplit, CardStateName.RightSplit);
        }
        List<CardStateName> order = new ArrayList<>();
        order.add(CardStateName.Original);
        for (CardStateName name : card.getStates()) {
            if (name != CardStateName.Original && !order.contains(name)) {
                order.add(name);
            }
        }
        return List.copyOf(order);
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
     * {@code hashCode()} is the identity hash, so that order differs between
     * the convert JVM and the collect JVM and an ordinal taken from it names a
     * different keyword on each side. Sorting the state's own keywords by their
     * printed text is stable in both processes.
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
