package com.pricepredictor.connector.effects;

import forge.card.CardRules;
import forge.card.CardStateName;
import forge.card.ICardFace;
import forge.game.card.Card;
import forge.game.card.CardCloneStates;
import forge.game.card.CardState;
import forge.game.keyword.KeywordCollection;
import forge.game.keyword.KeywordInterface;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Where a keyword of a state that is not a card's own printed state is
 * printed (finding 1.8): a mutated permanent's, which holds the keywords of
 * every card merged into it, and a token or spell copy's, which holds the
 * original's keywords plus those its copy effect added. A token copy of a
 * copy (Kiki-Jiki's token of a Clone copying Moldering Karok) is built from
 * the Clone's paper card, whose rules print none of its keywords: its
 * keywords are read through the Clone's copy state, to what the Clone
 * copies ({@link #copySourceOf}).
 *
 * <p>Neither is a copy {@code Card.isCloned} names, so
 * {@link ProvenanceResolver} does not read them through what they copy, and
 * numbering their keywords among themselves names the wrong file (a merged
 * permanent's keywords all landed in the under card's) or the wrong line (a
 * token copy's added haste sorted before the original's lifelink, and every
 * later keyword moved one line on).
 *
 * <p>Two instances with the same text are told apart by their order in the
 * state, Forge's insertion order for one keyword: a merged state lists the
 * top card's keywords, then each other card's in merge order, and a copy's
 * state lists the original's before the ones the copy effect added. A
 * keyword Forge treats as redundant (flying) is listed once, the first card
 * to bring it.
 */
final class KeywordPrinters {

    private KeywordPrinters() {
    }

    /**
     * The state {@code keyword} of {@code state} is read through: for a
     * mutated permanent's merged state, the printed state of the merged card
     * it came from, the top card's first; for a token copy of a copy, the
     * copied Clone's state ({@link #copiedPrinterOf}); null when none holds
     * it. {@code state} itself for any other state.
     */
    static CardState printerOf(CardState state, KeywordInterface keyword) {
        Card host = state.getCard();
        if (host == null) {
            return state;
        }
        if (isMergedState(host, state)) {
            return mergedPrinterOf(host, state, keyword);
        }
        Card source = copySourceOf(host);
        return source != null && source.isCloned() ? copiedPrinterOf(source, state, keyword) : state;
    }

    /**
     * The merged card's printed state that prints {@code keyword} of a
     * mutated permanent's merged state, the top card's first; null when
     * none prints it.
     */
    private static CardState mergedPrinterOf(Card host, CardState state, KeywordInterface keyword) {
        int rank = rankOf(state, keyword);
        for (Card component : host.getMergedCards()) {
            CardState printed = printedStateOf(component, host, state.getStateName());
            int count = countOf(printed, keyword.getOriginal());
            if (rank < count) {
                return printed;
            }
            rank -= count;
        }
        return null;
    }

    /**
     * The state of a copied Clone a token copy of it reads {@code keyword}
     * through: the Clone's state of the token state's name, else the one it
     * shows, whose keywords the copy took. Null for a keyword that state
     * does not hold: one the token's own copy effect added (Kiki-Jiki's
     * haste).
     */
    private static CardState copiedPrinterOf(Card source, CardState state, KeywordInterface keyword) {
        CardState copied = source.getState(state.getStateName());
        CardState printed = copied == null ? source.getCurrentState() : copied;
        return rankOf(state, keyword) < countOf(printed, keyword.getOriginal()) ? printed : null;
    }

    /**
     * The printed-keyword ordinal {@code keyword} keys to in {@code state}:
     * {@link FaceOrder#keywordIndex}, except on a token or spell copy, which
     * is numbered against the face its card prints, and answers -1 for a
     * keyword the copy effect added.
     */
    static int indexOf(CardState state, KeywordInterface keyword) {
        Card host = state.getCard();
        if (!isStateCopy(host)) {
            return FaceOrder.keywordIndex(state, keyword);
        }
        List<String> printed = printedOriginals(host.getRules(), state.getStateName());
        int count = Collections.frequency(printed, keyword.getOriginal());
        return rankOf(state, keyword) < count ? printed.indexOf(keyword.getOriginal()) : -1;
    }

    /**
     * Whether a card is a token or spell copy: one a copy effect built from
     * the original's states ({@code CopyPermanentEffect.getProtoType},
     * {@code CardFactory.copySpellHost}) and gave a {@code copiedPermanent},
     * a card other than itself. Its paper card, so its rules and its script,
     * are the original's.
     */
    static boolean isStateCopy(Card host) {
        Card copied = host == null ? null : host.getCopiedPermanent();
        return copied != null && copied != host;
    }

    /**
     * The permanent a token copy took its copiable values from: the one its
     * prototype names ({@code Card.getEffectRecordCopySource}, which the
     * patch sets in {@code CopyPermanentEffect.getProtoType}). The token's
     * own {@code copiedPermanent} is that throwaway prototype, which is no
     * clone and names nothing. Null for any other card, and on a checkout
     * without the hook.
     */
    static Card copySourceOf(Card host) {
        return isStateCopy(host)
                && PatchHooks.COPY_SOURCE.read(host.getCopiedPermanent()) instanceof Card source
                ? source : null;
    }

    /** Whether {@code state} is one of the card's mutated clone states. */
    private static boolean isMergedState(Card host, CardState state) {
        if (!host.isMutated()) {
            return false;
        }
        CardCloneStates merged = host.getCloneStates().get(host.getMutatedTimestamp());
        return merged != null && merged.values().stream().anyMatch(each -> each == state);
    }

    /**
     * A merged card's printed state as {@code CardFactory.getMutatedCloneStates}
     * read it: the host's own state of that name, any other card's current
     * one.
     */
    private static CardState printedStateOf(Card component, Card host, CardStateName name) {
        return component == host ? host.getOriginalState(name) : component.getCurrentState();
    }

    /**
     * How many keywords of {@code state} with the same text come before
     * {@code keyword}; 0 for a keyword that is not among them.
     */
    private static int rankOf(CardState state, KeywordInterface keyword) {
        int rank = 0;
        for (KeywordInterface each : state.getIntrinsicKeywords()) {
            if (each == keyword) {
                return rank;
            }
            rank += Objects.equals(each.getOriginal(), keyword.getOriginal()) ? 1 : 0;
        }
        return 0;
    }

    /** How many of {@code state}'s keywords read {@code original}; 0 for no state. */
    private static int countOf(CardState state, String original) {
        if (state == null) {
            return 0;
        }
        int count = 0;
        for (KeywordInterface each : state.getIntrinsicKeywords()) {
            count += Objects.equals(each.getOriginal(), original) ? 1 : 0;
        }
        return count;
    }

    /**
     * The keyword texts the face {@code rules} prints under {@code name},
     * sorted, as the converter's card holds them: each {@code K:} line,
     * a redundant repeat dropped as Forge drops it.
     */
    private static List<String> printedOriginals(CardRules rules, CardStateName name) {
        ICardFace face = rules == null ? null : FaceOrder.printedFace(rules, name);
        if (face == null) {
            return List.of();
        }
        KeywordCollection keywords = new KeywordCollection();
        keywords.addAll(face.getKeywords());
        List<String> originals = new ArrayList<>();
        keywords.forEach(keyword -> originals.add(keyword.getOriginal()));
        Collections.sort(originals);
        return originals;
    }
}
