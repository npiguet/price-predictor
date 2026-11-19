package com.pricepredictor.connector.effects;

import forge.card.CardRules;
import forge.card.CardStateName;
import forge.card.ICardFace;
import forge.game.card.Card;
import forge.game.card.CardCloneStates;
import forge.game.card.CardState;
import forge.game.keyword.KeywordCollection;
import forge.game.keyword.KeywordInterface;
import forge.game.spellability.SpellAbility;

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
 * keywords are read through the printed state of what the Clone copied
 * when the token was made ({@link #printingSourceOf}), which a later copy
 * effect on the Clone (Mirrorweave) does not change, as it does not change
 * the token (707.2). A keyword the Clone's own copy effect added as an
 * exception (707.9b: Dack's Duplicate's haste and dethrone) is printed in
 * that effect's line of the Clone's own card ({@link #copyClauseOf}).
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
     * printed state of what the Clone copied ({@link #copiedPrinterOf});
     * null when none holds it. {@code state} itself for any other state,
     * a plain token copy's among them, whose own rules print its keywords.
     */
    static CardState printerOf(CardState state, KeywordInterface keyword) {
        Card host = state.getCard();
        if (host == null) {
            return state;
        }
        if (isMergedState(host, state)) {
            return mergedPrinterOf(host, state, keyword);
        }
        Card source = printingSourceOf(host);
        return source != null && source.getRules() != host.getRules()
                ? copiedPrinterOf(source, state, keyword) : state;
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
     * The printed state of what a copied Clone copied that a token copy of
     * it reads {@code keyword} through: its state of the token state's name,
     * else of the face it shows (a Clone of a transformed card copies its
     * back face into its own front), the first holding the keyword. Its
     * printed state, not the one it shows, which a later copy effect on it
     * changes. Null for a keyword neither holds: one the token's own copy
     * effect added (Kiki-Jiki's haste), or one the Clone's did
     * ({@link #copyClauseOf}).
     */
    private static CardState copiedPrinterOf(Card source, CardState state, KeywordInterface keyword) {
        int rank = rankOf(state, keyword);
        for (CardStateName name : List.of(state.getStateName(), source.getCurrentStateName())) {
            CardState printed = source.getOriginalState(name);
            if (rank < countOf(printed, keyword.getOriginal())) {
                return printed;
            }
        }
        return null;
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
     * The copy clause of a copied Clone's own printed state that added
     * {@code keyword} of a token copy's {@code state} as an exception
     * (707.9b), through a token copy of a token copy to the Clone: the
     * {@code Clone} clause whose {@code AddKeywords$} lists it
     * ({@link CopyExceptions}). Null for none: a keyword the token's own
     * copy effect added (Kiki-Jiki's haste, where the Clone adds none).
     */
    static SpellAbility copyClauseOf(CardState state, KeywordInterface keyword) {
        Card link = state.getCard();
        for (int depth = 0; isStateCopy(link) && depth < MAX_COPY_DEPTH; depth++) {
            Card copied = copiedPermanentOf(link);
            SpellAbility clause = CopyExceptions.clauseAdding(
                    ownStateOf(copied, state.getStateName()), keyword.getOriginal());
            if (clause != null) {
                return clause;
            }
            link = copied;
        }
        return null;
    }

    /**
     * The permanent a token copy's prototype copied
     * ({@code Card.getEffectRecordCopiedPermanent}, which the patch sets in
     * {@code CopyPermanentEffect.getProtoType}); null on a checkout without
     * the hook.
     */
    private static Card copiedPermanentOf(Card host) {
        return PatchHooks.COPIED_PERMANENT.read(host.getCopiedPermanent()) instanceof Card copied ? copied : null;
    }

    /**
     * A card's own state of {@code name}, else its own front: its printed
     * state, which no copy effect on it changes. Null for no card.
     */
    private static CardState ownStateOf(Card card, CardStateName name) {
        if (card == null) {
            return null;
        }
        CardState own = card.getOriginalState(name);
        return own != null ? own : card.getOriginalState(CardStateName.Original);
    }

    /** How many token copies of token copies {@link #printingSourceOf} follows. */
    private static final int MAX_COPY_DEPTH = 8;

    /**
     * The card whose printed values a token copy's copiable values are, as
     * they were when the token was made: what its prototype names, through
     * a token copy of a token copy to one that is none. Null for any other
     * card, and on a checkout without the hook.
     */
    static Card printingSourceOf(Card host) {
        Card printer = null;
        Card next = copySourceOf(host);
        for (int depth = 0; next != null && depth < MAX_COPY_DEPTH; depth++) {
            printer = next;
            next = copySourceOf(next);
        }
        return printer;
    }

    /**
     * The card whose printed values a token copy took as its copiable
     * values: the one its prototype names ({@code Card.getEffectRecordCopySource},
     * which the patch sets in {@code CopyPermanentEffect.getProtoType} to
     * what the copied permanent copied then, through a copy of a copy, else
     * that permanent). The token's own {@code copiedPermanent} is that
     * throwaway prototype, which is no clone and names nothing. Null for any
     * other card, and on a checkout without the hook.
     */
    private static Card copySourceOf(Card host) {
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
