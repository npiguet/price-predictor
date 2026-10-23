package com.pricepredictor.connector;

import forge.deck.CardPool;
import forge.deck.Deck;
import forge.deck.DeckSection;
import forge.item.PaperCard;
import forge.model.FModel;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Optional;

/**
 * Turns card names into Forge cards and decks, and decks back into names.
 *
 * <p>Every worker that plays or builds from a name list goes through here, so
 * a name Forge's card database does not know is handled one way, under a
 * policy the caller picks deliberately. Dropping it silently is the one
 * behaviour none of them may have: the deck then plays short, and whatever the
 * match feeds — a corpus row, a tally, a baseline deck — records a result for
 * a deck nobody asked for.
 *
 * <p>{@link ForgeEnvironmentInitializer#initialize()} must have been called
 * before {@link #lookUp}, {@link #resolve} or {@link #materialize}.
 */
final class DeckMaterializer {

    /** What a name Forge cannot resolve does to the call. */
    enum UnresolvedPolicy {
        /**
         * Log the names and drop them. Only for names that came out of Forge
         * in the first place, where a miss is a surprise rather than a broken
         * input, and killing a long run over one is worse than saying so.
         */
        WARN,
        /** Throw, naming them: the caller's input is not what Forge plays. */
        FAIL
    }

    /** At most this many names are listed before the rest are counted. */
    static final int UNRESOLVED_NAMES_LISTED = 8;

    private DeckMaterializer() {
    }

    /**
     * The card Forge's card database holds under {@code name}, or empty when
     * it holds none.
     *
     * <p>The one name lookup every caller shares, so "a name Forge knows"
     * means the same thing to the deck a worker plays, the basic-land rule
     * and the list of variants a round may build from.
     */
    static Optional<PaperCard> lookUp(String name) {
        return Optional.ofNullable(FModel.getMagicDb().getCommonCards().getCard(name));
    }

    /**
     * Resolve {@code names} in order, duplicates kept.
     *
     * @param why appended to the exception under {@link UnresolvedPolicy#FAIL}:
     *            what playing the short deck would have corrupted
     */
    static List<PaperCard> resolve(List<String> names, UnresolvedPolicy policy, String why) {
        List<PaperCard> cards = new ArrayList<>(names.size());
        List<String> unresolved = new ArrayList<>();
        for (String name : names) {
            lookUp(name).ifPresentOrElse(cards::add, () -> unresolved.add(name));
        }
        reportUnresolved(unresolved, names.size(), policy, why);
        return cards;
    }

    /** {@link #resolve} into a deck's main section. */
    static Deck materialize(List<String> names, UnresolvedPolicy policy, String why) {
        return deckOf(resolve(names, policy, why));
    }

    /** A deck whose main section is exactly {@code cards}. */
    static Deck deckOf(List<PaperCard> cards) {
        Deck deck = new Deck();
        CardPool main = deck.getOrCreate(DeckSection.Main);
        for (PaperCard card : cards) {
            main.add(card);
        }
        return deck;
    }

    /** Every card in a deck's main section, duplicates repeated. */
    static List<String> names(Deck deck) {
        List<String> names = new ArrayList<>();
        for (PaperCard card : deck.getOrCreate(DeckSection.Main).toFlatList()) {
            names.add(card.getName());
        }
        return names;
    }

    /**
     * Say something about names {@link #lookUp} could not resolve:
     * a warning on stderr, or an exception under {@link UnresolvedPolicy#FAIL}.
     */
    static void reportUnresolved(
            List<String> unresolved, int requested, UnresolvedPolicy policy, String why) {
        if (unresolved.isEmpty()) {
            return;
        }
        List<String> distinct = new ArrayList<>(new LinkedHashSet<>(unresolved));
        String listed = String.join(", ",
                distinct.subList(0, Math.min(UNRESOLVED_NAMES_LISTED, distinct.size())));
        if (distinct.size() > UNRESOLVED_NAMES_LISTED) {
            listed += " (and " + (distinct.size() - UNRESOLVED_NAMES_LISTED) + " more)";
        }
        String detail = unresolved.size() + " of " + requested
                + " card names did not resolve against Forge's card database: " + listed;
        if (policy == UnresolvedPolicy.FAIL) {
            throw new IllegalStateException(
                    "deck cannot be materialized: " + detail + ". " + why);
        }
        System.err.println("WARNING: " + detail);
    }
}
