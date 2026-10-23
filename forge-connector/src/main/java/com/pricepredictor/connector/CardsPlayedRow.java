package com.pricepredictor.connector;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Per-game card-play record written to {@code output/sealed/cards-played.txt}
 * through a {@link CorpusWriter}.
 *
 * <p>Schema (eleven semicolon-separated fields, no trailing {@code ;}):
 * {@code timestamp;run_id;set_code;method_A;method_B;cards_played_A;
 * cards_played_B;cards_not_played_A;cards_not_played_B;winner;starter}
 *
 * <p>Card list columns are pipe-separated and may be empty (encoded as the
 * empty string between two {@code ;}). Each is a set of distinct names, with
 * basic lands excluded: a card is listed once however many copies the deck
 * holds or were played, and for each side the played and not-played columns
 * are disjoint, a card with any copy played appearing only in the played one.
 *
 * @param header           The parent match's header, timestamp included: every game of one
 *                         match carries the same one.
 * @param cardsPlayedA     Non-basic cards in deck A whose name entered the battlefield or stack at least once.
 * @param cardsPlayedB     Same for deck B.
 * @param cardsNotPlayedA  Deck A cards (minus basics) that were not played in this game.
 * @param cardsNotPlayedB  Same for deck B.
 * @param winner           Matches the corresponding char of the parent's {@code games}.
 * @param starter          Matches the corresponding char of the parent's {@code play}.
 */
public record CardsPlayedRow(
        MatchHeader header,
        List<String> cardsPlayedA,
        List<String> cardsPlayedB,
        List<String> cardsNotPlayedA,
        List<String> cardsNotPlayedB,
        Side winner,
        Side starter
) implements CorpusRow {

    public CardsPlayedRow {
        Objects.requireNonNull(header, "header");
        Objects.requireNonNull(winner, "winner");
        Objects.requireNonNull(starter, "starter");
    }

    /** Format this row as one line of {@code cards-played.txt}. */
    @Override
    public String toLine() {
        return header.line(
                CardNameList.format(cardsPlayedA), CardNameList.format(cardsPlayedB),
                CardNameList.format(cardsNotPlayedA), CardNameList.format(cardsNotPlayedB),
                winner.toString(), starter.toString());
    }

    /**
     * The row for one game of {@code parent}, from the cards that game played.
     *
     * <p>The card columns are distinct sets: a name appears once however many
     * copies the deck holds, basic lands appear in none of them, and a card is
     * in at most one of a side's two columns — played if any copy was.
     */
    static CardsPlayedRow of(MatchResult parent, GamePlayer.GameOutcome game) {
        List<String> playedA = distinctNonBasic(game.cardsPlayedA());
        List<String> playedB = distinctNonBasic(game.cardsPlayedB());
        return new CardsPlayedRow(
                parent.header(),
                playedA,
                playedB,
                nonBasicMinus(parent.deckA(), playedA),
                nonBasicMinus(parent.deckB(), playedB),
                game.winner(),
                game.playFirst());
    }

    private static List<String> distinctNonBasic(List<String> names) {
        Set<String> kept = new LinkedHashSet<>();
        for (String name : names) {
            if (!BasicLands.isBasic(name)) {
                kept.add(name);
            }
        }
        return new ArrayList<>(kept);
    }

    /** The deck's distinct non-basic names, in deck order, that {@code played} lacks. */
    private static List<String> nonBasicMinus(List<String> deck, List<String> played) {
        Set<String> playedSet = new HashSet<>(played);
        Set<String> remaining = new LinkedHashSet<>();
        for (String name : deck) {
            if (!BasicLands.isBasic(name) && !playedSet.contains(name)) {
                remaining.add(name);
            }
        }
        return new ArrayList<>(remaining);
    }
}
