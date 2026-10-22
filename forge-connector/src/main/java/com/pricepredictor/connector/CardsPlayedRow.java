package com.pricepredictor.connector;

import java.time.Instant;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Per-game card-play record written to {@code output/sealed/cards-played.txt}
 * by {@link CardsPlayedWriter}.
 *
 * <p>Schema (eleven semicolon-separated fields, no trailing {@code ;}):
 * {@code timestamp;run_id;set_code;method_A;method_B;cards_played_A;
 * cards_played_B;cards_not_played_A;cards_not_played_B;winner;starter}
 *
 * <p>Card list columns are pipe-separated and may be empty (encoded as the
 * empty string between two {@code ;}). Multiplicities are preserved.
 *
 * @param timestamp        When the parent match was recorded (UTC). Same value across all games of one match.
 * @param runId            UUID identifying the supervisor invocation that produced the parent match.
 * @param setCode          MTG set code both decks were drawn from.
 * @param methodA          Deck A build method (forge-best/forge-3sub/forge-8sub/random or a LABEL).
 * @param methodB          Deck B build method.
 * @param cardsPlayedA     Non-basic cards in deck A whose name entered the battlefield or stack at least once.
 * @param cardsPlayedB     Same for deck B.
 * @param cardsNotPlayedA  Deck A cards (minus basics) that were not played in this game.
 * @param cardsNotPlayedB  Same for deck B.
 * @param winner           Matches the corresponding char of the parent's {@code games}.
 * @param starter          Matches the corresponding char of the parent's {@code play}.
 */
public record CardsPlayedRow(
        Instant timestamp,
        String runId,
        String setCode,
        String methodA,
        String methodB,
        List<String> cardsPlayedA,
        List<String> cardsPlayedB,
        List<String> cardsNotPlayedA,
        List<String> cardsNotPlayedB,
        Side winner,
        Side starter
) {

    public CardsPlayedRow {
        Objects.requireNonNull(winner, "winner");
        Objects.requireNonNull(starter, "starter");
    }

    /** Format this row as one line of {@code cards-played.txt}. */
    public String toLine() {
        return CorpusLine.of(timestamp, runId, setCode, methodA, methodB,
                CardNameList.format(cardsPlayedA), CardNameList.format(cardsPlayedB),
                CardNameList.format(cardsNotPlayedA), CardNameList.format(cardsNotPlayedB),
                winner.toString(), starter.toString());
    }

    /**
     * The row for one game of {@code parent}, from the cards that game played.
     *
     * <p>The card columns are distinct sets: a name appears once however many
     * copies the deck holds, basic lands never appear in the not-played
     * columns, and a card is in at most one of a side's two columns — played
     * if any copy was.
     */
    static CardsPlayedRow of(MatchResult parent, GamePlayer.GameOutcome game) {
        List<String> playedA = distinct(game.cardsPlayedA());
        List<String> playedB = distinct(game.cardsPlayedB());
        return new CardsPlayedRow(
                parent.timestamp(),
                parent.runId(),
                parent.setCode(),
                parent.methodA(),
                parent.methodB(),
                playedA,
                playedB,
                nonBasicMinus(parent.deckA(), playedA),
                nonBasicMinus(parent.deckB(), playedB),
                game.winner(),
                game.playFirst());
    }

    private static List<String> distinct(List<String> names) {
        return new ArrayList<>(new LinkedHashSet<>(names));
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
