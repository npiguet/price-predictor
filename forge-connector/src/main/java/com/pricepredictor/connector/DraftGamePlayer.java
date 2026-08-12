package com.pricepredictor.connector;

import forge.deck.CardPool;
import forge.deck.Deck;
import forge.deck.DeckSection;
import forge.item.PaperCard;
import forge.model.FModel;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

/**
 * Plays one drawn pairing and appends the result as a match-outcome row.
 *
 * <p>A {@code deck} seat plays the deck the corpus recorded for it. A
 * {@code pool} seat's deck is built from its drafted pool by the same builder Forge
 * uses for a drafted pool, {@link forge.gamemodes.limited.BoosterDeckBuilder}, so the
 * same drafted cards can be compared under two builders. Either way the row records
 * the deck <em>as played</em>, never the pool.
 *
 * <p>Unlike {@link ValidationMatchPlayer} nothing here is positionally aligned
 * with an input file, so a pairing that throws simply writes no row — there is
 * no filler outcome to keep line counts in step, and a fabricated row would
 * corrupt the tally.
 *
 * <p>{@link ForgeEnvironmentInitializer#initialize()} must have been called
 * before use.
 */
public class DraftGamePlayer {

    private final GamePlayer gamePlayer;
    private final MatchResultWriter writer;
    private final DeckBuilder deckBuilder;
    private final String runId;

    public DraftGamePlayer(GamePlayer gamePlayer, MatchResultWriter writer, String runId) {
        this(gamePlayer, writer, new DeckBuilder(), runId);
    }

    DraftGamePlayer(
            GamePlayer gamePlayer,
            MatchResultWriter writer,
            DeckBuilder deckBuilder,
            String runId) {
        this.gamePlayer = gamePlayer;
        this.writer = writer;
        this.deckBuilder = deckBuilder;
        this.runId = runId;
    }

    /**
     * Play one pairing and append its row.
     *
     * @return the row written, for tests and logging
     */
    public MatchResult play(SeatTableIndex.Pairing pairing) {
        Deck deckA = resolveDeck(pairing.a());
        Deck deckB = resolveDeck(pairing.b());

        GamePlayer.PlayedMatch played = gamePlayer.playMatch(deckA, deckB);

        MatchResult result = new MatchResult(
                Instant.now(),
                runId,
                pairing.a().setCode(),
                pairing.a().label(),
                pairing.b().label(),
                cardNames(deckA),
                cardNames(deckB),
                winnerSequence(played),
                playSequence(played),
                played.durationSeconds());

        writer.write(result);
        return result;
    }

    /**
     * A {@code deck} seat plays what the corpus recorded; a {@code pool} seat's
     * deck comes from Forge's own draft builder.
     *
     * <p>The pool arrives in pick order and must stay that way — {@code buildDrafted}
     * replays it to recover the colours the seat committed to while drafting.
     */
    private Deck resolveDeck(SeatTableIndex.SeatEntry seat) {
        List<PaperCard> cards = resolveCards(seat.cardNames());
        if (seat.isPool()) {
            return deckBuilder.buildDrafted(cards);
        }
        Deck deck = new Deck();
        CardPool main = deck.getOrCreate(DeckSection.Main);
        for (PaperCard card : cards) {
            main.add(card);
        }
        return deck;
    }

    private static List<PaperCard> resolveCards(List<String> names) {
        List<PaperCard> cards = new ArrayList<>(names.size());
        for (String name : names) {
            PaperCard card = FModel.getMagicDb().getCommonCards().getCard(name);
            if (card != null) {
                cards.add(card);
            }
        }
        return cards;
    }

    // ── Static helpers (package-visible for testing) ────────────────

    /** All cards in a deck's main section, duplicates repeated. */
    static List<String> cardNames(Deck deck) {
        List<String> names = new ArrayList<>();
        for (var entry : deck.getOrCreate(DeckSection.Main)) {
            for (int i = 0; i < entry.getValue(); i++) {
                names.add(entry.getKey().getName());
            }
        }
        return names;
    }

    /** Per-game winner sequence, e.g. {@code "ABA"}. */
    static String winnerSequence(GamePlayer.PlayedMatch played) {
        StringBuilder sb = new StringBuilder();
        for (GamePlayer.GameOutcome game : played.games()) {
            sb.append(game.winner());
        }
        return sb.toString();
    }

    /** Per-game play-first sequence, same length as the winner sequence. */
    static String playSequence(GamePlayer.PlayedMatch played) {
        StringBuilder sb = new StringBuilder();
        for (GamePlayer.GameOutcome game : played.games()) {
            sb.append(game.playFirst());
        }
        return sb.toString();
    }
}
