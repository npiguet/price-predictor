package com.pricepredictor.connector;

import forge.deck.Deck;
import forge.item.PaperCard;

import java.time.Instant;
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
                DeckMaterializer.names(deckA),
                DeckMaterializer.names(deckB),
                played.winnerSequence(),
                played.playSequence(),
                played.durationSeconds());

        writer.write(result);
        return result;
    }

    /**
     * Why a seat with a name Forge cannot resolve is refused. The row this
     * match would write is a tally entry for a seat's drafted deck; played
     * short, it would credit or blame the agent for a deck it never drafted.
     * Refusing writes no row, the outcome the class contract already gives a
     * pairing that throws.
     */
    static final String UNRESOLVED_SEAT =
            "A draft-game row records the seat's drafted deck, so a short deck "
                    + "would put a result in the tally for a deck nobody drafted.";

    /**
     * A {@code deck} seat plays what the corpus recorded; a {@code pool} seat's
     * deck comes from Forge's own draft builder.
     *
     * <p>The pool arrives in pick order and must stay that way — {@code buildDrafted}
     * replays it to recover the colours the seat committed to while drafting.
     */
    private Deck resolveDeck(SeatTableIndex.SeatEntry seat) {
        List<PaperCard> cards = DeckMaterializer.resolve(
                seat.cardNames(), DeckMaterializer.UnresolvedPolicy.FAIL, UNRESOLVED_SEAT);
        if (seat.isPool()) {
            return deckBuilder.buildDrafted(cards);
        }
        return DeckMaterializer.deckOf(cards);
    }
}
