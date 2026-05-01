package com.pricepredictor.connector;

import com.pricepredictor.connector.GeneratedDecksIndex.GeneratedDeck;
import forge.deck.CardPool;
import forge.deck.Deck;
import forge.deck.DeckSection;
import forge.item.PaperCard;
import forge.model.FModel;
import forge.util.MyRandom;

import java.time.Instant;
import java.util.List;
import java.util.Random;

/**
 * Generates one self-play match outcome per call.
 *
 * <p>Picks deck A from a {@link GeneratedDecksIndex} (scorer-built decks loaded
 * from a generated-decks file), then builds deck B by one of 5 weighted methods:
 *
 * <ul>
 *   <li>Methods 1-4 (weights 4:3:2:1, total 10/14): generate a fresh pool from
 *       deck A's set via {@link PoolGenerator}, then call {@link DeckBuilder}
 *       which itself rolls method 1-4 with weights 4:3:2:1.</li>
 *   <li>Method 5 (weight 4/14): pick a random other deck from the index that
 *       has the same set code as deck A.</li>
 * </ul>
 *
 * <p>Same-set pairing is enforced for all methods. Combined deck B method
 * weights are 4:3:2:1:4 across methods 1-5.
 *
 * <p>{@link ForgeEnvironmentInitializer#initialize()} must have been called
 * before invoking {@link #generateMatch()}.
 */
public class SelfPlayMatchGenerator {

    /**
     * Cumulative thresholds for method 5. Below {@code METHOD5_THRESHOLD} → use
     * method 5; otherwise delegate to {@link DeckBuilder#buildDeck(List)} which
     * picks methods 1-4 with weights 4:3:2:1. The combined effective weights
     * are 4:3:2:1:4 (total 14).
     */
    static final double METHOD5_THRESHOLD = 4.0 / 14.0;

    private final GeneratedDecksIndex index;
    private final DeckBuilder deckBuilder;
    private final PoolGenerator poolGenerator;
    private final GamePlayer gamePlayer;
    private final List<String> eligibleSets;
    private final String runId;
    private final Random random;

    public SelfPlayMatchGenerator(
            GeneratedDecksIndex index,
            DeckBuilder deckBuilder,
            PoolGenerator poolGenerator,
            GamePlayer gamePlayer,
            List<String> eligibleSets,
            String runId) {
        this(index, deckBuilder, poolGenerator, gamePlayer, eligibleSets,
                runId, MyRandom.getRandom());
    }

    SelfPlayMatchGenerator(
            GeneratedDecksIndex index,
            DeckBuilder deckBuilder,
            PoolGenerator poolGenerator,
            GamePlayer gamePlayer,
            List<String> eligibleSets,
            String runId,
            Random random) {
        if (runId == null || runId.isBlank()) {
            throw new IllegalArgumentException("runId must be non-empty");
        }
        this.index = index;
        this.deckBuilder = deckBuilder;
        this.poolGenerator = poolGenerator;
        this.gamePlayer = gamePlayer;
        this.eligibleSets = List.copyOf(eligibleSets);
        this.runId = runId;
        this.random = random;
    }

    /**
     * Roll the deck B method.
     *
     * @return 5 if method 5 is selected, otherwise 0 to indicate "delegate to
     *         {@link DeckBuilder} which picks 1-4 internally".
     */
    int rollIsMethod5() {
        return random.nextDouble() < METHOD5_THRESHOLD ? 5 : 0;
    }

    /** Pick deck A uniformly at random from the index. */
    GeneratedDeck pickDeckA() {
        return index.randomDeck(random);
    }

    /**
     * Pick deck B for method 5: a random scorer-built deck from the same set as
     * deck A, never returning deck A itself. Returns {@code null} if no
     * other deck with this set exists in the index.
     */
    GeneratedDeck pickMethod5DeckB(GeneratedDeck deckA) {
        return index.randomDeckFromSet(deckA.setCode(), deckA, random);
    }

    /**
     * Generate one self-play match.
     *
     * @return {@link MatchResult} with deck A from the index and deck B built
     *         by the rolled method, with same-set pairing enforced.
     */
    public MatchResult generateMatch() {
        GeneratedDeck deckA = pickDeckA();
        int methodGroup = rollIsMethod5();

        List<String> deckBNames;
        Deck deckBForge;
        String methodB;

        if (methodGroup == 5) {
            GeneratedDeck deckBFromIndex = pickMethod5DeckB(deckA);
            if (deckBFromIndex == null) {
                // Fallback: not enough decks of this set in the index — fall through
                // to the builder path. The data-model assumption (≥2 decks per set)
                // makes this rare.
                List<PaperCard> pool = poolGenerator.generatePool(deckA.setCode());
                DeckBuilder.BuiltDeck built = deckBuilder.buildDeck(pool);
                deckBNames = toNames(built.deck());
                deckBForge = built.deck();
                methodB = built.method();
            } else {
                deckBNames = deckBFromIndex.cardNames();
                deckBForge = materializeDeck(deckBNames);
                methodB = deckBFromIndex.label();
            }
        } else {
            List<PaperCard> pool = poolGenerator.generatePool(deckA.setCode());
            DeckBuilder.BuiltDeck built = deckBuilder.buildDeck(pool);
            deckBNames = toNames(built.deck());
            deckBForge = built.deck();
            methodB = built.method();
        }

        Deck deckAForge = materializeDeck(deckA.cardNames());

        GamePlayer.PlayedMatch played = gamePlayer.playMatch(deckAForge, deckBForge);

        StringBuilder games = new StringBuilder(played.games().size());
        StringBuilder play = new StringBuilder(played.games().size());
        for (GamePlayer.GameOutcome g : played.games()) {
            games.append(g.winner());
            play.append(g.playFirst());
        }

        return new MatchResult(
                Instant.now(),
                runId,
                deckA.setCode(),
                deckA.label(),
                methodB,
                deckA.cardNames(),
                deckBNames,
                games.toString(),
                play.toString(),
                played.durationSeconds()
        );
    }

    /** Resolve a list of card names into a Forge {@link Deck}. */
    static Deck materializeDeck(List<String> cardNames) {
        Deck deck = new Deck();
        CardPool main = deck.getOrCreate(DeckSection.Main);
        for (String name : cardNames) {
            PaperCard card = FModel.getMagicDb().getCommonCards().getCard(name);
            if (card != null) {
                main.add(card);
            }
        }
        return deck;
    }

    static List<String> toNames(Deck deck) {
        return deck.get(DeckSection.Main).toFlatList().stream()
                .map(PaperCard::getName)
                .toList();
    }
}
