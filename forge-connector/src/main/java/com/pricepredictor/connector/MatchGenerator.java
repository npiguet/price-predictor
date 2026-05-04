package com.pricepredictor.connector;

import com.pricepredictor.connector.GeneratedDecksIndex.GeneratedDeck;
import forge.StaticData;
import forge.card.CardEdition;
import forge.deck.CardPool;
import forge.deck.Deck;
import forge.deck.DeckSection;
import forge.item.PaperCard;
import forge.item.SealedTemplate;
import forge.item.generation.UnOpenedProduct;
import forge.model.FModel;
import forge.util.MyRandom;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Generates one complete sealed match outcome per call.
 *
 * <p>Each match is composed by independently choosing how to produce deck A
 * and deck B, with a same-set constraint: deck B is always built from a pool
 * (or sampled from a deck file) matching deck A's set code.
 *
 * <p><b>Side-A source</b> (controlled by {@code sideAIndex}):
 * <ul>
 *   <li>{@code null} (default): pick a random eligible set, generate a fresh
 *       6-booster pool, and call {@link DeckBuilder#buildDeck} which rolls
 *       the 4 Forge methods with weights 4:3:2:1.</li>
 *   <li>non-{@code null}: sample a deck uniformly at random from the index;
 *       the deck's recorded {@code label} becomes {@code method_A}.</li>
 * </ul>
 *
 * <p><b>Side-B source</b> (controlled by {@code sideBIndex} +
 * {@code sideBWeight}):
 * <ul>
 *   <li>{@code sideBIndex == null}: deck B is always Forge-built from a
 *       fresh pool of deck A's set, via {@link DeckBuilder#buildDeck}
 *       (weights 4:3:2:1).</li>
 *   <li>{@code sideBIndex != null}: roll between Forge methods (weights
 *       4:3:2:1, total 10) and sampling from the index (weight
 *       {@code sideBWeight}). On the file-sample branch, {@link
 *       GeneratedDecksIndex#randomDeckFromSet} filters to deck A's set code
 *       and excludes mirror matches by content equality. If no non-mirror
 *       candidate exists for that set, fall back to Forge methods.</li>
 * </ul>
 *
 * <p>{@link ForgeEnvironmentInitializer#initialize()} must have been called before use.
 */
public class MatchGenerator {

    private static final int BOOSTERS_PER_POOL = 6;

    /** Total weight of the 4 Forge deck-build methods (4 + 3 + 2 + 1). */
    static final int FORGE_METHODS_TOTAL_WEIGHT = 10;

    private final List<String> eligibleSetCodes;
    private final DeckBuilder deckBuilder;
    private final GamePlayer gamePlayer;
    private final String runId;
    private final GeneratedDecksIndex sideAIndex;
    private final GeneratedDecksIndex sideBIndex;
    private final int sideBWeight;
    private final Random random;

    public MatchGenerator(
            List<String> eligibleSetCodes,
            DeckBuilder deckBuilder,
            GamePlayer gamePlayer,
            String runId,
            GeneratedDecksIndex sideAIndex,
            GeneratedDecksIndex sideBIndex,
            int sideBWeight) {
        this(eligibleSetCodes, deckBuilder, gamePlayer, runId,
                sideAIndex, sideBIndex, sideBWeight, MyRandom.getRandom());
    }

    MatchGenerator(
            List<String> eligibleSetCodes,
            DeckBuilder deckBuilder,
            GamePlayer gamePlayer,
            String runId,
            GeneratedDecksIndex sideAIndex,
            GeneratedDecksIndex sideBIndex,
            int sideBWeight,
            Random random) {
        if (eligibleSetCodes.isEmpty()) {
            throw new IllegalArgumentException("Eligible set list must not be empty");
        }
        if (runId == null || runId.isBlank()) {
            throw new IllegalArgumentException("runId must be non-empty");
        }
        if (sideBIndex != null && sideBWeight < 1) {
            throw new IllegalArgumentException(
                    "sideBWeight must be >= 1 when sideBIndex is provided, got: " + sideBWeight);
        }
        this.eligibleSetCodes = List.copyOf(eligibleSetCodes);
        this.deckBuilder = deckBuilder;
        this.gamePlayer = gamePlayer;
        this.runId = runId;
        this.sideAIndex = sideAIndex;
        this.sideBIndex = sideBIndex;
        this.sideBWeight = sideBWeight;
        this.random = random;
    }

    /**
     * Phase-0 default constructor: no file sources on either side. Equivalent
     * to the pre-flag Phase-0 behavior — pick a random eligible set, build
     * both decks via {@link DeckBuilder}.
     */
    public static MatchGenerator phaseZero(
            List<String> eligibleSetCodes,
            DeckBuilder deckBuilder,
            GamePlayer gamePlayer,
            String runId) {
        return new MatchGenerator(eligibleSetCodes, deckBuilder, gamePlayer, runId,
                null, null, 0);
    }

    /**
     * Convenience factory: Phase-0 with default {@link DeckBuilder} and
     * {@link GamePlayer}, computing the eligible-set list from Forge's
     * {@code StaticData}. Used by integration tests that don't care about
     * the file-source flags.
     */
    public static MatchGenerator withDefaultBuilders(String runId) {
        return phaseZero(computeEligibleSets(), new DeckBuilder(), new GamePlayer(), runId);
    }

    /**
     * Minimum booster size for a set to be considered sealed-viable. Matches the
     * rule Forge itself uses in {@code AdventureEventData.isValidDraftBlock()};
     * excludes historical sets (DRK, FEM, …) whose 8-card boosters are too small
     * to yield a real sealed pool.
     */
    static final int MIN_BOOSTER_CARDS = 12;

    /**
     * Compute the list of eligible sealed-format set codes.
     * Eligible sets have a draft booster template, are not un-sets (Type.FUNNY),
     * and have at least {@link #MIN_BOOSTER_CARDS} cards per booster.
     */
    public static List<String> computeEligibleSets() {
        List<String> eligible = new ArrayList<>();
        for (CardEdition edition : StaticData.instance().getEditions().getOrderedEditions()) {
            if (edition.getType() == CardEdition.Type.FUNNY) {
                continue;
            }
            SealedTemplate template = edition.getBoosterTemplate("Draft");
            if (template == null) {
                continue;
            }
            if (template.getNumberOfCardsExpected() < MIN_BOOSTER_CARDS) {
                continue;
            }
            eligible.add(edition.getCode());
        }
        return eligible;
    }

    /**
     * Generate one complete match outcome.
     *
     * <p>Returns both the per-match {@link MatchResult} (written to
     * {@code match-outcomes.txt}) and one {@link CardsPlayedRow} per played
     * game (written to {@code cards-played.txt}).
     */
    public MatchGenerationResult generateMatch() {
        DeckSelection a = pickDeckA();
        DeckSelection b = pickDeckB(a.setCode, a.cardNames);

        GamePlayer.PlayedMatch played = gamePlayer.playMatch(a.deck, b.deck);

        StringBuilder games = new StringBuilder(played.games().size());
        StringBuilder play = new StringBuilder(played.games().size());
        for (GamePlayer.GameOutcome g : played.games()) {
            games.append(g.winner());
            play.append(g.playFirst());
        }

        Instant timestamp = Instant.now();
        MatchResult matchResult = new MatchResult(
                timestamp,
                runId,
                a.setCode,
                a.method,
                b.method,
                a.cardNames,
                b.cardNames,
                games.toString(),
                play.toString(),
                played.durationSeconds()
        );

        java.util.LinkedHashSet<String> deckASet = distinctNonBasic(a.cardNames);
        java.util.LinkedHashSet<String> deckBSet = distinctNonBasic(b.cardNames);

        List<CardsPlayedRow> rows = new ArrayList<>(played.games().size());
        for (GamePlayer.GameOutcome g : played.games()) {
            rows.add(buildRow(timestamp, matchResult, g, deckASet, deckBSet));
        }
        return new MatchGenerationResult(matchResult, rows);
    }

    private CardsPlayedRow buildRow(
            Instant timestamp,
            MatchResult parent,
            GamePlayer.GameOutcome g,
            java.util.LinkedHashSet<String> deckASet,
            java.util.LinkedHashSet<String> deckBSet) {
        List<String> playedA = distinct(g.cardsPlayedA());
        List<String> playedB = distinct(g.cardsPlayedB());
        return new CardsPlayedRow(
                timestamp,
                parent.runId(),
                parent.setCode(),
                parent.methodA(),
                parent.methodB(),
                playedA,
                playedB,
                deckMinusPlayed(deckASet, playedA),
                deckMinusPlayed(deckBSet, playedB),
                g.winner().charAt(0),
                g.playFirst().charAt(0));
    }

    private static java.util.LinkedHashSet<String> distinctNonBasic(List<String> deck) {
        java.util.LinkedHashSet<String> distinct = new java.util.LinkedHashSet<>();
        for (String name : deck) {
            if (BASIC_LAND_NAMES.contains(name)) {
                continue;
            }
            distinct.add(name);
        }
        return distinct;
    }

    private static List<String> distinct(List<String> names) {
        java.util.LinkedHashSet<String> seen = new java.util.LinkedHashSet<>(names);
        return new ArrayList<>(seen);
    }

    private static List<String> deckMinusPlayed(
            java.util.LinkedHashSet<String> deckSet, List<String> playedNames) {
        java.util.LinkedHashSet<String> playedSet = new java.util.LinkedHashSet<>(playedNames);
        List<String> remaining = new ArrayList<>();
        for (String name : deckSet) {
            if (!playedSet.contains(name)) {
                remaining.add(name);
            }
        }
        return remaining;
    }

    /** Card names whose printed type is "Basic Land — X" or Wastes / snow basics. */
    private static final java.util.Set<String> BASIC_LAND_NAMES = java.util.Set.of(
            "Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes",
            "Snow-Covered Plains", "Snow-Covered Island", "Snow-Covered Swamp",
            "Snow-Covered Mountain", "Snow-Covered Forest", "Snow-Covered Wastes"
    );

    /**
     * Pick deck A. When {@code sideAIndex} is null, roll the 4 Forge methods
     * on a fresh pool from a random eligible set. When non-null, sample
     * uniformly from the index.
     */
    DeckSelection pickDeckA() {
        if (sideAIndex == null) {
            String setCode = eligibleSetCodes.get(random.nextInt(eligibleSetCodes.size()));
            return forgeBuilt(setCode);
        }
        GeneratedDeck sampled = sideAIndex.randomDeck(random);
        return DeckSelection.fromFile(sampled, sampled.setCode());
    }

    /**
     * Pick deck B for a match whose deck A has the given set code and card
     * list. Roll between the 4 Forge methods (weights 4:3:2:1, total
     * {@link #FORGE_METHODS_TOTAL_WEIGHT}) and — if {@code sideBIndex} is
     * present — sampling from the index (weight {@code sideBWeight}). The
     * file-sample branch falls back to Forge methods when no non-mirror
     * candidate exists for {@code setCode}.
     */
    DeckSelection pickDeckB(String setCode, List<String> deckACards) {
        if (sideBIndex != null && rollIsFileSample()) {
            GeneratedDeck pick = sideBIndex.randomDeckFromSet(
                    setCode, deckACards, random);
            if (pick != null) {
                return DeckSelection.fromFile(pick, setCode);
            }
            // No non-mirror deck available for this set; fall through to Forge.
        }
        return forgeBuilt(setCode);
    }

    /**
     * Build a deck via the 4 Forge methods (weights 4:3:2:1) on a fresh
     * 6-booster pool of {@code setCode}.
     */
    private DeckSelection forgeBuilt(String setCode) {
        List<PaperCard> pool = generatePool(setCode);
        DeckBuilder.BuiltDeck built = deckBuilder.buildDeck(pool);
        return new DeckSelection(
                built.deck(), toNames(built.deck()), built.method(), setCode);
    }

    /**
     * Roll {@code true} with probability {@code sideBWeight / (10 + sideBWeight)}
     * — the "sample from sideBIndex" branch. Always returns {@code false}
     * when {@code sideBIndex} is null.
     */
    boolean rollIsFileSample() {
        if (sideBIndex == null) {
            return false;
        }
        double total = FORGE_METHODS_TOTAL_WEIGHT + sideBWeight;
        return random.nextDouble() < sideBWeight / total;
    }

    List<PaperCard> generatePool(String setCode) {
        SealedTemplate template = StaticData.instance().getBoosters().get(setCode);
        if (template == null) {
            CardEdition edition = StaticData.instance().getEditions().get(setCode);
            template = edition.getBoosterTemplate("Draft");
        }

        List<PaperCard> pool = new ArrayList<>();
        for (int i = 0; i < BOOSTERS_PER_POOL; i++) {
            List<PaperCard> booster = new UnOpenedProduct(template).get();
            for (PaperCard card : booster) {
                if (!card.getRules().getMainPart().getType().isBasicLand()) {
                    pool.add(card);
                }
            }
        }
        return pool;
    }

    /** Materialize a list of card names into a Forge {@link Deck}. */
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
        List<String> names = new ArrayList<>();
        for (PaperCard card : deck.getMain().toFlatList()) {
            names.add(card.getName());
        }
        return names;
    }

    /**
     * One side of a match: the playable {@link Deck}, its card-name list, the
     * method tag, and the set code. Bundled together so {@link #pickDeckA}
     * and {@link #pickDeckB} can return both file-sampled and Forge-built
     * decks through a single shape.
     */
    record DeckSelection(Deck deck, List<String> cardNames, String method, String setCode) {

        /**
         * Build a {@code DeckSelection} from a generated-decks-file entry,
         * materializing the card list into a Forge {@link Deck} and using
         * the deck's recorded {@code label} as the method tag. {@code setCode}
         * is taken explicitly so callers can either propagate the file's
         * own set code (side A) or substitute deck A's set code (side B,
         * where same-set pairing is enforced upstream).
         */
        static DeckSelection fromFile(GeneratedDeck deck, String setCode) {
            return new DeckSelection(
                    materializeDeck(deck.cardNames()),
                    deck.cardNames(),
                    deck.label(),
                    setCode);
        }
    }
}
