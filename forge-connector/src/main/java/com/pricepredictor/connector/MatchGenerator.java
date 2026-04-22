package com.pricepredictor.connector;

import forge.StaticData;
import forge.card.CardEdition;
import forge.deck.Deck;
import forge.item.PaperCard;
import forge.item.SealedTemplate;
import forge.item.generation.UnOpenedProduct;
import forge.util.MyRandom;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Generates one complete sealed match outcome per call.
 *
 * <p>Each call picks a random eligible set, generates two 6-booster pools,
 * builds one deck per pool via {@link DeckBuilder}, plays a best-of-3 via
 * {@link GamePlayer}, and returns a {@link MatchResult}.
 *
 * <p>Eligible sets: sets with a draft/play booster template that are not
 * un-sets ({@code CardEdition.Type.FUNNY}).
 *
 * <p>{@link ForgeEnvironmentInitializer#initialize()} must have been called before use.
 */
public class MatchGenerator {

    private static final int BOOSTERS_PER_POOL = 6;

    private final List<String> eligibleSetCodes;
    private final DeckBuilder deckBuilder;
    private final GamePlayer gamePlayer;
    private final String runId;
    private final Random random;

    public MatchGenerator(
            List<String> eligibleSetCodes,
            DeckBuilder deckBuilder,
            GamePlayer gamePlayer,
            String runId) {
        if (eligibleSetCodes.isEmpty()) {
            throw new IllegalArgumentException("Eligible set list must not be empty");
        }
        if (runId == null || runId.isBlank()) {
            throw new IllegalArgumentException("runId must be non-empty");
        }
        this.eligibleSetCodes = List.copyOf(eligibleSetCodes);
        this.deckBuilder = deckBuilder;
        this.gamePlayer = gamePlayer;
        this.runId = runId;
        this.random = MyRandom.getRandom();
    }

    /**
     * Create a MatchGenerator with default DeckBuilder and GamePlayer.
     * Computes the eligible set list from Forge's StaticData.
     */
    public static MatchGenerator withDefaultBuilders(String runId) {
        return new MatchGenerator(computeEligibleSets(), new DeckBuilder(), new GamePlayer(), runId);
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
     * @return MatchResult with both decks and win counts
     */
    public MatchResult generateMatch() {
        String setCode = eligibleSetCodes.get(random.nextInt(eligibleSetCodes.size()));

        List<PaperCard> poolA = generatePool(setCode);
        List<PaperCard> poolB = generatePool(setCode);

        DeckBuilder.BuiltDeck builtA = deckBuilder.buildDeck(poolA);
        DeckBuilder.BuiltDeck builtB = deckBuilder.buildDeck(poolB);

        GamePlayer.PlayedMatch played = gamePlayer.playMatch(builtA.deck(), builtB.deck());

        StringBuilder games = new StringBuilder(played.games().size());
        StringBuilder play = new StringBuilder(played.games().size());
        for (GamePlayer.GameOutcome g : played.games()) {
            games.append(g.winner());
            play.append(g.playFirst());
        }

        return new MatchResult(
                Instant.now(),
                runId,
                setCode,
                builtA.method(),
                builtB.method(),
                toNames(builtA.deck()),
                toNames(builtB.deck()),
                games.toString(),
                play.toString(),
                played.durationSeconds()
        );
    }

    List<PaperCard> generatePool(String setCode) {
        SealedTemplate template = StaticData.instance().getBoosters().get(setCode);
        if (template == null) {
            // Fall back to draft booster template from the edition
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

    private static List<String> toNames(Deck deck) {
        List<String> names = new ArrayList<>();
        for (PaperCard card : deck.getMain().toFlatList()) {
            names.add(card.getName());
        }
        return names;
    }
}
