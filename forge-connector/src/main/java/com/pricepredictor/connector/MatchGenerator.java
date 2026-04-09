package com.pricepredictor.connector;

import forge.StaticData;
import forge.card.CardEdition;
import forge.deck.Deck;
import forge.item.PaperCard;
import forge.item.SealedTemplate;
import forge.item.generation.UnOpenedProduct;
import forge.util.MyRandom;

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
    private final Random random;

    public MatchGenerator(List<String> eligibleSetCodes, DeckBuilder deckBuilder, GamePlayer gamePlayer) {
        if (eligibleSetCodes.isEmpty()) {
            throw new IllegalArgumentException("Eligible set list must not be empty");
        }
        this.eligibleSetCodes = List.copyOf(eligibleSetCodes);
        this.deckBuilder = deckBuilder;
        this.gamePlayer = gamePlayer;
        this.random = MyRandom.getRandom();
    }

    /**
     * Create a MatchGenerator with default DeckBuilder and GamePlayer.
     * Computes the eligible set list from Forge's StaticData.
     */
    public static MatchGenerator withDefaultBuilders() {
        return new MatchGenerator(computeEligibleSets(), new DeckBuilder(), new GamePlayer());
    }

    /**
     * Compute the list of eligible sealed-format set codes.
     * Eligible sets have a draft booster template and are not un-sets (Type.FUNNY).
     */
    public static List<String> computeEligibleSets() {
        List<String> eligible = new ArrayList<>();
        for (CardEdition edition : StaticData.instance().getEditions().getOrderedEditions()) {
            if (edition.getBoosterTemplate("Draft") != null
                    && edition.getType() != CardEdition.Type.FUNNY) {
                eligible.add(edition.getCode());
            }
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

        Deck deckA = deckBuilder.buildDeck(poolA);
        Deck deckB = deckBuilder.buildDeck(poolB);

        int[] wins = gamePlayer.playMatch(deckA, deckB);

        List<String> deckANames = toNames(deckA);
        List<String> deckBNames = toNames(deckB);

        return new MatchResult(deckANames, deckBNames, wins[0], wins[1]);
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
