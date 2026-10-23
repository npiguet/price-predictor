package com.pricepredictor.connector;

import forge.StaticData;
import forge.card.CardDb;
import forge.card.CardEdition;
import forge.card.MagicColor;
import forge.card.mana.ManaCost;
import forge.card.mana.ManaCostShard;
import forge.deck.Deck;
import forge.gamemodes.limited.BoosterDeckBuilder;
import forge.gamemodes.limited.DeckColors;
import forge.gamemodes.limited.DraftedColorCommitment;
import forge.gamemodes.limited.SealedDeckBuilder;
import forge.item.PaperCard;
import forge.model.FModel;
import forge.util.MyRandom;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Builds a 40-card sealed deck from a booster pool using one of the four
 * weighted {@link BuildMethod}s.
 *
 * <p>"Nonland card" means a spell (creature, instant, sorcery, enchantment, artifact,
 * planeswalker) — cards whose type line does not include "Land". Non-basic lands (dual
 * lands, utility lands) are treated as lands throughout.
 *
 * <p>{@link ForgeEnvironmentInitializer#initialize()} must have been called before use.
 */
public class DeckBuilder {

    static final int DECK_SIZE = 40;

    /** Spells {@link BuildMethod#RANDOM} draws: a typical 23-spell, 17-land deck. */
    static final int RANDOM_DECK_SPELLS = 23;

    /** Where basics come from when no spell's set prints them; GRN always does. */
    static final String FALLBACK_BASIC_LAND_SET = "GRN";

    private final Random random;

    /** A constructed deck together with the method that produced it. */
    public record BuiltDeck(Deck deck, BuildMethod method) {}

    public DeckBuilder() {
        this.random = MyRandom.getRandom();
    }

    DeckBuilder(Random random) {
        this.random = random;
    }

    /**
     * Build a deck from the given pool using a randomly selected weighted method,
     * returning the deck alongside the method tag so callers can record which
     * strategy was used.
     *
     * @param pool booster pool cards (no basic lands)
     * @return the 40-card deck and its method tag
     */
    public BuiltDeck buildDeck(List<PaperCard> pool) {
        BuildMethod method = selectMethod();
        Deck deck = switch (method) {
            case FORGE_BEST -> buildStandard(pool);
            case FORGE_3SUB, FORGE_8SUB -> buildWithSwaps(pool, method.swaps());
            case RANDOM -> buildRandom(pool);
        };
        return new BuiltDeck(deck, method);
    }

    BuildMethod selectMethod() {
        return BuildMethod.forRoll(random.nextDouble());
    }

    Deck buildStandard(List<PaperCard> pool) {
        return new SealedDeckBuilder(new ArrayList<>(pool)).buildDeck();
    }

    /**
     * Build a deck from a <em>drafted</em> pool the way Forge itself does.
     *
     * <p>Not interchangeable with {@link #buildStandard}. Both end up in the same
     * {@code LimitedDeckBuilder}, but they differ in where the colour pair comes from, and a
     * drafted pool is the case where that matters. {@code SealedDeckBuilder} re-derives colours
     * from the top third of the ranked pool, which is right for a sealed pool nobody has made
     * decisions about. A drafted pool was assembled by a drafter that committed to two colours
     * early and let that commitment steer every later pick, so the pool only makes sense read
     * against those colours — Forge passes them to {@link BoosterDeckBuilder} for exactly that
     * reason ({@code LimitedPlayerAI.buildDeck}). Re-deriving them can land on a pair the pool
     * was never drafted for, leaving too few on-colour playables and a deck padded out with
     * basic lands.
     *
     * @param picksInPickOrder the seat's drafted cards, earliest pick first — the order is
     *                         load-bearing, since the colour commitment is replayed from it
     * @return the 40-card deck
     */
    Deck buildDrafted(List<PaperCard> picksInPickOrder) {
        DeckColors colors = DraftedColorCommitment.replay(picksInPickOrder);
        return new BoosterDeckBuilder(new ArrayList<>(picksInPickOrder), colors).buildDeck();
    }

    Deck buildWithSwaps(List<PaperCard> pool, int swapCount) {
        List<PaperCard> standard = buildStandard(pool).getMain().toFlatList();
        // Basic lands are dropped from both halves: they are rebalanced to the
        // swapped spells' pips at the end.
        NonBasics chosen = NonBasics.of(standard);
        NonBasics remaining = NonBasics.of(poolMinus(pool, standard));
        swapTypeMatched(chosen, remaining, swapCount);
        return rebalanceLands(chosen.spells(), chosen.lands());
    }

    /**
     * A card list's nonbasic cards split by type: spells, and nonbasic lands.
     * Both lists are mutable, so a swap can move cards between them in place.
     */
    record NonBasics(List<PaperCard> spells, List<PaperCard> lands) {

        static NonBasics of(List<PaperCard> cards) {
            NonBasics split = new NonBasics(new ArrayList<>(), new ArrayList<>());
            for (PaperCard card : cards) {
                if (BasicLands.isBasic(card)) {
                    continue;
                }
                boolean land = card.getRules().getMainPart().getType().isLand();
                (land ? split.lands : split.spells).add(card);
            }
            return split;
        }

        int size() {
            return spells.size() + lands.size();
        }
    }

    /** {@code pool} with one copy of each of {@code taken} removed. */
    private static List<PaperCard> poolMinus(List<PaperCard> pool, List<PaperCard> taken) {
        List<PaperCard> remaining = new ArrayList<>(pool);
        for (PaperCard card : taken) {
            remaining.remove(card);
        }
        return remaining;
    }

    /**
     * Swap {@code swapCount} of {@code chosen}'s cards for {@code remaining}'s,
     * spells only for spells and nonbasic lands only for nonbasic lands. Each
     * swap picks uniformly from all of {@code chosen}; a pick whose type has no
     * replacement left still uses up a swap.
     */
    private void swapTypeMatched(NonBasics chosen, NonBasics remaining, int swapCount) {
        for (int swapsLeft = swapCount; swapsLeft > 0 && chosen.size() > 0; swapsLeft--) {
            int pick = random.nextInt(chosen.size());
            boolean spell = pick < chosen.spells().size();
            List<PaperCard> from = spell ? chosen.spells() : chosen.lands();
            List<PaperCard> replacements = spell ? remaining.spells() : remaining.lands();
            if (replacements.isEmpty()) {
                continue;
            }
            from.remove(spell ? pick : pick - chosen.spells().size());
            from.add(replacements.remove(random.nextInt(replacements.size())));
        }
    }

    Deck buildRandom(List<PaperCard> pool) {
        // Pick 23 random spells from pool; non-basic lands in the pool are excluded
        List<PaperCard> spellPool = new ArrayList<>();
        for (PaperCard card : pool) {
            if (!card.getRules().getMainPart().getType().isLand()) {
                spellPool.add(card);
            }
        }

        int pickCount = Math.min(RANDOM_DECK_SPELLS, spellPool.size());
        List<PaperCard> chosen = new ArrayList<>(pickCount);
        List<PaperCard> mutablePool = new ArrayList<>(spellPool);
        for (int i = 0; i < pickCount; i++) {
            int idx = random.nextInt(mutablePool.size());
            chosen.add(mutablePool.remove(idx));
        }

        return rebalanceLands(chosen, List.of());
    }

    /**
     * Build a 40-card deck from the given spells and non-basic lands by adding basic
     * lands proportional to the WUBRG mana pip counts of the spells.
     *
     * <p>Note: {@code LimitedDeckBuilder.addLands()} is private and cannot be called
     * externally. Passing cards to {@code SealedDeckBuilder} as a pool re-selects a
     * subset rather than treating all cards as already chosen, producing severely
     * land-heavy decks. This method reimplements the pip-proportional allocation directly.
     *
     * @param spells        chosen nonland spells (creatures, instants, etc.)
     * @param nonbasicLands non-basic lands to retain unchanged from the standard deck
     * @return 40-card Deck
     */
    Deck rebalanceLands(List<PaperCard> spells, List<PaperCard> nonbasicLands) {
        int basicLandsNeeded = DECK_SIZE - spells.size() - nonbasicLands.size();
        if (basicLandsNeeded < 0) {
            throw new IllegalArgumentException("Cannot build a " + DECK_SIZE + "-card deck from "
                    + spells.size() + " spells and " + nonbasicLands.size() + " non-basic lands");
        }

        int[] basicCounts = distributeBasics(countColourPips(spells), basicLandsNeeded);

        // Source basic land cards from a set in the spell pool, else the fallback
        String setCode = findBasicLandSet(spells);
        List<PaperCard> cards = new ArrayList<>(spells);
        cards.addAll(nonbasicLands);
        for (int i = 0; i < basicCounts.length; i++) {
            if (basicCounts[i] == 0) continue;
            PaperCard landCard = basicLand(MagicColor.Constant.BASIC_LANDS.get(i), setCode);
            for (int j = 0; j < basicCounts[i]; j++) {
                cards.add(landCard);
            }
        }
        return DeckMaterializer.deckOf(cards);
    }

    /**
     * Count WUBRG mana pips across spells to determine basic land proportions.
     * Non-basic lands are intentionally excluded — they have no mana cost.
     */
    private static int[] countColourPips(List<PaperCard> spells) {
        int[] pips = new int[MagicColor.WUBRG.length]; // W=0, U=1, B=2, R=3, G=4
        for (PaperCard card : spells) {
            ManaCost mc = card.getRules().getMainPart().getManaCost();
            for (ManaCostShard shard : mc) {
                for (int i = 0; i < MagicColor.WUBRG.length; i++) {
                    if (shard.canBePaidWithManaOfColor(MagicColor.WUBRG[i])) {
                        pips[i]++;
                    }
                }
            }
        }
        return pips;
    }

    /**
     * Distribute basic land slots proportionally to the pips (greedy, WUBRG order;
     * the last required colour absorbs any rounding remainder).
     */
    private static int[] distributeBasics(int[] pips, int basicLandsNeeded) {
        int totalPips = Arrays.stream(pips).sum();
        if (totalPips == 0) {
            // Colorless deck: distribute equally across all five types
            pips = new int[pips.length];
            Arrays.fill(pips, 1);
            totalPips = pips.length;
        }

        int[] counts = new int[pips.length];
        int remaining = basicLandsNeeded;
        int remainingPips = totalPips;
        for (int i = 0; i < pips.length; i++) {
            if (pips[i] == 0 || remaining == 0) continue;
            int count = (remainingPips == pips[i])
                    ? remaining
                    : Math.round((float) remaining * pips[i] / remainingPips);
            counts[i] = Math.max(0, Math.min(count, remaining));
            remaining -= counts[i];
            remainingPips -= pips[i];
        }
        return counts;
    }

    /**
     * The basic land from {@code setCode}, or its default printing when that set has none —
     * a missing land must fail loudly, since a silently skipped slot leaves the deck short.
     */
    static PaperCard basicLand(String landName, String setCode) {
        CardDb cards = FModel.getMagicDb().getCommonCards();
        PaperCard land = cards.getCard(landName, setCode);
        if (land == null) {
            land = cards.getCard(landName);
        }
        if (land == null) {
            throw new IllegalStateException(
                    "No printing of basic land '" + landName + "' (requested set " + setCode + ")");
        }
        return land;
    }

    private String findBasicLandSet(List<PaperCard> cards) {
        for (PaperCard card : cards) {
            CardEdition ed = StaticData.instance().getEditions().get(card.getEdition());
            if (ed != null && CardEdition.Predicates.hasBasicLands.test(ed)) {
                return card.getEdition();
            }
        }
        return FALLBACK_BASIC_LAND_SET;
    }
}
