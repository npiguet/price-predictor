package com.pricepredictor.connector;

import forge.StaticData;
import forge.card.CardDb;
import forge.card.CardEdition;
import forge.card.MagicColor;
import forge.card.mana.ManaCostShard;
import forge.item.PaperCard;
import forge.model.FModel;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * The basic lands that fill a deck's remaining slots, in proportion to the
 * WUBRG mana pips of its spells.
 *
 * <p>Needs an initialized Forge: the lands are Forge's own printings.
 */
final class BasicLandAllocation {

    /** Where basics come from when no spell's set prints them; GRN always does. */
    static final String FALLBACK_BASIC_LAND_SET = "GRN";

    private BasicLandAllocation() {
    }

    /**
     * {@code basicLandsNeeded} basic land cards, split across the five colours by
     * the pips of {@code spells}, each colour's lands grouped together in WUBRG order.
     */
    static List<PaperCard> basicsFor(List<PaperCard> spells, int basicLandsNeeded) {
        int[] basicCounts = distributeBasics(countColourPips(spells), basicLandsNeeded);

        // Source basic land cards from a set in the spell pool, else the fallback
        String setCode = findBasicLandSet(spells);
        List<PaperCard> basics = new ArrayList<>();
        for (int i = 0; i < basicCounts.length; i++) {
            if (basicCounts[i] == 0) continue;
            PaperCard landCard = basicLand(MagicColor.Constant.BASIC_LANDS.get(i), setCode);
            for (int j = 0; j < basicCounts[i]; j++) {
                basics.add(landCard);
            }
        }
        return basics;
    }

    /**
     * Count WUBRG mana pips across spells to determine basic land proportions.
     * Non-basic lands are intentionally excluded — they have no mana cost.
     */
    private static int[] countColourPips(List<PaperCard> spells) {
        int[] pips = new int[MagicColor.WUBRG.length]; // W=0, U=1, B=2, R=3, G=4
        for (PaperCard card : spells) {
            for (ManaCostShard shard : card.getRules().getMainPart().getManaCost()) {
                addShardPips(pips, shard);
            }
        }
        return pips;
    }

    /** One more pip for each colour of mana that can pay {@code shard}. */
    private static void addShardPips(int[] pips, ManaCostShard shard) {
        for (int i = 0; i < MagicColor.WUBRG.length; i++) {
            if (shard.canBePaidWithManaOfColor(MagicColor.WUBRG[i])) {
                pips[i]++;
            }
        }
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

    private static String findBasicLandSet(List<PaperCard> cards) {
        for (PaperCard card : cards) {
            CardEdition ed = StaticData.instance().getEditions().get(card.getEdition());
            if (ed != null && ed.hasBasicLands()) {
                return card.getEdition();
            }
        }
        return FALLBACK_BASIC_LAND_SET;
    }
}
