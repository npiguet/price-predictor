package com.pricepredictor.connector;

import forge.StaticData;
import forge.card.CardEdition;
import forge.item.SealedTemplate;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/** Which booster a set's packs are opened from, for the pool, match and draft workers. */
final class BoosterTemplates {

    private BoosterTemplates() {
    }

    /**
     * The booster type every worker opens: Forge keys an edition's templates
     * by type, and the draft booster is what a sealed pool or a draft pod is
     * opened from in paper.
     */
    static final String DRAFT_BOOSTER = "Draft";

    /**
     * Minimum booster size for a set to be considered sealed-viable. Matches the
     * rule Forge itself uses in {@code AdventureEventData.isValidDraftBlock()};
     * excludes historical sets (DRK, FEM, …) whose 8-card boosters are too small
     * to yield a real sealed pool.
     */
    static final int MIN_BOOSTER_CARDS = 12;

    /**
     * The set's draft booster template, or empty.
     *
     * <p>Read off the edition rather than {@code StaticData.getBoosters()}: that
     * map holds the same Draft templates (keyed by bare code, other booster
     * types under an affixed key), but only under each set's own code, while
     * the editions lookup also resolves a set's alias and two-letter code.
     *
     * <p>Empty rather than an exception for a set Forge does not know: the set
     * code can come from a decks file or a {@code -D} property, and an unknown
     * edition is a configuration error the caller names, not a
     * {@code NullPointerException} from inside the lookup.
     */
    static Optional<SealedTemplate> forSet(String setCode) {
        return edition(setCode).map(edition -> edition.getBoosterTemplate(DRAFT_BOOSTER));
    }

    /**
     * The codes of every set a sealed pool can be opened from, in Forge's
     * edition order.
     *
     * <p>Eligible sets have a draft booster template, are not un-sets
     * ({@code Type.FUNNY}), and have at least {@link #MIN_BOOSTER_CARDS} cards
     * per booster. Kept beside {@link #forSet} because both answer which
     * booster a worker opens: every set listed here is one {@code forSet}
     * finds a template for.
     */
    static List<String> eligibleSealedSets() {
        List<String> eligible = new ArrayList<>();
        for (CardEdition edition : StaticData.instance().getEditions().getOrderedEditions()) {
            if (edition.getType() == CardEdition.Type.FUNNY) {
                continue;
            }
            SealedTemplate template = edition.getBoosterTemplate(DRAFT_BOOSTER);
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
     * The code the set's edition goes by, which an alias resolves to, or
     * {@code setCode} unchanged when Forge knows no such set.
     *
     * <p>What a corpus line records: every downstream join compares set codes
     * as strings, so a pool opened under {@code CON} and one opened under
     * {@code CFX} must not read as two sets.
     */
    static String canonicalCode(String setCode) {
        return edition(setCode).map(CardEdition::getCode).orElse(setCode);
    }

    private static Optional<CardEdition> edition(String setCode) {
        return Optional.ofNullable(StaticData.instance().getEditions().get(setCode));
    }
}
