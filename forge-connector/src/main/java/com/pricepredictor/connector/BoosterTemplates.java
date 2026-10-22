package com.pricepredictor.connector;

import forge.StaticData;
import forge.card.CardEdition;
import forge.item.SealedTemplate;

import java.util.Optional;

/** Which booster a set's packs are opened from, for the match and draft workers. */
final class BoosterTemplates {

    private BoosterTemplates() {
    }

    /**
     * The set's booster template: its sealed booster where Forge has one,
     * else its edition's draft booster, else empty.
     *
     * <p>Empty rather than an exception for a set Forge does not know: the set
     * code can come from a decks file or a {@code -D} property, and an unknown
     * edition is a configuration error the caller names, not a
     * {@code NullPointerException} from inside the lookup.
     */
    static Optional<SealedTemplate> forSet(String setCode) {
        SealedTemplate template = StaticData.instance().getBoosters().get(setCode);
        if (template != null) {
            return Optional.of(template);
        }
        CardEdition edition = StaticData.instance().getEditions().get(setCode);
        return edition == null
                ? Optional.empty()
                : Optional.ofNullable(edition.getBoosterTemplate("Draft"));
    }
}
