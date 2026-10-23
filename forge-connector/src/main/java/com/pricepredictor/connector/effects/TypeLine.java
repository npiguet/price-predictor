package com.pricepredictor.connector.effects;

import forge.card.CardTypeView;

import java.util.ArrayList;
import java.util.List;

/**
 * A type line as one flat list of names.
 *
 * <p>Core types first, then supertypes, then subtypes: the order a card's
 * computed type changes and a static's type contributions are both listed in,
 * so the same type line reads the same in both. Each name is spelled by
 * {@link CardSpelling#typeName}, as the snapshot spells it.
 */
final class TypeLine {

    private TypeLine() {
    }

    /** Every name on {@code types}, core → super → sub. */
    static List<String> names(CardTypeView types) {
        List<String> names = new ArrayList<>();
        for (var type : types.getCoreTypes()) {
            names.add(CardSpelling.typeName(type));
        }
        for (var supertype : types.getSupertypes()) {
            names.add(CardSpelling.typeName(supertype));
        }
        for (String subtype : types.getSubtypes()) {
            names.add(CardSpelling.typeName(subtype));
        }
        return names;
    }
}
