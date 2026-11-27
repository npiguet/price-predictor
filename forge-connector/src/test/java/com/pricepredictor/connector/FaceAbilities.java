package com.pricepredictor.connector;

import java.util.List;
import java.util.stream.Stream;

/** What a converted face holds and prints, for the tests that read the parser's output. */
final class FaceAbilities {

    private FaceAbilities() {
    }

    /** The converted text BatchConverter writes for a card. */
    static String rendered(MultiCard card) {
        return RenderedLine.join(card.renderLines());
    }

    /** The converted text of one face, as BatchConverter writes it. */
    static String rendered(CardFace face) {
        return RenderedLine.join(face.renderLines(0));
    }

    /** The face's top-level abilities of {@code type}, in order. */
    static List<Ability> abilitiesOfType(CardFace card, AbilityType type) {
        return card.abilities().stream().filter(a -> a.type() == type).toList();
    }

    /** How many of the face's top-level abilities are of {@code type}. */
    static long countOfType(CardFace card, AbilityType type) {
        return abilitiesOfType(card, type).size();
    }

    /** All OPTION abilities, both top-level and nested one level down as sub-abilities. */
    static List<Ability> allOptions(CardFace card) {
        return card.abilities().stream()
                .flatMap(a -> Stream.concat(Stream.of(a), a.subAbilities().stream()))
                .filter(a -> a.type() == AbilityType.OPTION)
                .toList();
    }

    /** The OPTION sub-abilities of {@code ability}, in order. */
    static List<Ability> optionsOf(Ability ability) {
        return ability.subAbilities().stream().filter(a -> a.type() == AbilityType.OPTION).toList();
    }
}
