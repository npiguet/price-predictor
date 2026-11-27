package com.pricepredictor.connector;

import java.util.List;
import java.util.function.Predicate;
import java.util.stream.IntStream;
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

    /** The index of the face's first top-level ability whose type matches, or -1. */
    static int firstIndexOf(CardFace card, Predicate<AbilityType> type) {
        return indicesOf(card, type).findFirst().orElse(-1);
    }

    /** The index of the face's last top-level ability whose type matches, or -1. */
    static int lastIndexOf(CardFace card, Predicate<AbilityType> type) {
        return indicesOf(card, type).reduce((earlier, later) -> later).orElse(-1);
    }

    private static IntStream indicesOf(CardFace card, Predicate<AbilityType> type) {
        List<Ability> abilities = card.abilities();
        return IntStream.range(0, abilities.size()).filter(i -> type.test(abilities.get(i).type()));
    }
}
