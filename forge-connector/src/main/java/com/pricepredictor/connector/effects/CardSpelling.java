package com.pricepredictor.connector.effects;

import forge.card.ColorSet;
import forge.card.MagicColor;
import forge.game.card.Card;
import forge.game.keyword.KeywordInterface;

import java.util.ArrayList;
import java.util.List;

/**
 * How a record spells a card's characteristics.
 *
 * <p>One spelling for the snapshot's entities and for every other part of a
 * record that describes the same characteristics — a static's contributions,
 * a stat change, a made token — because a model that saw {@code Creature} in
 * one and {@code creature} in the other would learn them as different types.
 */
final class CardSpelling {

    private static final char[] COLORS = {'W', 'U', 'B', 'R', 'G'};

    private CardSpelling() {
    }

    /**
     * One type-line name as the snapshot spells it.
     *
     * <p>Core types and supertypes arrive as enums and subtypes as strings, and
     * a continuous record's contributions mix all three into one list.
     */
    static String typeName(Object type) {
        String name = type instanceof Enum<?> value ? value.name() : String.valueOf(type);
        return name.toLowerCase(java.util.Locale.ROOT);
    }

    /** A colour set as the snapshot's letters, in WUBRG order. */
    static List<String> colorLetters(ColorSet colors) {
        List<String> letters = new ArrayList<>();
        if (colors == null) {
            return letters;
        }
        for (char color : COLORS) {
            if (colors.hasAnyColor(MagicColor.fromName(String.valueOf(color)))) {
                letters.add(String.valueOf(color));
            }
        }
        return letters;
    }

    /**
     * An entity's keywords as the snapshot spells them: each keyword's
     * original text, lower-cased, skipping the ones with none.
     *
     * <p>Shared with the {@code token_created} outcome, whose description of
     * a token has to read the same as that token's entry on the board.
     */
    static List<String> keywordNames(Iterable<KeywordInterface> keywords) {
        List<String> names = new ArrayList<>();
        for (KeywordInterface keyword : keywords) {
            if (!keyword.getOriginal().isEmpty()) {
                names.add(keyword.getOriginal().toLowerCase(java.util.Locale.ROOT));
            }
        }
        return names;
    }

    /**
     * The token's script stem — what FR-078 means by a token-script id — or
     * the printed name when no script can be named (an engine-built token with
     * no image key), so a token is never reported as a non-token.
     */
    static String tokenScriptIdOf(Card card) {
        if (!card.isToken()) {
            return null;
        }
        return ScriptLocator.tokenScriptStem(card).orElseGet(card::getName);
    }
}
