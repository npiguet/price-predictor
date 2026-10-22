package com.pricepredictor.connector;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * A card face owns its identity fields and abilities, and can format itself as text.
 */
public record CardFace(
        String name,
        String manaCost,
        String types,
        String powerToughness,
        String loyalty,
        String defense,
        String colors,
        String text,
        List<Ability> abilities
) {

    public CardFace {
        if (name == null || name.isEmpty()) {
            throw new IllegalArgumentException("name must not be null or empty");
        }
        if (types == null || types.isEmpty()) {
            throw new IllegalArgumentException("types must not be null or empty");
        }
        Objects.requireNonNull(abilities, "abilities must not be null");
        abilities = List.copyOf(abilities);
    }

    /** This face with its abilities replaced. */
    public CardFace withAbilities(List<Ability> replacement) {
        return new CardFace(name, manaCost, types, powerToughness, loyalty, defense,
                colors, text, replacement);
    }

    /** Format as text output. */
    public String formatText() {
        return RenderedLine.join(renderLines(0));
    }

    /**
     * Render this face one output line at a time, each line carrying the
     * ability that produced it — none for the header lines.
     *
     * <p>{@code formatText} delegates here, so the sidecar's line indices always
     * describe the file that was actually written.
     *
     * @param face this face's ordinal within its card
     */
    public List<RenderedLine> renderLines(int face) {
        List<RenderedLine> lines = new ArrayList<>();
        addHeader(lines, face, "name: " + name);
        if (manaCost != null) addHeader(lines, face, "mana cost: " + manaCost);
        addHeader(lines, face, "types: " + types);
        if (powerToughness != null) {
            addHeader(lines, face, "power toughness: " + powerToughness);
        }
        if (loyalty != null) addHeader(lines, face, "loyalty: " + loyalty);
        if (defense != null) addHeader(lines, face, "defense: " + defense);
        if (colors != null) addHeader(lines, face, "colors: " + colors);
        if (text != null) addHeader(lines, face, "text: " + text);

        ActionCounter counter = new ActionCounter(0);
        for (Ability ability : abilities) {
            ability.appendBlock(counter, face, lines);
        }
        return lines;
    }

    private static void addHeader(List<RenderedLine> lines, int face, String rendered) {
        for (String line : rendered.split("\n", -1)) {
            lines.add(new RenderedLine(line, null, face));
        }
    }
}
