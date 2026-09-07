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

    /** Format as text output. */
    public String formatText() {
        return String.join("\n", renderLines(null));
    }

    /**
     * Render this face one output line at a time, optionally recording which
     * ability produced each line.
     *
     * <p>{@code formatText} delegates here, so the sidecar's line indices always
     * describe the file that was actually written.
     *
     * @param owners collects one entry per line — null for the header lines,
     *               which belong to no ability — or null to skip the recording
     */
    public List<String> renderLines(List<Ability> owners) {
        List<String> lines = new ArrayList<>();
        addHeader(lines, owners, "name: " + name);
        if (manaCost != null) addHeader(lines, owners, "mana cost: " + manaCost);
        addHeader(lines, owners, "types: " + types);
        if (powerToughness != null) {
            addHeader(lines, owners, "power toughness: " + powerToughness);
        }
        if (loyalty != null) addHeader(lines, owners, "loyalty: " + loyalty);
        if (defense != null) addHeader(lines, owners, "defense: " + defense);
        if (colors != null) addHeader(lines, owners, "colors: " + colors);
        if (text != null) addHeader(lines, owners, "text: " + text);

        ActionCounter counter = new ActionCounter(0);
        for (Ability ability : abilities) {
            ability.appendBlock(counter, lines, owners);
        }
        return lines;
    }

    private static void addHeader(List<String> lines, List<Ability> owners,
                                  String rendered) {
        for (String line : rendered.split("\n", -1)) {
            lines.add(line);
            if (owners != null) owners.add(null);
        }
    }
}
