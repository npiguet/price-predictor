package com.pricepredictor.connector;

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
        StringBuilder sb = new StringBuilder();
        sb.append("name: ").append(name);
        if (manaCost != null) {
            sb.append('\n').append("mana cost: ").append(manaCost);
        }
        sb.append('\n').append("types: ").append(types);
        if (powerToughness != null) {
            sb.append('\n').append("power toughness: ").append(powerToughness);
        }
        if (loyalty != null) {
            sb.append('\n').append("loyalty: ").append(loyalty);
        }
        if (defense != null) {
            sb.append('\n').append("defense: ").append(defense);
        }
        if (colors != null) {
            sb.append('\n').append("colors: ").append(colors);
        }
        if (text != null) {
            sb.append('\n').append("text: ").append(text);
        }
        for (Ability ability : abilities) {
            sb.append('\n').append(ability.formatLine());
            for (Ability sub : ability.subAbilities()) {
                sb.append('\n').append(sub.formatLine());
            }
        }
        return sb.toString();
    }
}
