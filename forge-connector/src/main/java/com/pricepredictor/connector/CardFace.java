package com.pricepredictor.connector;

import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * A card face owns its identity fields and abilities, and can format itself as text.
 */
public record CardFace(
        NonBlankString name,
        Optional<NonBlankString> manaCost,
        NonBlankString types,
        Optional<NonBlankString> powerToughness,
        Optional<NonBlankString> loyalty,
        Optional<NonBlankString> defense,
        Optional<NonBlankString> colors,
        Optional<NonBlankString> text,
        List<Ability> abilities
) {

    public CardFace {
        Objects.requireNonNull(name, "name must not be null");
        Objects.requireNonNull(manaCost, "manaCost must not be null");
        Objects.requireNonNull(types, "types must not be null");
        Objects.requireNonNull(powerToughness, "powerToughness must not be null");
        Objects.requireNonNull(loyalty, "loyalty must not be null");
        Objects.requireNonNull(defense, "defense must not be null");
        Objects.requireNonNull(colors, "colors must not be null");
        Objects.requireNonNull(text, "text must not be null");
        Objects.requireNonNull(abilities, "abilities must not be null");
        abilities = List.copyOf(abilities);
    }

    /** Return a copy of this face with a different ability list. */
    public CardFace withAbilities(List<Ability> newAbilities) {
        return new CardFace(name, manaCost, types, powerToughness, loyalty, defense, colors, text, newAbilities);
    }

    /** Format as text output. */
    public String formatText() {
        StringBuilder sb = new StringBuilder();
        sb.append("name: ").append(name);
        manaCost.ifPresent(mc -> sb.append('\n').append("mana cost: ").append(mc));
        sb.append('\n').append("types: ").append(types);
        powerToughness.ifPresent(pt -> sb.append('\n').append("power toughness: ").append(pt));
        loyalty.ifPresent(lo -> sb.append('\n').append("loyalty: ").append(lo));
        defense.ifPresent(de -> sb.append('\n').append("defense: ").append(de));
        colors.ifPresent(co -> sb.append('\n').append("colors: ").append(co));
        text.ifPresent(tx -> sb.append('\n').append("text: ").append(tx));
        ActionCounter counter = new ActionCounter(0);
        for (Ability ability : abilities) {
            sb.append('\n').append(ability.formatBlock(counter));
        }
        return sb.toString();
    }
}
