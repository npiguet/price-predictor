package com.pricepredictor.connector;

import java.util.List;

/**
 * A single ability on a card face. Implementations hold Forge data directly
 * and defer formatting to output time.
 */
public interface Ability {
    AbilityType type();
    String descriptionText();
    default List<Ability> subAbilities() { return List.of(); }
    default int ordinal() { return 0; }

    default String formatLine(Integer actionNumber) {
        String prefix = type().getOutputPrefix();
        if (type().isActionable() && actionNumber != null) {
            return prefix + "[" + actionNumber + "]: " + descriptionText();
        }
        return prefix + ": " + descriptionText();
    }

    default String formatLine() {
        return formatLine(null);
    }

    default String formatBlock(ActionCounter counter) {
        Integer num;
        if (ordinal() > 0) {
            num = ordinal();
        } else if (type().isActionable()) {
            num = counter.next();
        } else {
            num = null;
        }
        StringBuilder sb = new StringBuilder(formatLine(num));
        for (Ability sub : subAbilities()) {
            sb.append('\n').append(sub.formatBlock(counter));
        }
        return sb.toString();
    }
}
