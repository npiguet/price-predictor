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

    /**
     * Render this ability and its sub-abilities one output line per element,
     * each line carrying the ability that produced it.
     *
     * <p>A description that already contains newlines contributes several
     * output lines, all attributed to this ability.
     *
     * @param face the ordinal of the face being rendered
     */
    default void appendBlock(ActionCounter counter, int face, List<RenderedLine> lines) {
        Integer num;
        if (ordinal() > 0) {
            num = ordinal();
        } else if (type().isActionable()) {
            num = counter.next();
        } else {
            num = null;
        }
        for (String line : formatLine(num).split("\n", -1)) {
            lines.add(new RenderedLine(line, this, face));
        }
        for (Ability sub : subAbilities()) {
            sub.appendBlock(counter, face, lines);
        }
    }
}
