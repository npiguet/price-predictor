package com.pricepredictor.connector;

import java.util.List;

/**
 * A single ability on a card face. Implementations hold no Forge objects:
 * each factory reads its trait while the card is parsed and stores the
 * description already formatted, so rendering only adds the line prefix and
 * action number.
 */
public interface Ability {
    AbilityType type();
    String descriptionText();
    default List<Ability> subAbilities() { return List.of(); }
    default int ordinal() { return 0; }

    default String formatLine(Integer actionNumber) {
        return linePrefix(actionNumber) + descriptionText();
    }

    /** The {@code kind[n]: } a line of this ability opens with, separator included. */
    default String linePrefix(Integer actionNumber) {
        String prefix = type().getOutputPrefix();
        if (type().isActionable() && actionNumber != null) {
            return prefix + "[" + actionNumber + "]: ";
        }
        return prefix + ": ";
    }

    /**
     * Render this ability and its sub-abilities one output line per element,
     * each line carrying the ability that produced it.
     *
     * <p>A description that already contains newlines contributes several
     * output lines, all attributed to this ability; only the first carries
     * the prefix.
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
        RenderedLine.appendLines(
                lines, linePrefix(num), String.valueOf(descriptionText()), this, face);
        for (Ability sub : subAbilities()) {
            sub.appendBlock(counter, face, lines);
        }
    }
}
