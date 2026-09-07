package com.pricepredictor.connector;

import java.util.ArrayList;
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
        List<String> lines = new ArrayList<>();
        appendBlock(counter, lines, null);
        return String.join("\n", lines);
    }

    /**
     * Render this ability and its sub-abilities one output line per element,
     * optionally recording which ability produced each line.
     *
     * <p>{@code formatBlock} delegates here so the rendered text and the
     * line-to-ability map cannot drift apart: the provenance sidecar indexes
     * rendered lines, and a sidecar that indexes a different rendering than the
     * one on disk is worse than no sidecar at all.
     *
     * <p>A description that already contains newlines contributes several
     * output lines, all attributed to this ability.
     *
     * @param owners collects one entry per appended line, or null to skip
     */
    default void appendBlock(ActionCounter counter, List<String> lines,
                             List<Ability> owners) {
        Integer num;
        if (ordinal() > 0) {
            num = ordinal();
        } else if (type().isActionable()) {
            num = counter.next();
        } else {
            num = null;
        }
        for (String line : formatLine(num).split("\n", -1)) {
            lines.add(line);
            if (owners != null) owners.add(this);
        }
        for (Ability sub : subAbilities()) {
            sub.appendBlock(counter, lines, owners);
        }
    }
}
