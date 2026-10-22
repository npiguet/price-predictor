package com.pricepredictor.connector;

import java.util.ArrayList;
import java.util.List;

/**
 * The front face's {@code Name:}, {@code Oracle:} and {@code Draft:} lines of a
 * Forge card script, read straight from its text.
 *
 * <p>Read from the text because Forge's card model drops what is needed here:
 * the game engine ignores {@code Draft:} lines entirely, and the converter's
 * oracle fallback needs the printed oracle of the front face. Reading stops at
 * {@code ALTERNATE}, so a back face's lines never stand in for the front's.
 *
 * @param name       the first {@code Name:} value, or null if there is none
 * @param oracle     the first {@code Oracle:} value, or null if there is none
 * @param draftLines every {@code Draft:} value, in script order
 */
public record ScriptHeader(String name, String oracle, List<String> draftLines) {

    private static final String NAME = "Name:";
    private static final String ORACLE = "Oracle:";
    private static final String DRAFT = "Draft:";

    public ScriptHeader {
        draftLines = List.copyOf(draftLines);
    }

    /** The header of the script whose lines are {@code scriptLines}; values are trimmed. */
    public static ScriptHeader of(List<String> scriptLines) {
        String name = null;
        String oracle = null;
        List<String> draftLines = new ArrayList<>();
        for (String line : scriptLines) {
            String trimmed = line.trim();
            if ("ALTERNATE".equals(trimmed)) break;
            if (trimmed.startsWith(ORACLE) && oracle == null) {
                oracle = trimmed.substring(ORACLE.length()).trim();
            }
            if (trimmed.startsWith(NAME) && name == null) {
                name = trimmed.substring(NAME.length()).trim();
            }
            if (trimmed.startsWith(DRAFT)) {
                draftLines.add(trimmed.substring(DRAFT.length()).trim());
            }
        }
        return new ScriptHeader(name, oracle, draftLines);
    }
}
