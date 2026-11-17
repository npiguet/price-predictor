package com.pricepredictor.connector.ability;

import com.pricepredictor.connector.AbilityDescription;

import java.util.ArrayList;
import java.util.List;

/**
 * A face's {@code Oracle:} text, read for the sentence of a keyword-built
 * trait whose script states none: a saga chapter whose effect has no
 * {@code SpellDescription$} (Fall of the Thran's II, III), an
 * {@code ETBReplacement} whose effect has none of its own (Sutured Ghoul).
 * The same fallback {@link UndescribedSpellText} gives an instant or sorcery.
 *
 * @param text the {@code Oracle:} value, its line breaks written as {@code \n}
 *             escapes, or null
 * @param name the face's name, written as CARDNAME
 */
public record FaceOracle(String text, String name) {

    /** No Oracle text: every lookup finds nothing. */
    public static final FaceOracle NONE = new FaceOracle(null, null);

    /**
     * What follows {@code prefix} on the Oracle line opening with it (a
     * chapter's "II, III —"), its name written as CARDNAME; null when no line
     * opens with it or nothing follows.
     */
    public String lineAfter(String prefix) {
        for (String line : lines()) {
            if (line.startsWith(prefix)) return named(line.substring(prefix.length()).trim());
        }
        return null;
    }

    /**
     * The first Oracle line {@code stated} contains, compared as the output
     * writes both (CARDNAME, casing), in the output's form; null when none.
     */
    public String firstLineIn(String stated) {
        String statedLine = AbilityDescription.scriptLine(stated, name);
        if (statedLine == null) return null;
        for (String line : lines()) {
            String written = AbilityDescription.scriptLine(line, name);
            if (written != null && statedLine.contains(written)) return written;
        }
        return null;
    }

    private List<String> lines() {
        if (text == null) return List.of();
        List<String> lines = new ArrayList<>();
        for (String line : text.replace("\\n", "\n").split("\n")) {
            if (!line.isBlank()) lines.add(line.trim());
        }
        return lines;
    }

    private String named(String line) {
        if (line.isEmpty()) return null;
        return name == null || name.isEmpty() ? line : line.replace(name, "CARDNAME");
    }
}
