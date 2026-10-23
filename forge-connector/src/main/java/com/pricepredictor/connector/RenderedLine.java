package com.pricepredictor.connector;

import java.util.List;
import java.util.StringJoiner;
import java.util.regex.Pattern;

/**
 * One line of a converted card, together with what produced it.
 *
 * <p>The text and its producer travel as one value so the provenance sidecar,
 * which indexes rendered lines, can never pair a line with another line's
 * ability: a sidecar that indexes a different rendering than the file on disk
 * is worse than no sidecar at all.
 *
 * <p>The line is held as its two parts rather than as one string because the
 * readers need the split and cannot recover it from the text: the prefix is
 * structure the tokenizer never sees, and role spans are offsets into the
 * prose after it. Cutting the text at its first {@code ": "} finds the end of
 * a prefix, and mis-cuts a description's continuation line, which has none,
 * wherever the line's own text holds one — {@code {T}: add {G}.}.
 *
 * @param prefix the structural opening, separator included —
 *               {@code activated[2]: }, {@code name: } — or empty for a
 *               continuation line of a multi-line text and for a separator
 *               line; always ASCII, so its length is the prose's offset in
 *               any encoding a reader counts characters in
 * @param prose  the rest of the line, as written
 * @param owner  the ability that rendered the line, or null for a header or
 *               separator line, which belong to no ability
 * @param face   the ordinal of the face the line belongs to, or -1 for the
 *               layout and face-separator lines outside every face
 */
public record RenderedLine(String prefix, String prose, Ability owner, int face) {

    /**
     * A line break inside a text: Forge joins some generated descriptions
     * with {@code \r\n}, and a carriage return left on the end of a line is a
     * character the converted file would carry into every reader.
     */
    private static final Pattern LINE_BREAK = Pattern.compile("\r?\n");

    /**
     * The lines a text renders as: the first opened by {@code prefix}, every
     * further one a continuation with none.
     */
    public static void appendLines(
            List<RenderedLine> lines, String prefix, String text, Ability owner, int face) {
        String[] parts = LINE_BREAK.split(text, -1);
        for (int i = 0; i < parts.length; i++) {
            lines.add(new RenderedLine(i == 0 ? prefix : "", parts[i], owner, face));
        }
    }

    /** The line as the converted file holds it. */
    public String text() {
        return prefix + prose;
    }

    /** The lines as the converted file holds them. */
    public static String join(List<RenderedLine> lines) {
        StringJoiner joined = new StringJoiner("\n");
        for (RenderedLine line : lines) joined.add(line.text());
        return joined.toString();
    }
}
