package com.pricepredictor.connector;

import java.util.List;
import java.util.StringJoiner;

/**
 * One line of a converted card, together with what produced it.
 *
 * <p>The text and its producer travel as one value so the provenance sidecar,
 * which indexes rendered lines, can never pair a line with another line's
 * ability: a sidecar that indexes a different rendering than the file on disk
 * is worse than no sidecar at all.
 *
 * @param owner the ability that rendered the line, or null for a header or
 *              separator line, which belong to no ability
 * @param face  the ordinal of the face the line belongs to, or -1 for the
 *              layout and face-separator lines outside every face
 */
public record RenderedLine(String text, Ability owner, int face) {

    /** The lines as the converted file holds them. */
    public static String join(List<RenderedLine> lines) {
        StringJoiner joined = new StringJoiner("\n");
        for (RenderedLine line : lines) joined.add(line.text());
        return joined.toString();
    }
}
