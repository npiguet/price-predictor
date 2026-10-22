package com.pricepredictor.connector;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Represents a complete card with one or more faces.
 */
public record MultiCard(String layout, List<CardFace> faces) {

    public MultiCard {
        Objects.requireNonNull(faces, "faces must not be null");
        if (faces.isEmpty()) {
            throw new IllegalArgumentException("faces must have at least one entry");
        }
        faces = List.copyOf(faces);
    }

    public static MultiCard singleFace(CardFace face) {
        return new MultiCard(null, List.of(face));
    }

    public static MultiCard multiFace(String layout, List<CardFace> faces) {
        return new MultiCard(layout, faces);
    }

    /** This card with its first face replaced. */
    public MultiCard withFirstFace(CardFace replacement) {
        List<CardFace> replaced = new ArrayList<>(faces);
        replaced.set(0, replacement);
        return new MultiCard(layout, replaced);
    }

    /** Format the complete card as text output. */
    String formatText() {
        return RenderedLine.join(renderLines());
    }

    /**
     * Render the card one output line at a time, each line carrying the
     * ability and the face that produced it.
     *
     * <p>{@code formatText} delegates here, so a sidecar's line indices always
     * index the file that was written rather than a parallel rendering.
     */
    public List<RenderedLine> renderLines() {
        if (layout == null) {
            return faces.get(0).renderLines(0);
        }
        List<RenderedLine> lines = new ArrayList<>();
        lines.add(new RenderedLine("layout: " + layout, null, -1));
        for (int i = 0; i < faces.size(); i++) {
            if (i > 0) {
                // The blank/ALTERNATE/blank separator between faces.
                for (String separator : List.of("", "ALTERNATE", "")) {
                    lines.add(new RenderedLine(separator, null, -1));
                }
            }
            lines.addAll(faces.get(i).renderLines(i));
        }
        return lines;
    }
}
