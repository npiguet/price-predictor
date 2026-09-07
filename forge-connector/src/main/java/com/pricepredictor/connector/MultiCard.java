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

    /** Format the complete card as text output. */
    public String formatText() {
        return String.join("\n", renderLines(null, null));
    }

    /**
     * Render the card one output line at a time, optionally recording which
     * ability and which face produced each line.
     *
     * <p>{@code formatText} delegates here, so a sidecar's line indices always
     * index the file that was written rather than a parallel rendering.
     *
     * @param owners     collects the producing ability per line, null for
     *                   header and separator lines; may be null to skip
     * @param faceOfLine collects the face ordinal per line, -1 for lines
     *                   outside any face; may be null to skip
     */
    public List<String> renderLines(List<Ability> owners, List<Integer> faceOfLine) {
        List<String> lines = new ArrayList<>();
        if (layout == null) {
            appendFace(lines, owners, faceOfLine, 0);
            return lines;
        }

        lines.add("layout: " + layout);
        if (owners != null) owners.add(null);
        if (faceOfLine != null) faceOfLine.add(-1);

        for (int i = 0; i < faces.size(); i++) {
            if (i > 0) {
                // The blank/ALTERNATE/blank separator between faces.
                for (String separator : List.of("", "ALTERNATE", "")) {
                    lines.add(separator);
                    if (owners != null) owners.add(null);
                    if (faceOfLine != null) faceOfLine.add(-1);
                }
            }
            appendFace(lines, owners, faceOfLine, i);
        }
        return lines;
    }

    private void appendFace(List<String> lines, List<Ability> owners,
                            List<Integer> faceOfLine, int faceIndex) {
        int before = lines.size();
        lines.addAll(faces.get(faceIndex).renderLines(owners));
        if (faceOfLine != null) {
            for (int i = before; i < lines.size(); i++) {
                faceOfLine.add(faceIndex);
            }
        }
    }
}
