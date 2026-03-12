package com.pricepredictor.connector;

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
        if (layout == null) {
            return faces.get(0).formatText();
        }

        StringBuilder sb = new StringBuilder();
        sb.append("layout: ").append(layout);
        for (int i = 0; i < faces.size(); i++) {
            sb.append('\n');
            if (i > 0) {
                sb.append("\nALTERNATE\n\n");
            }
            sb.append(faces.get(i).formatText());
        }
        return sb.toString();
    }
}
