package com.pricepredictor.connector;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.function.UnaryOperator;

/**
 * Represents a complete card with one or more faces.
 */
public record MultiCard(Optional<String> layout, List<CardFace> faces) {

    public MultiCard {
        Objects.requireNonNull(layout, "layout must not be null");
        Objects.requireNonNull(faces, "faces must not be null");
        if (faces.isEmpty()) {
            throw new IllegalArgumentException("faces must have at least one entry");
        }
        faces = List.copyOf(faces);
    }

    public static MultiCard singleFace(CardFace face) {
        return new MultiCard(Optional.empty(), List.of(face));
    }

    public static MultiCard multiFace(String layout, List<CardFace> faces) {
        return new MultiCard(Optional.of(layout), faces);
    }

    /** Return a copy of this card with the primary face replaced by {@code transform.apply(faces.get(0))}. */
    public MultiCard withPrimaryFace(UnaryOperator<CardFace> transform) {
        List<CardFace> newFaces = new ArrayList<>(faces);
        newFaces.set(0, transform.apply(faces.get(0)));
        return new MultiCard(layout, newFaces);
    }

    /** Format the complete card as text output. */
    public String formatText() {
        if (layout.isEmpty()) {
            return faces.get(0).formatText();
        }

        StringBuilder sb = new StringBuilder();
        sb.append("layout: ").append(layout.get());
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
