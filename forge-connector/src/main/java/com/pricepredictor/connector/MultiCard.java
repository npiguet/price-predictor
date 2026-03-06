package com.pricepredictor.connector;

import java.util.List;
import java.util.Objects;

/**
 * Represents a complete card with one or more faces.
 */
public record MultiCard(String layout, List<ConvertedCard> faces) {

    public MultiCard {
        Objects.requireNonNull(faces, "faces must not be null");
        if (faces.isEmpty()) {
            throw new IllegalArgumentException("faces must have at least one entry");
        }
        faces = List.copyOf(faces);
    }

    public static MultiCard singleFace(ConvertedCard face) {
        return new MultiCard(null, List.of(face));
    }

    public static MultiCard multiFace(String layout, List<ConvertedCard> faces) {
        return new MultiCard(layout, faces);
    }
}
