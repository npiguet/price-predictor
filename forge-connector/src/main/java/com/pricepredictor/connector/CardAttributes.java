package com.pricepredictor.connector;

import lombok.Builder;
import lombok.Singular;
import lombok.Value;

import java.util.List;

/**
 * Immutable card attributes used as input for a price prediction.
 * Use {@link #builder()} to construct instances.
 */
@Value
@Builder
public class CardAttributes {
    String name;
    String manaCost;
    @Singular List<String> types;
    @Singular List<String> supertypes;
    @Singular List<String> subtypes;
    String oracleText;
    @Singular List<String> keywords;
    String power;
    String toughness;
    String loyalty;
}
