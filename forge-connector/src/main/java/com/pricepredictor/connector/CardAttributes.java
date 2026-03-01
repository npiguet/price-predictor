package com.pricepredictor.connector;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Immutable card attributes used as input for a price prediction.
 * Use {@link #builder()} to construct instances.
 */
public final class CardAttributes {

    private final String name;
    private final String manaCost;
    private final List<String> types;
    private final List<String> supertypes;
    private final List<String> subtypes;
    private final String oracleText;
    private final List<String> keywords;
    private final String power;
    private final String toughness;
    private final String loyalty;

    private CardAttributes(Builder builder) {
        this.name = builder.name;
        this.manaCost = builder.manaCost;
        this.types = Collections.unmodifiableList(new ArrayList<>(builder.types));
        this.supertypes = Collections.unmodifiableList(new ArrayList<>(builder.supertypes));
        this.subtypes = Collections.unmodifiableList(new ArrayList<>(builder.subtypes));
        this.oracleText = builder.oracleText;
        this.keywords = Collections.unmodifiableList(new ArrayList<>(builder.keywords));
        this.power = builder.power;
        this.toughness = builder.toughness;
        this.loyalty = builder.loyalty;
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getName() { return name; }
    public String getManaCost() { return manaCost; }
    public List<String> getTypes() { return types; }
    public List<String> getSupertypes() { return supertypes; }
    public List<String> getSubtypes() { return subtypes; }
    public String getOracleText() { return oracleText; }
    public List<String> getKeywords() { return keywords; }
    public String getPower() { return power; }
    public String getToughness() { return toughness; }
    public String getLoyalty() { return loyalty; }

    public static final class Builder {
        private String name;
        private String manaCost;
        private List<String> types = new ArrayList<>();
        private List<String> supertypes = new ArrayList<>();
        private List<String> subtypes = new ArrayList<>();
        private String oracleText;
        private List<String> keywords = new ArrayList<>();
        private String power;
        private String toughness;
        private String loyalty;

        private Builder() {}

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder manaCost(String manaCost) {
            this.manaCost = manaCost;
            return this;
        }

        public Builder types(String... types) {
            this.types = new ArrayList<>(List.of(types));
            return this;
        }

        public Builder types(List<String> types) {
            this.types = new ArrayList<>(types);
            return this;
        }

        public Builder supertypes(String... supertypes) {
            this.supertypes = new ArrayList<>(List.of(supertypes));
            return this;
        }

        public Builder supertypes(List<String> supertypes) {
            this.supertypes = new ArrayList<>(supertypes);
            return this;
        }

        public Builder subtypes(String... subtypes) {
            this.subtypes = new ArrayList<>(List.of(subtypes));
            return this;
        }

        public Builder subtypes(List<String> subtypes) {
            this.subtypes = new ArrayList<>(subtypes);
            return this;
        }

        public Builder oracleText(String oracleText) {
            this.oracleText = oracleText;
            return this;
        }

        public Builder keywords(String... keywords) {
            this.keywords = new ArrayList<>(List.of(keywords));
            return this;
        }

        public Builder keywords(List<String> keywords) {
            this.keywords = new ArrayList<>(keywords);
            return this;
        }

        public Builder power(String power) {
            this.power = power;
            return this;
        }

        public Builder toughness(String toughness) {
            this.toughness = toughness;
            return this;
        }

        public Builder loyalty(String loyalty) {
            this.loyalty = loyalty;
            return this;
        }

        public CardAttributes build() {
            if (types == null || types.isEmpty()) {
                throw new IllegalStateException("types must be non-null and non-empty");
            }
            return new CardAttributes(this);
        }
    }
}
