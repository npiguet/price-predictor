package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class CardAttributesTest {

    @Test
    void builderWithTypesBuildSuccessfully() {
        CardAttributes card = CardAttributes.builder()
                .type("Creature")
                .build();

        assertEquals(List.of("Creature"), card.getTypes());
    }

    @Test
    void allOptionalFieldsCanBeSetAndRetrieved() {
        CardAttributes card = CardAttributes.builder()
                .name("Test Card")
                .manaCost("2 W W")
                .type("Creature")
                .supertype("Legendary")
                .subtype("Human").subtype("Wizard")
                .oracleText("Some ability text")
                .keyword("Flying").keyword("Vigilance")
                .power("3")
                .toughness("4")
                .loyalty("5")
                .build();

        assertEquals("Test Card", card.getName());
        assertEquals("2 W W", card.getManaCost());
        assertEquals(List.of("Creature"), card.getTypes());
        assertEquals(List.of("Legendary"), card.getSupertypes());
        assertEquals(List.of("Human", "Wizard"), card.getSubtypes());
        assertEquals("Some ability text", card.getOracleText());
        assertEquals(List.of("Flying", "Vigilance"), card.getKeywords());
        assertEquals("3", card.getPower());
        assertEquals("4", card.getToughness());
        assertEquals("5", card.getLoyalty());
    }

    @Test
    void typesSingularAndCollectionOverloadsBothWork() {
        CardAttributes fromSingular = CardAttributes.builder()
                .type("Creature").type("Artifact")
                .build();

        CardAttributes fromCollection = CardAttributes.builder()
                .types(List.of("Creature", "Artifact"))
                .build();

        assertEquals(fromSingular.getTypes(), fromCollection.getTypes());
    }

    @Test
    void builtObjectIsImmutable() {
        CardAttributes card = CardAttributes.builder()
                .type("Creature")
                .keyword("Flying")
                .build();

        assertThrows(UnsupportedOperationException.class, () ->
                card.getTypes().add("Instant")
        );
        assertThrows(UnsupportedOperationException.class, () ->
                card.getKeywords().add("Trample")
        );
    }
}
