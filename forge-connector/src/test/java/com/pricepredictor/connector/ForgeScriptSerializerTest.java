package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class ForgeScriptSerializerTest {

    @Test
    void fullCardAttributesProduceCorrectScript() {
        CardAttributes card = CardAttributes.builder()
                .name("Serra Angel")
                .manaCost("3 W W")
                .supertype("Legendary")
                .type("Creature")
                .subtype("Angel")
                .oracleText("Flying, vigilance")
                .keyword("Flying").keyword("Vigilance")
                .power("4").toughness("4")
                .build();

        String script = ForgeScriptSerializer.serialize(card);

        assertTrue(script.contains("Name:Serra Angel\n"));
        assertTrue(script.contains("ManaCost:3 W W\n"));
        assertTrue(script.contains("Types:Legendary Creature Angel\n"));
        assertTrue(script.contains("Oracle:Flying, vigilance\n"));
        assertTrue(script.contains("K:Flying\n"));
        assertTrue(script.contains("K:Vigilance\n"));
        assertTrue(script.contains("PT:4/4\n"));
    }

    @Test
    void partialAttributesOnlyTypesProduceMinimalScript() {
        CardAttributes card = CardAttributes.builder()
                .type("Instant")
                .build();

        String script = ForgeScriptSerializer.serialize(card);

        assertTrue(script.contains("Types:Instant\n"));
        assertFalse(script.contains("Name:"));
        assertFalse(script.contains("ManaCost:"));
        assertFalse(script.contains("Oracle:"));
        assertFalse(script.contains("PT:"));
    }

    @Test
    void nullOptionalFieldsAreOmitted() {
        CardAttributes card = CardAttributes.builder()
                .name("Test")
                .type("Sorcery")
                .build();

        String script = ForgeScriptSerializer.serialize(card);

        assertTrue(script.contains("Name:Test\n"));
        assertTrue(script.contains("Types:Sorcery\n"));
        assertFalse(script.contains("ManaCost:"));
        assertFalse(script.contains("Oracle:"));
        assertFalse(script.contains("K:"));
        assertFalse(script.contains("PT:"));
        assertFalse(script.contains("Loyalty:"));
    }

    @Test
    void keywordsEachGetSeparateKLine() {
        CardAttributes card = CardAttributes.builder()
                .type("Creature")
                .keyword("Flying").keyword("Trample").keyword("Haste")
                .power("3").toughness("3")
                .build();

        String script = ForgeScriptSerializer.serialize(card);

        assertTrue(script.contains("K:Flying\n"));
        assertTrue(script.contains("K:Trample\n"));
        assertTrue(script.contains("K:Haste\n"));
    }

    @Test
    void powerToughnessSerializeAsPTLine() {
        CardAttributes card = CardAttributes.builder()
                .type("Creature")
                .power("2").toughness("3")
                .build();

        String script = ForgeScriptSerializer.serialize(card);
        assertTrue(script.contains("PT:2/3\n"));
    }

    @Test
    void loyaltySerializesAsLoyaltyLine() {
        CardAttributes card = CardAttributes.builder()
                .type("Planeswalker")
                .loyalty("3")
                .build();

        String script = ForgeScriptSerializer.serialize(card);
        assertTrue(script.contains("Loyalty:3\n"));
    }
}
