package com.pricepredictor.connector;

import forge.game.cost.Cost;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests for Forge's Cost class.
 * Requires the full Forge transitive dependency chain (Guava, etc.) on the classpath.
 * Run via: mvn test -Pintegration
 */
@Tag("integration")
class CostTranslationTest {

    @Test
    void tapCostTranslatesToSymbol() {
        Cost cost = new Cost("T", true);
        String display = cost.toSimpleString();
        assertTrue(display.contains("{T}"), "Expected {T} in: " + display);
    }

    @Test
    void untapCostTranslatesToSymbol() {
        Cost cost = new Cost("Q", true);
        String display = cost.toSimpleString();
        assertTrue(display.contains("{Q}"), "Expected {Q} in: " + display);
    }

    @Test
    void manaCostTranslatesCorrectly() {
        Cost cost = new Cost("2 W W", true);
        String display = cost.toSimpleString();
        assertTrue(display.contains("{2}"), "Expected {2} in: " + display);
        assertTrue(display.contains("{W}"), "Expected {W} in: " + display);
    }

    @Test
    void sacrificeCostContainsSacrifice() {
        Cost cost = new Cost("Sac<1/CARDNAME>", true);
        String display = cost.toSimpleString();
        assertTrue(display.toLowerCase().contains("sacrifice"), "Expected sacrifice in: " + display);
    }

    @Test
    void discardCostContainsDiscard() {
        Cost cost = new Cost("Discard<1/Card>", true);
        String display = cost.toSimpleString();
        assertTrue(display.toLowerCase().contains("discard"), "Expected discard in: " + display);
    }

    @Test
    void payLifeCostContainsPayLife() {
        Cost cost = new Cost("PayLife<3>", true);
        String display = cost.toSimpleString();
        assertTrue(display.toLowerCase().contains("pay") && display.contains("3")
                        && display.toLowerCase().contains("life"),
                "Expected 'Pay 3 life' in: " + display);
    }

    @Test
    void payEnergyCostContainsEnergySymbol() {
        Cost cost = new Cost("PayEnergy<2>", true);
        String display = cost.toSimpleString();
        assertTrue(display.contains("{E}"), "Expected {E} in: " + display);
    }

    @Test
    void multipleCostPartsJoinedWithComma() {
        Cost cost = new Cost("T Sac<1/CARDNAME>", true);
        String display = cost.toSimpleString();
        assertTrue(display.contains("{T}"), "Expected {T} in: " + display);
        assertTrue(display.toLowerCase().contains("sacrifice"), "Expected sacrifice in: " + display);
        assertTrue(display.contains(","), "Expected comma separator in: " + display);
    }
}
