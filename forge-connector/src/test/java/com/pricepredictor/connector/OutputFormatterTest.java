package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class OutputFormatterTest {

    @Test
    void vanillaCreatureFormatsCorrectly() {
        CardFace card = new CardFace(
                "grizzly bears", "{1}{G}", "creature bear",
                "2/2", null, null, null, null, List.of());
        String output = card.formatText();
        assertEquals("""
                name: grizzly bears
                mana cost: {1}{G}
                types: creature bear
                power toughness: 2/2""", output);
    }

    @Test
    void nullOptionalFieldsOmitted() {
        CardFace card = new CardFace(
                "lightning bolt", "{R}", "instant",
                null, null, null, null, null,
                List.of(new Ability(AbilityType.SPELL,
                        "CARDNAME deals 3 damage to any target.", 1)));
        String output = card.formatText();
        assertFalse(output.contains("power toughness:"));
        assertFalse(output.contains("loyalty:"));
        assertFalse(output.contains("defense:"));
        assertFalse(output.contains("colors:"));
        assertFalse(output.contains("text:"));
        assertTrue(output.contains("spell[1]: CARDNAME deals 3 damage to any target."));
    }

    @Test
    void abilitiesFormattedInOrder() {
        CardFace card = new CardFace(
                "test card", "{W}", "creature human",
                "1/1", null, null, null, null,
                List.of(
                        new Ability(AbilityType.STATIC, "flying", null),
                        new Ability(AbilityType.ACTIVATED, "{T}: add {W}.", 1),
                        new Ability(AbilityType.TRIGGERED, "when test card enters, draw a card.", null)
                ));
        String output = card.formatText();
        int staticPos = output.indexOf("static: flying");
        int actPos = output.indexOf("activated[1]: {T}: add {W}.");
        int trigPos = output.indexOf("triggered: when test card enters, draw a card.");
        assertTrue(staticPos < actPos, "static should come before activated");
        assertTrue(actPos < trigPos, "activated should come before triggered");
    }

    @Test
    void actionNumbersFormattedCorrectly() {
        Ability activated = new Ability(AbilityType.ACTIVATED, "{T}: add {G}.", 1);
        assertEquals("activated[1]: {T}: add {G}.", activated.formatLine());

        Ability keywordActive = new Ability(AbilityType.ACTIVATED, "kicker {2}", 2);
        assertEquals("activated[2]: kicker {2}", keywordActive.formatLine());
    }

    @Test
    void nonActionableTypesHaveNoBrackets() {
        Ability staticAbility = new Ability(AbilityType.STATIC, "flying", null);
        assertEquals("static: flying", staticAbility.formatLine());

        Ability triggered = new Ability(AbilityType.TRIGGERED, "when this enters, gain 3 life.", null);
        assertEquals("triggered: when this enters, gain 3 life.", triggered.formatLine());

        Ability staticLine = new Ability(AbilityType.STATIC, "creatures you control get +1/+1.", null);
        assertEquals("static: creatures you control get +1/+1.", staticLine.formatLine());
    }

    @Test
    void multiFaceCardWithLayout() {
        CardFace front = new CardFace(
                "daring sleuth", "{1}{U}", "creature human rogue",
                "2/1", null, null, null, null, List.of());
        CardFace back = new CardFace(
                "bearer of overwhelming truths", null, "creature human wizard",
                "3/2", null, null, null, null, List.of());
        MultiCard multi = MultiCard.multiFace("transform", List.of(front, back));
        String output = multi.formatText();

        assertTrue(output.startsWith("layout: transform\n"));
        assertTrue(output.contains("ALTERNATE"));
        assertTrue(output.contains("name: daring sleuth"));
        assertTrue(output.contains("name: bearer of overwhelming truths"));
    }

    @Test
    void singleFaceCardNoLayoutLine() {
        CardFace card = new CardFace(
                "grizzly bears", "{1}{G}", "creature bear",
                "2/2", null, null, null, null, List.of());
        MultiCard single = MultiCard.singleFace(card);
        String output = single.formatText();
        assertFalse(output.contains("layout:"));
    }

    @Test
    void loyaltyFieldIncluded() {
        CardFace card = new CardFace(
                "jace beleren", "{1}{U}{U}", "legendary planeswalker jace",
                null, "3", null, null, null,
                List.of(new Ability(AbilityType.PLANESWALKER, "+2: each player draws a card.", 1)));
        String output = card.formatText();
        assertTrue(output.contains("loyalty: 3"));
        assertTrue(output.contains("planeswalker[1]: +2: each player draws a card."));
    }
}
