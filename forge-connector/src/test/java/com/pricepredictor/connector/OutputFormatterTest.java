package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class OutputFormatterTest {

    @Test
    void vanillaCreatureFormatsCorrectly() {
        ConvertedCard card = new ConvertedCard(
                "grizzly bears", "{1}{G}", "creature bear",
                "2/2", null, null, null, null, List.of());
        String output = OutputFormatter.formatCard(card);
        assertEquals("""
                name: grizzly bears
                mana cost: {1}{G}
                types: creature bear
                power toughness: 2/2""", output);
    }

    @Test
    void nullOptionalFieldsOmitted() {
        ConvertedCard card = new ConvertedCard(
                "lightning bolt", "{R}", "instant",
                null, null, null, null, null,
                List.of(new AbilityLine(AbilityType.SPELL,
                        "CARDNAME deals 3 damage to any target.", 1)));
        String output = OutputFormatter.formatCard(card);
        assertFalse(output.contains("power toughness:"));
        assertFalse(output.contains("loyalty:"));
        assertFalse(output.contains("defense:"));
        assertFalse(output.contains("colors:"));
        assertFalse(output.contains("text:"));
        assertTrue(output.contains("spell[1]: CARDNAME deals 3 damage to any target."));
    }

    @Test
    void abilitiesFormattedInOrder() {
        ConvertedCard card = new ConvertedCard(
                "test card", "{W}", "creature human",
                "1/1", null, null, null, null,
                List.of(
                        new AbilityLine(AbilityType.KEYWORD_PASSIVE, "flying", null),
                        new AbilityLine(AbilityType.ACTIVATED, "{T}: add {W}.", 1),
                        new AbilityLine(AbilityType.TRIGGERED, "when test card enters, draw a card.", null)
                ));
        String output = OutputFormatter.formatCard(card);
        int kwPos = output.indexOf("keyword: flying");
        int actPos = output.indexOf("activated[1]: {T}: add {W}.");
        int trigPos = output.indexOf("triggered: when test card enters, draw a card.");
        assertTrue(kwPos < actPos, "keyword should come before activated");
        assertTrue(actPos < trigPos, "activated should come before triggered");
    }

    @Test
    void actionNumbersFormattedCorrectly() {
        AbilityLine activated = new AbilityLine(AbilityType.ACTIVATED, "{T}: add {G}.", 1);
        assertEquals("activated[1]: {T}: add {G}.", activated.formatLine());

        AbilityLine keywordActive = new AbilityLine(AbilityType.KEYWORD_ACTIVE, "kicker {2}", 2);
        assertEquals("keyword[2]: kicker {2}", keywordActive.formatLine());
    }

    @Test
    void nonActionableTypesHaveNoBrackets() {
        AbilityLine keyword = new AbilityLine(AbilityType.KEYWORD_PASSIVE, "flying", null);
        assertEquals("keyword: flying", keyword.formatLine());

        AbilityLine triggered = new AbilityLine(AbilityType.TRIGGERED, "when this enters, gain 3 life.", null);
        assertEquals("triggered: when this enters, gain 3 life.", triggered.formatLine());

        AbilityLine staticLine = new AbilityLine(AbilityType.STATIC, "creatures you control get +1/+1.", null);
        assertEquals("static: creatures you control get +1/+1.", staticLine.formatLine());
    }

    @Test
    void multiFaceCardWithLayout() {
        ConvertedCard front = new ConvertedCard(
                "daring sleuth", "{1}{U}", "creature human rogue",
                "2/1", null, null, null, null, List.of());
        ConvertedCard back = new ConvertedCard(
                "bearer of overwhelming truths", null, "creature human wizard",
                "3/2", null, null, null, null, List.of());
        MultiCard multi = MultiCard.multiFace("transform", List.of(front, back));
        String output = OutputFormatter.formatMultiCard(multi);

        assertTrue(output.startsWith("layout: transform\n"));
        assertTrue(output.contains("ALTERNATE"));
        assertTrue(output.contains("name: daring sleuth"));
        assertTrue(output.contains("name: bearer of overwhelming truths"));
    }

    @Test
    void singleFaceCardNoLayoutLine() {
        ConvertedCard card = new ConvertedCard(
                "grizzly bears", "{1}{G}", "creature bear",
                "2/2", null, null, null, null, List.of());
        MultiCard single = MultiCard.singleFace(card);
        String output = OutputFormatter.formatMultiCard(single);
        assertFalse(output.contains("layout:"));
    }

    @Test
    void loyaltyFieldIncluded() {
        ConvertedCard card = new ConvertedCard(
                "jace beleren", "{1}{U}{U}", "legendary planeswalker jace",
                null, "3", null, null, null,
                List.of(new AbilityLine(AbilityType.PLANESWALKER, "+2: each player draws a card.", 1)));
        String output = OutputFormatter.formatCard(card);
        assertTrue(output.contains("loyalty: 3"));
        assertTrue(output.contains("planeswalker[1]: +2: each player draws a card."));
    }
}
