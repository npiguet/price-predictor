package com.pricepredictor.connector;

import com.pricepredictor.connector.ability.TextAbility;
import org.junit.jupiter.api.Test;

import java.util.List;

import static com.pricepredictor.connector.AbilityRendering.lineOf;
import static org.junit.jupiter.api.Assertions.*;

/** How a card face, a multi-face card and an ability line render as converted text. */
class CardFaceRenderingTest {

    @Test
    void vanillaCreatureFormatsCorrectly() {
        CardFace card = new CardFace(
                "grizzly bears", "{1}{G}", "creature bear",
                "2/2", null, null, null, List.of());
        String output = RenderedLine.join(card.renderLines(0));
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
                null, null, null, null,
                List.of(new TextAbility(AbilityType.SPELL,
                        "CARDNAME deals 3 damage to any target.")));
        String output = RenderedLine.join(card.renderLines(0));
        assertFalse(output.contains("power toughness:"));
        assertFalse(output.contains("loyalty:"));
        assertFalse(output.contains("defense:"));
        assertFalse(output.contains("text:"));
        assertTrue(output.contains("spell[1]: CARDNAME deals 3 damage to any target."));
    }

    @Test
    void abilitiesFormattedInOrder() {
        CardFace card = new CardFace(
                "test card", "{W}", "creature human",
                "1/1", null, null, null,
                List.of(
                        new TextAbility(AbilityType.STATIC, "flying"),
                        new TextAbility(AbilityType.ACTIVATED, "{T}: add {W}."),
                        new TextAbility(AbilityType.TRIGGERED, "when test card enters, draw a card.")
                ));
        String output = RenderedLine.join(card.renderLines(0));
        int staticPos = output.indexOf("static: flying");
        int actPos = output.indexOf("activated[1]: {T}: add {W}.");
        int trigPos = output.indexOf("triggered: when test card enters, draw a card.");
        assertTrue(staticPos < actPos, "static should come before activated");
        assertTrue(actPos < trigPos, "activated should come before triggered");
    }

    @Test
    void actionNumbersFormattedCorrectly() {
        CardFace card = new CardFace(
                "test card", "{G}", "creature", "1/1", null, null, null,
                List.of(new TextAbility(AbilityType.ACTIVATED, "{T}: add {G}."),
                        new TextAbility(AbilityType.ACTIVATED, "kicker {2}")));
        List<String> lines = card.renderLines(0).stream().map(RenderedLine::text).toList();

        assertTrue(lines.contains("activated[1]: {T}: add {G}."), lines.toString());
        assertTrue(lines.contains("activated[2]: kicker {2}"), lines.toString());
    }

    /** A non-actionable type still prints no action number, whatever the counter has reached. */
    @Test
    void nonActionableTypesHaveNoBrackets() {
        assertEquals("static: flying",
                lineOf(new TextAbility(AbilityType.STATIC, "flying")));
        assertEquals("triggered: when this enters, gain 3 life.",
                lineOf(new TextAbility(AbilityType.TRIGGERED, "when this enters, gain 3 life.")));
        assertEquals("static: creatures you control get +1/+1.",
                lineOf(new TextAbility(AbilityType.STATIC, "creatures you control get +1/+1.")));
    }

    @Test
    void multiFaceCardWithLayout() {
        CardFace front = new CardFace(
                "daring sleuth", "{1}{U}", "creature human rogue",
                "2/1", null, null, null, List.of());
        CardFace back = new CardFace(
                "bearer of overwhelming truths", null, "creature human wizard",
                "3/2", null, null, null, List.of());
        MultiCard multi = MultiCard.multiFace("transform", List.of(front, back));
        String output = RenderedLine.join(multi.renderLines());

        assertTrue(output.startsWith("layout: transform\n"));
        assertTrue(output.contains("ALTERNATE"));
        assertTrue(output.contains("name: daring sleuth"));
        assertTrue(output.contains("name: bearer of overwhelming truths"));
    }

    @Test
    void singleFaceCardNoLayoutLine() {
        CardFace card = new CardFace(
                "grizzly bears", "{1}{G}", "creature bear",
                "2/2", null, null, null, List.of());
        MultiCard single = MultiCard.singleFace(card);
        String output = RenderedLine.join(single.renderLines());
        assertFalse(output.contains("layout:"));
    }

    @Test
    void loyaltyFieldIncluded() {
        CardFace card = new CardFace(
                "jace beleren", "{1}{U}{U}", "legendary planeswalker jace",
                null, "3", null, null,
                List.of(new TextAbility(AbilityType.PLANESWALKER, "[+2]: each player draws a card.")));
        String output = RenderedLine.join(card.renderLines(0));
        assertTrue(output.contains("loyalty: 3"));
        assertTrue(output.contains("planeswalker[1]: [+2]: each player draws a card."));
    }
}
