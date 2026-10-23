package com.pricepredictor.connector.effects;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Prose role spans over a rendered ability line.
 *
 * <p>The spans are what let the encoder learn that {@code {R}} in a cost means
 * the opposite of {@code {R}} in an effect; the role-polarity probe measures
 * exactly that, so a span that mislabels a cost as an effect would show up
 * there as a model failure rather than a data one.
 */
class RoleSpansTest {

    private static String textOf(String prose, RoleSpans.Span span) {
        return prose.substring(span.start(), span.end());
    }

    private static RoleSpans.Span roleIn(List<RoleSpans.Span> spans, SpanRole role) {
        return spans.stream().filter(s -> s.role() == role).findFirst()
                .orElse(null);
    }

    @Test
    void anActivatedAbilitySplitsAtItsCostColon() {
        String prose = "{T}, pay 2 life: draw a card.";
        List<RoleSpans.Span> spans = RoleSpans.of(prose, true);
        assertEquals("{T}, pay 2 life:", textOf(prose, roleIn(spans, SpanRole.COST)));
        assertEquals(" draw a card.", textOf(prose, roleIn(spans, SpanRole.EFFECT)));
    }

    @Test
    void aTriggeredAbilitySplitsAtItsFirstTopLevelComma() {
        String prose = "whenever this creature attacks, draw a card.";
        List<RoleSpans.Span> spans = RoleSpans.of(prose, false);
        assertEquals("whenever this creature attacks,",
                textOf(prose, roleIn(spans, SpanRole.TRIGGER_CONDITION)));
        assertEquals(" draw a card.",
                textOf(prose, roleIn(spans, SpanRole.EFFECT)));
    }

    @Test
    void aReminderTextCommaDoesNotSplitTheCondition() {
        String prose = "when this enters (from anywhere, somehow), draw a card.";
        List<RoleSpans.Span> spans = RoleSpans.of(prose, false);
        assertEquals("when this enters (from anywhere, somehow),",
                textOf(prose, roleIn(spans, SpanRole.TRIGGER_CONDITION)));
    }

    @Test
    void atAlsoOpensATriggerCondition() {
        String prose = "at the beginning of your upkeep, gain 1 life.";
        List<RoleSpans.Span> spans = RoleSpans.of(prose, false);
        assertEquals(SpanRole.TRIGGER_CONDITION, spans.get(0).role());
    }

    @Test
    void aStaticLineIsAllEffect() {
        String prose = "creatures you control get +1/+1.";
        List<RoleSpans.Span> spans = RoleSpans.of(prose, false);
        assertEquals(1, spans.size());
        assertEquals(SpanRole.EFFECT, spans.get(0).role());
        assertEquals(prose, textOf(prose, spans.get(0)));
    }

    @Test
    void anActivatedAbilityWithNoCostColonIsAllEffect() {
        String prose = "draw a card.";
        List<RoleSpans.Span> spans = RoleSpans.of(prose, true);
        assertEquals(1, spans.size());
        assertEquals(SpanRole.EFFECT, spans.get(0).role());
    }

    @Test
    void aTargetPhraseIsSpannedInsideTheClauseItAppearsIn() {
        String prose = "deal 3 damage to target creature or player.";
        List<RoleSpans.Span> spans = RoleSpans.of(prose, false);
        RoleSpans.Span target = roleIn(spans, SpanRole.TARGET_SPEC);
        assertEquals("target creature or player", textOf(prose, target));
    }

    @Test
    void severalTargetPhrasesEachGetASpan() {
        String prose = "target creature fights target creature you don't control.";
        List<RoleSpans.Span> spans = RoleSpans.of(prose, false);
        assertEquals(2, spans.stream()
                .filter(s -> s.role() == SpanRole.TARGET_SPEC).count());
    }

    @Test
    void aWordMerelyEndingInTargetIsNotOne() {
        String prose = "untarget the creature.";
        List<RoleSpans.Span> spans = RoleSpans.of(prose, false);
        assertTrue(spans.stream().noneMatch(
                s -> s.role() == SpanRole.TARGET_SPEC));
    }

    @Test
    void anEmptyLineHasNoSpans() {
        assertTrue(RoleSpans.of("", false).isEmpty());
        assertTrue(RoleSpans.of(null, false).isEmpty());
    }

    @Test
    void everySpanStaysInsideTheProse() {
        String prose = "{2}{R}: deal 2 damage to target creature, then draw a card.";
        List<RoleSpans.Span> spans = RoleSpans.of(prose, true);
        assertFalse(spans.isEmpty(), "no spans at all would pass vacuously");
        for (RoleSpans.Span span : spans) {
            assertTrue(span.start() >= 0);
            assertTrue(span.end() <= prose.length());
            assertFalse(span.start() > span.end());
        }
    }

    @Test
    void aSpanRendersAsTheJsonTheSidecarCarries() {
        RoleSpans.Span span = new RoleSpans.Span(0, 34, SpanRole.TRIGGER_CONDITION);
        assertEquals("{\"start\":0,\"end\":34,\"role\":\"trigger-condition\"}",
                span.toJson());
    }
}
