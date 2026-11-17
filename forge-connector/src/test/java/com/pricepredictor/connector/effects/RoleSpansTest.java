package com.pricepredictor.connector.effects;

import org.junit.jupiter.api.Test;

import com.pricepredictor.connector.AbilityType;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
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

    private static List<RoleSpans.Span> of(String prose, AbilityType type) {
        return LineRoleSpans.of(prose, type, 0);
    }

    private static String roleText(String prose, AbilityType type, SpanRole role) {
        RoleSpans.Span span = roleIn(of(prose, type), role);
        return span == null ? null : textOf(prose, span);
    }

    @Test
    void aLoyaltyCostIsACostSpan() {
        String prose = "[+2]: look at the top card of target player's library.";
        assertEquals("[+2]:", roleText(prose, AbilityType.PLANESWALKER, SpanRole.COST));
        assertEquals(" look at the top card of target player's library.",
                roleText(prose, AbilityType.PLANESWALKER, SpanRole.EFFECT));
        assertEquals("target player's library",
                roleText(prose, AbilityType.PLANESWALKER, SpanRole.TARGET_SPEC));
    }

    @Test
    void aZeroLoyaltyCostWithoutBracketsIsACostSpan() {
        String prose = "0: draw three cards, then put two cards from your hand on top of your library in any order.";
        assertEquals("0:", roleText(prose, AbilityType.PLANESWALKER, SpanRole.COST));
    }

    @Test
    void aLevelCostIsACostSpanAndItsTriggerKeepsItsCondition() {
        String prose = "{B/G}: whenever a land you control enters, creature cards in your graveyard perpetually get +1/+1.";
        assertEquals("{B/G}:", roleText(prose, AbilityType.LEVEL, SpanRole.COST));
        assertEquals("whenever a land you control enters,",
                roleText(prose, AbilityType.LEVEL, SpanRole.TRIGGER_CONDITION));
    }

    @Test
    void aLevelLineWithNoCostHasNoCostSpan() {
        String prose = "landfall — whenever a land you control enters, mill two cards.";
        assertEquals(null, roleText(prose, AbilityType.LEVEL, SpanRole.COST));
        assertEquals("whenever a land you control enters,",
                roleText(prose, AbilityType.LEVEL, SpanRole.TRIGGER_CONDITION));
    }

    @Test
    void aSpellsAdditionalCostSentenceIsACostSpan() {
        String prose = "as an additional cost to cast this spell, sacrifice a land. search your library for up to two basic land cards.";
        int costLength = "as an additional cost to cast this spell, sacrifice a land.".length();
        List<RoleSpans.Span> spans = LineRoleSpans.of(prose, AbilityType.SPELL, costLength);
        assertEquals("as an additional cost to cast this spell, sacrifice a land.",
                textOf(prose, roleIn(spans, SpanRole.COST)));
        assertEquals(" search your library for up to two basic land cards.",
                textOf(prose, roleIn(spans, SpanRole.EFFECT)));
    }

    @Test
    void anAdditionalCostLineIsAllCost() {
        String prose = "exile six cards from your graveyard.";
        assertEquals(prose, roleText(prose, AbilityType.ADDITIONAL_COST, SpanRole.COST));
        assertEquals(null, roleText(prose, AbilityType.ADDITIONAL_COST, SpanRole.EFFECT));
    }

    @Test
    void anAlternateCostLineIsAllCost() {
        String prose = "cleave {1}{B}{B}{G}";
        assertEquals(prose, roleText(prose, AbilityType.ALTERNATE_COST, SpanRole.COST));
    }

    @Test
    void aCostReductionLineIsAllCost() {
        String prose = "affinity for allies";
        assertEquals(prose, roleText(prose, AbilityType.COST_REDUCTION, SpanRole.COST));
    }

    @Test
    void anAbilityWordBeforeATriggerIsLeftOutOfItsCondition() {
        String prose = "landfall — whenever a land you control enters, create a 2/2 green elemental creature token.";
        assertEquals("whenever a land you control enters,",
                roleText(prose, AbilityType.TRIGGERED, SpanRole.TRIGGER_CONDITION));
        assertEquals(" create a 2/2 green elemental creature token.",
                roleText(prose, AbilityType.TRIGGERED, SpanRole.EFFECT));
    }

    @Test
    void aDashInsideAnEffectIsNotAnAbilityWord() {
        String prose = "creatures you control get +1/+1 — whenever that matters, draw.";
        assertEquals(null, roleText(prose, AbilityType.STATIC, SpanRole.TRIGGER_CONDITION));
    }

    @Test
    void aSpanRendersAsTheJsonTheSidecarCarries() {
        RoleSpans.Span span = new RoleSpans.Span(0, 34, SpanRole.TRIGGER_CONDITION);
        assertEquals("{\"start\":0,\"end\":34,\"role\":\"trigger-condition\"}",
                span.toJson());
    }
}
