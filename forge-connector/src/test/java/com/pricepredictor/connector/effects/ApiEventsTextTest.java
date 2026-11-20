package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.ability.AbilityFactory;
import forge.game.card.Card;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;

import static com.pricepredictor.connector.effects.ApiClauseObservations.observe;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

/**
 * The {@link TextApiRules}: a card's text changed.
 */
@ExtendWith(ForgeExtension.class)
class ApiEventsTextTest {

    // ── text_change: ChangeText ──────────────────────────────────────────

    @Test
    void changeTextNamesEveryTargetedCard() {
        SpellAbility sa = AbilityFactory.getAbility(
                "SP$ ChangeText | ValidTgts$ Card | ChangeColorWord$ Choose Choose",
                TestCards.build("Grizzly Bears"));
        Card c1 = TestCards.build("Runeclaw Bear");
        Card c2 = TestCards.build("Grizzly Bears");
        sa.resetTargets();
        sa.getTargets().add(c1);
        sa.getTargets().add(c2);

        EffectEvent event = TestCards.onlyEvent(observe(sa));

        assertEquals(EventType.TEXT_CHANGE, event.type());
        assertEquals(
                List.of(Refs.of(c1), Refs.of(c2)),
                event.subjects());
    }

    /**
     * {@code artificial_evolution}/{@code sleight_of_mind}/{@code
     * magical_hack}-shaped: {@code ValidTgts$ Card | TgtZone$
     * Stack,Battlefield}, targeting a spell on the stack rather than a
     * permanent. {@code TargetChoices} stores that as the targeted {@code
     * SpellAbility} itself, not as a {@code Card} -- {@code getTargetCards()}
     * filters to {@code Card.class} instances and would see nothing -- so
     * {@code cardsFromTargets}'s {@code getTargetSpells()} union is the only
     * path that can name a subject here, so the union is not dead code. Built the way {@code
     * changeTargetsNamesWhatTheRetargetedSpellPointsAtNow} already builds a
     * "spell on the stack" target: a real, distinct {@code SpellAbility}
     * added to {@code sa}'s own targets via {@code TargetChoices.add(GameObject)}'s
     * {@code SpellAbility} overload, the same way real targeting leaves it.
     * The plausible wrong implementation this falsifies is {@link
     * #affectedCards} alone (no {@code getTargetSpells()} union at all),
     * which would report zero subjects here instead of one.
     */
    @Test
    void changeTextTargetingASpellOnTheStackNamesItsHostCard() {
        SpellAbility sa = AbilityFactory.getAbility(
                "SP$ ChangeText | ValidTgts$ Card | TgtZone$ Stack,Battlefield | "
                        + "ChangeColorWord$ Choose Choose",
                TestCards.build("Grizzly Bears"));
        SpellAbility spellOnStack = AbilityFactory.getAbility(
                "SP$ Pump | ValidTgts$ Creature | NumAtt$ +0 | NumDef$ +0",
                TestCards.build("Runeclaw Bear"));
        sa.resetTargets();
        sa.getTargets().add(spellOnStack);

        EffectEvent event = TestCards.onlyEvent(observe(sa));

        assertEquals(EventType.TEXT_CHANGE, event.type());
        assertEquals(
                List.of(Refs.of(spellOnStack.getHostCard())),
                event.subjects());
    }

    /**
     * {@code ChangeTextEffect.resolve()}'s target loop still fires {@code
     * GameEventCardStatsChanged} even when the clause names no word to
     * replace at all -- the plausible wrong implementation this falsifies is
     * a rule that reports every target regardless.
     */
    @Test
    void changeTextWithNoWordToReplaceNamesNoOne() {
        SpellAbility sa = AbilityFactory.getAbility(
                "SP$ ChangeText | ValidTgts$ Card", TestCards.build("Grizzly Bears"));
        sa.resetTargets();
        sa.getTargets().add(TestCards.build("Runeclaw Bear"));

        EffectEvent event = TestCards.onlyEvent(observe(sa));

        assertNull(event, "no ChangeColorWord$/ChangeTypeWord$ means nothing textual changed");
    }
}
