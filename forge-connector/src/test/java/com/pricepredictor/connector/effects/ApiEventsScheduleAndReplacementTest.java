package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.ability.AbilityFactory;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.List;

import static com.pricepredictor.connector.effects.ApiClauseObservations.observe;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

/**
 * The {@link ScheduleAndReplacementApiRules}: delayed and immediate
 * triggers, and the six {@code Replace*} APIs.
 */
@ExtendWith(ForgeExtension.class)
class ApiEventsScheduleAndReplacementTest {

    // ── delayed_trigger_created ───────────────────────────────────────────

    @Test
    void aDelayedTriggerNamesTheSchedulingCard() {
        SpellAbility sa = AbilityFactory.getAbility(
                "DB$ DelayedTrigger | Mode$ Phase | Phase$ EndCombat | ValidPlayer$ Player | "
                        + "TriggerDescription$ test",
                TestCards.build("Grizzly Bears"));

        EffectEvent event = TestCards.onlyEvent(observe(sa));

        assertEquals(EventType.DELAYED_TRIGGER_CREATED, event.type());
        assertEquals(List.of(Refs.of(sa.getHostCard())), event.subjects());
    }

    @Test
    void anImmediateTriggerNamesTheSchedulingCard() {
        SpellAbility sa = AbilityFactory.getAbility(
                "DB$ ImmediateTrigger | TriggerDescription$ test",
                TestCards.build("Grizzly Bears"));

        EffectEvent event = TestCards.onlyEvent(observe(sa));

        assertEquals(EventType.DELAYED_TRIGGER_CREATED, event.type());
        assertEquals(List.of(Refs.of(sa.getHostCard())), event.subjects());
    }

    /**
     * CR 603.12a: "once for each of those times" can be zero times.
     * {@code ImmediateTriggerEffect.resolve()} returns before registering
     * anything when {@code TriggerAmount$} calculates to <= 0. The plausible
     * wrong implementation this falsifies is a rule that always fires
     * regardless of the amount.
     */
    @Test
    void anImmediateTriggerWithZeroAmountRegistersNothing() {
        SpellAbility sa = AbilityFactory.getAbility(
                "DB$ ImmediateTrigger | TriggerAmount$ 0 | TriggerDescription$ test",
                TestCards.build("Grizzly Bears"));

        EffectEvent event = TestCards.onlyEvent(observe(sa));

        assertNull(event, "TriggerAmount$ 0 registers no delayed trigger at all");
    }

    // ── replacement_applied ───────────────────────────────────────────────

    /**
     * Every {@code Replace*} API shares one emitter, which names the card whose
     * replacement applied -- the host, not the activator (no activator is even
     * set here, which is the point: this rule reads nothing but the host). This
     * pins that each of the six is actually keyed in {@code ApiEvents.RULES}
     * rather than trusting the six hand-written entries agree with each other;
     * a missing or misspelled key would make {@code ruleFor} return null and
     * fail with a {@code NullPointerException} on {@code event.type()}.
     */
    @ParameterizedTest(name = "{0}")
    @CsvSource(delimiter = ';', value = {
            "ReplaceEffect;      VarName$ DamageAmount | VarValue$ X",
            "ReplaceCounter;     ValidCounterType$ ENERGY | ChooseCounter$ True | Amount$ 1",
            "ReplaceDamage;      Amount$ 0",
            "ReplaceMana;        ReplaceType$ C",
            "ReplaceSplitDamage; DamageTarget$ Remembered | VarName$ Y",
            "ReplaceToken;       Type$ ReplaceToken | TokenScript$ c_a_clue_draw",
    })
    void aReplaceApiNamesTheHostCardAsReplacementApplied(String api, String minimalParams) {
        SpellAbility sa = AbilityFactory.getAbility(
                "DB$ " + api + " | " + minimalParams, TestCards.build("Grizzly Bears"));

        EffectEvent event = TestCards.onlyEvent(observe(sa));

        assertEquals(EventType.REPLACEMENT_APPLIED, event.type());
        assertEquals(List.of(Refs.of(sa.getHostCard())), event.subjects());
    }
}
