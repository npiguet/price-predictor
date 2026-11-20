package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Direction;
import forge.game.EvenOdd;
import forge.game.ability.AbilityFactory;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.function.Consumer;

import static com.pricepredictor.connector.effects.ApiClauseObservations.observe;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * The {@link ChoiceApiRules}: every Choose* API, read back off what it left
 * on its host.
 */
@ExtendWith(ForgeExtension.class)
class ApiEventsChoiceTest {

    @Test
    void aChosenColourIsAChoice() {
        SpellAbility sa = TestCards.scriptedAbility("Akroma's Blessing", "ChooseColor");
        ApiEvents.ClauseObservation<?> observed = observe(sa);
        sa.getHostCard().setChosenColors(com.google.common.collect.ImmutableList.of("red"));

        EffectEvent event = TestCards.onlyEvent(observed);

        assertEquals(EventType.CHOICE_MADE, event.type());
        assertEquals("color", event.param(EventParam.CHOICE_KIND));
        assertEquals("red", event.param(EventParam.VALUE));
    }

    /**
     * For every {@code RULES} entry the schema
     * declares as {@code choice_made} (fixture keys asserted equal to every
     * {@code ApiType} whose rule emits {@code choice_made}, read live off the
     * table rather than hand-counted), simulate that API's own effect class writing its
     * one host field, and require {@code choice()} to read it back as that
     * API's own kind. An invariant, not a count: it keeps holding as rules
     * are added, and it fails loudly -- either on the set comparison, if a
     * new {@code choice_made} API has no fixture yet, or on the per-API
     * assertions, if a fixture exists but the matching {@code choice()}
     * branch does not -- the moment either half drifts from the other.
     *
     * <p>{@code choice()}'s first branch checks {@code host.hasChosenColor()},
     * not {@code host.getChosenColor() != null}: {@code
     * Card.getChosenColor()} never returns {@code null} for an unset colour --
     * it returns {@code ""} -- so that check would report <em>every</em>
     * clause routed through {@code choice()} as a colour choice. Every fixture
     * but {@code ChooseColor} catches that on its kind, and {@code
     * ChooseDirection} on its value as well. {@code ChooseDirectionEffect.resolve()}
     * always calls {@code setChosenDirection}.
     *
     * <p>{@code ChooseSector} is deliberately not a key here: it has no rule
     * in {@code RULES} (see the comment there), so no rule emits {@code
     * choice_made} for it either -- not an exemption carved out of this test.
     *
     * <p>Each fixture calls the exact setter that API's real effect class
     * calls: {@code ChooseColorEffect.setChosenColors}, {@code
     * ChooseTypeEffect.setChosenType}, {@code ChooseCardEffect}/{@code
     * ChooseSourceEffect.setChosenCards} (both share the one accessor, so
     * both correctly read back as {@code "cards"}), {@code
     * ChooseCardNameEffect.addNamedCard}, {@code
     * ChooseNumberEffect.setChosenNumber}, {@code
     * ChoosePlayerEffect.setChosenPlayer}, {@code
     * ChooseDirectionEffect.setChosenDirection}, {@code
     * ChooseEvenOddEffect.setChosenEvenOdd}, and {@code
     * ChooseGenericEffect}'s {@code SetChosenMode$ True} path's {@code
     * setChosenMode} -- on a fresh host each time, which already holds a
     * colour and a number chosen earlier, so the reading follows the API
     * that ran rather than whichever accessor holds something.
     */
    @Test
    void everyChoiceMadeApiHasAReadableChoice() {
        record ChoiceFixture(String expectedKind, Consumer<Card> mutate, Object expectedValue) {
            /** A fixture whose value is only required to be there. */
            ChoiceFixture(String expectedKind, Consumer<Card> mutate) {
                this(expectedKind, mutate, null);
            }
        }

        Map<String, ChoiceFixture> fixtures = new TreeMap<>();
        fixtures.put("ChooseColor", new ChoiceFixture("color",
                host -> host.setChosenColors(List.of("blue"))));
        fixtures.put("ChooseType", new ChoiceFixture("type",
                host -> host.setChosenType("Goblin")));
        fixtures.put("ChooseCard", new ChoiceFixture("cards",
                host -> host.setChosenCards(List.of(TestCards.build("Grizzly Bears")))));
        fixtures.put("ChooseSource", new ChoiceFixture("cards",
                host -> host.setChosenCards(List.of(TestCards.build("Grizzly Bears")))));
        fixtures.put("NameCard", new ChoiceFixture("card_name",
                host -> host.addNamedCard("Grizzly Bears")));
        fixtures.put("ChooseNumber", new ChoiceFixture("number",
                host -> host.setChosenNumber(3)));
        fixtures.put("ChoosePlayer", new ChoiceFixture("player",
                host -> host.setChosenPlayer(new Player("chooser", TestCards.game(), 93001))));
        fixtures.put("ChooseDirection", new ChoiceFixture("direction",
                host -> host.setChosenDirection(Direction.Left), "Left"));
        fixtures.put("ChooseEvenOdd", new ChoiceFixture("even_odd",
                host -> host.setChosenEvenOdd(EvenOdd.Odd)));
        fixtures.put("GenericChoice", new ChoiceFixture("mode",
                host -> host.setChosenMode("Abzan")));

        Set<String> choiceMadeApis = new TreeSet<>();
        for (ApiType api : ApiType.values()) {
            ApiEvents.Rule<?> rule = ApiEvents.ruleFor(api);
            if (rule != null && EventType.CHOICE_MADE.equals(rule.eventType())) {
                choiceMadeApis.add(api.name());
            }
        }
        assertEquals(fixtures.keySet(), choiceMadeApis,
                "the fixture table above and RULES's live choice_made keys have "
                        + "drifted apart -- add a fixture (and, if it is new, a "
                        + "choice() branch to read it) for whichever API changed");

        for (Map.Entry<String, ChoiceFixture> fixture : fixtures.entrySet()) {
            Card host = TestCards.build("Grizzly Bears");
            SpellAbility sa = AbilityFactory.getAbility("SP$ " + fixture.getKey(), host);
            // Earlier choices the host still holds, which every other API's
            // reading would find first: the clause's own API decides the
            // reading, not what the host happens to hold.
            if (!"ChooseColor".equals(fixture.getKey())) {
                host.setChosenColors(List.of("red"));
            }
            if (!"ChooseNumber".equals(fixture.getKey())) {
                host.setChosenNumber(9);
            }
            fixture.getValue().mutate().accept(host);

            EffectEvent event = TestCards.onlyEvent(observe(sa));

            assertNotNull(event, fixture.getKey() + " -- choice() reported nothing");
            assertEquals(EventType.CHOICE_MADE, event.type(), fixture.getKey());
            assertEquals(
                    fixture.getValue().expectedKind(), event.param(EventParam.CHOICE_KIND),
                    fixture.getKey());
            assertNotNull(event.param(EventParam.VALUE), fixture.getKey() + " -- empty value");
            if (fixture.getValue().expectedValue() != null) {
                assertEquals(fixture.getValue().expectedValue(), event.param(EventParam.VALUE),
                        fixture.getKey());
            }
        }
    }

    /**
     * {@code ChooseNumberEffect.resolve}
     * calls {@code setChosenNumber(chosen, true)} -- the secret variant that
     * skips the trackable view update -- when {@code Secretly$} is set and
     * {@code KeepSecret$} is not. The general guard's {@code ChooseNumber}
     * fixture only exercises the plain setter; both variants write the same
     * {@code chosenNumber} field
     * ({@code setChosenNumber(int, boolean)}), so {@code choice()} reads
     * either -- this is coverage for that claim.
     */
    @Test
    void aSecretlyChosenNumberIsStillAChoice() {
        Card host = TestCards.build("Grizzly Bears");
        // A colour chosen earlier, which the number's API does not read.
        host.setChosenColors(List.of("red"));
        host.setChosenNumber(7, true);
        SpellAbility sa = AbilityFactory.getAbility("SP$ ChooseNumber", host);

        EffectEvent event = TestCards.onlyEvent(observe(sa));

        assertEquals(EventType.CHOICE_MADE, event.type());
        assertEquals("number", event.param(EventParam.CHOICE_KIND));
        assertEquals("7", event.param(EventParam.VALUE));
    }
}
