package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.ability.AbilityFactory;
import forge.game.card.Card;
import forge.game.spellability.AbilitySub;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Which clause of a line an event came from, and how long what it did
 * lasts.
 */
@ExtendWith(ForgeExtension.class)
class EventAttributionTest {

    /** A three-clause line: gain life, then draw, then scry. */
    private static SpellAbility threeClauseAbility() {
        Card host = TestCards.build("Fountain of Youth");
        host.setSVar("DBDraw",
                "DB$ Draw | Defined$ You | NumCards$ 1 | SubAbility$ DBScry");
        host.setSVar("DBScry", "DB$ Scry | Defined$ You | ScryNum$ 1");
        SpellAbility root = AbilityFactory.getAbility(
                "AB$ GainLife | Cost$ 1 | Defined$ You | LifeAmount$ 1"
                        + " | SubAbility$ DBDraw", host);
        assertNotNull(root.getSubAbility(), "the chain has to be three deep");
        assertNotNull(root.getSubAbility().getSubAbility());
        return root;
    }

    /**
     * Each clause of a multi-clause line is its own index.
     *
     * <p>The engine's pointer was set only around the line the stack resolved,
     * so every clause after the first was attributed to the root: 298 of 74,952
     * events carried an {@code attributed_to} at all, and those were the dozen
     * branching effects that resolve a sub-ability through
     * {@code AbilityUtils.resolve}. "Put a counter" and "draw a card" are
     * exactly the distinction the per-entity head learns, so a whole line
     * attributed to its root teaches neither.
     */
    @Test
    void eachClauseOfAMultiClauseAbilityAttributesToItself() {
        SpellAbility root = threeClauseAbility();
        AbilitySub first = (AbilitySub) root.getSubAbility();
        AbilitySub second = (AbilitySub) first.getSubAbility();

        assertEquals("0", EventAttribution.attributedTo(root, first));
        assertEquals("1", EventAttribution.attributedTo(root, second));
    }

    /**
     * The root acting and no pointer at all are different answers.
     *
     * <p>Both used to be null, which is why the 3.2% of events carrying an
     * attribution could not be read: a single-clause ability legitimately has
     * no sub-ability and the root is the right answer, and that looked exactly
     * like the pointer failing to land. Separating them is what turns the rate
     * into a measurement instead of a number.
     */
    @Test
    void theRootActingAndNoPointerAreDifferentAnswers() {
        SpellAbility root = threeClauseAbility();

        assertEquals(EventAttribution.ROOT,
                EventAttribution.attributedTo(root, root));
        assertEquals(EventAttribution.UNRESOLVED,
                EventAttribution.attributedTo(root, null));
        assertNotEquals(EventAttribution.ROOT, EventAttribution.UNRESOLVED);
    }

    /** Neither sentinel can be mistaken for a chain index. */
    @Test
    void neitherSentinelReadsAsAClauseIndex() {
        for (String sentinel
                : List.of(EventAttribution.ROOT, EventAttribution.UNRESOLVED)) {
            assertThrows(NumberFormatException.class,
                    () -> Integer.parseInt(sentinel), sentinel);
        }
    }

    /**
     * A clause with no line above it names nothing a reader could join to.
     *
     * <p>Its index counts clauses of a chain this collector never saw, so the
     * number would address a line in some other script. Saying so is the third
     * state, not the root.
     */
    @Test
    void aDetachedClauseCannotBePlacedAndSaysSo() {
        SpellAbility root = threeClauseAbility();
        AbilitySub orphan = (AbilitySub) root.getSubAbility();
        orphan.setParent(null);

        assertEquals(EventAttribution.UNRESOLVED,
                EventAttribution.attributedTo(root, orphan));
    }

    /** And attribution never returns null, whatever it is handed. */
    @Test
    void attributionAlwaysNamesOneOfTheThreeStates() {
        SpellAbility root = threeClauseAbility();

        for (Object pointer
                : new Object[]{null, "not an ability", root, root.getSubAbility()}) {
            assertNotNull(EventAttribution.attributedTo(root, pointer),
                    String.valueOf(pointer));
        }
    }

    /** And an event stamped inside a clause carries that clause's index. */
    @Test
    void anEventCarriesTheClauseThatProducedIt() {
        SpellAbility root = threeClauseAbility();
        AbilitySub second = (AbilitySub) root.getSubAbility().getSubAbility();

        String json = EventAttribution.stamp(
                new EffectEvent(EffectEvent.CARD_DRAWN).subject("P0"),
                root, second).toJson();

        assertTrue(json.contains("\"attributed_to\":\"1\""), json);
    }

    /**
     * A clause's own {@code Duration$} outranks the line's.
     *
     * <p>Both fields come from one read of the pointer, so they cannot describe
     * different instants of a nested resolution.
     */
    @Test
    void aClauseDeclaringItsOwnDurationOutranksTheLine() {
        Card host = TestCards.build("Fountain of Youth");
        host.setSVar("DBPump",
                "DB$ Pump | Defined$ Self | NumAtt$ 1 | Duration$ Permanent");
        SpellAbility root = AbilityFactory.getAbility(
                "AB$ GainLife | Cost$ 1 | Defined$ You | LifeAmount$ 1"
                        + " | SubAbility$ DBPump", host);

        assertEquals(EventAttribution.PERMANENT,
                EventAttribution.duration(root, root.getSubAbility()));
        assertEquals(EventAttribution.INSTANT,
                EventAttribution.duration(root, root));
    }

    /**
     * An undeclared duration means what Forge does with it. A pump with no
     * {@code Duration$} ends at end of turn — Giant Growth — and a counter
     * lasts; only an outcome with no lasting state is instant.
     */
    @Test
    void anUndeclaredDurationIsTheOneForgeApplies() {
        Card host = TestCards.build("Grizzly Bears");
        SpellAbility pump = AbilityFactory.getAbility(
                "SP$ Pump | ValidTgts$ Creature | NumAtt$ +3 | NumDef$ +3", host);
        SpellAbility counter = AbilityFactory.getAbility(
                "SP$ PutCounter | Defined$ Self | CounterType$ P1P1 | CounterNum$ 1", host);

        assertEquals(EventAttribution.END_OF_TURN,
                EventAttribution.duration(pump, pump, EffectEvent.PT_CHANGE));
        assertEquals(EventAttribution.PERMANENT,
                EventAttribution.duration(counter, counter, EffectEvent.PT_CHANGE));
        assertEquals(EventAttribution.INSTANT,
                EventAttribution.duration(pump, pump, EffectEvent.DAMAGE_DEALT));

        String json = EventAttribution.stamp(
                new EffectEvent(EffectEvent.PT_CHANGE).subject("E1"), pump, pump).toJson();
        assertTrue(json.contains("\"duration\":\"end_of_turn\""), json);
    }

    /**
     * With nothing resolving and nothing named, the record says nothing.
     *
     * <p>An honest absence rather than a placeholder: Forge genuinely names no
     * causer for a turn-based action, and a reader can act on the difference.
     */
    @Test
    void anEventNothingCausedCarriesNoCause() {
        assertNull(EventAttribution.cause(null, null));

        String json = EventAttribution.stamp(
                new EffectEvent(EffectEvent.UNTAPPED).subject("E7"), null, null)
                .toJson();

        assertFalse(json.contains("cause"), json);
    }
}
