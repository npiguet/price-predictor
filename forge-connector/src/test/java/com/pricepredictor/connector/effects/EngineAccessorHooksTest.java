package com.pricepredictor.connector.effects;

import com.google.common.collect.Table;
import com.pricepredictor.connector.ForgeExtension;
import forge.card.mana.ManaCost;
import forge.game.card.Card;
import forge.game.card.CardState;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The branch accessors that replaced the connector's reflective reads of
 * private Forge members ({@link ForgeFields}, {@link StackActions}): each is
 * named here, required, present, and read the way its caller reads it.
 */
@ExtendWith(ForgeExtension.class)
class EngineAccessorHooksTest {

    /** Every accessor, by the hook that reaches it. */
    private static final List<InstanceAccessor> ACCESSORS = List.of(
            PatchHooks.PRIMARY_ABILITY, PatchHooks.STATIC_EFFECTS_BY_STATIC,
            PatchHooks.CHANGED_CARD_TYPES, PatchHooks.CHANGED_CARD_COLORS, PatchHooks.COLOR_LAYERS,
            PatchHooks.CHANGED_CARD_MANA_COST, PatchHooks.HIDDEN_EXTRINSIC_KEYWORDS,
            PatchHooks.CANT_HAVE_KEYWORDS, PatchHooks.TEMP_CONTROLLERS,
            PatchHooks.SPELL_ABILITIES_THROUGH, PatchHooks.TRIGGERS_THROUGH,
            PatchHooks.STATIC_ABILITIES_THROUGH, PatchHooks.REPLACEMENT_EFFECTS_THROUGH);

    @Test
    void everyAccessorIsRequiredAndThePatchedCheckoutHasIt() {
        assertTrue(PatchHooks.REQUIRED.containsAll(ACCESSORS));
        for (InstanceAccessor accessor : ACCESSORS) {
            assertTrue(accessor.present(), "the checkout lacks " + accessor);
        }
    }

    @Test
    void theCardTablesAreTheEnginesOwnAndStayTheSameObject() {
        Card card = TestBoard.cardInAnotherGame("Grizzly Bears");

        for (InstanceAccessor table : List.of(PatchHooks.CHANGED_CARD_TYPES, PatchHooks.CHANGED_CARD_COLORS,
                PatchHooks.CHANGED_CARD_MANA_COST, PatchHooks.HIDDEN_EXTRINSIC_KEYWORDS,
                PatchHooks.CANT_HAVE_KEYWORDS)) {
            assertTrue(table.read(card) instanceof Table<?, ?, ?>, table.toString());
            assertSame(table.read(card), table.read(card), table.toString());
        }
        assertTrue(PatchHooks.TEMP_CONTROLLERS.read(card) instanceof Map<?, ?>);
        assertEquals(3, ForgeFields.colorLayers(card).size());
        assertSame(PatchHooks.COLOR_LAYERS.read(card), PatchHooks.COLOR_LAYERS.read(card));
    }

    @Test
    void aManaCostChangeReadsBackItsCostAndWhetherItAdds() {
        Card card = TestBoard.cardInAnotherGame("Grizzly Bears");
        ManaCost cost = ManaCost.TWO;
        card.addChangedManaCost(cost, true, 7L, 3L);

        Table<?, ?, ?> changes = ForgeFields.changedCardManaCost(card);
        Map.Entry<ManaCost, Boolean> change = ForgeFields.manaCostChange(changes.get(7L, 3L));

        assertEquals(cost, change.getKey());
        assertEquals(true, change.getValue());
        assertNull(ForgeFields.manaCostChange(null));
    }

    @Test
    void theTraitsThroughNoChangesAreTheStatesOwn() {
        CardState state = TestBoard.cardInAnotherGame("Paralyze").getCurrentState();

        assertEquals(List.copyOf(state.getTriggersWithoutChanges()),
                ForgeFields.traitsThrough(state, TraitKind.TRIGGER, List.of()));
        assertEquals(List.copyOf(state.getSpellAbilitiesWithoutChanges()),
                ForgeFields.traitsThrough(state, TraitKind.SPELL, List.of()));
        assertEquals(List.copyOf(state.getStaticAbilitiesWithoutChanges()),
                ForgeFields.traitsThrough(state, TraitKind.STATIC, List.of()));
        assertEquals(List.copyOf(state.getReplacementEffectsWithoutChanges()),
                ForgeFields.traitsThrough(state, TraitKind.REPLACEMENT, List.of()));
        assertNull(ForgeFields.traitsThrough(state, TraitKind.KEYWORD, List.of()));
    }

    @Test
    void theStackNamesNoActionWhileNothingFreezesIt() {
        Card card = TestBoard.cardInAnotherGame("Grizzly Bears");

        assertNull(PatchHooks.PRIMARY_ABILITY.read(card.getGame().getStack()));
        assertNull(StackActions.inProgress(card.getGame().getStack()));
        assertNull(StackActions.inProgress(null));
    }

    /**
     * What a stock Forge gives: a hook whose method the class lacks reads as
     * null, so each read of {@link ForgeFields} answers null or empty.
     */
    @Test
    void anAbsentAccessorReadsNullAndEveryReadFallsBack() {
        Card card = TestBoard.cardInAnotherGame("Paralyze");
        InstanceAccessor absent = new InstanceAccessor(
                new HookSite("forge.game.card.Card", "getEffectRecordNoSuchTable"), "nothing");
        InstanceAccessor absentThrough = new InstanceAccessor(
                new HookSite("forge.game.card.CardState", "getEffectRecordNoSuchThrough"), "nothing");

        assertNull(absent.read(card));
        assertNull(absentThrough.read(
                card.getCurrentState(), new InstanceAccessor.Argument(Iterable.class, List.of())));
        assertNull(absent.read(null));
        assertNull(ForgeFields.manaCostChange(new Object()));
    }
}
