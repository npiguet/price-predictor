package com.pricepredictor.connector.effects;

import com.google.common.collect.Table;
import com.pricepredictor.connector.effects.PatchHooks.Hook;
import forge.card.mana.ManaCost;
import forge.game.CardTraitBase;
import forge.game.Game;
import forge.game.StaticEffect;
import forge.game.card.Card;
import forge.game.card.CardState;
import forge.game.card.ICardTraitChanges;
import forge.game.staticability.StaticAbility;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * The few engine tables a snapshot reads that Forge declares private and
 * offers no side-effect-free getter for, reached through the branch's own
 * accessors ({@link PatchHooks}) rather than by forcing access. It writes
 * none: what a continuous record lends the live board for a read goes through
 * {@link PatchHooks#LEND_TYPE} and {@link PatchHooks#SET_ASIDE_COLORS}.
 *
 * <p>Each read answers null (or empty, for the lists) when the accessor is
 * absent, which is every read on a stock Forge, so the snapshot loses that
 * fact rather than the record. Every accessor hands out the engine's own
 * table or map, never a copy.
 *
 * <ul>
 *   <li>{@code StaticEffects}: {@code getStaticEffect(stAb)}
 *       <em>creates</em> an entry for a static it has never applied, so asking
 *       it which permanents a static affects would write to the game.</li>
 *   <li>{@code Card}'s layer tables, keyed (timestamp, static id), which the
 *       public accessors hand out only regrouped by static or flattened,
 *       without the timestamp a resolved effect's row is joined by; and the
 *       colour layers in order, which the tests that check a lend of a
 *       continuous record's board put every change back read.</li>
 *   <li>{@code CardState}'s trait assembly: the public getters assemble a
 *       state's traits through the card's own trait changes only, and a
 *       printed line's index is counted through those changes with no line
 *       taken away.</li>
 * </ul>
 */
final class ForgeFields {

    private ForgeFields() {
    }

    /** The static effect Forge applied for {@code ability}, or null when it applied none. */
    static StaticEffect staticEffectOf(Game game, StaticAbility ability) {
        return as(StaticEffect.class, staticEffects(game).get(ability));
    }

    /** Every static effect Forge applies now by its static; empty when unreadable. */
    private static Map<?, ?> staticEffects(Game game) {
        Map<?, ?> effects = game == null ? null
                : as(Map.class, PatchHooks.STATIC_EFFECTS_BY_STATIC.read(game.getStaticEffects()));
        return effects == null ? Map.of() : effects;
    }

    /** The card's layer-4 type changes keyed (timestamp, static id), or null. */
    static Table<?, ?, ?> changedCardTypes(Card card) {
        return as(Table.class, PatchHooks.CHANGED_CARD_TYPES.read(card));
    }

    /**
     * The card's hidden extrinsic keywords keyed (timestamp, static id), or
     * null: a {@code Pump}'s or {@code Animate}'s {@code KW$ HIDDEN …} under
     * static id 0, a static's {@code AddHiddenKeyword$} under its own id.
     * Forge hands out only the concatenated values.
     */
    static Table<?, ?, ?> hiddenExtrinsicKeywords(Card card) {
        return as(Table.class, PatchHooks.HIDDEN_EXTRINSIC_KEYWORDS.read(card));
    }

    /** The card's can't-have keyword rows keyed (timestamp, static id), or null. */
    static Table<?, ?, ?> cantHaveKeywords(Card card) {
        return as(Table.class, PatchHooks.CANT_HAVE_KEYWORDS.read(card));
    }

    /**
     * The card's temporary controllers keyed by the timestamp of the change
     * that gave each (a resolved {@code GainControl}'s, a static's), or null.
     */
    static Map<?, ?> tempControllers(Card card) {
        return as(Map.class, PatchHooks.TEMP_CONTROLLERS.read(card));
    }

    /** The timestamps of every static effect Forge applies now; empty when unreadable. */
    static Set<Long> staticEffectTimestamps(Game game) {
        return staticEffects(game).values().stream()
                .map(effect -> as(StaticEffect.class, effect))
                .filter(Objects::nonNull)
                .map(StaticEffect::getTimestamp)
                .collect(Collectors.toCollection(HashSet::new));
    }

    /**
     * The card's colour change tables in Forge's layer order
     * ({@code Card.colorLayers}: by text, characteristic-defining, layer 5),
     * each keyed (timestamp, static id); the tables themselves, none when
     * unreadable.
     */
    static List<Table<Long, Long, Object>> colorLayers(Card card) {
        List<?> layers = as(List.class, PatchHooks.COLOR_LAYERS.read(card));
        return layers == null ? new ArrayList<>() : layers.stream()
                .filter(Table.class::isInstance).map(ForgeFields::asColorTable)
                .collect(Collectors.toCollection(ArrayList::new));
    }

    /** One of {@code Card.colorLayers}, a {@code Table<Long, Long, CardColor>}. */
    @SuppressWarnings("unchecked")
    private static Table<Long, Long, Object> asColorTable(Object layer) {
        return (Table<Long, Long, Object>) layer;
    }

    /** The card's layer-3 mana cost changes keyed (timestamp, static id), or null. */
    static Table<?, ?, ?> changedCardManaCost(Card card) {
        return as(Table.class, PatchHooks.CHANGED_CARD_MANA_COST.read(card));
    }

    /**
     * One mana cost change's cost and whether it adds to the cost before it
     * ({@code Card.CardManaCost}, read by its accessors); null when
     * unreadable.
     */
    static Map.Entry<ManaCost, Boolean> manaCostChange(Object change) {
        ManaCost cost = as(ManaCost.class, PatchHooks.MANA_COST_CHANGE_MANA.read(change));
        Boolean additional = as(Boolean.class, PatchHooks.MANA_COST_CHANGE_ADDITIONAL.read(change));
        return cost == null || additional == null ? null : Map.entry(cost, additional);
    }

    /** The card's layer-5 colour changes keyed (timestamp, static id), or null. */
    static Table<?, ?, ?> changedCardColors(Card card) {
        return as(Table.class, PatchHooks.CHANGED_CARD_COLORS.read(card));
    }

    /**
     * {@code state}'s {@code kind} traits assembled as Forge assembles them,
     * but through {@code changes} rather than the card's own trait changes;
     * null when the assembly cannot be read, and for {@link TraitKind#KEYWORD},
     * which has no slice.
     */
    static List<? extends CardTraitBase> traitsThrough(
            CardState state, TraitKind kind, Iterable<? extends ICardTraitChanges> changes) {
        Hook assembly = assemblyOf(kind);
        Collection<?> traits = assembly == null ? null
                : as(Collection.class, assembly.read(state, Iterable.class, changes));
        return traits == null ? null : traits.stream().map(CardTraitBase.class::cast).toList();
    }

    private static Hook assemblyOf(TraitKind kind) {
        return switch (kind) {
            case SPELL -> PatchHooks.SPELL_ABILITIES_THROUGH;
            case TRIGGER -> PatchHooks.TRIGGERS_THROUGH;
            case STATIC -> PatchHooks.STATIC_ABILITIES_THROUGH;
            case REPLACEMENT -> PatchHooks.REPLACEMENT_EFFECTS_THROUGH;
            case KEYWORD -> null;
        };
    }

    /** {@code value} as a {@code type}, or null when it is another or none. */
    private static <T> T as(Class<T> type, Object value) {
        return type.isInstance(value) ? type.cast(value) : null;
    }
}
