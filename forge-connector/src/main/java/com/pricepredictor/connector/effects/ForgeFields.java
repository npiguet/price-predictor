package com.pricepredictor.connector.effects;

import com.google.common.collect.Table;
import forge.game.CardTraitBase;
import forge.game.Game;
import forge.game.StaticEffect;
import forge.game.StaticEffects;
import forge.game.card.Card;
import forge.game.card.CardState;
import forge.game.card.ICardTraitChanges;
import forge.game.spellability.SpellAbility;
import forge.game.staticability.StaticAbility;
import forge.util.collect.FCollection;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * The few engine fields a snapshot reads that Forge declares private and
 * offers no side-effect-free getter for.
 *
 * <p>Each read answers null when the field is gone or unreadable, so a Forge
 * that renamed one costs the snapshot that fact rather than the record.
 *
 * <ul>
 *   <li>{@code StaticEffects.staticEffects}: {@code getStaticEffect(stAb)}
 *       <em>creates</em> an entry for a static it has never applied, so asking
 *       it which permanents a static affects would write to the game.</li>
 *   <li>{@code Card.changedCardTypes} and {@code Card.changedCardColors}: the
 *       layer-4 and layer-5 tables keyed (timestamp, static id), which the
 *       public accessors hand out only regrouped by static, without the
 *       timestamp a resolved effect's row is joined by.</li>
 *   <li>{@code CardState}'s trait assembly ({@code abilities},
 *       {@code updateSpellAbilities(FCollection)}, {@code triggers(Iterable)},
 *       {@code staticAbilities(Iterable)}, {@code replacementEffects(boolean,
 *       Iterable)}): the public getters assemble a state's traits through the
 *       card's own trait changes only, and a printed line's index is counted
 *       through those changes with no line taken away.</li>
 * </ul>
 */
final class ForgeFields {

    private ForgeFields() {
    }

    private static final Map<String, Field> FIELDS = new ConcurrentHashMap<>();
    /** Stands for a field looked up and not found, since the map takes no null. */
    private static final Field MISSING;

    static {
        try {
            MISSING = ForgeFields.class.getDeclaredField("MISSING");
        } catch (NoSuchFieldException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    /** The static effect Forge applied for {@code ability}, or null when it applied none. */
    static StaticEffect staticEffectOf(Game game, StaticAbility ability) {
        if (read(StaticEffects.class, "staticEffects", game.getStaticEffects())
                instanceof Map<?, ?> effects
                && effects.get(ability) instanceof StaticEffect effect) {
            return effect;
        }
        return null;
    }

    /** The card's layer-4 type changes keyed (timestamp, static id), or null. */
    static Table<?, ?, ?> changedCardTypes(Card card) {
        return read(Card.class, "changedCardTypes", card) instanceof Table<?, ?, ?> table
                ? table : null;
    }

    /** The card's layer-5 colour changes keyed (timestamp, static id), or null. */
    static Table<?, ?, ?> changedCardColors(Card card) {
        return read(Card.class, "changedCardColors", card) instanceof Table<?, ?, ?> table
                ? table : null;
    }

    /**
     * {@code state}'s {@code kind} traits assembled as Forge assembles them,
     * but through {@code changes} rather than the card's own trait changes;
     * null when the assembly cannot be read, and for {@link TraitKind#KEYWORD},
     * which has no slice.
     */
    static List<? extends CardTraitBase> traitsThrough(
            CardState state, TraitKind kind, Iterable<? extends ICardTraitChanges> changes) {
        try {
            return switch (kind) {
                case SPELL -> spellAbilitiesThrough(state, changes);
                case TRIGGER -> listOf(invoke("triggers", new Class<?>[]{Iterable.class},
                        state, changes));
                case STATIC -> listOf(invoke("staticAbilities", new Class<?>[]{Iterable.class},
                        state, changes));
                case REPLACEMENT -> listOf(invoke("replacementEffects",
                        new Class<?>[]{boolean.class, Iterable.class}, state, true, changes));
                case KEYWORD -> null;
            };
        } catch (RuntimeException e) {
            return null;
        }
    }

    /** As {@code CardState.getSpellAbilities}, through {@code changes}; null when unreadable. */
    private static List<SpellAbility> spellAbilitiesThrough(
            CardState state, Iterable<? extends ICardTraitChanges> changes) {
        FCollection<SpellAbility> assembled = new FCollection<>();
        if (invoke("updateSpellAbilities", new Class<?>[]{FCollection.class}, state, assembled)
                == MISSING
                || !(read(CardState.class, "abilities", state) instanceof Collection<?> own)) {
            return null;
        }
        for (Object ability : own) {
            assembled.add((SpellAbility) ability);
        }
        state.getCard().updateSpellAbilities(assembled, state, changes);
        return assembled;
    }

    private static List<? extends CardTraitBase> listOf(Object assembled) {
        if (!(assembled instanceof Collection<?> traits)) {
            return null;
        }
        return traits.stream().map(CardTraitBase.class::cast).toList();
    }

    private static final Map<String, Object> METHODS = new ConcurrentHashMap<>();

    /**
     * Call {@code CardState}'s non-public {@code name} on {@code state}: what
     * it returns (null for a void method), or {@link #MISSING} when it is gone
     * or throws.
     */
    private static Object invoke(String name, Class<?>[] parameters, Object state, Object... args) {
        Object looked = METHODS.computeIfAbsent(name, key -> {
            try {
                Method found = CardState.class.getDeclaredMethod(name, parameters);
                found.setAccessible(true);
                return found;
            } catch (ReflectiveOperationException | RuntimeException e) {
                return MISSING;
            }
        });
        if (!(looked instanceof Method method)) {
            return MISSING;
        }
        try {
            return method.invoke(state, args);
        } catch (ReflectiveOperationException | RuntimeException e) {
            return MISSING;
        }
    }

    private static Object read(Class<?> owner, String name, Object target) {
        if (target == null) {
            return null;
        }
        Field field = FIELDS.computeIfAbsent(owner.getName() + "#" + name, key -> {
            try {
                Field found = owner.getDeclaredField(name);
                found.setAccessible(true);
                return found;
            } catch (ReflectiveOperationException | RuntimeException e) {
                return MISSING;
            }
        });
        if (field == MISSING) {
            return null;
        }
        try {
            return field.get(target);
        } catch (ReflectiveOperationException | RuntimeException e) {
            return null;
        }
    }
}
