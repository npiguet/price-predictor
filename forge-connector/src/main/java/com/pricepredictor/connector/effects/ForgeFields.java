package com.pricepredictor.connector.effects;

import com.google.common.collect.Table;
import forge.game.Game;
import forge.game.StaticEffect;
import forge.game.StaticEffects;
import forge.game.card.Card;
import forge.game.staticability.StaticAbility;

import java.lang.reflect.Field;
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
