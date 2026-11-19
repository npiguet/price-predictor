package com.pricepredictor.connector.effects.directed.rules;

import forge.game.Game;
import forge.game.trigger.Trigger;
import forge.trackable.TrackableTypes;

import java.lang.reflect.Field;
import java.util.List;

/**
 * What a directed precondition, which runs on the live game, must leave as it
 * was: the game's timestamp and card id counters, the trigger id counter, the
 * tracker's lookups for card id -1 (what a probe card would register), and
 * the number of static effects the game applies (a probe in no zone makes
 * no trigger or static a game registers, which the trigger id counter and
 * this count would show).
 */
record LiveGameMarks(List<Object> values) {

    static LiveGameMarks of(Game game) {
        return new LiveGameMarks(List.of(
                game.getTimestamp(),
                read(Game.class, "cardIdCounter", game),
                read(Game.class, "hiddenCardIdCounter", game),
                read(Trigger.class, "maxId", null),
                game.getTracker().hasObj(TrackableTypes.CardViewType, -1),
                game.getTracker().hasObj(TrackableTypes.CardStateViewType, -1),
                com.google.common.collect.Iterables.size(game.getStaticEffects().getEffects())));
    }

    private static Object read(Class<?> owner, String name, Object target) {
        try {
            Field field = owner.getDeclaredField(name);
            field.setAccessible(true);
            return field.get(target);
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException(e);
        }
    }
}
