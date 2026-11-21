package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.GameStage;
import forge.game.player.Player;
import forge.game.trigger.Trigger;
import forge.game.zone.ZoneType;
import forge.util.MyRandom;

import java.lang.reflect.Field;
import java.util.List;

/**
 * What a fork must leave as it was in the live game, the part of it that
 * is cheap to read: its timestamp and card id counters, the random
 * stream Forge draws from, each player's life, poison, controller and
 * zone sizes, and its stack's size; and Forge's trigger id counter.
 * {@link Forks.ForkRun#take} reads them before every fork and compares
 * after it.
 *
 * <p>The trigger id counter ({@code Trigger.maxId}) is static, shared by
 * every game of the JVM, and every trigger a game makes, a copy's
 * included, takes the next id, so a fork moves it forward and that is no
 * difference. A restart of the game ({@code RestartGameEffect}) sets it
 * to 50000 instead: the live game's later triggers then take ids a run
 * without the fork would not give them, which its records write (a
 * delayed trigger's {@code D<id>}), and from a counter above 50000 ids
 * its own triggers already hold, which Forge compares triggers by. A fork
 * that lowered the counter, or whose game was restarted, moved the live
 * game.
 *
 * @param triggerIds the trigger id counter, or -1 where it cannot be read
 */
record LiveMarks(String marks, int triggerIds) {

    private static final Field TRIGGER_IDS = triggerCounter();

    private static Field triggerCounter() {
        try {
            Field field = Trigger.class.getDeclaredField("maxId");
            field.setAccessible(true);
            return field;
        } catch (ReflectiveOperationException | RuntimeException e) {
            return null;
        }
    }

    /** Forge's trigger id counter now, or -1 where it cannot be read. */
    static int currentTriggerIds() {
        try {
            return TRIGGER_IDS == null ? -1 : TRIGGER_IDS.getInt(null);
        } catch (ReflectiveOperationException e) {
            return -1;
        }
    }

    private static final Field CARD_IDS = counter("cardIdCounter");
    private static final Field HIDDEN_IDS = counter("hiddenCardIdCounter");

    private static Field counter(String name) {
        try {
            Field field = Game.class.getDeclaredField(name);
            field.setAccessible(true);
            return field;
        } catch (ReflectiveOperationException | RuntimeException e) {
            return null;
        }
    }

    /** {@code live}'s marks, or none for no game. */
    static LiveMarks of(Game live) {
        if (live == null) {
            return new LiveMarks("", currentTriggerIds());
        }
        StringBuilder out = new StringBuilder()
                .append("timestamp ").append(live.getTimestamp())
                .append(", card ids ").append(read(CARD_IDS, live))
                .append(", hidden card ids ").append(read(HIDDEN_IDS, live))
                .append(", random stream ").append(System.identityHashCode(MyRandom.getRandom()))
                .append(", stack ").append(live.getStack().size());
        for (Player player : live.getRegisteredPlayers()) {
            out.append(", ").append(Refs.of(player))
                    .append(" life ").append(player.getLife())
                    .append(" poison ").append(player.getPoisonCounters())
                    .append(" controller ").append(System.identityHashCode(player.getController()))
                    .append(" zones");
            for (ZoneType zone : MARKED_ZONES) {
                out.append(' ').append(player.getZone(zone).size());
            }
        }
        return new LiveMarks(out.toString(), currentTriggerIds());
    }

    private static final List<ZoneType> MARKED_ZONES = List.of(
            ZoneType.Hand, ZoneType.Library,
            ZoneType.Battlefield, ZoneType.Graveyard,
            ZoneType.Exile, ZoneType.Command);

    private static Object read(Field field, Game game) {
        try {
            return field == null ? "?" : field.get(game);
        } catch (ReflectiveOperationException e) {
            return "?";
        }
    }

    /**
     * Count a difference from {@code live}'s marks now under {@code key},
     * the first printed; {@code fork} is the fork's game, null when no
     * copy was made.
     */
    void compareAfter(String key, Game live, Game fork) {
        LiveMarks after;
        try {
            after = of(live);
        } catch (RuntimeException e) {
            return;
        }
        boolean restarted = restarted(fork);
        if (movedBy(after) || restarted) {
            HookFailures.countContained(new HookFailures.FailureKey(key), "Effect records: " + key + "; before the fork: " + marks
                    + ", trigger ids " + triggerIds + "; after: " + after.marks() + ", trigger ids "
                    + after.triggerIds() + (restarted ? "; the fork restarted its game" : "")
                    + " (later ones are counted, not printed)");
        }
    }

    /** Whether {@code after} differs from these marks, or its trigger id counter went down. */
    private boolean movedBy(LiveMarks after) {
        return !after.marks().equals(marks) || after.triggerIds() < triggerIds;
    }

    /** Whether {@code fork}'s game was restarted, false when no copy was made. */
    private static boolean restarted(Game fork) {
        return fork != null && fork.getAge() == GameStage.RestartedByKarn;
    }
}
