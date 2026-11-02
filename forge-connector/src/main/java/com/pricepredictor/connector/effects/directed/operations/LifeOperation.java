package com.pricepredictor.connector.effects.directed.operations;

import com.pricepredictor.connector.effects.directed.ForkOperation;
import forge.game.Game;
import forge.game.player.Player;

import java.lang.reflect.Field;

/**
 * {@code life}: set a player's life total.
 *
 * <p>Written to Forge's private {@code Player.life} field, because Forge has
 * no silent setter: {@code Player.setLife} goes through {@code gainLife} or
 * {@code loseLife}, which fire the life-gained and life-lost triggers (Ajani's
 * Pridemate, Sanguine Bond), run the can't-gain and can't-lose replacements
 * and add to the life gained or lost this turn. A variant's "the player is at
 * 5 life" states none of that. The player view, which only the desktop
 * interface reads, is left as it was; the engine, the AI and the snapshot read
 * the field.
 *
 * <p>A Forge that renamed the field fails the edit, and the fork is discarded:
 * the edit never falls back on {@code setLife}'s triggers.
 *
 * @param player the fork's player
 * @param life   the life total afterwards
 */
public record LifeOperation(Player player, int life) implements BoardEdit {

    /** {@code Player.life}, or null when this Forge has no such field. */
    private static final Field LIFE = lifeField();

    private static Field lifeField() {
        try {
            Field field = Player.class.getDeclaredField("life");
            field.setAccessible(true);
            return field;
        } catch (ReflectiveOperationException | RuntimeException e) {
            return null;
        }
    }

    @Override
    public ForkOperation operation() {
        return ForkOperation.LIFE;
    }

    @Override
    public void apply(Game fork) {
        if (LIFE == null) {
            throw new IllegalStateException(
                    "Forge's Player.life field is gone: a life total cannot be set silently");
        }
        try {
            LIFE.setInt(player, life);
        } catch (IllegalAccessException e) {
            throw new IllegalStateException("cannot write Player.life", e);
        }
        BoardEdit.settle(fork);
    }
}
