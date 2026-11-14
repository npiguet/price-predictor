package com.pricepredictor.connector.effects;

import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

import java.util.function.BiConsumer;

/** {@link InterventionScheduler}'s watch on the lines it forces, for a test outside this package. */
public final class TestInterventions {

    private TestInterventions() {
    }

    /** Tell {@code watch} of each line forced from now on, or stop telling with null. */
    public static void watch(BiConsumer<SpellAbility, Player> watch) {
        InterventionScheduler.forcedWatch = watch;
    }

    /** The scheduler's spread over a typical game, in game turns. */
    public static int typicalGameTurns() {
        return InterventionScheduler.TYPICAL_GAME_TURNS;
    }
}
