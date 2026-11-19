package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.ability.EffectRecordHook;
import forge.game.phase.PhaseHandler;

import java.lang.reflect.Field;

/**
 * Forge's priority-settled hook fired as its main loop fires it, for a test
 * that sets a board's step by hand: {@code devModeSet} publishes the step's
 * phase event and gives no priority.
 */
final class TestPriority {

    private TestPriority() {
    }

    /**
     * Tell the installed priority-settled listener that priority settled in
     * {@code game}, the player whose turn it is holding it, as {@code
     * PhaseHandler.mainLoopStep} does before that player chooses; nothing
     * when no listener is installed.
     */
    static void settle(Game game) {
        PhaseHandler.EffectRecordPrioritySettledListener listener = installed();
        if (listener != null) {
            listener.onPrioritySettled(game, game.getPhaseHandler().getPlayerTurn());
        }
    }

    private static PhaseHandler.EffectRecordPrioritySettledListener installed() {
        try {
            Field field = PhaseHandler.class.getDeclaredField("PRIORITY_SETTLED_HOOK");
            field.setAccessible(true);
            EffectRecordHook<?> hook = (EffectRecordHook<?>) field.get(null);
            return (PhaseHandler.EffectRecordPrioritySettledListener) hook.get();
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("no priority-settled hook on this checkout", e);
        }
    }
}
