package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.player.Player;

/**
 * The phase handler's two signals for one game: where each run of the
 * state-based loop begins and ends, and each moment priority settles.
 *
 * <p>Neither is observable otherwise. The tracker's freeze covers one pass of
 * the loop and not the triggers put on the stack between passes, and the
 * bus's priority event fires before the loop has run, so the board it
 * describes may still hold a creature about to die.
 *
 * <p>The depth of the loop is kept here and nowhere else: the sink hears
 * the outermost begin and the matching end only, so a loop begun inside a
 * loop leaves the outer one open.
 *
 * <p>Both hooks are JVM statics, and a fork's own phase handler fires them
 * too, so each handler keeps to its game by the {@code Game} the engine hands
 * it, compared by reference as {@link LiveGame} compares every hook's
 * argument.
 */
final class PhaseSignals {

    /** What this game's signals are delivered to. */
    interface Sink {
        default void stateBasedLoopBegan() {
        }

        default void stateBasedLoopEnded() {
        }

        default void prioritySettled(Player priorityPlayer) {
        }

        /** Both sinks, {@code first} told of each signal before {@code second}. */
        static Sink both(Sink first, Sink second) {
            return new Sink() {
                @Override
                public void stateBasedLoopBegan() {
                    try {
                        first.stateBasedLoopBegan();
                    } finally {
                        second.stateBasedLoopBegan();
                    }
                }

                @Override
                public void stateBasedLoopEnded() {
                    try {
                        first.stateBasedLoopEnded();
                    } finally {
                        second.stateBasedLoopEnded();
                    }
                }

                @Override
                public void prioritySettled(Player priorityPlayer) {
                    try {
                        first.prioritySettled(priorityPlayer);
                    } finally {
                        second.prioritySettled(priorityPlayer);
                    }
                }
            };
        }
    }

    private final LiveGame live;
    private final Sink sink;

    /** How many begins are open: a loop begun inside a loop leaves the outer one open at its end. */
    private int loopDepth;

    PhaseSignals(LiveGame live, Sink sink) {
        this.live = live;
        this.sink = sink;
    }

    /** Whether this game is between a state-based loop's begin and its end. */
    boolean inStateBasedLoop() {
        return loopDepth > 0;
    }

    /**
     * The state-based loop hook's listener.
     *
     * <p>Package-private, as every handler is: the argument order has no
     * compiler behind it, so its game-identity test calls it with a
     * synthesized argument array.
     */
    PatchHooks.Listener stateBasedLoopHandler() {
        return PatchHooks.STATE_BASED_LOOP_LISTENER.listener((callback, args) -> {
            if (!ours(args)) {
                return;
            }
            if (callback == PatchHooks.Callback.ON_STATE_BASED_LOOP_BEGIN) {
                begin();
            } else {
                end();
            }
        });
    }

    private void begin() {
        loopDepth++;
        if (loopDepth == 1) {
            sink.stateBasedLoopBegan();
        }
    }

    private void end() {
        loopDepth = Math.max(0, loopDepth - 1);
        if (loopDepth == 0) {
            sink.stateBasedLoopEnded();
        }
    }

    /** The priority-settled hook's listener. */
    PatchHooks.Listener prioritySettledHandler() {
        return PatchHooks.PRIORITY_SETTLED_LISTENER.listener((callback, args) -> {
            if (ours(args)) {
                sink.prioritySettled(args.get(1, Player.class));
            }
        });
    }

    /** Whether a signal's first argument is this game. */
    private boolean ours(PatchHooks.Args args) {
        Game game = args.get(0, Game.class);
        return game != null && game == live.game();
    }
}
