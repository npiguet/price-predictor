package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.directed.DirectedRules;

/**
 * Prints the rule ids registered for directed forks, one per line, in
 * registration order.
 *
 * <p>The supervisor's startup check (FR-014) runs this once: a rule listed in
 * {@code --directed-fork-rules} that it does not print has no board
 * precondition in the worker, and the run exits naming it before any game.
 * The registry is plain code, so Forge's environment is not initialized.
 */
public final class DirectedRulesMain {

    private DirectedRulesMain() {
    }

    public static void main(String[] args) {
        ToolMain.run(() -> DirectedRules.registered().ids().forEach(System.out::println));
    }
}
