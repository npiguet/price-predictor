package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Binding;
import forge.game.Game;

import java.util.Map;
import java.util.Set;

/**
 * What a directed fork of the live game looks like once the variant is
 * applied: the changes from the live board, both snapshots, and the
 * fork's binding and game.
 */
record RuleTwin(Set<String> changes, Map<String, Object> real, Map<String, Object> forked,
                Binding fork, Game forkGame) {
}
