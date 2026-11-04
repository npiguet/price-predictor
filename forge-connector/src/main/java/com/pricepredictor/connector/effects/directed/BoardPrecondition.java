package com.pricepredictor.connector.effects.directed;

import forge.game.Game;

import java.util.Optional;

/**
 * Whether the live board meets a scenario's base variant (FR-014a): a
 * hand-written predicate over the live game, asked at every phase boundary of
 * a game whose run enables the rule.
 *
 * <p>It decides only when to fork. The fork's records are tagged afterwards by
 * the rule's Python detectors like any other record, so a precondition that
 * binds a board the detector then does not tag costs a fork and nothing else.
 * It reads the live game and changes nothing in it.
 */
@FunctionalInterface
public interface BoardPrecondition {

    /** The objects the variants act on, or empty when the board does not meet the base variant. */
    Optional<Binding> bind(Game live);
}
