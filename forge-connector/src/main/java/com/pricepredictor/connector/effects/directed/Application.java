package com.pricepredictor.connector.effects.directed;

import com.pricepredictor.connector.effects.directed.operations.BoardEdit;

import java.util.Objects;
import java.util.function.Function;

/**
 * How one scenario variant is made out of a board its rule's precondition
 * bound: the {@link ForkOperation} the scenario classifies the variant into,
 * and the edit that applies it to the fork's objects.
 *
 * <p>The operation is named apart from the edit so the registry can be
 * checked against {@code rules/scenarios/*.json} without a game
 * ({@code DirectedRulesContractTest}); {@link #edit} refuses an edit of
 * another operation, so the record's {@code directed.operation} is always the
 * scenario's.
 *
 * @param operation the variant's classified operation
 * @param edit      builds the edit from the fork's binding
 */
public record Application(ForkOperation operation, Function<Binding, BoardEdit> edit) {

    public Application {
        Objects.requireNonNull(operation, "operation");
        Objects.requireNonNull(edit, "edit");
    }

    /**
     * The edit for a fork's binding.
     *
     * @throws IllegalStateException when the edit is of another operation
     *                               than the variant's, which discards the
     *                               fork rather than mislabel its records
     */
    public BoardEdit edit(Binding fork) {
        BoardEdit built = edit.apply(fork);
        if (built.operation() != operation) {
            throw new IllegalStateException("a " + operation.wireName()
                    + " variant built a " + built.operation().wireName() + " edit");
        }
        return built;
    }
}
