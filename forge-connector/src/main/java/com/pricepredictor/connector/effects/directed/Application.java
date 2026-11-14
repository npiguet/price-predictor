package com.pricepredictor.connector.effects.directed;

import com.pricepredictor.connector.effects.directed.operations.BoardEdit;

import java.util.Objects;
import java.util.function.Function;

/**
 * How one scenario variant is made out of a board its rule's precondition
 * bound: the {@link ForkOperation} the scenario classifies the variant into,
 * the edit that applies it to the fork's objects, and the records of the
 * fork kept whatever the samplers say ({@link KeptRecords}, FR-010c).
 *
 * <p>The operation is named apart from the edit so the registry can be
 * checked against {@code rules/scenarios/*.json} without a game
 * ({@code DirectedRulesContractTest}); {@link #edit} refuses an edit of
 * another operation, so the record's {@code directed.operation} is always the
 * scenario's.
 *
 * @param operation the variant's classified operation
 * @param edit      builds the edit from the fork's binding
 * @param kept      names, from the fork's binding, the cards whose records
 *                  the fork keeps
 */
public record Application(
        ForkOperation operation, Function<Binding, BoardEdit> edit, Function<Binding, KeptRecords> kept) {

    public Application {
        Objects.requireNonNull(operation, "operation");
        Objects.requireNonNull(edit, "edit");
        Objects.requireNonNull(kept, "kept");
    }

    /** A variant whose fork keeps nothing beyond what the samplers draw. */
    public Application(ForkOperation operation, Function<Binding, BoardEdit> edit) {
        this(operation, edit, fork -> KeptRecords.NONE);
    }

    /** This variant, its fork keeping the records {@code kept} names (FR-010c). */
    public Application keeping(Function<Binding, KeptRecords> kept) {
        return new Application(operation, edit, kept);
    }

    /** This variant, its fork keeping the decision candidates of the card bound as {@code role}. */
    public Application keepingCandidatesOf(String role) {
        return keeping(fork -> KeptRecords.candidatesOf(fork.card(role)));
    }

    /** This variant, its fork keeping the not-fired evaluations of the triggers the card bound as {@code role} hosts. */
    public Application keepingNegativesOf(String role) {
        return keeping(fork -> KeptRecords.triggersOf(fork.card(role)));
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

    /** The records the fork of {@code fork}'s binding keeps. */
    public KeptRecords kept(Binding fork) {
        return Objects.requireNonNull(kept.apply(fork), "kept");
    }
}
