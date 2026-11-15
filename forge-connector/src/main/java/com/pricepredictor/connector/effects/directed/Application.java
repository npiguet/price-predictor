package com.pricepredictor.connector.effects.directed;

import com.pricepredictor.connector.effects.directed.operations.BoardEdit;
import forge.game.Game;
import forge.game.card.Card;

import java.util.HashSet;
import java.util.Objects;
import java.util.Set;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * How one scenario variant is made out of a board its rule's precondition
 * bound: the {@link ForkOperation} the scenario classifies the variant into,
 * the edit that applies it to the fork's objects, the records of the fork
 * kept whatever the samplers say ({@link KeptRecords}, FR-010c), and the one
 * play its fork forces, if any ({@link ForcedPlay}, FR-010b).
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
 * @param forced    the play the fork forces, or null when its seats make
 *                  every decision
 * @param made      the fork's binding once the edit is applied: the binding
 *                  with the roles of cards the edit made, found on the edited
 *                  copy (the card an {@code add_permanent} put there, which
 *                  no live card maps to), for a forced play to name
 * @param makes     the roles {@code made} binds: on the edited copy only,
 *                  never on the live board
 */
public record Application(
        ForkOperation operation, Function<Binding, BoardEdit> edit, Function<Binding, KeptRecords> kept,
        ForcedPlay forced, BiFunction<Binding, Game, Binding> made, Set<String> makes) {

    public Application {
        Objects.requireNonNull(operation, "operation");
        Objects.requireNonNull(edit, "edit");
        Objects.requireNonNull(kept, "kept");
        Objects.requireNonNull(made, "made");
        makes = Set.copyOf(makes);
    }

    /** A variant whose fork keeps nothing beyond what the samplers draw and forces no play. */
    public Application(ForkOperation operation, Function<Binding, BoardEdit> edit) {
        this(operation, edit, fork -> KeptRecords.NONE, null, (fork, game) -> fork, Set.of());
    }

    /**
     * This variant, its fork also keeping the records {@code more} names
     * (FR-010c), with those it kept already.
     */
    public Application keeping(Function<Binding, KeptRecords> more) {
        Objects.requireNonNull(more, "more");
        Function<Binding, KeptRecords> already = kept;
        return new Application(operation, edit, fork -> already.apply(fork).and(more.apply(fork)), forced, made,
                makes);
    }

    /** This variant, its fork forcing {@code play} (FR-010b). */
    public Application forcing(ForcedPlay play) {
        return new Application(operation, edit, kept, Objects.requireNonNull(play, "play"), made, makes);
    }

    /**
     * This variant, the card its edit makes bound as {@code role} once the
     * edit is applied, as {@code find} reads it on the edited copy: a role
     * no live card maps to, so what its forced play names of it is checked
     * on the copy only, never by the dry run on the live board.
     */
    public Application making(String role, BiFunction<Binding, Game, Card> find) {
        Objects.requireNonNull(role, "role");
        Objects.requireNonNull(find, "find");
        BiFunction<Binding, Game, Binding> already = made;
        Set<String> more = new HashSet<>(makes);
        more.add(role);
        return new Application(operation, edit, kept, forced, (fork, game) -> {
            Binding found = already.apply(fork, game);
            return found.with(role, find.apply(found, game));
        }, more);
    }

    /** The fork's binding once its edit is applied to {@code fork}: with the roles of the cards it made. */
    public Binding afterEdit(Binding forked, Game fork) {
        return Objects.requireNonNull(made.apply(forked, fork), "made");
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
