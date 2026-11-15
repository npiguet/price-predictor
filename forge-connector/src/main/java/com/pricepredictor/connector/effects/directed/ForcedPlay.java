package com.pricepredictor.connector.effects.directed;

import forge.game.keyword.Keyword;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Predicate;

/**
 * The one play a directed rule's variant forces in its fork (FR-010b): made by
 * the seat bound as {@link #actor}, in the fork's own step (an attack in the
 * turn's declare-attackers step, an answer whenever its question is asked),
 * while every other decision in the fork stays the seat's own.
 *
 * <p>Four kinds. A {@link Kind#CAST} or {@link Kind#ACTIVATE} plays a line of
 * the card bound as {@link #source}, chosen by {@link Line}, its targets named
 * by role clause by clause (a clause named by no role drawn from its legal
 * targets by the fork's own random stream), X drawn, its cost paid through the
 * engine's own payment, what the cost sacrifices, discards or exiles named by
 * role ({@link #costs}, else the seat's choice). A modal line (a charm) is no
 * line a forced play makes: the engine's cast asks its modes afresh. An
 * {@link Kind#ANSWER} answers yes or no the "you may" or "you may pay" a line
 * of the source asks, the line chosen by {@link Line} (by default the source's
 * triggered abilities). An {@link Kind#ATTACK} adds the source's attack on the
 * object bound as {@link #defender} to the seat's own declaration.
 *
 * <p>A value of the registry, built when the registry is: a keyword is named
 * by its {@code Keyword} constant's name, read only inside a fork, so building
 * one touches no localised Forge enum ({@code DirectedRules} is loaded where
 * Forge is not). Which variants force what is fixed per rule here, and no
 * record member marks a forced play: a fork's records hold it as a live play.
 *
 * @param kind     what is forced
 * @param actor    the role of the player who makes the play
 * @param source   the role of the card whose line is played, whose "you may"
 *                 is answered, or which attacks
 * @param line     which of the source's lines a cast or activation plays, or
 *                 whose question an answer answers; null for an attack
 * @param targets  per targeting clause of the line, in chain order, the roles
 *                 of its targets; an empty list draws that clause's targets
 * @param costs    the roles of the cards the cost sacrifices, discards or
 *                 exiles; empty leaves those choices to the seat
 * @param answer   an answer's yes or no
 * @param defender the role of the attacked player or permanent, for an attack
 */
public record ForcedPlay(
        Kind kind, String actor, String source, Line line,
        List<List<String>> targets, List<String> costs, boolean answer, String defender) {

    /** What a variant forces. */
    public enum Kind {
        /** Cast the source, a spell. */
        CAST,
        /** Activate a line of the source. */
        ACTIVATE,
        /** Answer a "you may" or a "you may pay" a line of the source asks. */
        ANSWER,
        /** Add the source's attack to the seat's declaration. */
        ATTACK
    }

    /**
     * Which of a source's lines a cast or activation plays, or whose question
     * an answer answers: its spell (not an alternative cost), its triggered
     * abilities, the activated ability of an API type, the line a keyword
     * gives it, or a line a rule describes in its own words.
     *
     * @param name what the line is, for a reader
     * @param test whether an ability of the source is the line; read only
     *             inside a fork or on a live board, never at registration
     */
    public record Line(String name, Predicate<SpellAbility> test) {

        /** The card's own cast, no alternative cost. */
        public static final Line SPELL = new Line("spell", ability ->
                ability.isSpell() && ability.getAlternativeCost() == null && ability.getKeyword() == null);

        /**
         * The source's triggered abilities, any clause of their chains: the
         * lines an answer answers by default.
         */
        public static final Line TRIGGER = new Line("trigger", SpellAbility::isTrigger);

        public Line {
            Objects.requireNonNull(name, "name");
            Objects.requireNonNull(test, "test");
        }

        /** The activated ability of {@code api}, an {@code ApiType} name. */
        public static Line api(String api) {
            Objects.requireNonNull(api, "api");
            return new Line(api, ability -> ability.isActivatedAbility() && ability.getApi() != null
                    && ability.getApi().name().equals(api));
        }

        /** The line {@code keyword}, a {@code Keyword} name, gives the source. */
        public static Line keyword(String keyword) {
            Objects.requireNonNull(keyword, "keyword");
            return new Line(keyword, ability -> ability.isKeyword(Keyword.valueOf(keyword)));
        }

        /** Whether {@code ability} is this line. */
        public boolean matches(SpellAbility ability) {
            return test.test(ability);
        }
    }

    public ForcedPlay {
        Objects.requireNonNull(kind, "kind");
        Objects.requireNonNull(actor, "actor");
        Objects.requireNonNull(source, "source");
        if ((kind != Kind.ATTACK) != (line != null)) {
            throw new IllegalArgumentException("a line belongs to a cast, an activation or an answer, and only there");
        }
        if ((kind == Kind.ATTACK) != (defender != null)) {
            throw new IllegalArgumentException("a defender belongs to an attack, and only there");
        }
        targets = targets.stream().map(List::copyOf).toList();
        costs = List.copyOf(costs);
    }

    /** {@code actor} casts the spell bound as {@code card}. */
    public static ForcedPlay cast(String actor, String card) {
        return new ForcedPlay(Kind.CAST, actor, card, Line.SPELL, List.of(), List.of(), false, null);
    }

    /** {@code actor} activates {@code line} of the card bound as {@code permanent}. */
    public static ForcedPlay activate(String actor, String permanent, Line line) {
        return new ForcedPlay(Kind.ACTIVATE, actor, permanent, line, List.of(), List.of(), false, null);
    }

    /**
     * {@code actor} answers {@code yes} to the "you may" or "you may pay" a
     * triggered ability of the card bound as {@code source} asks; another of
     * its lines with {@link #playing}.
     */
    public static ForcedPlay answer(String actor, String source, boolean yes) {
        return new ForcedPlay(Kind.ANSWER, actor, source, Line.TRIGGER, List.of(), List.of(), yes, null);
    }

    /** {@code actor}'s creature bound as {@code attacker} attacks what is bound as {@code defender}. */
    public static ForcedPlay attack(String actor, String attacker, String defender) {
        return new ForcedPlay(Kind.ATTACK, actor, attacker, null, List.of(), List.of(), false,
                Objects.requireNonNull(defender, "defender"));
    }

    /**
     * This play, its next targeting clause aimed at the objects bound as
     * {@code roles}; no role draws that clause's targets.
     */
    public ForcedPlay targeting(String... roles) {
        List<List<String>> more = new ArrayList<>(targets);
        more.add(List.of(roles));
        return new ForcedPlay(kind, actor, source, line, more, costs, answer, defender);
    }

    /** This play, its cost sacrificing, discarding or exiling the cards bound as {@code roles}. */
    public ForcedPlay paying(String... roles) {
        return new ForcedPlay(kind, actor, source, line, targets, List.of(roles), answer, defender);
    }

    /** This cast, activation or answer, of {@code other} rather than the line it named. */
    public ForcedPlay playing(Line other) {
        if (line == null) {
            throw new IllegalStateException("only a cast, an activation or an answer names a line");
        }
        return new ForcedPlay(kind, actor, source, Objects.requireNonNull(other, "line"), targets, costs,
                answer, defender);
    }

    /** Whether this play is a cast or an activation, played at a priority. */
    public boolean playsALine() {
        return kind == Kind.CAST || kind == Kind.ACTIVATE;
    }
}
