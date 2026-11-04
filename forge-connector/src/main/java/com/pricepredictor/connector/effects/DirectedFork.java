package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.DirectedTag;
import forge.game.Game;
import forge.game.player.Player;

import java.util.List;
import java.util.function.Consumer;
import java.util.function.Supplier;

/**
 * A <b>directed fork</b> (FR-010): copy the live game at a phase boundary
 * where a rule's board precondition holds, apply one contrasting variant's
 * change through its operation, play the copy on to the end of the turn and
 * write what it recorded.
 *
 * <p>The order inside the fork is the one the operations need. The copy is
 * score-checked against the live game before anything is changed (FR-013),
 * so a copy that already disagrees is discarded as measuring the copier. The
 * edit is applied before the fork's collectors start listening: the edits are
 * silent to the rules, but they still publish bus events (a zone change, an
 * attachment, a stats change), which would otherwise land in the fork's
 * records as if the turn had done them. {@link ForkPlayForward#play} opens the
 * fork's recording only after the edit.
 *
 * <p>An edit that cannot describe its variant on this board, a bound object
 * the copy does not hold, or a fork that throws while playing on, discards
 * the fork with nothing written ({@link Forks.ForkRun}).
 */
final class DirectedFork {

    private final Game game;
    private final ForkPlayForward playForward;
    /** The id of a live record still open at the fork point, or null (D5). */
    private final Supplier<String> openRecordId;
    /** The live player the copy's score is checked from, read after the copy. */
    private final Supplier<Player> perspective;

    DirectedFork(
            Game game, ForkPlayForward playForward, Supplier<String> openRecordId,
            Supplier<Player> perspective) {
        this.game = game;
        this.playForward = playForward;
        this.openRecordId = openRecordId;
        this.perspective = perspective;
    }

    /**
     * Take one directed fork. The budget is the scheduler's to spend, before
     * this is called, so a discarded fork counts as a kept one does.
     *
     * @param binding the live objects the precondition bound
     * @return whether the fork was kept and its records written
     */
    boolean take(DirectedRule rule, String variant, Application application, Binding binding) {
        Consumer<EffectRecord> flags = flags(
                rule.rule(), new DirectedTag(variant, application.operation()),
                openRecordId.get());
        Forks.ForkRun run = new Forks.ForkRun(game, () -> { }, perspective);
        ForkPlayForward.Played played = run.take((verified, snapshots) -> {
            Game fork = verified.fork();
            application.edit(binding.inFork(verified.copier())).apply(fork);
            return playForward.play(fork, flags);
        }, (verified, result) -> result);
        return played != null;
    }

    /**
     * The flags every record of a directed fork carries (FR-009a, D5): the one
     * rule it was taken for as {@code aimed_at}, narrower than a run's list; a
     * non-interventional fork; the variant and operation applied; and
     * {@code mirror_of} only when a live record was open at the fork point.
     * A fork taken at a phase boundary usually has none, because the live
     * turn's records are written after the fork ends; the records are paired
     * with the live turn by {@code game_id}, turn and {@code directed}.
     */
    static Consumer<EffectRecord> flags(String rule, DirectedTag tag, String mirrorOf) {
        return record -> {
            record.fork(true).interventional(false).aimedAt(List.of(rule)).directed(tag);
            if (mirrorOf != null) {
                record.mirrorOf(mirrorOf);
            }
        };
    }
}
