package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.DirectedTag;
import com.pricepredictor.connector.effects.directed.KeptRecords;
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
 * fork's recording only after the edit, runs the state-based check the edit
 * makes as the fork's premise ({@link ForkPremise}), keeps the records the variant
 * names whatever the samplers draw ({@link KeptRecords}, FR-010c), and
 * attaches the variant's forced play to its actor's seat in the copy
 * ({@link ForcedPlayRun}, FR-010b).
 *
 * <p>An edit that cannot describe its variant on this board, a bound object
 * the copy does not hold, or a fork that throws while playing on, discards
 * the fork with nothing written ({@link Forks.ForkRun}).
 */
final class DirectedFork {

    /** The fork's kind, as the failure report and the dropped-event counts name it. */
    static final String KIND = ForkKind.DIRECTED.wireName();

    private final Game game;
    private final ForkPlayForward playForward;
    /** The live player the copy's score is checked from, read after the copy. */
    private final Supplier<Player> perspective;

    DirectedFork(Game game, ForkPlayForward playForward, Supplier<Player> perspective) {
        this.game = game;
        this.playForward = playForward;
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
                rule.rules(), new DirectedTag(variant, application.operation()));
        Forks.ForkRun run = new Forks.ForkRun(KIND, game, () -> { }, perspective);
        ForkPlayForward.Played played = run.take((verified, snapshots) -> {
            Game fork = verified.fork();
            Binding forked = binding.inFork(verified.copier());
            // Named before the edit, which may move a named card: it keeps its id.
            KeptRecords kept = application.kept(forked);
            application.edit(forked).apply(fork);
            // Its line looked for on the edited copy, which may have given it
            // the line or made the card that holds it.
            ForcedPlayRun forced = ForcedPlayRun.start(application, application.afterEdit(forked, fork), fork);
            return playForward.play(fork, flags.andThen(verified::stamp), kept, forced);
        }, (verified, result) -> playForward.countDropped(KIND, result));
        return played != null;
    }

    /**
     * The flags every record of a directed fork carries (FR-009a, D5): the
     * rules its registration names as {@code aimed_at}, one unless rules
     * share a board, a binding and an edit, narrower than a run's list; a
     * non-interventional fork; and the variant and operation applied. No
     * {@code mirror_of}: a fork is taken at a phase boundary, which closes
     * every live record first, so none is open to mirror; the records are
     * paired with the live turn by {@code game_id}, turn and {@code directed}.
     */
    static Consumer<EffectRecord> flags(List<String> rules, DirectedTag tag) {
        return record -> record.fork(ForkKind.DIRECTED).interventional(false).aimedAt(rules).directed(tag);
    }
}
