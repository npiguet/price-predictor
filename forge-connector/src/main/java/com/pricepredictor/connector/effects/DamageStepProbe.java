package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.card.Card;
import forge.game.player.Player;

import java.util.HashSet;
import java.util.List;
import java.util.Optional;

/**
 * A <b>damage-step probe</b>: a fork that re-runs one combat with a keyword
 * stripped, to isolate what that keyword did.
 *
 * <p>It is built only for keywords gate 2 routed to it, and taken only when
 * {@code --probe-keywords} names one — the flag is the runtime switch,
 * separate from the build decision. It names all eight damage-step keywords
 * unless a run narrows or empties it.
 */
final class DamageStepProbe {

    /**
     * No turn could be read, so the pairing guard stands down.
     *
     * <p>A probe with no live game — the unit tests' shape — has no turn to
     * compare against, and refusing to write there would test the guard
     * rather than the pairing.
     */
    static final int TURN_UNKNOWN = -1;

    private final Game game;
    private final CollectorContext context;
    private final ForkBudget budget;

    DamageStepProbe(Game game, CollectorContext context, ForkBudget budget) {
        this.game = game;
        this.context = context;
        this.budget = budget;
    }

    /**
     * Re-run this damage step without one creature's keyword.
     *
     * <p>Called <b>before</b> the real step, which is the only moment the fork
     * can diverge: the phase event fires ahead of the turn-based action that
     * deals the damage, so the copy is taken from a board where nothing has
     * been dealt yet. The fork then assigns and deals its own damage with the
     * keyword gone, and what the bus publishes is what that keyword was worth.
     *
     * <p>The branch is <b>held rather than written</b>. It mirrors the real
     * combat record, and that record does not exist until the step it forked
     * ahead of has finished — so the caller completes the pairing once the id
     * is known. Reconstructing the pairing from board state instead would break
     * on two identical-looking combats in one turn.
     *
     * @param step the damage step about to be dealt, which the fork deals too
     * @return the held branch, or null when the fork could not be taken
     */
    HeldProbe probe(String keyword, Card carrier, DamageStep step) {
        if (!budget.mayProbe(keyword) || carrier == null) {
            return null;
        }
        // The whole method is the fork, not just the damage it deals: see
        // ForkCollector.FORK_RUNNING for what the narrower scope left uncovered.
        return ForkCollector.asFork(() -> {
            // Counted before the score check, like an intervention: a copy that
            // systematically fails must not retry until the game ends.
            budget.spendProbe();

            Optional<ForkCollector.VerifiedFork> verified = ForkCollector.copyAndVerify(
                    game,
                    () -> game.getPhaseHandler() == null
                            ? null : game.getPhaseHandler().getPlayerTurn());
            if (verified.isEmpty()) {
                return null;
            }
            Game fork = verified.get().fork();
            Player perspective = verified.get().perspective();
            Card forkCarrier = ForkCollector.mappedCard(verified.get().copier(), carrier);
            if (forkCarrier == null || fork.getCombat() == null) {
                return null;
            }

            SnapshotBuilder snapshots = new SnapshotBuilder(fork);
            String state;
            ProbedStep dealt;
            try {
                stripKeyword(fork, forkCarrier, keyword);
                // The state is the board as the step began, with the keyword
                // already gone: that is the input the counterfactual answers for.
                state = snapshots.ofBoard().render();
                dealt = runDamageStep(fork, step == DamageStep.FIRST_STRIKE);
            } catch (RuntimeException | StackOverflowError e) {
                // A stripped keyword reaches combat states ordinary play does not.
                // Discarding is right; failing the worker is not.
                return null;
            }
            return new HeldProbe(
                    ProbeScheduler.normalizeKeyword(keyword),
                    SnapshotBuilder.entityId(carrier), state, dealt.events(),
                    SnapshotBuilder.playerId(perspective), dealt.combat(),
                    step,
                    // The *live* game's turn, which is also the fork's: the copier
                    // carries the counter across (GameCopier hands it to
                    // devModeSet). Read here so the branch remembers the step it
                    // belongs to.
                    liveTurn());
        });
    }

    /**
     * Take {@code keyword} away from the fork's copy of the carrier.
     *
     * <p>Through the layer system rather than off the printed list: the
     * keyword may be printed, equipped or granted until end of turn, and a
     * removal layer takes it away in all three cases the way the rules would.
     */
    private static void stripKeyword(Game fork, Card forkCarrier, String keyword) {
        forkCarrier.addChangedCardKeywords(
                null, List.of(keyword), false,
                fork.getNextTimestamp(), null);
    }

    /**
     * What a probe's damage step did: the fork's combat, read as the observed
     * record reads it, and the events the bus published.
     */
    private record ProbedStep(CombatShape combat, List<EffectEvent> events) {
    }

    /**
     * Assign and deal the fork's damage for one step, then let the deaths
     * happen, listening throughout.
     *
     * @param firstStrike whether this is the first-strike damage step
     */
    private static ProbedStep runDamageStep(Game fork, boolean firstStrike) {
        // Listening from here, once the keyword is gone: the stripped
        // keyword is the branch's input, and a sink primed before it
        // would report its removal as something the step did. No root:
        // a damage step is a turn-based action, not a resolution, so
        // this branch's events attribute the way the observed combat
        // record's do -- from the resolving-clause pointer alone, which
        // names nothing.
        ForkEventSink sink = ForkEventSink.forProbe(fork);
        fork.subscribeToEvents(sink);
        fork.getCombat().removeAbsentCombatants();
        // Who is blocking whom, before the damage removes any of them; the
        // assignment only exists after. Both are read as the observed
        // record reads them, so the branch and the record it mirrors agree.
        CombatShape shape = CombatShape.of(fork.getCombat());
        if (fork.getCombat().assignCombatDamage(firstStrike)) {
            shape.addAssignment(fork.getCombat());
            fork.getCombat().dealAssignedDamage();
            // And then let the deaths happen. Damage alone kills nothing;
            // a creature with lethal damage on it leaves the battlefield
            // during state-based actions, and stopping before them would make
            // the branch systematically miss every death the real step had —
            // biasing the difference gate 2 computes towards "the keyword
            // changed nothing". The same call Forge's own simulator makes
            // after resolving a stack.
            fork.getAction().checkStateEffects(false, new HashSet<>());
        }
        return new ProbedStep(shape, sink.events());
    }

    /**
     * A probe branch waiting for the record it mirrors.
     *
     * <p>Holds what the fork observed until the real combat record has an id.
     * The probe's own budget was already spent taking the fork, so a branch
     * that is never completed still counted — the cost was the simulation, not
     * the write.
     *
     * <p>{@code substep} names the damage step this branched from, the one
     * the bracket collector's record names. Without it a branch taken at the
     * first-strike step would be completed against whichever combat record
     * came next, the <b>regular</b> step's, and gate 2 would read a
     * counterfactual for a step that did not happen against a real record for
     * a different one.
     *
     * <p>Package-private, as {@link CombatShape} is: nothing outside this
     * package takes a probe or completes one.
     */
    record HeldProbe(
            String keyword, String carrier, String state,
            List<EffectEvent> events, String actor, CombatShape combat,
            DamageStep substep, int turn) {

        /**
         * The branch's payload, naming what was perturbed.
         *
         * <p>Both the keyword and the creature it came from: a board with two
         * tramplers gives the keyword alone two readings, and the evaluator
         * reproducing this perturbation model-side would strip the wrong one.
         *
         * <p>{@code combat} is the fork's own combat, the shape the probe's
         * damage step filled in, rendered here rather than when the branch
         * was taken. Nothing writes to that shape once the step has run, so it
         * renders as the step left it. It is the counterfactual's half of
         * what gate 2 compares: the observed record says who blocked whom and
         * how the damage was split with the keyword, and this says the same
         * about the board without it.
         */
        String payload() {
            return RecordPayloads.probePayload(combat.fields(), events, keyword, carrier);
        }
    }

    /**
     * Write a held branch, now that the record it mirrors has an id.
     *
     * <p>As an ordinary combat record with {@code fork = true} and {@code
     * interventional = false} — which is what distinguishes a probe from an
     * intervention by flags alone — and {@code mirror_of} naming the real
     * record it varies. The real-versus-fork <b>difference is never computed
     * here</b> and never becomes a training target: it is a diagnostic about
     * the model, computed at evaluation time, and training on it would teach
     * the model to reproduce its own errors.
     *
     * <p>Only if the two describe the same moment. A branch is completed
     * against the next combat record of its own substep, and a damage step that
     * wrote no real record leaves its branch waiting for one, which can then
     * be completed against a mirror turns later.
     *
     * <p>Dropped rather than relabelled, which is the choice worth stating.
     * The fork's turn is not stale: the copier carries the counter across, and
     * the branch's snapshot honestly describes the board of the turn it was
     * taken on. Stamping the mirror's turn over it would make the state block
     * lie about a board that is genuinely several turns old, and gate 2 would
     * read the difference between two unrelated combats as something the
     * keyword did. The branch's budget was spent taking the fork either way;
     * the only question is whether an unreadable counterfactual is written,
     * and it is not — the same policy {@code discardStaleProbes} applies to a
     * branch of the wrong substep.
     */
    boolean writeHeldProbe(HeldProbe held, String mirrorOfRecordId) {
        if (held == null || mirrorOfRecordId == null) {
            return false;
        }
        int now = liveTurn();
        if (held.turn() != TURN_UNKNOWN && now != TURN_UNKNOWN
                && held.turn() != now) {
            System.err.println(
                    "Effect records: dropping a probe branch taken on turn "
                            + held.turn() + " that would mirror "
                            + mirrorOfRecordId + " from turn " + now
                            + "; its own damage step wrote no record");
            return false;
        }
        context.emit(context.newRecord(RecordKind.COMBAT)
                .fork(true)
                .interventional(false)
                .mirrorOf(mirrorOfRecordId)
                .actor(held.actor())
                .state(held.state())
                .payload(held.payload()));
        return true;
    }

    /**
     * The live game's turn, or {@link #TURN_UNKNOWN}.
     *
     * <p>Read from the game this probe forks from rather than from a fork, so
     * a branch and the record it will mirror are timed by the same clock: the
     * mirror's snapshot reads this same counter at the instant the branch is
     * completed.
     */
    private int liveTurn() {
        if (game == null || game.getPhaseHandler() == null) {
            return TURN_UNKNOWN;
        }
        return game.getPhaseHandler().getTurn();
    }
}
