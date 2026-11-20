package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.effects.directed.KeywordEdit;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.spellability.SpellAbility;

import java.util.List;
import java.util.Map;

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

    /** A refusal's reason: the copy holds no combat to deal. */
    static final String NO_COMBAT = "no combat in the copy";

    /** The fork's kind, as the failure report and the refusal tally name it. */
    static final String KIND = ForkKind.PROBE.wireName();

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
        Forks.ForkRun run = new Forks.ForkRun(
                KIND, game,
                budget::spendProbe,
                () -> game.getPhaseHandler() == null
                        ? null : game.getPhaseHandler().getPlayerTurn());
        return run.take((verified, snapshots) -> {
            Game fork = withoutKeyword(verified, carrier, keyword);
            // The state is the board as the step began, with the keyword
            // already gone: that is the input the counterfactual answers for.
            RecordState state = snapshots.ofBoard().state();
            ProbedStep dealt = runDamageSteps(fork, stepsDealt(keyword, step));
            return new HeldProbe(
                    KeywordEdit.normalize(keyword),
                    Refs.of(carrier), state, dealt.events(),
                    Refs.of(verified.perspective()), dealt.combat(),
                    step,
                    // The *live* game's turn, which is also the fork's: the copier
                    // carries the counter across (GameCopier hands it to
                    // devModeSet). Read here so the branch remembers the step it
                    // belongs to.
                    liveTurn(), verified.copyDropped());
        }, (verified, held) -> held);
    }

    /**
     * The fork with {@code keyword} stripped from the copy of {@code carrier}.
     *
     * @throws ForkRefusal when the copy has no counterpart of the carrier or holds no combat
     */
    private static Game withoutKeyword(Forks.VerifiedFork verified, Card carrier, String keyword) {
        Game fork = verified.fork();
        Card forkCarrier = Forks.mappedCard(verified.copier(), carrier);
        if (forkCarrier == null) {
            throw new ForkRefusal(ForkRefusal.NO_COUNTERPART);
        }
        if (fork.getCombat() == null) {
            throw new ForkRefusal(NO_COMBAT);
        }
        // A stripped keyword reaches combat states ordinary play does
        // not; Forks.ForkRun discards the fork when one throws. Every
        // spelling the carrier has of it (trample, and trample over
        // planeswalkers): one left behind keeps the keyword working.
        List<String> spellings = KeywordEdit.carried(forkCarrier, keyword);
        for (String spelling : spellings.isEmpty() ? List.of(keyword) : spellings) {
            KeywordEdit.remove(fork, forkCarrier, spelling);
        }
        return fork;
    }

    /**
     * What a probe's damage step did: the fork's combat, read as the observed
     * record reads it, and the events the bus published.
     */
    private record ProbedStep(CombatShape combat, List<EffectEvent> events) {
    }

    /**
     * The damage steps a branch deals: the one it forked ahead of, and after
     * a first-strike step whose first or double strike was stripped, the
     * regular step as well.
     *
     * <p>The stripped creature's damage moves to the regular step: Forge lets
     * every combatant that dealt no first-strike damage deal damage there
     * ({@code Combat.dealDamageThisPhase}), and the branch's creature never
     * had first strike when its first-strike damage was dealt. A branch that
     * stopped after the first-strike step would hold nothing of what the
     * creature did without the keyword.
     */
    static List<DamageStep> stepsDealt(String keyword, DamageStep step) {
        String normalized = KeywordEdit.normalize(keyword);
        boolean strikeStripped = ProbeScheduler.FIRST_STRIKE.equals(normalized)
                || ProbeScheduler.DOUBLE_STRIKE.equals(normalized);
        if (step == DamageStep.FIRST_STRIKE && strikeStripped) {
            return List.of(DamageStep.FIRST_STRIKE, DamageStep.REGULAR);
        }
        return List.of(step);
    }

    /**
     * Assign and deal the fork's damage for each of {@code steps}, letting
     * the deaths happen after each, listening throughout.
     */
    private static ProbedStep runDamageSteps(Game fork, List<DamageStep> steps) {
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
        Forks.listeningForHooks(hooksFiling(sink), () -> {
            for (DamageStep step : steps) {
                if (!fork.isGameOver()) {
                    dealStep(fork, shape, step);
                }
            }
            return null;
        });
        return new ProbedStep(shape, sink.events());
    }

    /**
     * What the observed combat record takes from the hooks rather than the
     * bus, damage a shield or a protection prevented above all, is the
     * branch's too: without it the difference gate 2 reads would hold a
     * prevention the stripped keyword did not cause.
     */
    private static Forks.ForkHooks hooksFiling(ForkEventSink sink) {
        return new Forks.ForkHooks() {
            @Override
            public void clauseEvent(SpellAbility clause, EffectEvent event) {
                sink.fileOutcome(event, null);
            }

            @Override
            public void outcome(SpellAbility ability, EffectEvent event) {
                sink.fileOutcome(event, null);
            }
        };
    }

    /** One damage step's turn-based action, as {@code PhaseHandler.onPhaseBegin} deals it. */
    private static void dealStep(Game fork, CombatShape shape, DamageStep step) {
        Combat combat = fork.getCombat();
        combat.removeAbsentCombatants();
        if (combat.assignCombatDamage(step == DamageStep.FIRST_STRIKE)) {
            shape.addAssignment(combat);
            combat.dealAssignedDamage();
            StateBasedCheck.run(fork);
        }
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
            String keyword, String carrier, RecordState state,
            List<EffectEvent> events, String actor, CombatShape combat,
            DamageStep substep, int turn, Map<String, Integer> copyDropped) {

        HeldProbe {
            copyDropped = copyDropped == null ? Map.of() : Map.copyOf(copyDropped);
        }

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
        RecordPayload payload() {
            return RecordPayloads.probePayload(combat, events, keyword, carrier);
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
        if (isStale(held.turn(), now)) {
            System.err.println(
                    "Effect records: dropping a probe branch taken on turn "
                            + held.turn() + " that would mirror "
                            + mirrorOfRecordId + " from turn " + now
                            + "; its own damage step wrote no record");
            return false;
        }
        context.emit(context.newRecord(RecordKind.COMBAT)
                .fork(ForkKind.PROBE)
                .interventional(false)
                .mirrorOf(mirrorOfRecordId)
                .copyDropped(held.copyDropped())
                .actor(held.actor())
                .state(held.state())
                .payload(held.payload()));
        return true;
    }

    /** Whether a branch taken on {@code taken} no longer matches the live {@code now}, both known. */
    private static boolean isStale(int taken, int now) {
        return taken != TURN_UNKNOWN && now != TURN_UNKNOWN && taken != now;
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
