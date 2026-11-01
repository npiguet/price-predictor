package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.combat.Combat;
import forge.game.phase.PhaseType;

import java.util.function.Consumer;

/**
 * The live collector's damage-step bracket: one {@code combat} record per
 * damage step, closed at the phase boundary, so a first-strike combat writes
 * two — the only way a keyword that acts by splitting the step is visible in
 * the corpus at all.
 *
 * <p>{@code ability} is absent from the record: no single line acts in a
 * damage step, and inventing one would attribute the whole step to whichever
 * creature happened to be first.
 */
final class CombatStepBracket {

    private final Game game;
    private final CollectorContext context;
    private final SnapshotBuilder snapshots;
    /** Written before the step's bracket opens: combat damage is the next action. */
    private final HeldResolutionWindow heldResolution;
    /** Told each combat record's id, for a probe that mirrors it. */
    private final Consumer<String> onCombatRecord;

    /** The damage step's combat being gathered, or null outside one. */
    private StepCombat combat;

    /**
     * The board as this damage step began, before any damage, or null outside
     * a damage step; taken by {@link #beginStep} and used by the step's record.
     */
    private String stepStart;

    /**
     * Whether this damage step's combat record has been written because
     * something used the stack after the damage.
     *
     * <p>A damage step has a priority window after its damage, in which
     * combat-damage triggers and anything cast resolve. Their events are
     * theirs, not the combat's, so the first bracket to open in the step writes
     * the combat record, and nothing after it joins one. Reset at the next
     * phase.
     */
    private boolean windowClosed;

    /**
     * @param onCombatRecord told the id of each {@code combat} record as it is
     *                       written; see {@link BusBracketCollector}'s
     *                       constructor
     */
    CombatStepBracket(
            Game game, CollectorContext context, SnapshotBuilder snapshots,
            HeldResolutionWindow heldResolution, Consumer<String> onCombatRecord) {
        this.game = game;
        this.context = context;
        this.snapshots = snapshots;
        this.heldResolution = heldResolution;
        this.onCombatRecord = onCombatRecord;
    }

    /**
     * One damage step's combat, from the first event dealt in it until the
     * step ends or something uses the stack.
     *
     * <p>The state is the board as the step began ({@link #beginStep}): read
     * at the first damage event instead, it already shows that event's damage
     * marked, so the record's input holds part of its own answer.
     *
     * <p>The structure is read at the first damage event rather than at the
     * flush, because the flush happens at the phase boundary — after the
     * damage has been dealt and the creatures it killed have left. Read there,
     * an attacker that traded with its blocker is in neither list, and the
     * record describes a combat nobody fought. At the step's start the
     * assignment does not exist yet: Forge assigns combat damage in the
     * step's turn-based action, after the phase event.
     */
    private static final class StepCombat {
        private final String state;
        private final CombatShape shape;
        private final EventBuffer events = new EventBuffer();

        private StepCombat(String state, Combat combat) {
            this.state = state;
            // Nothing has died yet, so this is the moment the attackers and
            // blockers are all still in combat. The assignment is Forge's copy of
            // this step's, taken before any replacement, so it reads the same
            // whenever in the step it is read.
            this.shape = CombatShape.of(combat);
            shape.addAssignment(combat);
        }

        String payload() {
            return RecordPayloads.combatPayload(shape, events.events());
        }
    }

    /**
     * Whether an event now is the combat's: dealt in a damage step, before
     * anything has used the stack in it ({@link #windowClosed}).
     */
    boolean takesEventsNow() {
        return !windowClosed && inDamageStep();
    }

    /**
     * A step has begun: in a combat damage step, the board is snapshotted now,
     * before the step deals any damage (FR-027). Taken for every damage step
     * with a combat and dropped at the step's end when no damage came.
     */
    void beginStep(PhaseType step) {
        stepStart = (step == PhaseType.COMBAT_FIRST_STRIKE_DAMAGE
                || step == PhaseType.COMBAT_DAMAGE) && game.getCombat() != null
                ? snapshots.ofBoard().render() : null;
    }

    /** File an event into the damage step's bracket, opening it first if need be. */
    void file(EffectEvent event) {
        if (combat == null) {
            // Combat damage is the next action. Ordinarily the step boundary
            // before it has written the held resolution already.
            heldResolution.close();
            // The step's start where it was seen; a step entered without its
            // phase event (a hand-driven board) is read now, as late as that.
            String state = stepStart != null ? stepStart : snapshots.ofBoard().render();
            stepStart = null;
            combat = new StepCombat(state, game.getCombat());
        }
        combat.events.file(event);
    }

    /**
     * Something is going on the stack. In a damage step that ends the
     * combat's share of the step: the record is written now, and what
     * resolves for the rest of the step is not the combat's.
     */
    void closeWindow() {
        if (inDamageStep()) {
            flush();
            windowClosed = true;
        }
    }

    /**
     * A step or phase has ended: the step's combat record is written and the
     * next damage step starts with its window open again.
     */
    void endStep() {
        flush();
        windowClosed = false;
        stepStart = null;
    }

    /** Write the open combat bracket, if any, as one {@code combat} record. */
    void flush() {
        if (combat == null) {
            return;
        }
        var phase = game.getPhaseHandler();
        String actor = phase == null ? null : Refs.ofNullable(phase.getPlayerTurn());
        EffectRecord record = context.newRecord(RecordKind.COMBAT)
                .actor(actor)
                .state(combat.state)
                .payload(combat.payload());
        context.emit(record);
        // A damage-step probe forked before this step and has been holding its
        // branch since: it needs this record's id to say which combat it
        // mirrors, and the id does not exist until the record is written.
        onCombatRecord.accept(record.recordId());
        combat = null;
    }

    /**
     * Whether the game is in either combat damage step.
     *
     * <p>Both names are tested: Forge's first-strike step is {@code
     * COMBAT_FIRST_STRIKE_DAMAGE}, which does not contain {@code
     * COMBAT_DAMAGE}, so a test on that substring alone would send the step's
     * creature damage, life changes and deaths to the ordinary bracket.
     */
    private boolean inDamageStep() {
        return DamageStep.of(game.getPhaseHandler()) != null;
    }
}
