package com.pricepredictor.connector.effects;

import forge.game.card.Card;
import forge.game.card.CardView;
import forge.game.event.GameEventCardChangeZone;
import forge.game.spellability.SpellAbility;

import java.util.List;

/**
 * The live collector's resolution whose effect halves wait for their
 * state-based consequences.
 *
 * <p>Forge does not check state-based actions when an ability finishes:
 * {@code MagicStack.resolveStack} publishes {@code GameEventSpellResolved}
 * and returns, and the creature a Lightning Bolt left with lethal damage
 * dies at the next priority, inside {@code GameAction.checkStateEffects} —
 * after {@code GameEventPlayerPriority}, so not even that event is a
 * boundary. Written at the resolved event, the effect half would hold the
 * damage and never the death, which a combat record, whose bracket stays
 * open until the step ends, does hold.
 *
 * <p>So the halves are held from the resolution until the end of the
 * state-based loop that follows it ({@code PhaseHandler.checkStateBasedEffects},
 * whose end signal {@link PhaseSignals} delivers), and written then, exactly
 * once ({@link #close}). A spell's card leaves the stack between the two,
 * after the resolution is published: that move is the last event of the
 * resolution's own ({@link #takeMoveOffStack}, FR-042b), and the deaths come
 * after it. The loop is the whole check: each pass, and the
 * triggers it puts on the stack between two passes, which open brackets of
 * their own without ending this one (FR-028). On a worker without the loop
 * hook the window ends at the next real action instead — a bracket opening
 * for a cast or a trigger, another resolution ending, a combat bracket
 * opening, a step or phase boundary, a land played, the end of the game —
 * and those stay as the fallback with the hook too, for a resolution no loop
 * follows. One resolution at most is held: the next one's end is itself such
 * an action.
 */
final class HeldResolutionWindow {

    private final CollectorContext context;
    /** Where a free activation that changed nothing is reported (FR-040). */
    private final FruitlessActivations fruitless;

    /**
     * The resolution waiting for its consequences, or null when none is: the
     * one waiting that takes the state-based deaths its effects cause.
     */
    private Waiting held;

    /**
     * A fizzled or declined spell's effect halves, written with no event but
     * its card's move off the stack, which Forge makes after publishing the
     * resolution; null when none waits. It takes no deaths: it did nothing.
     */
    private Waiting fizzled;

    HeldResolutionWindow(CollectorContext context, FruitlessActivations fruitless) {
        this.context = context;
        this.fruitless = fruitless;
    }

    /**
     * What one held effect half's record says besides its events, all of it
     * fixed at the resolution — the id, the timestamp, the key and the board,
     * which is the one taken as the resolution started. The events can still
     * grow, by the spell's move off the stack and the deaths the
     * resolution's own effects cause.
     */
    private record HalfRecord(
            String recordId, String timestamp, ProvenanceKey.Resolved keys, String state,
            List<ConditionsMet> conditions) {

        static HalfRecord of(
                String recordId, ProvenanceKey.Resolved keys, String state,
                List<ConditionsMet> conditions) {
            return new HalfRecord(
                    recordId, RecordShardWriter.timestamp(), keys, state, conditions);
        }
    }

    /**
     * What a waiting resolution's records say about the resolution itself.
     *
     * @param root  the resolved ability, which the deaths and the move are stamped against
     * @param key   its line, for {@link FruitlessActivations}
     * @param actor who the records name as the actor
     */
    private record Source(SpellAbility root, ProvenanceKey key, String actor, String linkId) {
        static Source of(OpenResolution resolved) {
            return new Source(resolved.ability, resolved.keys.key(), resolved.actor, resolved.linkId);
        }
    }

    /**
     * A resolution's effect halves, their ids and boards fixed at the
     * resolution, waiting for the card's move off the stack (608.2b). The
     * one held waits for the deaths of the state-based loop too, and so
     * holds the events its halves gathered; a fizzled or declined spell's
     * halves hold none but the move.
     *
     * @param free whether it is an activation that paid nothing the game keeps
     *             ({@link PaidCosts#paidNothingKept}); never true of a fizzled spell
     */
    private static final class Waiting {
        final Source source;
        final ActivationOutcome outcome;
        final HeldHalves<HalfRecord> halves;
        final boolean free;
        /**
         * The id of the spell card whose move off the stack the last half
         * takes, or -1 when none is awaited: an ability, or a move taken.
         */
        int awaitedCard;

        Waiting(
                Source source, ActivationOutcome outcome, HeldHalves<HalfRecord> halves,
                boolean free, int awaitedCard) {
            this.source = source;
            this.outcome = outcome;
            this.halves = halves;
            this.free = free;
            this.awaitedCard = awaitedCard;
        }

        /** Whether it waits for {@code move}: its card leaving the stack. */
        boolean awaits(Move move) {
            return awaitedCard >= 0 && move.leftStackAs(awaitedCard);
        }

        /** Take the card's move off the stack into the last half. */
        void takeMove(EffectEvent moved) {
            awaitedCard = -1;
            halves.fileLast(EventAttribution.stampAt(moved, source.root(), null, null));
        }
    }

    /**
     * Hold a resolution's effect halves, under its cost half's link where it
     * has one.
     *
     * <p>Held rather than written: the spell's card has not left the stack
     * yet, and the deaths this resolution causes have not happened.
     *
     * @param outcome what became of the item: resolved, or partially fizzled
     * @param free    whether it is an activation that paid nothing the game keeps
     *                ({@link PaidCosts#paidNothingKept}), which is reported when it
     *                changed nothing
     */
    void hold(OpenResolution resolved, ActivationOutcome outcome, boolean free) {
        List<ModeSplit.ModeHalf> split = resolved.modes.halves(resolved.effectState());
        held = new Waiting(
                Source.of(resolved), outcome, halvesOf(resolved, split, true, true), free,
                awaitedCard(resolved.ability, true));
    }

    /**
     * Write the effect half of an item that did nothing (FR-042b): one that
     * fizzled, or whose controller declined what it offered. Its halves
     * have no events but one: a spell's own card's move off the stack
     * (608.2b), which Forge makes after publishing the resolution, so a
     * fizzled or declined spell's halves wait for it ({@link
     * #takeMoveOffStack}) and the last of them takes it; an ability, and a
     * copy of a spell, which ceases to exist without a move (707.10a), has
     * none and is written now. Nothing after the move joins them.
     * They are split as a resolved item's are: one root half, or on a
     * patched worker one per chosen mode of a modal item, through the mode's
     * option line, under the cost half's link (FR-029a, FR-029c). A declined
     * half keeps the conditions Forge decided on its way; a fizzled one
     * decided none.
     *
     * <p>Whatever its halves gathered anyway reaches no record and is
     * counted as dropped.
     *
     * @param outcome {@link ActivationOutcome#FIZZLED} or {@link ActivationOutcome#DECLINED}
     * @param dropped where what the halves gathered is counted
     * @param phase   the engine phase it is counted in
     */
    void writeEmpty(
            OpenResolution resolved, ActivationOutcome outcome, DroppedEvents dropped, String phase) {
        List<ModeSplit.ModeHalf> split = resolved.modes.halves(resolved.effectState());
        for (ModeSplit.ModeHalf half : split) {
            for (EffectEvent gathered : half.events()) {
                dropped.count(gathered.type(), phase);
            }
        }
        HeldHalves<HalfRecord> halves =
                halvesOf(resolved, split, false, outcome == ActivationOutcome.DECLINED);
        // A spell's card leaves the stack after a fizzle or a refusal as after
        // any resolution; a copy's ceases to exist instead (707.10a) and
        // publishes no move, so its halves are written now, empty.
        Waiting waiting = new Waiting(
                Source.of(resolved), outcome, halves, false, awaitedCard(resolved.ability, false));
        if (waiting.awaitedCard < 0) {
            emit(waiting);
            return;
        }
        fizzled = waiting;
    }

    /**
     * The id of the card whose move off the stack a spell's halves wait for,
     * or -1 for an ability, and for a copy of a spell unless {@code copiesWait}.
     */
    private static int awaitedCard(SpellAbility ability, boolean copiesWait) {
        Card host = ability.isSpell() ? ability.getHostCard() : null;
        if (host == null) {
            return -1;
        }
        return host.isCopiedSpell() && !copiesWait ? -1 : host.getId();
    }

    /**
     * One half per mode of a resolution, in order, each through the mode's own
     * option line (FR-029a, FR-029d).
     *
     * <p>Modes, X and the named card are set while the ability resolves, so
     * the block that carries them is re-read here and paired with the board
     * each half was captured with: the resolution's start for a root half,
     * the mode's first clause's for a mode's. A mode that fizzled on its own
     * carries an empty list under its key, which is what the schema says of
     * any resolution that did nothing observable.
     *
     * @param keepsEvents     whether a half holds the events it gathered, and its
     *                        board the library cards they name; false for an item
     *                        that did nothing
     * @param keepsConditions whether a half keeps the conditions Forge decided
     */
    private HeldHalves<HalfRecord> halvesOf(
            OpenResolution resolved, List<ModeSplit.ModeHalf> split, boolean keepsEvents,
            boolean keepsConditions) {
        HeldHalves<HalfRecord> halves = new HeldHalves<>();
        for (int index = 0; index < split.size(); index++) {
            ModeSplit.ModeHalf half = split.get(index);
            // The random results and the targets are the half's own, and the
            // library cards its events name join the board they were
            // captured with.
            List<EffectEvent> events = keepsEvents ? half.events() : List.of();
            ChainTargets.Half position = new ChainTargets.Half(index, split.size(), half.mode().option());
            String refs = refsOf(resolved, position, events);
            halves.add(HalfRecord.of(
                    context.nextRecordId(), resolved.keys.withOption(half.mode().option()),
                    half.mode().state().withRefs(refs).withEventCards(events).render(),
                    keepsConditions ? half.conditions() : List.of()),
                    events);
        }
        return halves;
    }

    /** Write a waiting resolution's halves, each with the events it holds. */
    private void emit(Waiting waiting) {
        Source source = waiting.source;
        for (HeldHalves.Half<HalfRecord> half : waiting.halves.halves()) {
            HalfRecord record = half.about();
            context.emit(context.newResolution(
                    Moment.RESOLUTION, record.recordId(), record.timestamp())
                    .actor(source.actor())
                    .ability(record.keys())
                    .linkId(source.linkId())
                    .state(record.state())
                    .payload(RecordPayloads.resolutionPayload(
                            half.events(), record.conditions(), waiting.outcome)));
        }
    }

    /**
     * One effect half's refs block: the half's random results, and the
     * targets of its own links (a mode's half: the charm's and its mode's)
     * as its resolution started and before Forge's check for illegal
     * targets.
     */
    private static String refsOf(
            OpenResolution resolved, ChainTargets.Half position, List<EffectEvent> events) {
        List<String> targets = resolved.targetsOfHalf(position);
        List<String> atCast = resolved.targetsAtCastOfHalf(position);
        return RefsJson.renderHalf(resolved.ability, events, resolved.x(), targets, atCast,
                resolved.onStackOf(targets, atCast));
    }

    /**
     * Take the waiting spell's own move off the stack into its last half, the
     * last event of the resolution's own (FR-042b): to the graveyard, to
     * exile, back to a library. Forge moves the card after publishing the
     * resolution, so the bracket has closed by then. A permanent spell's move
     * onto the battlefield is made while it resolves, and is filed with the
     * rest of its events. A fizzled spell's halves are written with the move.
     *
     * @return whether the move was taken
     */
    boolean takeMoveOffStack(GameEventCardChangeZone event, EffectEvent moved) {
        Move move = new Move(event);
        if (fizzled != null && fizzled.awaits(move)) {
            Waiting writing = fizzled;
            fizzled = null;
            writing.takeMove(moved);
            emit(writing);
            return true;
        }
        if (held == null || !held.awaits(move)) {
            return false;
        }
        held.takeMove(moved);
        return true;
    }

    /** Whether a resolution's halves are waiting for its consequences. */
    boolean isHolding() {
        return held != null;
    }

    /**
     * File a permanent's leaving, which a state-based check has just caused,
     * into the held resolution that left it behind.
     *
     * <p>Which permanent it is does not matter. Forge checks state-based
     * actions before every priority, so the check that runs while the half
     * is held is the first since the resolution began, and everything it
     * finds is something the resolution left behind — including what a
     * static it brought does: a 2/2 under the Dead Weight that just resolved,
     * the creatures an entering Elesh Norn shrinks to nothing, the loser of
     * the legend rule, the Aura that falls off a creature the resolution
     * killed. Only the move is filed here: the window takes no stat change,
     * and the static's own, published while the resolution resolves, is the
     * resolution's as any change after its snapshot is. That includes the
     * static check after the last clause (an Equip's +2/+0): the patched
     * branch runs it before publishing the resolution, so its changes reach
     * the bracket while it is still open, never this window.
     * The window closes at the loop's end (or, without the hook, at the next
     * action), so a check after a land or a new cast is not this
     * resolution's.
     *
     * <p>A modal resolution files the move into the half whose events named
     * the permanent, the last of them when several did — the mode that was
     * the last to change it — and into the last half when none did ({@link
     * HeldHalves#fileDeath}). The cost half and the link are the resolution's
     * and do not move.
     *
     * <p><b>Stamped against the resolved ability, with no clause.</b> The
     * death is a state-based action rather than anything a clause did, so
     * {@code attributed_to} is {@link EventAttribution#UNRESOLVED}, which is
     * what a combat record's state-based deaths carry too, while {@code
     * cause} is the resolved line's host card: the death is filed because
     * that line caused it, and it is what an interventional fork writes for
     * the same death, since its fork-bound recording is this same code
     * ({@link ForcedResolution}). One death, one rendering, observed or
     * forced.
     *
     * <p>Only while {@link #isHolding}.
     */
    void fileDeath(EffectEvent moved, CardView card) {
        fileConsequence(moved, Refs.of(card));
    }

    /**
     * File another event of the state-based check into the held resolution,
     * as {@link #fileDeath} files a death: an Aura or Equipment taken off a
     * permanent, filed with the half that names {@code subject}.
     *
     * <p>Only while {@link #isHolding}.
     */
    void fileConsequence(EffectEvent event, String subject) {
        held.halves.fileDeath(EventAttribution.stampAt(event, held.source.root(), null, null), subject);
    }

    /**
     * The state-based loop has ended, or a new action has begun, so the held
     * resolution's state-based consequences are over: write its effect
     * halves, if one is held.
     */
    void close() {
        if (fizzled != null) {
            // The card never left the stack where this collector saw it.
            Waiting writing = fizzled;
            fizzled = null;
            emit(writing);
        }
        if (held == null) {
            return;
        }
        Waiting writing = held;
        held = null;
        reportIfFruitless(writing);
        emit(writing);
    }

    /**
     * Reported once its state-based consequences are in: a free activation
     * that changed nothing the seat will not repeat this turn (FR-040).
     */
    private void reportIfFruitless(Waiting writing) {
        if (writing.free && writing.halves.halves().stream()
                .allMatch(half -> FruitlessActivations.changesNothing(half.events()))) {
            fruitless.report(writing.source.root(), writing.source.key());
        }
    }
}
