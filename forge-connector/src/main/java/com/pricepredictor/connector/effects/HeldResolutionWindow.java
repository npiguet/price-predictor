package com.pricepredictor.connector.effects;

import forge.game.card.Card;
import forge.game.card.CardView;
import forge.game.event.GameEventCardChangeZone;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.List;

/**
 * The live collector's resolution whose effect halves wait for its
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

    /** The resolution waiting for its consequences, or null when none is. */
    private HeldResolution held;

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
     * @param root the resolved ability, which the deaths are stamped against
     * @param key  its line, for {@link FruitlessActivations}
     * @param free whether it is an activation whose cost spent no mana and
     *             moved no card
     */
    private static final class HeldResolution {
        final SpellAbility root;
        final ProvenanceKey key;
        final boolean free;
        final String actor;
        final String linkId;
        final ActivationOutcome outcome;
        final HeldHalves<HalfRecord> halves;
        /**
         * The id of the spell card whose move off the stack the last half
         * takes, or -1 when none is awaited: an ability, or a move taken.
         */
        int awaitedCard;

        HeldResolution(
                SpellAbility root, ProvenanceKey key, boolean free, String actor, String linkId,
                ActivationOutcome outcome, HeldHalves<HalfRecord> halves) {
            this.root = root;
            this.key = key;
            this.free = free;
            this.actor = actor;
            this.linkId = linkId;
            this.outcome = outcome;
            this.halves = halves;
            Card host = root.getHostCard();
            this.awaitedCard = root.isSpell() && host != null ? host.getId() : -1;
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
     * @param free    whether it is an activation whose cost spent no mana and
     *                moved no card, which is reported when it changed nothing
     */
    void hold(OpenResolution resolved, ActivationOutcome outcome, boolean free) {
        // Modes, X and the named card are set while the ability resolves, so
        // the block that carries them is re-read here and paired with the
        // board each half was captured with: the resolution's start for a
        // root half, the mode's first clause's for a mode's.

        // A modal resolution on a patched worker writes one effect half per
        // chosen mode, each through the mode's own option line, every one of
        // them under the cost half's link (FR-029a, FR-029d). A mode that
        // fizzled on its own carries an empty list under its key, which is
        // what the schema says of any resolution that did nothing observable.
        HeldHalves<HalfRecord> halves = new HeldHalves<>();
        for (ModeSplit.ModeHalf half
                : resolved.modes.halves(resolved.effectState())) {
            // The random results are the half's own, and the library cards
            // its events name join the board they were captured with.
            String refs = RefsJson.renderResolved(resolved.ability, half.events(), resolved.x());
            halves.add(HalfRecord.of(
                    context.nextRecordId(), resolved.keys.withOption(half.mode().option()),
                    half.mode().state().withRefs(refs).withEventCards(half.events()).render(),
                    half.conditions()),
                    half.events());
        }
        held = new HeldResolution(
                resolved.ability, resolved.keys.key(), free, resolved.actor, resolved.linkId,
                outcome, halves);
    }

    /**
     * Write the effect half of an item that did nothing (FR-042b): one that
     * fizzled, or whose controller declined what it offered. Its one root
     * half has no events, and nothing joins it: a fizzled spell's move off
     * the stack and anything after go to no record. A declined half keeps
     * the conditions Forge decided on its way; a fizzled one decided none.
     *
     * @param outcome {@link ActivationOutcome#FIZZLED} or {@link ActivationOutcome#DECLINED}
     */
    void writeEmpty(OpenResolution resolved, ActivationOutcome outcome) {
        List<ConditionsMet> conditions = List.of();
        if (outcome == ActivationOutcome.DECLINED) {
            conditions = new java.util.ArrayList<>();
            for (ModeSplit.ModeHalf half : resolved.modes.halves(resolved.effectState())) {
                conditions.addAll(half.conditions());
            }
        }
        String refs = RefsJson.renderResolved(resolved.ability, List.of(), resolved.x());
        context.emit(context.newResolution(Moment.RESOLUTION)
                .actor(resolved.actor)
                .ability(resolved.keys)
                .linkId(resolved.linkId)
                .state(resolved.effectState().withRefs(refs).render())
                .payload(RecordPayloads.resolutionPayload(List.of(), conditions, outcome)));
    }

    /**
     * Take the held spell's own move off the stack into its last half, the
     * last event of the resolution's own (FR-042b): to the graveyard, to
     * exile, back to a library. Forge moves the card after publishing the
     * resolution, so the bracket has closed by then. A permanent spell's move
     * onto the battlefield is made while it resolves, and is filed with the
     * rest of its events.
     *
     * @return whether the move was taken
     */
    boolean takeMoveOffStack(GameEventCardChangeZone event, EffectEvent moved) {
        if (held == null || held.awaitedCard < 0 || event.card() == null
                || event.card().getId() != held.awaitedCard
                || event.from() == null || event.from().zoneType() != ZoneType.Stack) {
            return false;
        }
        held.awaitedCard = -1;
        held.halves.fileLast(EventAttribution.stampAt(moved, held.root, null, null));
        return true;
    }

    /** Whether a resolution's halves are waiting for its consequences. */
    boolean isHolding() {
        return held != null;
    }

    /**
     * The id of the held resolution's first effect half, or null when none
     * is held: the live record a fork taken now was forked beside (D5).
     */
    String heldRecordId() {
        if (held == null || held.halves.halves().isEmpty()) {
            return null;
        }
        return held.halves.halves().get(0).about().recordId();
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
     * resolution's as any change after its snapshot is.
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
     * the same death, since its sink is still listening when
     * {@code GameSimulator.resolveStack} runs the check against the forced
     * ability. One death, one rendering, observed or forced.
     *
     * <p>Only while {@link #isHolding}.
     */
    void fileDeath(EffectEvent moved, CardView card) {
        held.halves.fileDeath(
                EventAttribution.stampAt(moved, held.root, null, null), Refs.of(card));
    }

    /**
     * The state-based loop has ended, or a new action has begun, so the held
     * resolution's state-based consequences are over: write its effect
     * halves, if one is held.
     */
    void close() {
        if (held == null) {
            return;
        }
        HeldResolution writing = held;
        held = null;
        // Reported once its state-based consequences are in: a free activation
        // that changed nothing the seat will not repeat this turn (FR-040).
        if (writing.free && writing.halves.halves().stream()
                .allMatch(half -> FruitlessActivations.changesNothing(half.events()))) {
            fruitless.report(writing.root.getHostCard(), writing.key);
        }
        for (HeldHalves.Half<HalfRecord> half : writing.halves.halves()) {
            HalfRecord record = half.about();
            context.emit(context.newResolution(
                    Moment.RESOLUTION, record.recordId(), record.timestamp())
                    .actor(writing.actor)
                    .ability(record.keys())
                    .linkId(writing.linkId)
                    .state(record.state())
                    .payload(RecordPayloads.resolutionPayload(
                            half.events(), record.conditions(), writing.outcome)));
        }
    }
}
