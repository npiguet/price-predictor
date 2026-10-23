package com.pricepredictor.connector.effects;

import forge.game.card.CardView;
import forge.game.spellability.SpellAbility;

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
 * <p>So the halves are held from the resolution until the next real action
 * begins ({@link #close}) — a bracket opening for a cast or a trigger,
 * another resolution ending, a combat bracket opening, a step or phase
 * boundary, a land played, the end of the game — and written then, exactly
 * once. One resolution at most is held: the next one's end is itself such an
 * action.
 */
final class HeldResolutionWindow {

    private final CollectorContext context;
    private final SnapshotBuilder snapshots;

    /** The resolution waiting for its consequences, or null when none is. */
    private HeldResolution held;

    HeldResolutionWindow(CollectorContext context, SnapshotBuilder snapshots) {
        this.context = context;
        this.snapshots = snapshots;
    }

    /**
     * What one held effect half's record says besides its events, all of it
     * fixed at the resolution — the id, the timestamp, the key and the board,
     * which is the one taken before the resolution. The events can still
     * grow, by the deaths the resolution's own effects cause.
     */
    private record HalfRecord(
            String recordId, String timestamp, ProvenanceKey.Resolved keys, String state) {

        static HalfRecord of(String recordId, ProvenanceKey.Resolved keys, String state) {
            return new HalfRecord(recordId, RecordShardWriter.timestamp(), keys, state);
        }
    }

    /** @param root the resolved ability, which the deaths are stamped against */
    private record HeldResolution(
            SpellAbility root, String actor, String linkId, HeldHalves<HalfRecord> halves) {
    }

    /**
     * Hold a resolution's effect halves, under the cost half's link.
     *
     * <p>Held rather than written: the deaths this resolution causes have not
     * happened yet.
     */
    void hold(OpenResolution resolved, String link) {
        // Modes, X and the named card are set while the ability resolves, so
        // the block that carries them is re-read here and paired with the
        // board each half was captured with: the cast's for a root half, the
        // mode's first clause's for a mode's.
        String refs = snapshots.refsJson(resolved.ability);
        // A modal resolution on a patched worker writes one effect half per
        // chosen mode, each through the mode's own option line, every one of
        // them under the cost half's link (FR-029a, FR-029d). A mode that
        // fizzled on its own carries an empty list under its key, which is
        // what the schema says of any resolution that did nothing observable.
        HeldHalves<HalfRecord> halves = new HeldHalves<>();
        for (ModeSplit.ModeHalf half
                : resolved.modes.halves(resolved.state, resolved.events.events())) {
            halves.add(HalfRecord.of(
                    context.nextRecordId(), resolved.keys.withOption(half.mode().option()),
                    half.mode().state().withRefs(refs).render()), half.events());
        }
        held = new HeldResolution(resolved.ability, resolved.actor, link, halves);
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
     * killed. The static's own stat change stays out; it is the static's
     * continuous record, and only the move it causes is the resolution's.
     * The window closes at the next action, so a check after a land or a new
     * cast is not this resolution's.
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
        held.halves().fileDeath(
                EventAttribution.stampAt(moved, held.root(), null, null), Refs.of(card));
    }

    /**
     * A new action has begun, so the held resolution's state-based
     * consequences are over: write its effect halves, if one is held.
     */
    void close() {
        if (held == null) {
            return;
        }
        HeldResolution writing = held;
        held = null;
        for (HeldHalves.Half<HalfRecord> half : writing.halves().halves()) {
            HalfRecord record = half.about();
            context.emit(context.newResolution(
                    Moment.RESOLUTION, record.recordId(), record.timestamp())
                    .actor(writing.actor())
                    .ability(record.keys())
                    .linkId(writing.linkId())
                    .state(record.state())
                    .payload(EffectRecord.eventsPayload(half.events())));
        }
    }
}
