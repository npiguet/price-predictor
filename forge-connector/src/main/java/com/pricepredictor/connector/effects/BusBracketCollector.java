package com.pricepredictor.connector.effects;

import com.google.common.eventbus.Subscribe;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.event.GameEventCardChangeZone;
import forge.game.event.GameEventCombatEnded;
import forge.game.event.GameEventGameFinished;
import forge.game.event.GameEventGameOutcome;
import forge.game.event.GameEventLandPlayed;
import forge.game.event.GameEventSpellAbilityCast;
import forge.game.event.GameEventSpellResolved;
import forge.game.event.GameEventTurnPhase;
import forge.game.spellability.SpellAbility;

import java.lang.reflect.InvocationHandler;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.StringJoiner;

/**
 * Stage-one collection: the public event bus plus a bracket around resolution.
 *
 * <p>No patched Forge. Events arriving on the bus are attributed to whatever the
 * stack said was resolving when the bracket opened — correct for the common case
 * of one thing resolving at a time, and approximate where two resolve together.
 * Every record it writes is stamped {@code degraded} so a corpus never claims
 * more precision than it has.
 *
 * <p>The bracket is delimited by the bus itself: a cast opens one, a resolution
 * closes it. The resolving ability is read from {@code game.getStack()} rather
 * than from the event, because the event carries a view and the provenance key
 * needs the model object.
 *
 * <p>The cost half is <b>held</b> from the cast until the bracket closes, so it
 * can be stamped with what actually became of the spell. Every other record this
 * class writes goes out at the moment it describes; this one cannot, because
 * {@code outcome} is a field of it and a spell that is countered before it
 * resolves never publishes anything else the collector could attach the answer
 * to. A cast still on the stack when the game ends is dropped rather than given
 * one of the five outcomes it did not have — but it is now counted and named
 * rather than vanishing; see {@link #finishGame}.
 *
 * <p>Damage steps get their own brackets, closed at the phase boundary, so a
 * first-strike combat writes two records — the only way a keyword that acts by
 * splitting the step is visible in the corpus at all.
 */
public final class BusBracketCollector extends BusEventReader {

    private final CollectorContext context;
    private final SnapshotBuilder snapshots;

    /** The ability the open bracket attributes to, or null between brackets. */
    private SpellAbility resolving;
    private ProvenanceKey.Resolved resolvingKeys = ProvenanceKey.Resolved.UNKNOWN;
    private String resolvingActor;
    private Snapshot openBracketState;
    private final EventBuffer bracketEvents = new EventBuffer();

    /**
     * One chosen mode of the open modal resolution, once its first clause has
     * resolved (FR-029a).
     *
     * <p>The snapshot is taken at that clause, so a later mode's board shows
     * the earlier modes' effects (FR-029b); the events are what landed in the
     * bracket between this mode's first clause and the next mode's, or the
     * end of the resolution. {@code option} is the mode's {@code Choices$}
     * position, or null for a mode nothing could place, whose half keeps the
     * root key.
     */
    private static final class ModeHalf {
        private final Integer option;
        private final Snapshot state;
        private final List<EffectEvent> events = new ArrayList<>();

        private ModeHalf(Integer option, Snapshot state) {
            this.option = option;
            this.state = state;
        }
    }

    /** The modes of the open resolution whose halves are complete, in order. */
    private final List<ModeHalf> modeHalves = new ArrayList<>();
    /** The mode whose clauses are resolving now, or null outside one. */
    private ModeHalf openMode;

    /**
     * One effect half that has finished resolving and is not written yet.
     *
     * <p>Everything but the events is fixed at the resolution — the id, the
     * timestamp, the key and the board, which is the one taken before the
     * resolution as it always was. The events can still grow, by the deaths
     * the resolution's own effects cause; see {@link HeldResolution}.
     *
     * @param touched the entity refs this half's own events name as subjects,
     *                which choose the half a death joins on a modal resolution
     */
    private record HeldHalf(
            String recordId,
            String timestamp,
            ProvenanceKey.Resolved keys,
            String state,
            EventBuffer events,
            Set<String> touched) {

        static HeldHalf of(
                String recordId, ProvenanceKey.Resolved keys, String state,
                List<EffectEvent> events) {
            return new HeldHalf(
                    recordId, RecordShardWriter.timestamp(), keys, state,
                    EventBuffer.copyOf(events), touchedBy(events));
        }
    }

    /**
     * A resolution whose effect halves wait for its state-based consequences.
     *
     * <p>Forge does not check state-based actions when an ability finishes:
     * {@code MagicStack.resolveStack} publishes {@code GameEventSpellResolved}
     * and returns, and the creature a Lightning Bolt left with lethal damage
     * dies at the next priority, inside {@code GameAction.checkStateEffects} —
     * after {@code GameEventPlayerPriority}, so not even that event is a
     * boundary. Written at the resolved event, as it used to be, the effect
     * half held the damage and never the death: across twelve real shards,
     * none of 515 lethal-looking resolution damage events had its death in
     * the same record, while combat records, whose bracket stays open until
     * the step ends, held it 61% of the time.
     *
     * <p>So the halves are held from the resolution until the next real
     * action begins — a bracket opening for a cast or a trigger, another
     * resolution ending, a combat bracket opening, a step or phase boundary,
     * a land played, the end of the game — and written then, exactly once.
     *
     * @param root the resolved ability, which the deaths are stamped against
     */
    private record HeldResolution(
            SpellAbility root, String actor, String linkId, List<HeldHalf> halves) {
    }

    /** The resolution waiting for its consequences, or null when none is. */
    private HeldResolution held;

    /**
     * Casts whose outcome is not known yet, oldest first.
     *
     * <p>A list rather than a map because it is the stack: two spells can be
     * waiting at once and the inner one resolves first, so a lookup runs from
     * the newest end. Nothing here has been written yet — see
     * {@link PendingActivation} for why the record has to wait.
     */
    private final List<PendingActivation> pendingActivations = new ArrayList<>();

    /** Combat damage accumulates into its own bracket, closed at the phase end. */
    private final EventBuffer combatEvents = new EventBuffer();
    /** The step's combat, read when its bracket opens rather than at flush. */
    private CombatShape combatShape = CombatShape.empty();
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
    private boolean combatWindowClosed;
    private String combatState;

    /**
     * Whether the controller turned something down while this bracket was open.
     *
     * <p>Scoped to the bracket rather than matched against the declined
     * ability's id, because Forge resolves a copy of the ability it was handed
     * and an id comparison would silently never match — the same reason
     * {@link EventAttribution} counts a clause upward from the pointer instead
     * of downward from the root. What keeps the flag from over-claiming is the
     * second half of the test in {@link #endBracket}: a resolution that
     * produced any outcome at all is {@code resolved} however many optional
     * clauses inside it were declined.
     */
    private boolean bracketDeclined;

    /** Told each combat record's id, for a probe that mirrors it. */
    private java.util.function.Consumer<String> onCombatRecord;

    public BusBracketCollector(
            Game game, RecordShardWriter writer, String gameId) {
        super(game);
        this.context = new CollectorContext(writer, gameId);
        this.snapshots = new SnapshotBuilder(game);
        // Installed here and dropped at GameEventGameFinished, because this
        // collector has no other lifecycle: it is subscribed to the bus for
        // exactly one game and the bus tells it when that game is over. On an
        // unpatched checkout install() finds no hook and nothing changes.
        PatchHooks.CONFIRM_LISTENER.install(confirmHandler());
    }

    /**
     * Be told the id of each {@code combat} record as it is written.
     *
     * <p>A damage-step probe forks before the step and cannot know what it
     * mirrors until the real record exists, so the pairing is handed over here
     * rather than reconstructed. Reconstructing it from board state would not
     * work: two combats in one turn can look identical, and a wrong pairing
     * makes the difference between them meaningless.
     */
    public void onCombatRecord(java.util.function.Consumer<String> listener) {
        this.onCombatRecord = listener;
    }

    /** Name the match's random seat, so every record can say whether it acted. */
    public void setRandomSeatPlayerId(String playerId) {
        context.setRandomSeatPlayerId(playerId);
    }

    // ── brackets ────────────────────────────────────────────────────────

    /**
     * A cast opens a bracket and holds the cost half until its outcome is known.
     *
     * <p>The costs are read <b>now</b> — {@link EffectRecord#costsJson} says why
     * they are unreadable a moment later — but the record cannot be written yet,
     * because {@code outcome} is a field of the cost half and the answer to it
     * does not exist until the spell resolves or leaves the stack without
     * resolving. Writing it here is why every one of the corpus's 4,989
     * activation records said the literal {@code resolved}, and why 6.1% of
     * link halves had no partner: a fizzled spell issued a {@code link_id} whose
     * effect half never came.
     */
    @Subscribe
    public void onCast(GameEventSpellAbilityCast event) {
        // Read through peek() rather than peekAbility(), which dereferences the
        // top of an empty stack. The bus is not a promise about the stack.
        var top = game.getStack().peek();
        SpellAbility ability = top == null ? null : top.getSpellAbility();
        // A spell taken off the stack without resolving publishes nothing this
        // collector can tell from a resolution, so the absence is noticed
        // whenever the stack is next observed. A new cast is one such moment.
        reconcilePending();
        beginBracket(ability);
    }

    /**
     * Everything a cast does once the ability has been found on the stack.
     *
     * <p>Split from {@link #onCast} at the one line that reads the stack, so the
     * rules that decide what a cost record eventually says can be exercised
     * against a real ability without a game in progress — which is exactly what
     * they were missing while the outcome was a literal.
     */
    void beginBracket(SpellAbility ability) {
        // A cast, an activation or a trigger reaching the stack is the next
        // action: whatever the held resolution killed has died by now.
        flushHeldResolution();
        // And whatever the combat damage killed: state-based actions run
        // before anything goes on the stack.
        if (inCombatDamageStep()) {
            flushCombat();
            combatWindowClosed = true;
        }
        openBracket(ability);
        if (ability == null) {
            return;
        }
        pendingActivations.add(new PendingActivation(
                ability.getId(), context.nextRecordId(),
                RecordShardWriter.timestamp(), EffectRecord.costsJson(ability),
                actorOf(ability), ProvenanceKey.resolve(ability), openBracketState.render(),
                context.gameId() + ".link." + ability.getId(), targetCount(ability)));
    }

    /**
     * A resolution closes the bracket, stamps the cost half and holds the
     * effect half until the deaths it causes have happened ({@link
     * HeldResolution}).
     *
     * <p>A fizzle writes no effect half at all. {@code MagicStack.resolveStack}
     * skips resolution entirely when the targets are gone, so the alternative is
     * a record claiming an empty event list for something that never ran — and
     * a partnerless {@code link_id} on the cost half, which the schema bars.
     *
     * <p>A decline writes no effect half either, for the same reason and by the
     * same rule: {@code declined} is one of the three outcomes the schema calls
     * partnerless, and the Python loader rejects a {@code declined} activation
     * that reaches resolution.
     */
    @Subscribe
    public void onResolved(GameEventSpellResolved event) {
        endBracket(
                event.spell() == null ? -1 : event.spell().getId(),
                event.hasFizzled());
        // The counterspell that removed something has just resolved, so this is
        // the first moment its victim's absence from the stack is visible.
        reconcilePending();
    }

    /**
     * The half of {@link #onResolved} that does not read the stack.
     *
     * @param abilityId the ability the engine says has finished
     * @param fizzled   whether it was removed for having no legal target left,
     *                  in which case it never ran at all
     */
    void endBracket(int abilityId, boolean fizzled) {
        // Another resolution has run since the held one ended, so the held
        // one's state-based check is over: what dies after this is this
        // resolution's doing, or a mix nobody can untangle.
        flushHeldResolution();
        // The bracket is a single slot, so a spell cast in response to another
        // takes it over: when the outer one finally resolves the events that
        // belong to it were never gathered. The cost half is still stamped with
        // its real outcome; what it must not do is issue a link to an effect
        // half nobody is going to write.
        boolean bracketIsThisAbility = resolving != null && resolving.getId() == abilityId;
        boolean declined = bracketIsThisAbility && !fizzled && wasDeclined();
        boolean writesEffectHalf = bracketIsThisAbility && !fizzled && !declined;

        String link = settleActivation(
                abilityId,
                settledOutcome(abilityId, fizzled, declined, bracketIsThisAbility),
                writesEffectHalf);

        if (bracketIsThisAbility) {
            // The last mode's half closes with the resolution, since no next
            // mode's first clause comes to close it.
            closeOpenMode();
        }
        if (writesEffectHalf) {
            holdEffectHalves(link);
        }

        if (bracketIsThisAbility) {
            closeBracket();
        } else if (resolving == null) {
            bracketEvents.clear();
        }
    }

    /** What became of a cast that has left the stack through a resolution. */
    private ActivationOutcome settledOutcome(
            int abilityId, boolean fizzled, boolean declined, boolean bracketIsThisAbility) {
        if (fizzled) {
            return ActivationOutcome.FIZZLED;
        }
        if (declined) {
            return ActivationOutcome.DECLINED;
        }
        return outcomeOf(
                findPending(abilityId).map(PendingActivation::targetsAtCast).orElse(0),
                bracketIsThisAbility ? resolving : null);
    }

    /**
     * Hold the open resolution's effect halves, under the cost half's link.
     *
     * <p>Held rather than written: the deaths this resolution causes have not
     * happened yet. See {@link HeldResolution}.
     */
    private void holdEffectHalves(String link) {
        if (modeHalves.isEmpty()) {
            held = new HeldResolution(resolving, resolvingActor, link, List.of(HeldHalf.of(
                    context.nextRecordId(), resolvingKeys,
                    // Modes, X and the named card are set while the ability
                    // resolves, so the block that carries them is re-read here
                    // and paired with the board captured at the cast.
                    openBracketState.withRefs(snapshots.refsJson(resolving)).render(),
                    bracketEvents.events())));
            return;
        }
        // A modal resolution on a patched worker: one effect half per chosen
        // mode, each through the mode's own option line, every one of them
        // under the cost half's link (FR-029a, FR-029d). The events a mode
        // produced are its own and nothing else's; a mode that fizzled on its
        // own carries an empty list under its key, which is what the schema
        // says of any resolution that did nothing observable.
        String refs = snapshots.refsJson(resolving);
        List<HeldHalf> halves = new ArrayList<>();
        for (ModeHalf half : modeHalves) {
            halves.add(HeldHalf.of(
                    context.nextRecordId(), resolvingKeys.withOption(half.option),
                    half.state.withRefs(refs).render(), half.events));
        }
        held = new HeldResolution(resolving, resolvingActor, link, List.copyOf(halves));
    }

    /**
     * A phase change closes the combat bracket, if one is open.
     *
     * <p>One record per damage step: first strike and regular damage are
     * separate phases, so the phase boundary is exactly the step boundary.
     */
    @Subscribe
    public void onPhase(GameEventTurnPhase event) {
        flushHeldResolution();
        flushCombat();
        combatWindowClosed = false;
        // An effect that empties the stack -- ending the turn, ending combat --
        // resolves nothing afterwards, so a phase boundary is the last chance
        // to notice what it swept away.
        reconcilePending();
    }

    @Subscribe
    public void onCombatEnded(GameEventCombatEnded event) {
        flushHeldResolution();
        flushCombat();
    }

    /**
     * A land played is an action of its own, with no stack and so no bracket:
     * the held resolution's consequences are over by the time a player acts.
     *
     * <p>Needed even though the land's own arrival would never join the held
     * record — it arrives rather than leaves — because a land is followed by
     * another state-based check, and what that one kills is the land's doing
     * (a static it brought), not the held resolution's.
     */
    @Subscribe
    public void onLandPlayed(GameEventLandPlayed event) {
        flushHeldResolution();
    }

    // ── outcome events ──────────────────────────────────────────────────

    @Override
    boolean cardDamageIsCombat() {
        return inCombatDamage();
    }

    /**
     * Combat damage joins the step's bracket; any other damage the open one.
     *
     * <p>Routed by the caller's flag rather than by {@link #inCombatDamage},
     * because a player damage carries Forge's own combat flag and a card
     * damage has only the step to go by.
     */
    @Override
    void fileDamage(EffectEvent damage, String namedCause, boolean combat) {
        EventAttribution.stamp(damage, resolving, namedCause);
        if (combat) {
            openCombatBracket();
            combatEvents.file(damage);
        } else {
            bracketEvents.file(damage);
        }
    }

    @Override
    void fileMove(GameEventCardChangeZone event, EffectEvent moved) {
        if (!holdAsConsequence(event, moved)) {
            record(moved);
        }
    }

    /**
     * File a permanent's leaving into the held resolution that killed it.
     *
     * <p>Two conditions, each closing a way an unrelated move could join:
     * <ul>
     *   <li>it leaves the <b>battlefield</b> — a death, an exile instead of a
     *   death, a token ceasing to exist, an Aura falling off; a card arriving
     *   anywhere is not a consequence of this kind; and</li>
     *   <li>a state-based check is running, read as the game's view tracker
     *   being frozen, which {@code GameAction.checkStateEffects} holds for
     *   its whole loop. Between the check and the next cast event the player
     *   acts, and the costs of that action are paid before the cast is
     *   announced: a Treasure the resolution made, sacrificed for mana to
     *   cast the next spell, leaves the battlefield in this same window and
     *   is that spell's cost, not this resolution's consequence. Outside a
     *   resolution nothing else freezes the tracker while a card moves.</li>
     * </ul>
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
     * The window closes at the next action ({@link HeldResolution}), so a
     * check after a land or a new cast is not this resolution's.
     *
     * <p>A modal resolution files the move into the half whose events named
     * the permanent, the last of them when several did — the mode that was
     * the last to change it — and into the last half when none did. The cost
     * half and the link are the resolution's and do not move.
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
     * @return whether the move was filed into the held resolution, in which
     *         case it goes nowhere else
     */
    private boolean holdAsConsequence(GameEventCardChangeZone event, EffectEvent moved) {
        if (held == null || !isStateBasedLeave(event)) {
            return false;
        }
        List<Set<String>> touched = new ArrayList<>();
        for (HeldHalf half : held.halves()) {
            touched.add(half.touched());
        }
        HeldHalf into = held.halves().get(
                halfForDeath(touched, Refs.of(event.card())));
        into.events().file(EventAttribution.stamp(moved, held.root(), null, null));
        return true;
    }

    /**
     * The entity refs a half's own events name as subjects: what it damaged,
     * pumped, shrank, countered, animated or moved.
     */
    static Set<String> touchedBy(List<EffectEvent> events) {
        Set<String> touched = new LinkedHashSet<>();
        for (EffectEvent event : events) {
            for (String subject : event.subjects()) {
                if (Refs.isEntity(subject)) {
                    touched.add(subject);
                }
            }
        }
        return touched;
    }

    /**
     * Which of a resolution's halves a state-based death belongs to: the last
     * whose events named the permanent, or the last half when none did.
     *
     * <p>One rule for the live collector and for a modal fork, so an observed
     * and a forced resolution of the same charm file the same death the same
     * way.
     *
     * @param touchedPerHalf each half's {@link #touchedBy} set, in order;
     *                       never empty
     */
    static int halfForDeath(List<Set<String>> touchedPerHalf, String subject) {
        for (int i = touchedPerHalf.size() - 1; i >= 0; i--) {
            if (touchedPerHalf.get(i).contains(subject)) {
                return i;
            }
        }
        return touchedPerHalf.size() - 1;
    }

    /**
     * Write the held resolution's effect halves, if one is held.
     *
     * <p>Package-private so a test that ends a resolution by hand can close
     * the window the way the next action in a real game would.
     */
    void flushHeldResolution() {
        if (held == null) {
            return;
        }
        HeldResolution writing = held;
        held = null;
        for (HeldHalf half : writing.halves()) {
            context.emit(context.newResolution(
                    Moment.RESOLUTION, half.recordId(), half.timestamp())
                    .actor(writing.actor())
                    .ability(half.keys())
                    .linkId(writing.linkId())
                    .state(half.state())
                    .payload(EffectRecord.eventsPayload(half.events().events())));
        }
    }

    // ── plumbing ────────────────────────────────────────────────────────

    /**
     * Route an event into whichever bracket is open.
     *
     * <p>State-based-action deaths arrive with no ability resolving, after the
     * bracket that caused them has closed. The ones it caused never reach
     * here: {@link #holdAsConsequence} files them into its held effect half.
     * The rest land in the emptied bracket like any other event that arrives
     * with nothing resolving.
     */
    private void record(EffectEvent event) {
        fileOutcome(event, null);
    }

    /**
     * The same, for an event the bus itself named a causer for.
     *
     * @param namedCause the ref the engine named, which outranks the bracket's,
     *                   or null to take the bracket's
     */
    @Override
    void fileOutcome(EffectEvent event, String namedCause) {
        // Stamped here rather than where the event is built: which clause
        // produced it, how long it lasts and — absent a name off the event
        // itself — what caused it are properties of the bracket it landed in,
        // and only this side knows that.
        EventAttribution.stamp(event, resolving, namedCause);
        if (inCombatDamage()) {
            openCombatBracket();
            combatEvents.file(event);
        } else {
            bracketEvents.file(event);
        }
    }

    /**
     * An event an effect API described rather than the bus announced.
     *
     * <p>Package-private because {@code PatchedCollectors} owns the clause hook
     * and this collector owns the open bracket: the event belongs to whatever
     * resolution is in flight, which only this class knows.
     */
    void recordClauseEvent(EffectEvent event) {
        if (event != null) {
            record(event);
        }
    }

    private void openBracket(SpellAbility ability) {
        this.resolving = ability;
        this.bracketDeclined = false;
        this.resolvingKeys = ProvenanceKey.resolve(ability);
        this.resolvingActor = actorOf(ability);
        this.openBracketState = snapshots.ofAbility(ability, SnapshotBuilder.referencedOf(ability));
        this.bracketEvents.clear();
    }

    private void closeBracket() {
        this.resolving = null;
        this.bracketDeclined = false;
        this.resolvingKeys = ProvenanceKey.Resolved.UNKNOWN;
        this.resolvingActor = null;
        this.openBracketState = null;
        this.bracketEvents.clear();
        this.modeHalves.clear();
        this.openMode = null;
    }

    // ── the modes of a modal resolution ─────────────────────────────────

    /**
     * A chosen mode's first clause is resolving: close the previous mode's
     * half and open this one's, with a snapshot taken now (FR-029a, FR-029b).
     *
     * <p>Called by the clause hook, which is to say only on a patched worker:
     * a degraded one never reaches here and writes the one root half it
     * always wrote, with every mode's events in it (FR-029e). The clause's
     * host is checked against the open bracket's because the hook is a JVM
     * static that sees every resolution in the process, and the bracket is
     * one slot: a mode of some other charm — a copy resolving on a fork, a
     * nested resolution this slot does not hold — must not cut this one.
     *
     * <p>Events filed before the first mode's clause, between the cast and
     * the resolution, fall into the first mode's half rather than vanishing:
     * {@link #bracketEvents} is not cleared when the first mode opens.
     *
     * @param option the mode's {@code Choices$} position, or null for a mode
     *               {@link CharmModes} could not place
     */
    void beginMode(SpellAbility clause, Integer option) {
        if (resolving == null || clause == null
                || clause.getHostCard() != resolving.getHostCard()) {
            return;
        }
        closeOpenMode();
        openMode = new ModeHalf(
                option, snapshots.ofAbility(resolving, SnapshotBuilder.referencedOf(resolving)));
    }

    /** Move the bracket's events into the open mode's half and file it. */
    private void closeOpenMode() {
        if (openMode == null) {
            return;
        }
        openMode.events.addAll(bracketEvents.drain());
        modeHalves.add(openMode);
        openMode = null;
    }

    private static String actorOf(SpellAbility ability) {
        return ability == null || ability.getActivatingPlayer() == null
                ? null
                : SnapshotBuilder.playerId(ability.getActivatingPlayer());
    }

    // ── the outcome of a cast ───────────────────────────────────────────

    /**
     * A cast that has been observed but whose outcome is not settled.
     *
     * <p>Holds the rendered strings rather than the {@link SpellAbility},
     * because what was paid lives on the ability object and its next activation
     * clears it — the reason {@link EffectRecord#costsJson} must be called at
     * the cast and nowhere later. The record id and the timestamp are taken at
     * the cast too, so a held record still says when the spell was cast rather
     * than when it stopped being on the stack.
     */
    private record PendingActivation(
            int abilityId,
            String recordId,
            String timestamp,
            String costsJson,
            String actor,
            ProvenanceKey.Resolved ability,
            String state,
            String linkId,
            int targetsAtCast) {
    }

    /** The held cast for an ability id, newest first. */
    private Optional<PendingActivation> findPending(int abilityId) {
        for (int i = pendingActivations.size() - 1; i >= 0; i--) {
            if (pendingActivations.get(i).abilityId() == abilityId) {
                return Optional.of(pendingActivations.get(i));
            }
        }
        return Optional.empty();
    }

    /**
     * Write a held cost half now that its outcome is known.
     *
     * <p>{@code linked} is separate from the outcome because they answer
     * different questions: the outcome says what became of the spell, and
     * {@code link_id} promises that an effect half carrying the same id exists.
     * A resolution this collector could not describe -- the outer half of a
     * nested pair -- is a {@code resolved} activation with no link, which is
     * honest; issuing the link anyway is what left 306 halves dangling.
     *
     * @return the link the effect half may claim, or null when there is to be
     *         no effect half or nothing was held for this ability
     */
    String settleActivation(int abilityId, ActivationOutcome outcome, boolean linked) {
        Optional<PendingActivation> found = findPending(abilityId);
        if (found.isEmpty()) {
            return null;
        }
        PendingActivation activation = found.get();
        pendingActivations.remove(activation);
        emitActivation(activation, outcome, linked);
        return linked ? activation.linkId() : null;
    }

    private void emitActivation(
            PendingActivation activation, ActivationOutcome outcome, boolean linked) {
        context.emit(context.newResolution(
                Moment.ACTIVATION, activation.recordId(), activation.timestamp())
                .actor(activation.actor())
                .ability(activation.ability())
                .linkId(linked ? activation.linkId() : null)
                .state(activation.state())
                .payload(EffectRecord.costPayload(activation.costsJson(), outcome)));
    }

    /**
     * Whether a resolution lost some of its targets on the way.
     *
     * <p>{@code MagicStack.hasFizzled} strips the targets that became illegal
     * from the ability before resolving it, and only then does the ability
     * resolve — so comparing the count with the one read at the cast is the
     * engine's own answer to "did part of this spell do nothing". Nothing else
     * in the record distinguishes a Lightning Helix that hit from one whose
     * creature had already died.
     *
     * <p>Only the root line's targets are compared. A partial fizzle confined to
     * a sub-ability reads as {@code resolved}, which understates rather than
     * invents. Losing <b>every</b> target without the engine calling it a fizzle
     * is the {@code CantFizzle} case (Gilded Drake), where the spell still does
     * what it does: that is {@code resolved} too.
     */
    static ActivationOutcome outcomeOf(int targetsAtCast, SpellAbility resolved) {
        if (resolved == null) {
            return ActivationOutcome.RESOLVED;
        }
        int surviving = targetCount(resolved);
        return surviving > 0 && surviving < targetsAtCast
                ? ActivationOutcome.PARTIALLY_FIZZLED
                : ActivationOutcome.RESOLVED;
    }

    private static int targetCount(SpellAbility ability) {
        return ability == null || ability.getTargets() == null
                ? 0 : ability.getTargets().size();
    }

    /**
     * Write off every held cast that has left the stack without resolving.
     *
     * <p>Counterspells, {@code CostExileFromStack}, a spell bounced off the
     * stack and {@code MagicStack.clear} all take an entry away with no event
     * this collector can tell apart from a resolution -- {@code finishResolving}
     * publishes the same removal for a spell that resolved normally. So the
     * question is asked of the stack itself, at every moment the stack can have
     * changed: a held cast that is no longer on it and did not resolve was
     * removed, and {@code countered} is what the schema calls that.
     *
     * <p>It is the nearest of the five outcomes rather than an exact one. A
     * spell exiled off the stack by a cost was not countered in the rules sense;
     * what the corpus needs to know, and what all of these share, is that the
     * cost was paid and no effect followed.
     */
    private void reconcilePending() {
        if (pendingActivations.isEmpty()) {
            return;
        }
        writeOffRemoved(idsOnStack());
    }

    /** The ability ids the stack holds right now. */
    private Set<Integer> idsOnStack() {
        Set<Integer> onStack = new LinkedHashSet<>();
        for (var instance : game.getStack()) {
            SpellAbility sa = instance == null ? null : instance.getSpellAbility();
            if (sa != null) {
                onStack.add(sa.getId());
            }
        }
        return onStack;
    }

    /**
     * The half of {@link #reconcilePending} that has already read the stack.
     *
     * @param onStack the ability ids still on the stack
     * @return how many held casts were written off
     */
    int writeOffRemoved(Set<Integer> onStack) {
        int removed = 0;
        for (Iterator<PendingActivation> held = pendingActivations.iterator();
                held.hasNext();) {
            PendingActivation activation = held.next();
            if (!onStack.contains(activation.abilityId())) {
                held.remove();
                emitActivation(activation, ActivationOutcome.COUNTERED, false);
                removed++;
            }
        }
        return removed;
    }

    // ── the end of the game ─────────────────────────────────────────────

    /**
     * The last thing that happens to a game, and the last chance to write.
     *
     * <p>Nothing subscribed this event, and three channels were being lost at it.
     *
     * <p>The <b>combat bracket</b> is closed at a phase boundary, and a game
     * that ends in combat damage never reaches one:
     * {@code PhaseHandler.mainLoopStep} returns the moment
     * {@code checkStateBasedEffects} says the game is over, so neither
     * {@code GameEventTurnPhase} nor {@code GameEventCombatEnded} follows the
     * lethal damage step. The record for the combat that decided the game was
     * therefore the one combat record no game had.
     *
     * <p>The <b>resolution bracket's own leftovers</b> are closed and
     * cleared by the killing ability's own {@link
     * #endBracket} the moment it resolves -- before the state-based check that
     * discovers the game is over ever runs (CR 704.3, next main-loop
     * iteration). A bus event filed after that point, with no bracket open
     * (the ordinary case for {@code player_won}: burn, drain, decking out,
     * anything that is not a combat-damage kill), had nowhere left to be
     * flushed to. See {@link #flushOrphanedEvents}.
     *
     * <p>The <b>held casts</b> split in two, and only one half is writable.
     * A spell that left the stack since the last reconcile point — countered,
     * exiled off the stack, swept away by the effect that ended the game — is
     * written off exactly as it would be at any other moment. What is genuinely
     * still on the stack is not: it neither resolved nor was removed, and
     * {@code outcome} has no member for that. Those are counted and named on
     * stderr rather than stamped with one of the five they did not have.
     */
    @Subscribe
    public void onGameFinished(GameEventGameFinished event) {
        finishGame(idsOnStack());
    }

    /**
     * The half of {@link #onGameFinished} that has already read the stack.
     *
     * @param onStack the ability ids the stack still holds
     * @return how many held casts were abandoned rather than written
     */
    long finishGame(Set<Integer> onStack) {
        // First, because it is the oldest: the resolution it describes ended
        // before anything the two below hold. The winner is not one of its
        // consequences -- player_won is not a permanent leaving the
        // battlefield -- so it stays where flushOrphanedEvents has always put
        // it, in a record of its own after this one.
        flushHeldResolution();
        flushCombat();
        flushOrphanedEvents();
        writeOffRemoved(onStack);
        long abandoned = pendingActivations.size();
        if (abandoned > 0) {
            StringJoiner names = new StringJoiner(", ");
            for (PendingActivation activation : pendingActivations) {
                names.add(activation.linkId());
            }
            System.err.println(
                    "effect records: " + context.gameId() + " ended with " + abandoned
                            + " cast(s) still on the stack, dropped because"
                            + " outcome has no member for a game that stopped: "
                            + names);
        }
        pendingActivations.clear();
        // The listener is static and this collector lives for one game, so the
        // game ending is where it has to go.
        PatchHooks.CONFIRM_LISTENER.uninstall();
        return abandoned;
    }

    // ── an offer the controller turned down ─────────────────────────────

    /**
     * Notice that the controller said no to something.
     *
     * <p>{@code declined} is the one outcome nothing ever wrote, and there is
     * no bus event for it: an optional effect nobody took produces no zone
     * change, no damage and no life total to notice, which is precisely why the
     * schema wanted it — the counterfactual "it was offered and refused" is
     * otherwise indistinguishable from "it was never offered".
     *
     * <p>Only the refusal is kept. The hook is called with both answers, the
     * way the trigger-condition hook is, but an accepted offer is already
     * described by the events it produced.
     *
     * <p>Package-visible so a test can drive it through a proxy of its own. The
     * method name and the argument positions are the whole contract with the
     * patch — {@link PatchHooks} looks both up by string — and a typo in either
     * degrades this channel silently.
     */
    InvocationHandler confirmHandler() {
        return PatchHooks.CONFIRM_LISTENER.listener((callback, args) -> {
            // This hook is a JVM static, and PlayerControllerAi.confirmAction
            // is exactly what a fork's own forced resolution asks when it
            // pays a cost or resolves an optional clause (GameSimulator builds
            // its own PlayerControllerAi over the forked Game, but the static
            // listener it calls is this game's). args[0] is declared
            // SpellAbility on Forge's side; a value this cannot identify is
            // left alone rather than newly dropped.
            SpellAbility offered = args.get(0, SpellAbility.class);
            if (Boolean.FALSE.equals(args.get(1))
                    && (offered == null || belongsToLiveGame(offered))) {
                noteDeclined();
            }
        });
    }

    /**
     * As {@code PatchedCollectors#belongsToLiveGame(CardTraitBase)}, kept as
     * this class's own copy rather than shared: this one needs only the
     * {@code SpellAbility} shape {@link
     * #confirmHandler()} calls it with, not the wider {@code CardTraitBase}
     * {@code PatchedCollectors} also needs for its rewrite/trigger hooks.
     */
    private boolean belongsToLiveGame(SpellAbility ability) {
        Card host = ability.getHostCard();
        return host != null && host.getGame() == game;
    }

    /** Record a refusal against the open bracket, if there is one. */
    void noteDeclined() {
        if (resolving != null) {
            bracketDeclined = true;
        }
    }

    /**
     * Whether the open bracket is a refusal rather than a resolution.
     *
     * <p>Both halves are required. A refusal alone is not enough: a line whose
     * mandatory clause drew a card and whose optional clause was declined did
     * something, and {@code resolved} is the honest word for it. An empty event
     * list alone is not enough either — plenty of resolutions produce nothing
     * this collector can observe. Together they are the shape the schema
     * describes: offered, turned down, and nothing happened.
     */
    private boolean wasDeclined() {
        if (!bracketDeclined || !bracketEvents.isEmpty()) {
            return false;
        }
        // The events of an earlier mode have already moved into its half.
        for (ModeHalf half : modeHalves) {
            if (!half.events.isEmpty()) {
                return false;
            }
        }
        return true;
    }

    /**
     * Whether the game is in either combat damage step.
     *
     * <p>Both names are tested: Forge's first-strike step is {@code
     * COMBAT_FIRST_STRIKE_DAMAGE}, which does not contain {@code
     * COMBAT_DAMAGE}, and a test on that substring alone sent the step's
     * creature damage, life changes and deaths to the ordinary bracket.
     */
    private boolean inCombatDamageStep() {
        return DamageStep.of(game.getPhaseHandler()) != null;
    }

    /**
     * Whether an event now is the combat's: dealt in a damage step, before
     * anything has used the stack in it ({@link #combatWindowClosed}).
     */
    private boolean inCombatDamage() {
        return !combatWindowClosed && inCombatDamageStep();
    }

    /**
     * Start a damage step's bracket, and capture the combat it is about.
     *
     * <p>The structure is read here rather than at the flush, because the flush
     * happens at the phase boundary — after the damage has been dealt and the
     * creatures it killed have left. Read there, an attacker that traded with
     * its blocker is in neither list, and the record describes a combat nobody
     * fought. Read here it matches the snapshot beside it, which is taken in
     * the same call.
     */
    private void openCombatBracket() {
        if (combatState != null) {
            return;
        }
        // Combat damage is the next action. Ordinarily the step boundary
        // before it has written the held resolution already.
        flushHeldResolution();
        combatState = snapshots.ofBoard().render();
        // Nothing has died yet, so this is the moment the attackers and
        // blockers are all still in combat. The assignment is Forge's copy of
        // this step's, taken before any replacement, so it reads the same
        // whenever in the step it is read.
        combatShape = CombatShape.of(game.getCombat());
        combatShape.addAssignment(game.getCombat());
    }

    /**
     * Write the open combat bracket, if any, as one {@code combat} record.
     *
     * <p>{@code ability} is absent: no single line acts in a damage step, and
     * inventing one would attribute the whole step to whichever creature
     * happened to be first.
     */
    private void flushCombat() {
        if (combatState == null) {
            return;
        }
        var phase = game.getPhaseHandler();
        String actor = phase == null || phase.getPlayerTurn() == null
                ? null : SnapshotBuilder.playerId(phase.getPlayerTurn());
        EffectRecord record = context.newRecord(RecordKind.COMBAT)
                .actor(actor)
                .state(combatState)
                .payload(combatPayload());
        context.emit(record);
        // A damage-step probe forked before this step and has been holding its
        // branch since: it needs this record's id to say which combat it
        // mirrors, and the id does not exist until the record is written.
        if (onCombatRecord != null) {
            onCombatRecord.accept(record.recordId());
        }
        combatEvents.clear();
        combatShape = CombatShape.empty();
        combatState = null;
    }

    /**
     * Write a leftover resolution-bracket buffer as its own record, the same
     * rescue {@link #flushCombat} already gives the combat bracket -- and
     * needed for exactly the same reason: something can end the game between
     * the last bracket close and the next one that never comes.
     *
     * <p>Reachable by the most ordinary kind of win. {@code MagicStack} fires
     * {@code GameEventSpellResolved} immediately after a resolving ability's
     * own effect returns, with no game-over check between, so {@link
     * #endBracket} runs -- holding that ability's own effect half and clearing
     * the bracket -- before {@code checkStateBasedEffects} ever asks whether
     * anyone lost (CR 704.3, checked at the start of the next main-loop
     * iteration). A
     * burn spell that leaves its target at 0 life closes its own bracket first
     * and only then is the game discovered to be over: {@code
     * GameEventGameOutcome} fires with no bracket open, {@link #fileOutcome} has
     * nowhere to put it but the now-empty {@link #bracketEvents}, and no
     * further ability ever resolves to trigger the {@link #endBracket} that
     * would otherwise have flushed it. Without this, {@code player_won} (and
     * anything else the bus files outside combat with no bracket open) was
     * silently dropped for every non-combat kill -- a burn spell, a drain, a
     * mill-out, a scripted "you lose the game" -- which is most of them.
     *
     * <p>{@code ability} is absent, the same as {@link #flushCombat}'s own
     * combat record and for the same reason, not a narrower one: {@link
     * #resolvingKeys} is already this collector's own "nothing resolving"
     * default here ({@code closeBracket} set it, and nothing since has
     * touched it), so this writes the same honest answer {@code endBracket}
     * would write for a bracket that closed with nothing resolving -- a game
     * outcome genuinely has no acting line, which is a fact about the record,
     * not a lookup that failed. {@code state} cannot reuse {@link
     * #openBracketState} (also cleared by {@code closeBracket}), so it is
     * read fresh, the same null-ability snapshot {@link #openCombatBracket}
     * takes for its own record.
     *
     * <p><b>{@code resolving == null} is required, not incidental.</b>
     * {@link #bracketEvents} is non-empty at {@code
     * finishGame()} in two reachable shapes, and only one of them means
     * "nothing was resolving": {@code openBracket(X)} runs at <em>cast</em>
     * time, not at resolution, so {@code resolving} stays {@code X} for as
     * long as {@code X} sits on the stack -- and {@code
     * checkStateBasedEffects} runs at the top of every main-loop iteration,
     * the same CR 704.3 mechanism this method's own rescue depends on, so a
     * cast whose own cost payment ends the game (life paid to 0) can leave
     * {@code X} still {@code resolving}, still on the stack, and never
     * resolved. Writing this method's record in that shape would name
     * {@code X} as the resolving ability in the <em>same</em> {@code
     * finishGame()} call that {@link #writeOffRemoved} separately and
     * correctly counts {@code X} as abandoned and never resolved -- two
     * contradictory claims about one ability, and worse than attributing to
     * nothing, because it names a real ability that demonstrably did not
     * resolve. Chosen over keeping the events under some other attribution:
     * whatever {@code X} was doing when the game cut it off is exactly as
     * unfinished as {@code X} itself, so counted-and-dropped (the treatment
     * {@code pendingActivations} already gets) is the honest answer, not a
     * partial one salvaged under a different name.
     */
    private void flushOrphanedEvents() {
        if (resolving != null || bracketEvents.isEmpty()) {
            return;
        }
        context.emit(context.newResolution(Moment.RESOLUTION)
                .actor(resolvingActor)
                .ability(resolvingKeys)
                .state(snapshots.ofBoard().render())
                .payload(EffectRecord.eventsPayload(bracketEvents.drain())));
    }

    private String combatPayload() {
        return EffectRecord.combatPayload(combatShape.fields(), combatEvents.events());
    }
}
