package com.pricepredictor.connector.effects;

import com.google.common.eventbus.Subscribe;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.event.GameEventCardChangeZone;
import forge.game.event.GameEventCombatEnded;
import forge.game.event.GameEventGameFinished;
import forge.game.event.GameEventLandPlayed;
import forge.game.event.GameEventPlayerPriority;
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
 * needs the model object. A cast made while something is resolving — a spell a
 * {@code Play} effect casts, a cascade — waits for that resolution to end
 * before its bracket opens; see {@link WaitingCasts}.
 *
 * <p>The cost half is <b>held</b> from the cast until the bracket closes, so it
 * can be stamped with what actually became of the spell. Every other record this
 * class writes goes out at the moment it describes; this one cannot, because
 * {@code outcome} is a field of it and a spell that is countered before it
 * resolves never publishes anything else the collector could attach the answer
 * to. A cast still on the stack when the game ends is dropped rather than given
 * one of the five outcomes it did not have, and counted and named on stderr;
 * see {@link #finishGame}.
 *
 * <p>Damage steps get their own brackets, closed at the phase boundary, so a
 * first-strike combat writes two records — the only way a keyword that acts by
 * splitting the step is visible in the corpus at all.
 */
public final class BusBracketCollector extends BusEventReader {

    private final CollectorContext context;
    private final SnapshotBuilder snapshots;

    /** The resolution the open bracket attributes to, or null between brackets. */
    private OpenResolution open;

    /**
     * What the bus filed since the bracket last opened or emptied.
     *
     * <p>Outlives {@link #open}: an event that arrives with nothing resolving
     * lands here too, where {@link #flushOrphanedEvents} finds it if the game
     * ends before a cast or a resolution empties the buffer again.
     */
    private final EventBuffer bracketEvents = new EventBuffer();

    /** Casts made while something was resolving, whose brackets have not opened yet. */
    private final WaitingCasts waiting = new WaitingCasts();

    /** The resolution waiting for its consequences, or null when none is. */
    private HeldResolution held;

    private final PendingActivations pending = new PendingActivations();

    /** The damage step's combat being gathered, or null outside one. */
    private CombatBracket combat;

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

    /** Told each combat record's id, for a probe that mirrors it. */
    private final java.util.function.Consumer<String> onCombatRecord;

    /**
     * The ability a bracket attributes to, from the cast that opened it until
     * a resolution closes it or another cast takes the slot over.
     *
     * <p>Built at the cast, and opened then, or at the next priority for a
     * cast made while something was resolving ({@link WaitingCasts}). A cast
     * whose stack top holds no ability builds none: the slot is empty,
     * exactly as between brackets.
     */
    private static final class OpenResolution {
        private final SpellAbility ability;
        private final ProvenanceKey.Resolved keys;
        private final String actor;
        /** The board at the cast, paired at the resolution with the refs read then. */
        private final Snapshot state;
        /** This resolution's chosen modes, split as they resolve; its own and no other's. */
        private final ModeSplit modes = new ModeSplit();

        /**
         * Whether the controller turned something down while this bracket was open.
         *
         * <p>Scoped to the bracket rather than matched against the declined
         * ability's id, because Forge resolves a copy of the ability it was handed
         * and an id comparison would silently never match — the same reason
         * {@link EventAttribution} counts a clause upward from the pointer instead
         * of downward from the root. What keeps the flag from over-claiming is the
         * second half of the test in {@link BusBracketCollector#wasDeclined}: a
         * resolution that produced any outcome at all is {@code resolved} however
         * many optional clauses inside it were declined.
         */
        private boolean declined;

        private OpenResolution(
                SpellAbility ability, ProvenanceKey.Resolved keys, String actor, Snapshot state) {
            this.ability = ability;
            this.keys = keys;
            this.actor = actor;
            this.state = state;
        }

        /** Whether this is the bracket of the ability the engine names by id. */
        boolean is(int abilityId) {
            return ability.getId() == abilityId;
        }

        /**
         * Whether a clause is on this resolution's card; see
         * {@link BusBracketCollector#beginMode}.
         */
        boolean hosts(SpellAbility clause) {
            return clause != null && clause.getHostCard() == ability.getHostCard();
        }
    }

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

    /**
     * A modal resolution's chosen modes, split at each mode's first clause.
     *
     * <p>Owned by the {@link OpenResolution} it splits, so it goes wherever
     * that resolution goes and no other resolution ever sees it. That matters
     * for a mode that casts a spell (Tetsuo, Imperial Champion's second mode):
     * Forge resolves nothing inside a resolution, so the cast spell's
     * {@code GameEventSpellAbilityCast} arrives in the middle of the mode, the
     * charm then finishes and publishes its {@code GameEventSpellResolved},
     * and the spell resolves afterwards, on its own. The charm keeps its
     * bracket, and so its split, until its own resolution closes them; the
     * spell's bracket opens later with a split of its own, empty.
     */
    private static final class ModeSplit {
        /** The modes whose halves are complete, in order. */
        private final List<ModeHalf> complete = new ArrayList<>();
        /** The mode whose clauses are resolving now, or null outside one. */
        private ModeHalf current;

        /** Complete the current mode's half and open one for the next mode. */
        void begin(Integer option, Snapshot state, EventBuffer bracketEvents) {
            close(bracketEvents);
            current = new ModeHalf(option, state);
        }

        /** Move the bracket's events into the current mode's half and complete it. */
        void close(EventBuffer bracketEvents) {
            if (current == null) {
                return;
            }
            current.events.addAll(bracketEvents.drain());
            complete.add(current);
            current = null;
        }

        /** Whether no mode's half is complete: the resolution writes one root half. */
        boolean isEmpty() {
            return complete.isEmpty();
        }

        /** Whether any complete half holds an event. */
        boolean anyEvents() {
            for (ModeHalf half : complete) {
                if (!half.events.isEmpty()) {
                    return true;
                }
            }
            return false;
        }

        List<ModeHalf> complete() {
            return complete;
        }
    }

    /**
     * Casts made while something was resolving, oldest first, each waiting
     * for that resolution to end before its bracket can open.
     *
     * <p>Forge freezes the stack while an ability resolves and parks what is
     * added to it until the resolution ends. A spell a {@code Play} effect
     * casts goes onto it anyway — cascade, discover, madness, Tetsuo's cast
     * mode and every other "you may cast" are that effect — and so does a
     * copy marked {@code IgnoreFreeze}. Its {@code GameEventSpellAbilityCast}
     * is published in the middle of the resolution; the resolution then
     * finishes its remaining clauses and publishes
     * {@code GameEventSpellResolved}, and the cast spell resolves afterwards,
     * after a priority. So the resolving ability keeps the bracket: what it
     * does after the cast is its own, and its effect half and link are
     * written as for any other resolution.
     *
     * <p>The cast's bracket opens at the next {@code GameEventPlayerPriority},
     * which {@code PhaseHandler.mainLoopStep} publishes before its state-based
     * check and before the cast spell can resolve — not at the resolution's
     * end, because the resolved card's own move off the stack comes after
     * that and is nobody's effect. The board it pairs with is still the one
     * taken at the cast, as for every other cast. When several wait, the
     * newest is the stack top and resolves first, so it is the one that
     * opens; the others are under it, exactly as a spell responded to. So is
     * everything waiting when an ordinary cast takes the slot before the
     * priority comes — a parked trigger the end of the resolution put on the
     * stack above them.
     */
    private static final class WaitingCasts {
        private final List<OpenResolution> casts = new ArrayList<>();

        void add(OpenResolution cast) {
            casts.add(cast);
        }

        /**
         * The newest waiting cast whose outcome is still unsettled, none of
         * them waiting any longer. A cast removed from the stack before the
         * priority has been written off and has no bracket to open.
         */
        Optional<OpenResolution> takeNewest(PendingActivations pending) {
            Optional<OpenResolution> newest = Optional.empty();
            for (int i = casts.size() - 1; i >= 0; i--) {
                if (pending.find(casts.get(i).ability.getId()).isPresent()) {
                    newest = Optional.of(casts.get(i));
                    break;
                }
            }
            casts.clear();
            return newest;
        }

        void clear() {
            casts.clear();
        }
    }

    /**
     * One effect half that has finished resolving and is not written yet.
     *
     * <p>Everything but the events is fixed at the resolution — the id, the
     * timestamp, the key and the board, which is the one taken before the
     * resolution. The events can still grow, by the deaths
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
     * boundary. Written at the resolved event, the effect half would hold the
     * damage and never the death, which a combat record, whose bracket stays
     * open until the step ends, does hold.
     *
     * <p>So the halves are held from the resolution until the next real
     * action begins ({@link #onNextAction}) — a bracket opening for a cast or
     * a trigger, another resolution ending, a combat bracket opening, a step
     * or phase boundary, a land played, the end of the game — and written
     * then, exactly once.
     *
     * @param root the resolved ability, which the deaths are stamped against
     */
    private record HeldResolution(
            SpellAbility root, String actor, String linkId, List<HeldHalf> halves) {
    }

    /**
     * Casts whose outcome is not known yet, oldest first.
     *
     * <p>A list rather than a map because it is the stack: two spells can be
     * waiting at once and the inner one resolves first, so a lookup runs from
     * the newest end. Nothing here has been written yet — see
     * {@link PendingActivation} for why the record has to wait.
     */
    private static final class PendingActivations {
        private final List<PendingActivation> casts = new ArrayList<>();

        void add(PendingActivation cast) {
            casts.add(cast);
        }

        boolean isEmpty() {
            return casts.isEmpty();
        }

        /** The held cast for an ability id, newest first. */
        Optional<PendingActivation> find(int abilityId) {
            for (int i = casts.size() - 1; i >= 0; i--) {
                if (casts.get(i).abilityId() == abilityId) {
                    return Optional.of(casts.get(i));
                }
            }
            return Optional.empty();
        }

        /** The held cast for an ability id, no longer held. */
        Optional<PendingActivation> settle(int abilityId) {
            Optional<PendingActivation> found = find(abilityId);
            found.ifPresent(casts::remove);
            return found;
        }

        /** Every held cast the stack no longer holds, oldest first, no longer held. */
        List<PendingActivation> removeAbsent(Set<Integer> onStack) {
            List<PendingActivation> removed = new ArrayList<>();
            for (Iterator<PendingActivation> it = casts.iterator(); it.hasNext();) {
                PendingActivation cast = it.next();
                if (!onStack.contains(cast.abilityId())) {
                    it.remove();
                    removed.add(cast);
                }
            }
            return removed;
        }

        /** Every held cast, oldest first, none of them held any longer. */
        List<PendingActivation> removeAll() {
            List<PendingActivation> all = List.copyOf(casts);
            casts.clear();
            return all;
        }
    }

    /**
     * One damage step's combat, from the first event dealt in it until the
     * step ends or something uses the stack.
     *
     * <p>The structure is read when the bracket opens rather than at the
     * flush, because the flush happens at the phase boundary — after the
     * damage has been dealt and the creatures it killed have left. Read there,
     * an attacker that traded with its blocker is in neither list, and the
     * record describes a combat nobody fought. Read here it matches the
     * snapshot beside it, which is taken in the same call.
     */
    private static final class CombatBracket {
        private final String state;
        private final CombatShape shape;
        private final EventBuffer events = new EventBuffer();

        private CombatBracket(String state, Combat combat) {
            this.state = state;
            // Nothing has died yet, so this is the moment the attackers and
            // blockers are all still in combat. The assignment is Forge's copy of
            // this step's, taken before any replacement, so it reads the same
            // whenever in the step it is read.
            this.shape = CombatShape.of(combat);
            shape.addAssignment(combat);
        }

        void file(EffectEvent event) {
            events.file(event);
        }

        String payload() {
            return EffectRecord.combatPayload(shape.fields(), events.events());
        }
    }

    /**
     * @param onCombatRecord told the id of each {@code combat} record as it is
     *                       written. A damage-step probe forks before the step
     *                       and cannot know what it mirrors until the real
     *                       record exists, so the pairing is handed over
     *                       rather than reconstructed. Reconstructing it from
     *                       board state would not work: two combats in one
     *                       turn can look identical, and a wrong pairing makes
     *                       the difference between them meaningless.
     */
    BusBracketCollector(
            Game game, CollectorContext context,
            java.util.function.Consumer<String> onCombatRecord) {
        super(game);
        this.context = context;
        this.snapshots = new SnapshotBuilder(game);
        this.onCombatRecord = onCombatRecord;
    }

    // ── brackets ────────────────────────────────────────────────────────

    /**
     * A cast opens a bracket and holds the cost half until its outcome is known.
     *
     * <p>The costs are read <b>now</b> — {@link EffectRecord#costsJson} says why
     * they are unreadable a moment later — but the record cannot be written yet,
     * because {@code outcome} is a field of the cost half and the answer to it
     * does not exist until the spell resolves or leaves the stack without
     * resolving. Written here, every activation would read {@code resolved},
     * and a fizzled spell would issue a {@code link_id} whose effect half never
     * comes.
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
     * against a real ability without a game in progress.
     */
    void beginBracket(SpellAbility ability) {
        // A cast, an activation or a trigger reaching the stack is the next
        // action: whatever the held resolution killed has died by now.
        onNextAction();
        // And whatever the combat damage killed: state-based actions run
        // before anything goes on the stack.
        closeCombatWindow();
        OpenResolution cast = ability == null ? null : new OpenResolution(
                ability, ProvenanceResolver.resolve(ability), actorOf(ability),
                snapshots.ofResolving(ability));
        if (game.getStack().isResolving()) {
            // Cast by the resolving ability itself: the bracket stays with
            // the resolution until it ends (WaitingCasts).
            if (cast != null) {
                waiting.add(cast);
            }
        } else {
            openBracket(cast);
        }
        if (cast == null) {
            return;
        }
        pending.add(new PendingActivation(
                ability.getId(), context.nextRecordId(),
                RecordShardWriter.timestamp(), EffectRecord.costsJson(ability),
                cast.actor, cast.keys, cast.state.render(),
                context.gameId() + ".link." + ability.getId(), targetCount(ability)));
    }

    /**
     * A player is about to get priority: a cast made during the resolution
     * that has just ended takes the bracket now ({@link WaitingCasts}).
     */
    @Subscribe
    public void onPriority(GameEventPlayerPriority event) {
        waiting.takeNewest(pending).ifPresent(this::openBracket);
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
        onNextAction();
        // The bracket is a single slot, so a spell cast in response to another
        // takes it over: when the outer one finally resolves the events that
        // belong to it were never gathered. The cost half is still stamped with
        // its real outcome; what it must not do is issue a link to an effect
        // half nobody is going to write.
        boolean bracketIsThisAbility = open != null && open.is(abilityId);
        boolean declined = bracketIsThisAbility && !fizzled && wasDeclined();
        boolean writesEffectHalf = bracketIsThisAbility && !fizzled && !declined;

        String link = settleActivation(
                abilityId,
                settledOutcome(abilityId, fizzled, declined, bracketIsThisAbility),
                writesEffectHalf);

        if (bracketIsThisAbility) {
            // The last mode's half closes with the resolution, since no next
            // mode's first clause comes to close it.
            open.modes.close(bracketEvents);
        }
        if (writesEffectHalf) {
            holdEffectHalves(link);
        }

        if (bracketIsThisAbility) {
            closeBracket();
        } else if (open == null) {
            // Only the buffer: no resolution holds the slot.
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
                pending.find(abilityId).map(PendingActivation::targetsAtCast).orElse(0),
                bracketIsThisAbility ? open.ability : null);
    }

    /**
     * Hold the open resolution's effect halves, under the cost half's link.
     *
     * <p>Held rather than written: the deaths this resolution causes have not
     * happened yet. See {@link HeldResolution}.
     */
    private void holdEffectHalves(String link) {
        if (open.modes.isEmpty()) {
            held = new HeldResolution(open.ability, open.actor, link, List.of(HeldHalf.of(
                    context.nextRecordId(), open.keys,
                    // Modes, X and the named card are set while the ability
                    // resolves, so the block that carries them is re-read here
                    // and paired with the board captured at the cast.
                    open.state.withRefs(snapshots.refsJson(open.ability)).render(),
                    bracketEvents.events())));
            return;
        }
        // A modal resolution on a patched worker: one effect half per chosen
        // mode, each through the mode's own option line, every one of them
        // under the cost half's link (FR-029a, FR-029d). The events a mode
        // produced are its own and nothing else's; a mode that fizzled on its
        // own carries an empty list under its key, which is what the schema
        // says of any resolution that did nothing observable.
        String refs = snapshots.refsJson(open.ability);
        List<HeldHalf> halves = new ArrayList<>();
        for (ModeHalf half : open.modes.complete()) {
            halves.add(HeldHalf.of(
                    context.nextRecordId(), open.keys.withOption(half.option),
                    half.state.withRefs(refs).render(), half.events));
        }
        held = new HeldResolution(open.ability, open.actor, link, List.copyOf(halves));
    }

    /**
     * A phase change closes the combat bracket, if one is open.
     *
     * <p>One record per damage step: first strike and regular damage are
     * separate phases, so the phase boundary is exactly the step boundary.
     */
    @Subscribe
    public void onPhase(GameEventTurnPhase event) {
        onStepBoundary();
        // An effect that empties the stack -- ending the turn, ending combat --
        // resolves nothing afterwards, so a phase boundary is the last chance
        // to notice what it swept away.
        reconcilePending();
    }

    /**
     * The end of combat writes the combat bracket without reopening the
     * window: the phase change that follows it does that.
     */
    @Subscribe
    public void onCombatEnded(GameEventCombatEnded event) {
        onNextAction();
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
        onNextAction();
    }

    // ── the two boundaries ──────────────────────────────────────────────

    /**
     * A new action has begun, so the held resolution's state-based
     * consequences are over: write its effect halves, if one is held.
     *
     * <p>Package-private so a test that ends a resolution by hand can close
     * the window the way the next action in a real game would.
     */
    void onNextAction() {
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

    /**
     * A step or phase has ended: the step's combat record is written and the
     * next damage step starts with its window open again.
     */
    private void onStepBoundary() {
        onNextAction();
        flushCombat();
        combatWindowClosed = false;
    }

    /**
     * Something is going on the stack. In a damage step that ends the
     * combat's share of the step: the record is written now, and what
     * resolves for the rest of the step is not the combat's.
     */
    private void closeCombatWindow() {
        if (inCombatDamageStep()) {
            flushCombat();
            combatWindowClosed = true;
        }
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
        EventAttribution.stamp(damage, resolving(), namedCause);
        if (combat) {
            fileCombat(damage);
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
        into.events().file(EventAttribution.stampAt(moved, held.root(), null, null));
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
        EventAttribution.stamp(event, resolving(), namedCause);
        if (inCombatDamage()) {
            fileCombat(event);
        } else {
            bracketEvents.file(event);
        }
    }

    /**
     * An event an effect API described rather than the bus announced.
     *
     * <p>Package-private because {@link ClauseOutcomeRouter} owns the clause hook
     * and this collector owns the open bracket: the event belongs to whatever
     * resolution is in flight, which only this class knows.
     */
    void recordClauseEvent(EffectEvent event) {
        if (event != null) {
            record(event);
        }
    }

    /** The ability the open bracket attributes to, or null between brackets. */
    private SpellAbility resolving() {
        return open == null ? null : open.ability;
    }

    /**
     * Give the bracket to {@code resolution}, taking it over from any ability
     * that held it and from every cast still waiting for it
     * ({@link WaitingCasts}): those are under this one on the stack.
     *
     * @param resolution the cast's resolution, or null for a cast whose stack
     *                   top held no ability, which empties the slot
     */
    private void openBracket(OpenResolution resolution) {
        open = resolution;
        bracketEvents.clear();
        waiting.clear();
    }

    private void closeBracket() {
        open = null;
        bracketEvents.clear();
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
     * charm whose bracket a response took over — must not cut this one.
     *
     * <p>Events filed before the first mode's clause, between the cast and
     * the resolution, fall into the first mode's half rather than vanishing:
     * {@link #bracketEvents} is not cleared when the first mode opens.
     *
     * @param option the mode's {@code Choices$} position, or null for a mode
     *               {@link CharmModes} could not place
     */
    void beginMode(SpellAbility clause, Integer option) {
        if (open == null || !open.hosts(clause)) {
            return;
        }
        open.modes.begin(option, snapshots.ofResolving(open.ability), bracketEvents);
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

    /**
     * Write a held cost half now that its outcome is known.
     *
     * <p>{@code linked} is separate from the outcome because they answer
     * different questions: the outcome says what became of the spell, and
     * {@code link_id} promises that an effect half carrying the same id exists.
     * A resolution this collector could not describe -- a spell whose bracket
     * a response took over -- is a {@code resolved} activation with no link,
     * which is honest; issuing the link anyway would leave it dangling.
     *
     * @return the link the effect half may claim, or null when there is to be
     *         no effect half or nothing was held for this ability
     */
    String settleActivation(int abilityId, ActivationOutcome outcome, boolean linked) {
        Optional<PendingActivation> found = pending.settle(abilityId);
        if (found.isEmpty()) {
            return null;
        }
        PendingActivation activation = found.get();
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
        if (pending.isEmpty()) {
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
        List<PendingActivation> removed = pending.removeAbsent(onStack);
        for (PendingActivation activation : removed) {
            emitActivation(activation, ActivationOutcome.COUNTERED, false);
        }
        return removed.size();
    }

    // ── the end of the game ─────────────────────────────────────────────

    /**
     * The last thing that happens to a game, and the last chance to write.
     *
     * <p>Three channels have nowhere else to be written.
     *
     * <p>The <b>combat bracket</b> is closed at a phase boundary, and a game
     * that ends in combat damage never reaches one:
     * {@code PhaseHandler.mainLoopStep} returns the moment
     * {@code checkStateBasedEffects} says the game is over, so neither
     * {@code GameEventTurnPhase} nor {@code GameEventCombatEnded} follows the
     * lethal damage step, and without this the combat that decided the game
     * would never be written.
     *
     * <p>The <b>resolution bracket's own leftovers</b> are closed and
     * cleared by the killing ability's own {@link
     * #endBracket} the moment it resolves -- before the state-based check that
     * discovers the game is over ever runs (CR 704.3, next main-loop
     * iteration). A bus event filed after that point, with no bracket open
     * (the ordinary case for {@code player_won}: burn, drain, decking out,
     * anything that is not a combat-damage kill), has nowhere left to be
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
        // battlefield -- so it stays where flushOrphanedEvents puts it, in a
        // record of its own after this one.
        onNextAction();
        flushCombat();
        flushOrphanedEvents();
        writeOffRemoved(onStack);
        List<PendingActivation> abandoned = pending.removeAll();
        if (!abandoned.isEmpty()) {
            StringJoiner names = new StringJoiner(", ");
            for (PendingActivation activation : abandoned) {
                names.add(activation.linkId());
            }
            System.err.println(
                    "effect records: " + context.gameId() + " ended with " + abandoned.size()
                            + " cast(s) still on the stack, dropped because"
                            + " outcome has no member for a game that stopped: "
                            + names);
        }
        return abandoned.size();
    }

    // ── an offer the controller turned down ─────────────────────────────

    /**
     * Notice that the controller said no to something.
     *
     * <p>There is no bus event for {@code declined}: an optional effect nobody
     * took produces no zone change, no damage and no life total to notice,
     * which is why the schema has the outcome — "it was offered and refused"
     * is otherwise indistinguishable from "it was never offered".
     *
     * <p>Only the refusal is kept. The hook is called with both answers, the
     * way the trigger-condition hook is, but an accepted offer is already
     * described by the events it produced.
     *
     * <p>Package-visible because {@link GameRecording} installs it, and so a
     * test can drive it through a proxy of its own. The method name and the
     * argument positions are the whole contract with the patch —
     * {@link PatchHooks} looks both up by string — and a typo in either
     * degrades this channel silently.
     */
    InvocationHandler confirmHandler() {
        return PatchHooks.CONFIRM_LISTENER.listener((callback, args) -> {
            // This hook is a JVM static, and PlayerControllerAi.confirmAction
            // is exactly what a fork's own forced resolution asks when it
            // pays a cost or resolves an optional clause (GameSimulator builds
            // its own PlayerControllerAi over the forked Game, but the static
            // listener it calls is this game's). args[0] is declared
            // SpellAbility on Forge's side; a value this cannot identify
            // still counts as this game's refusal.
            SpellAbility offered = args.get(0, SpellAbility.class);
            if (Boolean.FALSE.equals(args.get(1))
                    && (offered == null || belongsToLiveGame(offered))) {
                noteDeclined();
            }
        });
    }

    /**
     * As {@link LiveGame#owns(forge.game.CardTraitBase)}, kept as this class's
     * own copy rather than shared: this one needs only the
     * {@code SpellAbility} shape {@link #confirmHandler()} calls it with, not
     * the wider {@code CardTraitBase} {@code LiveGame} also takes for the
     * rewrite and trigger hooks.
     */
    private boolean belongsToLiveGame(SpellAbility ability) {
        Card host = ability.getHostCard();
        return host != null && host.getGame() == game;
    }

    /** Record a refusal against the open bracket, if there is one. */
    void noteDeclined() {
        if (open != null) {
            open.declined = true;
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
        // The events of an earlier mode have already moved into its half.
        return open.declined && bracketEvents.isEmpty() && !open.modes.anyEvents();
    }

    // ── combat ──────────────────────────────────────────────────────────

    /**
     * Whether the game is in either combat damage step.
     *
     * <p>Both names are tested: Forge's first-strike step is {@code
     * COMBAT_FIRST_STRIKE_DAMAGE}, which does not contain {@code
     * COMBAT_DAMAGE}, so a test on that substring alone would send the step's
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

    /** File an event into the damage step's bracket, opening it first if need be. */
    private void fileCombat(EffectEvent event) {
        if (combat == null) {
            // Combat damage is the next action. Ordinarily the step boundary
            // before it has written the held resolution already.
            onNextAction();
            combat = new CombatBracket(snapshots.ofBoard().render(), game.getCombat());
        }
        combat.file(event);
    }

    /**
     * Write the open combat bracket, if any, as one {@code combat} record.
     *
     * <p>{@code ability} is absent: no single line acts in a damage step, and
     * inventing one would attribute the whole step to whichever creature
     * happened to be first.
     */
    private void flushCombat() {
        if (combat == null) {
            return;
        }
        var phase = game.getPhaseHandler();
        String actor = phase == null || phase.getPlayerTurn() == null
                ? null : SnapshotBuilder.playerId(phase.getPlayerTurn());
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
     * Write a leftover resolution-bracket buffer as its own record, the rescue
     * {@link #flushCombat} gives the combat bracket, for the same reason:
     * something can end the game between the last bracket close and the next
     * one, which never comes.
     *
     * <p>Reachable by the most ordinary kind of win. {@code MagicStack} fires
     * {@code GameEventSpellResolved} as soon as a resolving ability's effect
     * returns, so {@link #endBracket} closes the bracket before {@code
     * checkStateBasedEffects} asks whether anyone lost (CR 704.3, at the start
     * of the next main-loop iteration). The {@code GameEventGameOutcome} of a
     * burn spell, a drain, a mill-out or a scripted "you lose the game" then
     * lands in the empty {@link #bracketEvents}, and no later resolution comes
     * to flush it: without this, {@code player_won} is lost for every
     * non-combat kill.
     *
     * <p>{@code ability} is absent, as on {@link #flushCombat}'s record: with
     * no resolution open there is no acting line. {@code state} is read fresh,
     * as {@link #fileCombat} reads it, since the board captured at a cast went
     * with the resolution that closed.
     *
     * <p><b>{@code open == null} is required, not incidental.</b> The bracket
     * opens at the cast, so a cast whose own cost payment ends the game (life
     * paid to 0) leaves it open on an ability {@code X} that is still on the
     * stack and never resolved. A record here would name {@code X} as
     * resolving in the same {@code finishGame()} call that counts {@code X}
     * as abandoned. Its events are as unfinished as {@code X}, so they are
     * dropped with it.
     */
    private void flushOrphanedEvents() {
        if (open != null || bracketEvents.isEmpty()) {
            return;
        }
        context.emit(context.newResolution(Moment.RESOLUTION)
                .ability(ProvenanceKey.Resolved.UNKNOWN)
                .state(snapshots.ofBoard().render())
                .payload(EffectRecord.eventsPayload(bracketEvents.drain())));
    }
}
