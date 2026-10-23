package com.pricepredictor.connector.effects;

import com.google.common.eventbus.Subscribe;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.event.GameEvent;
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
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Stage-one collection: the public event bus plus a bracket around resolution.
 *
 * <p>No patched Forge. Events arriving on the bus are attributed to whatever the
 * stack says is resolving — correct for the common case of one thing resolving
 * at a time, and approximate where two resolve together. Every record it writes
 * is stamped {@code degraded} so a corpus never claims more precision than it
 * has.
 *
 * <p>The brackets are delimited by the bus itself: a cast opens one, a
 * resolution closes it. There is one per stack item ({@link StackRecords}), so
 * a spell under a trigger it set off, or under a response, keeps its own until
 * it resolves. The resolving ability is read from {@code game.getStack()}
 * rather than from the event, because the event carries a view and the
 * provenance key needs the model object.
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

    /** One bracket per stack item this collector saw cast. */
    private final StackRecords stack = new StackRecords();

    /**
     * The bracket the bus files into now, or null when nothing is: from a cast
     * until the next action or resolution, the cast's own, and while something
     * resolves, the resolving item's ({@link #followResolution}). Which events
     * it takes is {@link #causeNow}'s to say.
     */
    private OpenResolution open;

    /**
     * What the bus filed while no bracket was open: the aftermath of a
     * resolution, the state-based check before anything is cast again.
     *
     * <p>{@link #flushOrphanedEvents} finds it if the game ends before a cast
     * or a resolution empties it again — the ordinary place of a win.
     */
    private final EventBuffer orphans = new EventBuffer();

    /**
     * The ability whose resolution was the last published, while Forge may
     * still be finishing it, or null; see {@link #followResolution}.
     */
    private Integer finishing;

    /** The cast on its way onto the stack, if one is, and what it set off ({@link Cause#CAST}). */
    private final CastsUnderWay castsUnderWay;

    /** The resolution waiting for its consequences, or null when none is. */
    private HeldResolution held;

    /** The cost halves held until their casts' outcomes are known. */
    private final ActivationLedger ledger;

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
     * One stack item's bracket, from its cast until it resolves or leaves the
     * stack without resolving.
     *
     * <p>Built at the cast. A cast whose stack top holds no ability builds
     * none, and nothing is filed for it.
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
         * What the bus filed for this item: the consequences of its cast —
         * what its card's move onto the stack set off, the state-based check
         * after it, until the next action — and what it does while it
         * resolves.
         */
        private final EventBuffer events = new EventBuffer();

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
     * The brackets of the stack items this collector saw cast, mirroring
     * Forge's stack: one per item, keyed by its ability's id, which is the key
     * {@code MagicStack} itself finds an item by
     * ({@code getInstanceMatchingSpellAbilityID}) and the id both the
     * resolution and the cost half carry.
     *
     * <p>An item's bracket outlives everything put on the stack above it. A
     * cast trigger — prowess, "whenever you cast", heroic, storm, cascade —
     * and any response go above the spell and resolve first; each writes its
     * own halves and is removed, and the spell's bracket is still there when
     * the spell resolves.
     *
     * <p>A cast made <b>while something is resolving</b> waits beside the
     * stack instead of going on top. Forge freezes the stack while an ability
     * resolves and parks what is added to it until the resolution ends; a
     * spell a {@code Play} effect casts goes onto it anyway — cascade,
     * discover, madness, Tetsuo's cast mode and every other "you may cast"
     * are that effect — and so does a copy marked {@code IgnoreFreeze}. Its
     * {@code GameEventSpellAbilityCast} is published in the middle of the
     * resolution; the resolution then finishes its remaining clauses and
     * publishes {@code GameEventSpellResolved}, and the cast spell resolves
     * afterwards. So the resolving item stays the top: what it does after the
     * cast is its own. When it resolves, what waited goes on top in the order
     * it was cast, where Forge put it, below anything the end of the
     * resolution then adds — a parked trigger, a parked copy. The board a
     * waiting cast pairs with is still the one taken at its cast.
     */
    private static final class StackRecords {
        /** Bottom first: the last is the top. */
        private final List<OpenResolution> items = new ArrayList<>();
        /** Cast while something was resolving, oldest first. */
        private final List<OpenResolution> waiting = new ArrayList<>();

        void push(OpenResolution item) {
            items.add(item);
        }

        void waitForResolution(OpenResolution cast) {
            waiting.add(cast);
        }

        /** The top item's bracket, or null when no item is on the stack. */
        OpenResolution top() {
            return items.isEmpty() ? null : items.get(items.size() - 1);
        }

        /** The bracket of {@code abilityId}, no longer on the stack, or null. */
        OpenResolution remove(int abilityId) {
            for (int i = items.size() - 1; i >= 0; i--) {
                if (items.get(i).is(abilityId)) {
                    return items.remove(i);
                }
            }
            for (int i = waiting.size() - 1; i >= 0; i--) {
                if (waiting.get(i).is(abilityId)) {
                    return waiting.remove(i);
                }
            }
            return null;
        }

        /** A resolution has ended: what was cast during it goes on top. */
        void releaseWaiting() {
            items.addAll(waiting);
            waiting.clear();
        }

        /** Remove the bracket of every item the stack no longer holds. */
        void removeAbsent(Set<Integer> onStack) {
            items.removeIf(item -> !onStack.contains(item.ability.getId()));
            waiting.removeIf(item -> !onStack.contains(item.ability.getId()));
        }

        /** Whether {@code item} is still one of the stack's brackets. */
        boolean holds(OpenResolution item) {
            return items.contains(item) || waiting.contains(item);
        }
    }

    /**
     * What one held effect half's record says besides its events, all of it
     * fixed at the resolution — the id, the timestamp, the key and the board,
     * which is the one taken before the resolution. The events can still
     * grow, by the deaths the resolution's own effects cause; see {@link
     * HeldResolution}.
     */
    private record HalfRecord(
            String recordId, String timestamp, ProvenanceKey.Resolved keys, String state) {

        static HalfRecord of(String recordId, ProvenanceKey.Resolved keys, String state) {
            return new HalfRecord(recordId, RecordShardWriter.timestamp(), keys, state);
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
            SpellAbility root, String actor, String linkId, HeldHalves<HalfRecord> halves) {
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
        this.ledger = new ActivationLedger(context);
        this.castsUnderWay = new CastsUnderWay(game, ledger::holds);
        this.onCombatRecord = onCombatRecord;
    }

    /**
     * A cast opens a bracket and holds the cost half until its outcome is known.
     *
     * <p>The costs are read <b>now</b> — {@link PaidCosts} says why
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
        if (ability != null) {
            writeOffEarlierCast(ability.getId());
        }
        OpenResolution cast = ability == null ? null : new OpenResolution(
                ability, ProvenanceResolver.resolve(ability), actorOf(ability),
                snapshots.ofResolving(ability));
        if (game.getStack().isResolving()) {
            // Cast by the resolving ability itself, which keeps the bracket
            // until it ends (StackRecords).
            followResolution();
            if (cast != null) {
                stack.waitForResolution(cast);
            }
        } else {
            openBracket(cast);
            // What the card's move onto the stack set off is this cast's.
            if (cast != null) {
                castsUnderWay.fileHeldInto(cast.ability, cast.events);
            }
            // Any other card still on its way was given up: one cast at a time.
            castsUnderWay.abandonAll();
        }
        castsUnderWay.dropHeld();
        if (cast == null) {
            return;
        }
        ledger.hold(ability, cast.actor, cast.keys, cast.state.render());
    }

    /**
     * A cast of an ability whose earlier cast is still held, written off as
     * any removal is ({@link ActivationLedger#writeOffEarlierCast}), with the
     * earlier cast's bracket.
     */
    private void writeOffEarlierCast(int abilityId) {
        ledger.writeOffEarlierCast(abilityId);
        OpenResolution gone = stack.remove(abilityId);
        if (gone != null && open == gone) {
            open = null;
        }
    }

    /**
     * While the stack resolves, the bracket the bus files into is the
     * resolving item's: the top of {@link StackRecords}, since what is cast
     * during a resolution waits beside the stack rather than going on top.
     *
     * <p>Forge publishes nothing when a resolution starts, so this is asked
     * wherever one first shows — an event filed, a mode beginning, a refusal,
     * a cast made by it — and its resolution's end finds its bracket by id
     * however little it showed.
     *
     * <p>The stack still reads as resolving after a resolution has been
     * published: {@code MagicStack.resolveStack} publishes
     * {@code GameEventSpellResolved} first, then runs the static check and
     * {@code finishResolving}, which moves the card off the stack — to the
     * graveyard, to exile, out of existence for a copy — before it stops
     * resolving. That window is the resolved item's, which has no bracket any
     * more, and not the item below it: while the resolved item is still on
     * the stack ({@link #finishing}), nothing is opened.
     */
    private void followResolution() {
        if (!game.getStack().isResolving() || stack.top() == null || open == stack.top()) {
            return;
        }
        if (finishing != null) {
            if (idsOnStack().contains(finishing)) {
                return;
            }
            finishing = null;
        }
        open = stack.top();
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
        // Found by id wherever it is: on top as a rule, and anywhere for an
        // item resolved by hand. An item this collector never saw cast has
        // none, and its cost half, if any, is still stamped with its real
        // outcome without a link to a half nobody is going to write.
        OpenResolution resolved = stack.remove(abilityId);
        boolean declined = resolved != null && !fizzled && wasDeclined(resolved);
        boolean writesEffectHalf = resolved != null && !fizzled && !declined;

        String link = ledger.settle(
                abilityId,
                ledger.settledOutcome(
                        abilityId, fizzled, declined, resolved == null ? null : resolved.ability),
                writesEffectHalf);

        if (resolved != null) {
            // The last mode's half closes with the resolution, since no next
            // mode's first clause comes to close it.
            resolved.modes.close(resolved.events);
        }
        if (writesEffectHalf) {
            holdEffectHalves(resolved, link);
        }
        // What the resolution cast goes on top, and until the next item
        // resolves or something is cast, nothing is open: what follows is
        // the resolution's aftermath.
        stack.releaseWaiting();
        open = null;
        orphans.clear();
        finishing = abilityId;
        castsUnderWay.dropHeld();
    }

    /**
     * Hold a resolution's effect halves, under the cost half's link.
     *
     * <p>Held rather than written: the deaths this resolution causes have not
     * happened yet. See {@link HeldResolution}.
     */
    private void holdEffectHalves(OpenResolution resolved, String link) {
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
     *
     * <p>No cast is under way across it either: a card a failed play left in
     * the stack zone is a cast given up from here ({@link CastsUnderWay}).
     */
    @Subscribe
    public void onLandPlayed(GameEventLandPlayed event) {
        onNextAction();
        castsUnderWay.abandonAll();
    }

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

    /**
     * A step or phase has ended: the step's combat record is written and the
     * next damage step starts with its window open again.
     */
    private void onStepBoundary() {
        onNextAction();
        flushCombat();
        combatWindowClosed = false;
        // No cast is under way across a step boundary either.
        castsUnderWay.abandonAll();
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
        if (combat) {
            EventAttribution.stamp(damage, resolving(), namedCause);
            fileCombat(damage);
        } else {
            route(damage, namedCause, now(), false);
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
     * @return whether the move was filed into the held resolution, in which
     *         case it goes nowhere else
     */
    private boolean holdAsConsequence(GameEventCardChangeZone event, EffectEvent moved) {
        if (held == null || !isStateBasedLeave(event)) {
            return false;
        }
        held.halves().fileDeath(
                EventAttribution.stampAt(moved, held.root(), null, null), Refs.of(event.card()));
        return true;
    }

    /**
     * Route an event into whichever bracket it is a consequence of.
     *
     * <p>State-based-action deaths arrive with no ability resolving, after the
     * bracket that caused them has closed. The ones it caused never reach
     * here: {@link #holdAsConsequence} files them into its held effect half.
     * The rest are routed by {@link #now} like any other event.
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
        route(event, namedCause, now(), true);
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

    /**
     * What an event published now is a consequence of, read off the engine
     * at the moment it is filed, which decides the record it joins.
     */
    private enum Cause {
        /**
         * An ability is resolving: the resolving item's bracket, or no
         * bracket while Forge finishes one whose resolution has been
         * published ({@link #followResolution}).
         */
        RESOLUTION,
        /**
         * A cost is being paid — for something being cast or activated, or,
         * while an ability resolves, for a spell it casts
         * ({@link #payingForACast}). The paid ability's cost half records
         * it; the event goes nowhere.
         */
        PAYMENT,
        /**
         * A cast is under way and not yet announced ({@link CastsUnderWay}):
         * the static check its card's move onto the stack set off ({@code
         * GameAction.changeZone} runs one before the cast is pushed), and the
         * one {@code MagicStack.push} runs after the push and before the cast
         * event. The cast's own consequence, held for its bracket until its
         * cast event opens it.
         */
        CAST,
        /**
         * A player's action at priority other than a cast: a special action
         * — unmorph, foretell, plot, which {@code MagicStack.add} resolves
         * with no push and no cast event — a mana ability activated outside
         * any payment, or an activation being announced. Every one goes
         * through {@code ComputerUtil.handlePlayingSpellAbility}, which names
         * it to the stack. Nobody's record: the event goes nowhere, and the
         * action ends the last cast's consequences, so the state-based check
         * after it is nobody's either ({@link #onAnyEvent}).
         */
        ACTION,
        /**
         * What follows a cast until the next action: the state-based check
         * after it, and the rest of {@code MagicStack.add} after the cast is
         * published. The open bracket's — the last cast's, while no action
         * or resolution has ended its consequences — or the orphans'; in a
         * combat damage step before anything used the stack, the combat's. A
         * land played and a turn-based action fall here too, and find no
         * bracket open: neither happens with anything on the stack.
         */
        CONSEQUENCE
    }

    /**
     * The {@link Cause} of an event published now, with the card being cast
     * when the cause is {@link Cause#CAST}, or null.
     */
    private record Now(Cause cause, Card casting) {
        Now(Cause cause) {
            this(cause, null);
        }
    }

    /**
     * What an event published now is a consequence of.
     *
     * <p>Read off the engine's own state, in this order: the stack resolving;
     * {@code game.costPaymentStack}, which {@code CostPayment} keeps for every
     * part being paid; a cast under way; the ability {@code
     * MagicStack.freezeStack} names as the action in progress ({@link
     * StackActions}). What is left — a state-based check, which freezes the
     * stack naming no action, or nothing at all — is a consequence of what
     * came before.
     *
     * <p>Not a pure read: it follows a resolution that has begun ({@link
     * #followResolution}), and keeps the marks {@link CastsUnderWay#cardBeingCast}
     * keeps.
     */
    private Now now() {
        if (game.getStack().isResolving()) {
            followResolution();
            return new Now(payingForACast(resolving()) ? Cause.PAYMENT : Cause.RESOLUTION);
        }
        if (game.costPaymentStack.peek() != null) {
            return new Now(Cause.PAYMENT);
        }
        SpellAbility acting = StackActions.inProgress(game.getStack());
        Card casting = castsUnderWay.cardBeingCast(acting);
        if (casting != null) {
            return new Now(Cause.CAST, casting);
        }
        if (acting != null) {
            return new Now(isOpenCast(acting) ? Cause.CONSEQUENCE : Cause.ACTION);
        }
        return new Now(Cause.CONSEQUENCE);
    }

    /** {@link #now}'s cause alone, for a caller with no event to file. */
    private Cause causeNow() {
        return now().cause();
    }

    /** Whether what has this cause belongs with the open bracket, if one is open. */
    private static boolean fromOpenBracket(Cause cause) {
        return cause == Cause.RESOLUTION || cause == Cause.CONSEQUENCE;
    }

    /**
     * File an event where its cause puts it.
     *
     * <p>Stamped here rather than where the event is built: which clause
     * produced it, how long it lasts and — absent a name off the event
     * itself — what caused it are properties of the bracket it landed in,
     * and only this side knows that.
     *
     * @param mayBeCombat whether a consequence in a combat damage step joins
     *                    the combat's record: any outcome does, while a damage
     *                    the bus says is not combat damage goes to the open
     *                    bracket ({@link #fileDamage})
     */
    private void route(EffectEvent event, String namedCause, Now now, boolean mayBeCombat) {
        if (now.cause() != Cause.CAST && now.cause() != Cause.PAYMENT) {
            // The cast under way has been announced or abandoned: what its
            // window held is taken or dropped.
            castsUnderWay.dropHeld();
        }
        switch (now.cause()) {
            case PAYMENT -> {
                if (!game.getStack().isResolving()) {
                    open = null;
                }
            }
            case CAST -> castsUnderWay.hold(now.casting(), event, namedCause);
            case ACTION -> open = null;
            case CONSEQUENCE -> {
                EventAttribution.stamp(event, resolving(), namedCause);
                if (mayBeCombat && inCombatDamage()) {
                    fileCombat(event);
                } else {
                    openEvents().file(event);
                }
            }
            case RESOLUTION ->
                    openEvents().file(EventAttribution.stamp(event, resolving(), namedCause));
        }
    }

    /**
     * Any event at all: an action at priority that has begun ends the last
     * cast's consequences even when it files nothing of its own, so the
     * state-based check after it is not the waiting spell's.
     */
    @Subscribe
    public void onAnyEvent(GameEvent event) {
        if (open != null && !game.getStack().isResolving() && causeNow() == Cause.ACTION) {
            open = null;
        }
    }

    /**
     * A priority: no cast is under way across one. A card still in the stack
     * zone with no stack item is a cast given up — a copy that failed its
     * targeting — and no longer counts as being cast.
     */
    @Subscribe
    public void onPriority(GameEventPlayerPriority event) {
        castsUnderWay.abandonAll();
    }

    @Override
    void onStackArrival(int cardId) {
        castsUnderWay.onStackArrival(cardId);
    }

    /**
     * Whether the action in progress is the open bracket's own cast, the rest
     * of whose {@code MagicStack.add} runs after the cast is published. An
     * activation is pushed as a copy of the ability the action names.
     */
    private boolean isOpenCast(SpellAbility acting) {
        return open != null
                && (open.ability == acting || open.ability.getOriginalAbility() == acting);
    }

    /** The ability the open bracket attributes to, or null when none is open. */
    private SpellAbility resolving() {
        return open == null ? null : open.ability;
    }

    /** Where an event filed now goes: the open bracket's buffer, or the orphans. */
    private EventBuffer openEvents() {
        return open == null ? orphans : open.events;
    }

    /**
     * A cast has gone on top of the stack: its bracket opens. The item it
     * went above keeps its own whole, with what it gathered since its cast —
     * the state-based check its cast set off, a creature it left without
     * toughness — which is that cast's consequence.
     *
     * @param cast the cast's bracket, or null for a cast whose stack top held
     *             no ability, after which no bracket is open
     */
    private void openBracket(OpenResolution cast) {
        // Nothing is cast before Forge has finished the last resolution, and
        // the id it watched may be cast again from here on.
        finishing = null;
        orphans.clear();
        if (cast != null) {
            stack.push(cast);
        }
        open = cast;
    }

    /**
     * A chosen mode's first clause is resolving: close the previous mode's
     * half and open this one's, with a snapshot taken now (FR-029a, FR-029b).
     *
     * <p>Called by the clause hook, which is to say only on a patched worker:
     * a degraded one never reaches here and writes the one root half it
     * always wrote, with every mode's events in it (FR-029e). The clause's
     * host is checked against the open bracket's because the hook is a JVM
     * static that sees every resolution in the process: a mode of some other
     * charm — a copy resolving on a fork — must not cut this one. Nor does a
     * clause of a player's action at priority ({@link Cause#ACTION}): a mana
     * ability's or a special action's, which {@code MagicStack.add} resolves
     * in place.
     *
     * <p>Events filed before the first mode's clause, between the cast and
     * the resolution, fall into the first mode's half rather than vanishing:
     * the bracket's buffer is not cleared when the first mode opens.
     *
     * @param option the mode's {@code Choices$} position, or null for a mode
     *               {@link CharmModes} could not place
     */
    void beginMode(SpellAbility clause, Integer option) {
        if (!fromOpenBracket(causeNow()) || open == null || !open.hosts(clause)) {
            return;
        }
        open.modes.begin(option, snapshots.ofResolving(open.ability), open.events);
    }

    private static String actorOf(SpellAbility ability) {
        return ability == null || ability.getActivatingPlayer() == null
                ? null
                : SnapshotBuilder.playerId(ability.getActivatingPlayer());
    }

    /** As {@link ActivationLedger#outcomeOf}; kept here for the tests that pin it. */
    static ActivationOutcome outcomeOf(int targetsAtCast, SpellAbility resolved) {
        return ActivationLedger.outcomeOf(targetsAtCast, resolved);
    }

    /**
     * Write off every held cast that has left the stack without resolving
     * ({@link ActivationLedger#writeOffRemoved}), asked of the stack itself at
     * every moment it can have changed. With nothing held there is nothing to
     * ask.
     */
    private void reconcilePending() {
        if (ledger.isEmpty()) {
            return;
        }
        writeOffRemoved(idsOnStack());
    }

    /** The ability ids the stack holds right now. */
    private Set<Integer> idsOnStack() {
        Set<Integer> onStack = new LinkedHashSet<>();
        for (var instance : game.getStack()) {
            SpellAbility sa = instance.getSpellAbility();
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
        int removed = ledger.writeOffRemoved(onStack);
        // Their brackets go with them: nothing they gathered is anyone's.
        stack.removeAbsent(onStack);
        if (open != null && !stack.holds(open)) {
            open = null;
        }
        return removed;
    }

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
        return ledger.abandonAll();
    }

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

    /**
     * Record a refusal against the open bracket, if there is one and the
     * refusal is a consequence of it ({@link Cause#RESOLUTION} or {@link
     * Cause#CONSEQUENCE}). One made paying for something else is that other
     * ability's; one made announcing a cast, or in an action at priority, is
     * no resolution's.
     */
    void noteDeclined() {
        // The cause first: reading it opens the resolving item's bracket.
        if (fromOpenBracket(causeNow()) && open != null) {
            open.declined = true;
        }
    }

    /**
     * Whether a resolution is a refusal rather than a resolution.
     *
     * <p>Both halves are required. A refusal alone is not enough: a line whose
     * mandatory clause drew a card and whose optional clause was declined did
     * something, and {@code resolved} is the honest word for it. An empty event
     * list alone is not enough either — plenty of resolutions produce nothing
     * this collector can observe. Together they are the shape the schema
     * describes: offered, turned down, and nothing happened.
     */
    private static boolean wasDeclined(OpenResolution resolved) {
        // The events of an earlier mode have already moved into its half.
        return resolved.declined && resolved.events.isEmpty() && !resolved.modes.anyEvents();
    }

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
     * lands in {@link #orphans}, with no bracket open, and no later
     * resolution comes to flush it: without this, {@code player_won} is lost
     * for every non-combat kill.
     *
     * <p>{@code ability} is absent, as on {@link #flushCombat}'s record: with
     * no resolution open there is no acting line. {@code state} is read fresh,
     * as {@link #fileCombat} reads it, since the board captured at a cast went
     * with the resolution that closed.
     *
     * <p>Only the orphans are written, never an open bracket's events. A cast
     * whose own cost payment ends the game (life paid to 0) leaves its bracket
     * open on an ability {@code X} that is still on the stack and never
     * resolved, and the win lands in it. A record here would name {@code X}
     * as resolving in the same {@code finishGame()} call that counts {@code
     * X} as abandoned. Its events are as unfinished as {@code X}, so they are
     * dropped with it.
     */
    private void flushOrphanedEvents() {
        if (orphans.isEmpty()) {
            return;
        }
        context.emit(context.newResolution(Moment.RESOLUTION)
                .ability(ProvenanceKey.Resolved.UNKNOWN)
                .state(snapshots.ofBoard().render())
                .payload(EffectRecord.eventsPayload(orphans.drain())));
    }
}
