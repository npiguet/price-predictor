package com.pricepredictor.connector.effects;

import com.google.common.eventbus.Subscribe;
import forge.game.Game;
import forge.game.card.Card;
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
import forge.game.zone.ZoneType;

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
 * resolution closes it. There is one per stack item ({@link StackBrackets}), so
 * a spell under a trigger it set off, or under a response, keeps its own until
 * it resolves. The resolving ability is read from {@code game.getStack()}
 * rather than from the event, because the event carries a view and the
 * provenance key needs the model object.
 *
 * <p>Each record describes one moment, its board read just before it (root
 * spec § 8.14, FR-042a–FR-042f). A cast or an activation writes its cost half
 * when its costs are paid — at its cast event — on the board read before the
 * card moved to the stack or the costs were paid ({@link #castHandler}); it
 * does not wait for the item's fate. The item's resolution writes its effect
 * half, on the board read when the resolution starts ({@link
 * #resolutionHandler}), carrying the item's {@code outcome}; a countered item
 * writes none, its fate being the countering clause's {@code spell_countered}
 * and the card's move off the stack. A triggered ability and a copy, which
 * nobody casts, write no cost half.
 *
 * <p>Damage steps get their own brackets, closed at the phase boundary, so a
 * first-strike combat writes two records — the only way a keyword that acts by
 * splitting the step is visible in the corpus at all.
 *
 * <p>The turn-based actions that begin a step — the untap, the draw, the
 * cleanup's discard — get a window of their own ({@link TurnActionWindow}),
 * from the phase event to the next priority or phase event, and a {@code
 * turn_action} record per action.
 *
 * <p>The state-based loop Forge runs before each priority is told by its
 * begin and end signals ({@link #phaseSink}, FR-028). A resolution's effect
 * half is held across the whole loop and written at its end; a trigger put on
 * the stack between two passes opens its own bracket without ending it. What
 * reaches no record is counted ({@link DroppedEvents}, FR-029, FR-041).
 *
 * <p>This class is the {@code @Subscribe} facade: each handler is the sequence
 * of steps an event means, and the state those steps change is owned by the
 * collaborator named for it — {@link StackBrackets} the stack's brackets and
 * the open one, {@link HeldResolutionWindow} the effect halves waiting for their state-based
 * consequences, {@link CombatStepBracket} the damage step's record and
 * {@link CastsUnderWay} the cast on its way onto the stack. What the bus
 * publishes is classified by {@link #classifyNow}.
 */
public final class BusBracketCollector extends BusEventReader {

    private final CollectorContext context;
    private final SnapshotBuilder snapshots;

    /** What this game's records never took, by event type and engine phase. */
    private final DroppedEvents dropped = new DroppedEvents();

    /** One bracket per stack item this collector saw cast, and the open one. */
    private final StackBrackets brackets = new StackBrackets(dropped);

    /** The cast on its way onto the stack, if one is, and what it set off ({@link Cause#CAST}). */
    private final CastsUnderWay castsUnderWay;

    /** The resolution waiting for its consequences, if one is. */
    private final HeldResolutionWindow heldResolution;
    /** The free activations that changed nothing this turn, for the random seat. */
    private final FruitlessActivations fruitless;

    /**
     * The board read as a cast or an activation started (FR-042a), until its
     * cast event writes its cost half; null when no cast is under way.
     */
    private PendingCast pendingCast;

    /** A cast's board, read before its card moved or its costs were paid. */
    private record PendingCast(SpellAbility ability, int hostId, Snapshot board) {

        /** Whether the item a cast event put on the stack is this cast's. */
        boolean isOf(SpellAbility pushed) {
            return pushed == ability || pushed.getOriginalAbility() == ability
                    || (pushed.getHostCard() != null && pushed.getHostCard().getId() == hostId
                            && pushed.isSpell() == ability.isSpell());
        }
    }

    /** The damage step's combat record, gathered until the step ends. */
    private final CombatStepBracket combatStep;

    /** The step's turn-based actions, gathered until anyone gets priority. */
    private final TurnActionWindow turnActions;

    /**
     * Whether the engine is between a state-based loop's begin and end
     * signals. Never set on a worker without the loop hook, where the held
     * half keeps its fallback window ({@link HeldResolutionWindow}).
     */
    private boolean inStateBasedLoop;

    /** Whether something went on the stack during the state-based loop running now. */
    private boolean stackUsedInLoop;

    /** Where this game's {@link PhaseSignals} deliver the loop's two signals. */
    private final PhaseSignals.Sink phaseSink = new PhaseSignals.Sink() {
        @Override
        public void stateBasedLoopBegan() {
            beginStateBasedLoop();
        }

        @Override
        public void stateBasedLoopEnded() {
            endStateBasedLoop();
        }
    };

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
        this.fruitless = new FruitlessActivations(game);
        this.heldResolution = new HeldResolutionWindow(context, fruitless);
        this.castsUnderWay = new CastsUnderWay(game, brackets::holds);
        this.combatStep = new CombatStepBracket(
                game, context, snapshots, heldResolution, onCombatRecord);
        this.turnActions = new TurnActionWindow(game, context, snapshots);
    }

    /**
     * A cast opens a bracket and writes its cost half (FR-042a).
     *
     * <p>The costs are read <b>now</b> — {@link PaidCosts} says why they are
     * unreadable a moment later — and the half is written now, its costs
     * paid: what became of the item is its effect half's to say.
     */
    @Subscribe
    public void onCast(GameEventSpellAbilityCast event) {
        // Read through peek() rather than peekAbility(), which dereferences the
        // top of an empty stack. The bus is not a promise about the stack.
        SpellAbility ability = StackActions.topAbility(game.getStack());
        // A spell taken off the stack without resolving publishes nothing this
        // collector can tell from a resolution, so the absence is noticed
        // whenever the stack is next observed. A new cast is one such moment.
        reconcilePending();
        beginBracket(ability);
    }

    /**
     * The cast hook's listener: a spell's card is about to move to the stack,
     * or an activated ability's costs are about to be paid (FR-042a). The
     * cost half's board is read here, before either, and kept until the cast
     * event writes the half; a cast given up before it is never written.
     *
     * <p>Fired for every game on this JVM, a fork's included, so a cast of
     * another game's is not this one's.
     */
    PatchHooks.Listener castHandler() {
        return PatchHooks.CAST_LISTENER.listener((callback, args) -> {
            SpellAbility cast = args.get(0, SpellAbility.class);
            if (cast != null && owns(cast)) {
                castStarting(cast);
            }
        });
    }

    /**
     * The half of {@link #castHandler} past the game check: read the board
     * the cost half describes. Starting a cast at priority is the next
     * action, so the held resolution's consequences are over; one started by
     * a resolving ability (cascade, a "you may cast") is part of that
     * resolution and ends nothing.
     */
    void castStarting(SpellAbility cast) {
        if (!inStateBasedLoop && !game.getStack().isResolving()) {
            onNextAction();
        }
        Card host = cast.getHostCard();
        pendingCast = new PendingCast(
                cast, host == null ? -1 : host.getId(), snapshots.ofResolving(cast));
        rebaselineAtSnapshot();
    }

    /**
     * The resolution hook's listener: the stack's top item starts to resolve
     * (FR-042b). Its effect half's board is read here, before its first
     * clause; another game's item goes to the fork running on this thread,
     * if one is ({@link Forks#forkHooks}).
     */
    PatchHooks.Listener resolutionHandler() {
        return PatchHooks.RESOLUTION_LISTENER.listener((callback, args) -> {
            SpellAbility item = args.get(0, SpellAbility.class);
            if (item == null) {
                return;
            }
            boolean fizzled = Boolean.TRUE.equals(args.get(1));
            if (owns(item)) {
                resolutionStarting(item);
            } else {
                Forks.forkHooks().resolutionStarting(item, fizzled);
            }
        });
    }

    /**
     * The half of {@link #resolutionHandler} past the game check.
     *
     * <p>The resolution is the next action: the held resolution's
     * consequences are over. Its bracket opens, and what it gathered since
     * its cast — the cast's consequences, the state-based check before
     * priority — precedes this board and reaches no record: it is counted as
     * dropped. The stat differ reads the board again (FR-042f).
     */
    void resolutionStarting(SpellAbility item) {
        if (!inStateBasedLoop) {
            onNextAction();
        }
        OpenResolution resolving = brackets.startResolving(item.getId());
        if (resolving == null) {
            return;
        }
        brackets.countDropped(resolving, DroppedEvents.phaseOf(game));
        resolving.resolutionStarted(snapshots.ofResolving(resolving.ability));
        rebaselineAtSnapshot();
    }

    /** Whether an ability a hook hands over is this game's. */
    private boolean owns(SpellAbility ability) {
        Card host = ability.getHostCard();
        return host != null && game != null && host.getGame() == game;
    }

    /**
     * Everything a cast does once the ability has been found on the stack.
     *
     * <p>Split from {@link #onCast} at the one line that reads the stack, so the
     * rules that decide what a cost record says can be exercised against a
     * real ability without a game in progress.
     */
    void beginBracket(SpellAbility ability) {
        if (inStateBasedLoop) {
            // A trigger reaching the stack between two passes of the
            // state-based check (FR-028). It opens its own bracket below, and
            // the check goes on: the held half and the damage step's record
            // take its deaths until the loop's end signal.
            stackUsedInLoop = true;
        } else {
            // A cast, an activation or a trigger reaching the stack is the
            // next action: whatever the held resolution killed has died by
            // now.
            onNextAction();
            // And whatever the combat damage killed: state-based actions run
            // before anything goes on the stack.
            combatStep.closeWindow();
        }
        if (ability != null) {
            brackets.writeOff(ability.getId());
        }
        PendingCast started = startedCast(ability);
        OpenResolution cast = ability == null ? null : openResolution(ability, started);
        if (game.getStack().isResolving()) {
            // Cast by the resolving ability itself, which keeps the bracket
            // until it ends (StackBrackets).
            brackets.followResolving(this::idsOnStack);
            if (cast != null) {
                brackets.waitForResolution(cast);
            }
        } else {
            if (cast != null && started == null) {
                // Its board was read just now, at the cast event, where no
                // hook told the cast's start: its stat changes are measured
                // from here. Not while an ability resolves, whose bracket
                // keeps the events, and whose snapshot is older.
                rebaselineAtSnapshot();
            }
            brackets.openCast(cast);
            // Any other card still on its way was given up: one cast at a time.
            castsUnderWay.abandonAll();
        }
        if (cast == null || cast.linkId == null) {
            return;
        }
        writeCostHalf(cast, started);
        if (ability.isActivatedAbility() && cast.costs.spentNoManaAndMovedNoCard()) {
            // Until it resolves, nobody knows whether it changes anything, and
            // the random seat, which keeps priority, would stack another.
            fruitless.pending(ability, cast.keys.key());
        }
    }

    /**
     * The cast whose start was told for the item a cast event put on the
     * stack, no longer pending; null when none was told (a copy, a worker
     * without the cast hook). A started cast the pushed item is not, other
     * than for a trigger, was given up.
     */
    private PendingCast startedCast(SpellAbility pushed) {
        PendingCast started = pendingCast;
        if (started == null || pushed == null) {
            return null;
        }
        if (started.isOf(pushed)) {
            pendingCast = null;
            return started;
        }
        if (!pushed.isTrigger()) {
            pendingCast = null;
        }
        return null;
    }

    /**
     * The bracket of an item a cast event put on the stack. An item somebody
     * cast or activated — not a triggered ability, not a copy — has a cost
     * half: its link and what it paid, read now.
     *
     * @param started its cast's start, or null
     */
    private OpenResolution openResolution(SpellAbility ability, PendingCast started) {
        // The board a degraded worker's effect half is read against, the
        // resolution's start being told only by the hook.
        Snapshot atCast = started != null ? started.board() : snapshots.ofResolving(ability);
        if (!hasCostHalf(ability)) {
            return new OpenResolution(
                    ability, ProvenanceResolver.resolve(ability), actorOf(ability), atCast);
        }
        return new OpenResolution(
                ability, ProvenanceResolver.resolve(ability), actorOf(ability), atCast,
                context.gameId() + ".link." + ability.getId(), PaidCosts.of(ability));
    }

    /**
     * Whether a stack item has a cost half: something cast or activated.
     * A triggered ability is not (root § 8.14), nor a copy of a spell or an
     * ability, which is put on the stack without being cast.
     */
    static boolean hasCostHalf(SpellAbility ability) {
        if (ability.isTrigger() || ability.isCopied()) {
            return false;
        }
        Card host = ability.getHostCard();
        return !(ability.isSpell() && host != null && host.isCopiedSpell());
    }

    /**
     * Write a cast's cost half (FR-042a): the board read as its cast started
     * — or at the cast, where its start was not told — with the refs block
     * read now, its targets, modes and X chosen; its events, a spell card's
     * {@code zone_change} to the stack; and its costs.
     */
    private void writeCostHalf(OpenResolution cast, PendingCast started) {
        Snapshot board = started != null
                ? started.board().withRefs(RefsJson.render(cast.ability))
                : cast.state;
        context.emit(context.newResolution(Moment.ACTIVATION)
                .actor(cast.actor)
                .ability(cast.keys)
                .linkId(cast.linkId)
                .state(board.render())
                .payload(RecordPayloads.costPayload(castMove(cast.ability), cast.costs)));
    }

    /**
     * A spell's own move to the stack, from the zone the bus said its card
     * came from (601.2a); nothing for an activated ability, which moves no
     * card, or a card whose arrival was not seen.
     */
    private List<EffectEvent> castMove(SpellAbility ability) {
        Card host = ability.getHostCard();
        if (!ability.isSpell() || host == null) {
            return List.of();
        }
        String from = castsUnderWay.arrivedFrom(host.getId());
        if (from == null) {
            return List.of();
        }
        EffectEvent moved = new EffectEvent(EventType.ZONE_CHANGE)
                .subject(Refs.of(host))
                .param("from_zone", from)
                .param("to_zone", CardSpelling.zoneName(ZoneType.Stack));
        return List.of(EventAttribution.stampAt(moved, ability, null, null));
    }

    /**
     * A resolution closes the bracket and writes or holds its effect half,
     * which carries the item's {@code outcome} (FR-042b).
     *
     * <p>A fizzled or declined item did nothing: its effect half is written
     * now, with no events. Any other is held until the deaths it causes have
     * happened ({@link HeldResolutionWindow}), and a spell's half takes its
     * card's move off the stack, which comes after the resolution is
     * published.
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
        // An item this collector never saw cast has no bracket, and no
        // effect half.
        OpenResolution resolved = brackets.closeAfterResolution(abilityId);
        if (resolved == null) {
            return;
        }
        // The last mode's half closes with the resolution, since no next
        // mode's first clause comes to close it.
        resolved.modes.close();
        ActivationOutcome outcome = resolved.outcome(fizzled);
        if (outcome == ActivationOutcome.FIZZLED || outcome == ActivationOutcome.DECLINED) {
            heldResolution.writeEmpty(resolved, outcome);
            return;
        }
        // Held rather than written: the deaths this resolution causes have
        // not happened yet. An activation that paid nothing the game keeps is
        // one the random seat could repeat at will (research C9).
        boolean free = resolved.costs != null && resolved.ability.isActivatedAbility()
                && resolved.costs.spentNoManaAndMovedNoCard();
        heldResolution.hold(resolved, outcome, free);
    }

    /**
     * The id of a live record still open, its id already taken, or null: the
     * held resolution's first effect half. A directed fork taken at a phase
     * boundary finds none, because the boundary closes the held window first.
     */
    String heldRecordId() {
        return heldResolution.heldRecordId();
    }

    /**
     * A phase change closes the combat bracket, if one is open, and the last
     * step's turn-action window, and opens this step's.
     *
     * <p>One record per damage step: first strike and regular damage are
     * separate phases, so the phase boundary is exactly the step boundary.
     * Forge publishes the phase event before the step's turn-based actions,
     * so the window opened last sees all of them, and a damage step's board
     * is snapshotted here before it deals any damage.
     */
    @Subscribe
    public void onPhase(GameEventTurnPhase event) {
        turnActions.close();
        onStepBoundary();
        // An effect that empties the stack -- ending the turn, ending combat --
        // resolves nothing afterwards, so a phase boundary is the last chance
        // to notice what it swept away.
        reconcilePending();
        turnActions.open(event.phase());
        if (combatStep.beginStep(event.phase())) {
            rebaselineAtSnapshot();
        }
    }

    /**
     * The end of combat writes the combat bracket without reopening the
     * window: the phase change that follows it does that.
     */
    @Subscribe
    public void onCombatEnded(GameEventCombatEnded event) {
        onNextAction();
        combatStep.flush();
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
        pendingCast = null;
    }

    /**
     * A new action has begun, so the held resolution's state-based
     * consequences are over: write its effect halves, if one is held.
     *
     * <p>Package-private so a test that ends a resolution by hand can close
     * the window the way the next action in a real game would.
     */
    void onNextAction() {
        heldResolution.close();
    }

    /**
     * Where this game's state-based-loop signals go: {@link Channels} hands
     * it to the game's {@link PhaseSignals}.
     */
    PhaseSignals.Sink phaseSink() {
        return phaseSink;
    }

    /**
     * The state-based loop has begun: from here to its end, every permanent
     * leaving the battlefield is one of its deaths, frozen tracker or not;
     * the triggers it puts on the stack go there between two passes, with
     * the tracker thawed.
     */
    void beginStateBasedLoop() {
        inStateBasedLoop = true;
        stackUsedInLoop = false;
    }

    /**
     * The state-based loop has ended: the held resolution's consequences are
     * over and its effect halves are written; and, if a trigger went on the
     * stack during the loop, so is a damage step's record, since what
     * resolves for the rest of the step is not the combat's.
     */
    void endStateBasedLoop() {
        inStateBasedLoop = false;
        onNextAction();
        if (stackUsedInLoop) {
            combatStep.closeWindow();
        }
        stackUsedInLoop = false;
    }

    /** This game's events that reached no record so far. */
    DroppedEvents droppedEvents() {
        return dropped;
    }

    /**
     * A step or phase has ended: the step's combat record is written and the
     * next damage step starts with its window open again.
     */
    private void onStepBoundary() {
        onNextAction();
        combatStep.endStep();
        // No cast is under way across a step boundary either.
        castsUnderWay.abandonAll();
        pendingCast = null;
    }

    @Override
    boolean cardDamageIsCombat() {
        return combatStep.takesEventsNow();
    }

    /**
     * Combat damage joins the step's bracket; any other damage the open one.
     *
     * <p>Routed by the caller's flag rather than by {@link
     * CombatStepBracket#takesEventsNow}, because a player damage carries
     * Forge's own combat flag and a card damage has only the step to go by.
     */
    @Override
    void fileDamage(EffectEvent damage, String namedCause, boolean combat) {
        if (combat) {
            EventAttribution.stamp(damage, brackets.resolving(), namedCause);
            combatStep.file(damage);
        } else {
            route(damage, namedCause, Filing.NON_COMBAT_DAMAGE);
        }
    }

    /**
     * A state-based death made while a resolution is held is that
     * resolution's consequence, filed into it
     * ({@link HeldResolutionWindow#fileDeath}); one made while none is held
     * is a damage step's, or nobody's (FR-029); any other move is routed.
     *
     * <p>A state-based death is a move with two properties, each closing a
     * way an unrelated move could join:
     * <ul>
     *   <li>it leaves the <b>battlefield</b> — a death, an exile instead of a
     *   death, a token ceasing to exist, an Aura falling off; a card arriving
     *   anywhere is not a consequence of this kind; and</li>
     *   <li>it is made inside the state-based loop, between its begin and
     *   end signals, every pass included; on a worker without the loop hook,
     *   while a state-based check runs, read as the game's view tracker
     *   being frozen, which {@code GameAction.checkStateEffects} holds for
     *   each pass. After the loop the player acts, and the costs of that
     *   action are paid before the cast is announced: a Treasure the
     *   resolution made, sacrificed for mana to cast the next spell, leaves
     *   the battlefield after the loop and is that spell's cost, not this
     *   resolution's consequence. Outside a resolution nothing else freezes
     *   the tracker while a card moves.</li>
     * </ul>
     */
    @Override
    void fileMove(GameEventCardChangeZone event, EffectEvent moved) {
        if (heldResolution.takeMoveOffStack(event, moved)) {
            return;
        }
        if (!isStateBasedDeath(event)) {
            record(moved);
        } else if (heldResolution.isHolding()) {
            heldResolution.fileDeath(moved, event.card());
        } else {
            route(moved, null, Filing.STATE_BASED_DEATH);
        }
    }

    /** Whether a move is a death of the state-based check running now ({@link #fileMove}). */
    private boolean isStateBasedDeath(GameEventCardChangeZone event) {
        return inStateBasedLoop ? leavesBattlefield(event) : isStateBasedLeave(event);
    }

    /**
     * Route an event into whichever bracket it is a consequence of.
     *
     * <p>State-based-action deaths arrive with no ability resolving, after the
     * bracket that caused them has closed. None reaches here: {@link
     * #fileMove} files the held resolution's into its effect half and routes
     * the rest as deaths. Every other event is routed by {@link #now}, a
     * turn-based action's into its window ({@link TurnActionWindow}).
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
        route(event, namedCause, Filing.OUTCOME);
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
         * published ({@link StackBrackets#followResolving}).
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
         * event. Reaches no record, and is counted as dropped: the cost half
         * records the card's move and the costs alone, and the effect half's
         * board, read at the resolution's start, already shows it.
         */
        CAST,
        /**
         * A player's action at priority other than a cast: a special action
         * — unmorph, foretell, plot, which {@code MagicStack.add} resolves
         * with no push and no cast event — a mana ability activated outside
         * any payment, or an activation being announced. Every one goes
         * through {@code ComputerUtil.handlePlayingSpellAbility}, which names
         * it to the stack. Nobody's record: the event goes nowhere and is
         * counted as dropped, and the action ends the last cast's
         * consequences, so the state-based check after it is nobody's either
         * ({@link #onAnyEvent}).
         */
        ACTION,
        /**
         * What follows a cast until the next action: the state-based check
         * after it, and the rest of {@code MagicStack.add} after the cast is
         * published. The open bracket's — the last cast's, while no action
         * or resolution has ended its consequences — or the orphans'; in a
         * combat damage step before anything used the stack, the combat's. A
         * land played and a turn-based action fall here too, and find no
         * bracket open: neither happens with anything on the stack. A
         * state-based death nothing held is never the open bracket's: the
         * combat's, or nobody's (FR-029).
         */
        CONSEQUENCE
    }

    /** What kind of event {@link #route} is filing, which narrows where it may go. */
    private enum Filing {
        /** An outcome: in a combat damage step before anything used the stack, the combat's. */
        OUTCOME,
        /** A damage the bus says is not combat damage: never the combat's. */
        NON_COMBAT_DAMAGE,
        /**
         * A permanent leaving the battlefield in a state-based check while no
         * resolution is held ({@link #fileMove}): as a consequence, a combat
         * damage step's or nobody's, counted as dropped; never the open
         * bracket's, whose cast did not leave the board in that state
         * (FR-029).
         */
        STATE_BASED_DEATH
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
     * What an event published now is a consequence of, with this collector's
     * view of the engine brought up to date first.
     *
     * <p>Called once at each entry point that classifies an event or a
     * refusal, before the open bracket is read: Forge publishes nothing when a
     * resolution starts, so the first thing it does is what opens its
     * bracket.
     */
    private Now classifyNow() {
        syncWithEngine();
        return now();
    }

    /**
     * Bring this collector's view of the engine up to date, so that what
     * {@link #now} reads off it is current: a resolution that has begun is
     * followed ({@link StackBrackets#followResolving}), and the marks of the
     * casts under way are kept ({@link CastsUnderWay#keepMarksCurrent}). The
     * two halves run under exactly the conditions {@link #now} reads them
     * under.
     */
    private void syncWithEngine() {
        if (game.getStack().isResolving()) {
            brackets.followResolving(this::idsOnStack);
        } else if (game.costPaymentStack.peek() == null) {
            castsUnderWay.keepMarksCurrent(StackActions.inProgress(game.getStack()));
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
     * <p>A pure read, of what {@link #syncWithEngine} has brought up to date.
     */
    private Now now() {
        if (game.getStack().isResolving()) {
            return new Now(
                    payingForACast(brackets.resolving()) ? Cause.PAYMENT : Cause.RESOLUTION);
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
            return new Now(brackets.isOpenCast(acting) ? Cause.CONSEQUENCE : Cause.ACTION);
        }
        return new Now(Cause.CONSEQUENCE);
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
     * <p>An event that arrives while no record is open — at a player's action
     * at priority, for the orphans, or a state-based death nothing takes — is
     * counted as dropped ({@link DroppedEvents}); an orphan only once it is
     * discarded rather than rescued at the game's end ({@link StackBrackets}).
     *
     * @param filing what kind of event it is, which narrows where it may go
     */
    private void route(EffectEvent event, String namedCause, Filing filing) {
        Now now = classifyNow();
        switch (now.cause()) {
            case PAYMENT -> {
                if (!game.getStack().isResolving()) {
                    brackets.closeForAction();
                }
            }
            // What a card's move onto the stack set off: after the cost
            // half's board, which records only the move and the costs, and
            // before the effect half's (FR-042a, FR-042b).
            case CAST -> dropped.count(event.type(), DroppedEvents.phaseOf(game));
            case ACTION -> {
                brackets.closeForAction();
                dropped.count(event.type(), DroppedEvents.phaseOf(game));
            }
            case CONSEQUENCE -> {
                if (filing != Filing.NON_COMBAT_DAMAGE && combatStep.takesEventsNow()) {
                    combatStep.file(
                            EventAttribution.stamp(event, brackets.resolving(), namedCause));
                } else if (filing == Filing.STATE_BASED_DEATH) {
                    dropped.count(event.type(), DroppedEvents.phaseOf(game));
                } else if (!turnActions.take(event, namedCause)) {
                    brackets.file(
                            EventAttribution.stamp(event, brackets.resolving(), namedCause),
                            DroppedEvents.phaseOf(game));
                }
            }
            case RESOLUTION -> brackets.file(
                    EventAttribution.stamp(event, brackets.resolving(), namedCause),
                    DroppedEvents.phaseOf(game));
        }
    }

    /**
     * Any event at all: an action at priority that has begun ends the last
     * cast's consequences even when it files nothing of its own, so the
     * state-based check after it is not the waiting spell's.
     */
    @Subscribe
    public void onAnyEvent(GameEvent event) {
        if (brackets.open() == null || game.getStack().isResolving()) {
            return;
        }
        if (classifyNow().cause() == Cause.ACTION) {
            brackets.closeForAction();
        }
    }

    /**
     * A priority: no cast is under way across one. A card still in the stack
     * zone with no stack item is a cast given up — a copy that failed its
     * targeting — and no longer counts as being cast. The step's turn-based
     * actions are over: Forge gives priority only after them.
     */
    @Subscribe
    public void onPriority(GameEventPlayerPriority event) {
        castsUnderWay.abandonAll();
        // A cast started before priority and never pushed was given up.
        pendingCast = null;
        turnActions.close();
    }

    @Override
    void onStackArrival(int cardId, String fromZone) {
        castsUnderWay.onStackArrival(cardId, fromZone);
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
        Cause cause = classifyNow().cause();
        OpenResolution open = brackets.open();
        if (open == null || !fromOpenBracket(cause) || !open.hosts(clause)) {
            return;
        }
        open.modes.begin(option, snapshots.ofResolving(open.ability));
        rebaselineAtSnapshot();
    }

    /**
     * A record that collects events has just taken its snapshot: the stat
     * differ reads the board again, so the record's stat changes are the
     * ones after it ({@link StatDiffer#rebaseline}). A change the snapshot
     * already shows that Forge had not yet published reaches no record and
     * is counted.
     */
    void rebaselineAtSnapshot() {
        for (EffectEvent unreported : rebaselineStats()) {
            dropped.count(unreported.type(), DroppedEvents.phaseOf(game));
        }
    }

    /**
     * Whether a conditional clause met its condition (FR-034), as Forge
     * decided it, filed into the half its events joined: called by the clause
     * hook when the clause's own effect is over, before a later mode's first
     * clause begins. Filed by the test {@link #beginMode} applies:
     * a consequence of the open bracket, of a clause on its card; a clause
     * of a mana ability or a special action resolves at an action and is no
     * resolution's.
     */
    void recordClauseCondition(SpellAbility clause, ConditionsMet condition) {
        if (condition == null) {
            return;
        }
        Cause cause = classifyNow().cause();
        OpenResolution open = brackets.open();
        if (open == null || !fromOpenBracket(cause) || !open.hosts(clause)) {
            return;
        }
        open.modes.noteCondition(condition);
    }

    /** The free activations that changed nothing this turn (FR-040). */
    FruitlessActivations fruitlessActivations() {
        return fruitless;
    }

    private static String actorOf(SpellAbility ability) {
        return ability == null ? null : Refs.ofNullable(ability.getActivatingPlayer());
    }

    /**
     * Drop the bracket of every item that has left the stack without
     * resolving — countered, exiled off it, swept away by an effect —
     * asked of the stack itself at every moment it can have changed: no
     * event tells such a removal from a resolution. With no bracket kept
     * there is nothing to ask.
     */
    private void reconcilePending() {
        if (brackets.isEmpty()) {
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
     * The half of {@link #reconcilePending} that has already read the stack:
     * the brackets of the items gone without resolving are removed, and what
     * they gathered is counted as dropped. Their cost halves were written at
     * their casts; their fate is the countering clause's effect half.
     *
     * @param onStack the ability ids still on the stack
     */
    void writeOffRemoved(Set<Integer> onStack) {
        brackets.removeAbsent(onStack, game == null ? null : DroppedEvents.phaseOf(game));
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
     * <p>An item still on the stack had its cost half written at its cast,
     * and its effect half never comes: nothing is written for it.
     */
    @Subscribe
    public void onGameFinished(GameEventGameFinished event) {
        finishGame(idsOnStack());
    }

    /**
     * The half of {@link #onGameFinished} that has already read the stack.
     *
     * @param onStack the ability ids the stack still holds
     */
    void finishGame(Set<Integer> onStack) {
        // A game can end in a step no priority followed: its turn-based
        // actions are written as any other step's. Nothing is held below
        // across the phase event that opened the window.
        turnActions.close();
        // Next, because it is the oldest of the rest: the resolution it
        // describes ended before anything the two below hold. The winner is not one of its
        // consequences -- player_won is not a permanent leaving the
        // battlefield -- so it stays where flushOrphanedEvents puts it, in a
        // record of its own after this one.
        onNextAction();
        combatStep.flush();
        flushOrphanedEvents();
        writeOffRemoved(onStack);
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
     * <p>Package-visible because {@link PatchedCollectors} installs it, and so a
     * test can drive it through a proxy of its own. The method name and the
     * argument positions are the whole contract with the patch —
     * {@link PatchHooks} looks both up by string — and a typo in either
     * degrades this channel silently.
     */
    PatchHooks.Listener confirmHandler() {
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
        Cause cause = classifyNow().cause();
        OpenResolution open = brackets.open();
        if (open != null && fromOpenBracket(cause)) {
            open.markDeclined();
        }
    }

    /**
     * Write a leftover resolution-bracket buffer as its own record, the
     * rescue {@link CombatStepBracket#flush} gives the combat bracket, for the
     * same reason: something can end the game between the last bracket close
     * and the next one, which never comes.
     *
     * <p>Reachable by the most ordinary kind of win. {@code MagicStack} fires
     * {@code GameEventSpellResolved} as soon as a resolving ability's effect
     * returns, so {@link #endBracket} closes the bracket before {@code
     * checkStateBasedEffects} asks whether anyone lost (CR 704.3, at the start
     * of the next main-loop iteration). The {@code GameEventGameOutcome} of a
     * burn spell, a drain, a mill-out or a scripted "you lose the game" then
     * lands in the orphans ({@link StackBrackets}), with no bracket open, and
     * no later resolution comes to flush it: without this, {@code player_won}
     * is lost for every non-combat kill.
     *
     * <p>{@code ability} is absent, as on a combat record: with no resolution
     * open there is no acting line. {@code state} is read fresh, as {@link
     * CombatStepBracket#file} reads it, since the board captured at a cast
     * went with the resolution that closed.
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
        if (!brackets.hasOrphans()) {
            return;
        }
        context.emit(context.newResolution(Moment.RESOLUTION)
                .ability(ProvenanceKey.Resolved.UNKNOWN)
                .state(snapshots.ofBoard().render())
                .payload(RecordPayloads.resolutionPayload(
                        brackets.drainOrphans(), List.of(), null)));
    }
}
