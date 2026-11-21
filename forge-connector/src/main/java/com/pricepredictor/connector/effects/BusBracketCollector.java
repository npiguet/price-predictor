package com.pricepredictor.connector.effects;

import com.google.common.eventbus.Subscribe;
import forge.game.Game;
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

import java.util.ArrayList;
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
 * card moved to the stack or the costs were paid ({@link #castHandler}), its
 * {@code targets} links read with its refs at the cast event, its modes and
 * targets chosen; it does not wait for the item's fate. The item's resolution writes its effect
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
 * {@link CastsUnderWay} the cast on its way onto the stack, {@link
 * CastTracker} a cast from its start to its cost half. What the bus
 * publishes is classified and routed by {@link EventRouter}.
 */
public final class BusBracketCollector extends BusEventReader {

    private final CollectorContext context;
    /** The game this collector records, asked whether an ability a hook hands over is its own. */
    private final LiveGame live;
    private final SnapshotBuilder snapshots;

    /** What this game's records never took, by event type and engine phase. */
    private final DroppedEvents dropped = new DroppedEvents();
    /** The counters and life each payment under way has moved, until its cast event reads them. */
    private final PaymentLedger payments;

    /** One bracket per stack item this collector saw cast, and the open one. */
    private final StackBrackets brackets = new StackBrackets(dropped);

    /** The game's outcome no resolution or combat took, for the game-end record. */
    private final List<EffectEvent> gameEnd = new ArrayList<>();
    /** Whether the game's outcome has been filed ({@link #fileGameOutcome}). */
    private boolean outcomeFiled;

    /** The cast on its way onto the stack, if one is, and what it set off ({@link EventRouter.Cause#CAST}). */
    private final CastsUnderWay castsUnderWay;

    /** The resolution waiting for its consequences, if one is. */
    private final HeldResolutionWindow heldResolution;
    /** The free activations that changed nothing this turn, for the random seat. */
    private final FruitlessActivations fruitless;

    /** What an event published now is a consequence of, and the record it joins. */
    private final EventRouter router;

    /** The cast from its start to its cost half. */
    private final CastTracker castTracker;

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
        this.live = new LiveGame(game);
        this.snapshots = live.snapshots();
        this.fruitless = new FruitlessActivations(game);
        this.payments = new PaymentLedger(game);
        this.heldResolution = new HeldResolutionWindow(context, fruitless);
        this.castsUnderWay = new CastsUnderWay(game, brackets::holds);
        this.castTracker = new CastTracker(context, snapshots, payments, castsUnderWay);
        this.combatStep = new CombatStepBracket(
                game, context, snapshots, heldResolution, onCombatRecord);
        this.turnActions = new TurnActionWindow(game, context, snapshots);
        this.router = new EventRouter(new EventRouter.Collaborators(
                game, brackets, castsUnderWay, payments, dropped, combatStep, turnActions,
                this::payingForACast));
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
    Listener castHandler() {
        return PatchHooks.CAST_LISTENER.listener((callback, args) -> {
            SpellAbility cast = args.get(0, SpellAbility.class);
            if (cast != null && live.ownsStackAbility(cast)) {
                castStarting(cast);
            }
        });
    }

    /**
     * The half of {@link #castHandler} past the game check: read the board
     * the cost half describes. Starting a cast at priority is the next
     * action, so the held resolution's consequences are over; one started by
     * a resolving ability (cascade, a "you may cast") is part of that
     * resolution and ends nothing, and leaves the stat differ alone: the
     * resolving item's bracket keeps its events, measured from its own
     * snapshot, and the cast spell's effect half reads the board again when
     * its resolution starts.
     */
    void castStarting(SpellAbility cast) {
        boolean resolving = game.getStack().isResolving();
        if (!inStateBasedLoop && !resolving) {
            onNextAction();
        }
        castTracker.started(cast);
        if (!resolving) {
            rebaselineAtSnapshot();
        }
    }

    /**
     * The resolution hook's listener: the stack's top item starts to resolve
     * (FR-042b). Its effect half's board is read here, before its first
     * clause. Another game's item is not this game's: a forced
     * resolution's is its fork-bound recording's ({@link ForcedResolution}).
     */
    Listener resolutionHandler() {
        return PatchHooks.RESOLUTION_LISTENER.listener((callback, args) -> {
            SpellAbility item = args.get(0, SpellAbility.class);
            if (item == null) {
                return;
            }
            if (live.ownsStackAbility(item)) {
                resolutionStarting(item, Boolean.TRUE.equals(args.get(1)), args.optional(2));
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
     *
     * <p>A modal item that fizzled as a whole resolves none of its clauses,
     * so no mode's first clause will cut its halves: each chosen mode's half
     * opens and closes here, on the board read now, and stays empty
     * (FR-029c). A degraded worker, never told this moment, keeps its one
     * root half (FR-029e).
     *
     * @param fizzled whether Forge found no legal target left, in which case
     *                none of the item's clauses will run
     * @param targetsAtCast each link's targets before Forge's check for
     *                illegal targets, root first (null where the hook could
     *                not copy them, or an older Forge did not pass them)
     */
    void resolutionStarting(SpellAbility item, boolean fizzled, Object targetsAtCast) {
        router.setStepTakenOver(false);
        if (!inStateBasedLoop) {
            onNextAction();
        }
        OpenResolution resolving = brackets.startResolving(AbilityId.of(item));
        if (resolving == null) {
            return;
        }
        brackets.countDropped(resolving, DroppedEvents.phaseOf(game));
        ChainTargets atCast = ChainTargets.atCast(item, targetsAtCast, resolving.allocation());
        Snapshot board = snapshots.ofResolving(resolving.ability, atCast);
        resolving.resolutionStarted(board, ChainTargets.now(item), atCast);
        if (fizzled) {
            beginFizzledModes(resolving, item);
        }
        rebaselineAtSnapshot();
    }

    /**
     * A fizzled charm's halves, one per chosen mode, none of whose clauses
     * resolves: each on the board its resolution started on, linking its
     * own mode's targets as cast, as a resolving mode's half does.
     */
    private void beginFizzledModes(OpenResolution resolving, SpellAbility item) {
        for (Integer option : CharmModes.chosenOptions(item)) {
            resolving.modes.begin(option,
                    snapshots.ofResolving(resolving.ability, resolving.castTargetsOfMode(option)));
        }
        resolving.modes.close();
    }

    /**
     * Everything a cast does once the ability has been found on the stack:
     * end the previous action's consequences, place the item's bracket, and
     * write its cost half.
     *
     * <p>Split from {@link #onCast} at the one line that reads the stack, so the
     * rules that decide what a cost record says can be exercised against a
     * real ability without a game in progress.
     */
    void beginBracket(SpellAbility ability) {
        endPreviousConsequences();
        if (ability != null) {
            brackets.writeOff(AbilityId.of(ability));
        }
        CastTracker.PendingCast started = castTracker.startedCast(ability);
        OpenResolution cast = ability == null ? null : castTracker.openResolution(ability, started);
        placeBracket(cast, started);
        writeCostHalf(ability, cast, started);
    }

    /**
     * The first step of {@link #beginBracket}: an item reaching the stack
     * ends whatever the last action set off, unless the state-based loop or
     * a finishing resolution is still running.
     */
    private void endPreviousConsequences() {
        if (inStateBasedLoop) {
            // A trigger reaching the stack between two passes of the
            // state-based check (FR-028). It opens its own bracket next, and
            // the check goes on: the held half and the damage step's record
            // take its deaths until the loop's end signal.
            stackUsedInLoop = true;
        } else if (brackets.finishing()) {
            // A copy the resolution made while it froze the stack, pushed as
            // Forge finishes that resolution (Chain of Acid's, a "may copy"):
            // no player acted, and the state-based check still to come is
            // the resolution's, whose deaths its held half takes.
        } else {
            // A cast, an activation or a trigger reaching the stack is the
            // next action: whatever the held resolution killed has died by
            // now.
            onNextAction();
            // And whatever the combat damage killed: state-based actions run
            // before anything goes on the stack.
            combatStep.closeWindow();
        }
    }

    /**
     * The second step of {@link #beginBracket}: the item's bracket follows
     * the resolving ability that cast it, or opens as the cast's own.
     *
     * @param cast    the item's bracket, or null for an item none is opened for
     * @param started the cast's start the cast hook told, or null
     */
    private void placeBracket(OpenResolution cast, CastTracker.PendingCast started) {
        if (game.getStack().isResolving()) {
            // Cast by the resolving ability itself, which keeps the bracket
            // until it ends (StackBrackets).
            followResolving(cast);
        } else {
            openCast(cast, started);
        }
    }

    private void followResolving(OpenResolution cast) {
        brackets.followResolving(router::idsOnStack);
        if (cast != null) {
            brackets.waitForResolution(cast);
        }
    }

    private void openCast(OpenResolution cast, CastTracker.PendingCast started) {
        if (measuredFromTheCastEvent(cast, started)) {
            // Its board was read just now, at the cast event, where no
            // hook told the cast's start: its stat changes are measured
            // from here. Not while an ability resolves, whose bracket
            // keeps the events, and whose snapshot is older; nor for a
            // trigger the state-based loop puts on the stack between
            // two passes, where the loop has applied statics it
            // publishes only at its end, which a damage step's record
            // still takes (FR-028): the trigger's effect half reads the
            // board again when its resolution starts.
            rebaselineAtSnapshot();
        }
        brackets.openCast(cast);
        // Any other card still on its way was given up: one cast at a time.
        castsUnderWay.abandonAll();
    }

    /** Whether a cast opened outside a resolution measures its stat changes from its cast event. */
    private boolean measuredFromTheCastEvent(OpenResolution cast, CastTracker.PendingCast started) {
        return cast != null && started == null && !inStateBasedLoop;
    }

    /**
     * The third step of {@link #beginBracket}: a cast's or an activation's
     * cost half, and a free activation noted as possibly fruitless. An item
     * nobody cast (a trigger, a copy) has no link and writes none.
     */
    private void writeCostHalf(SpellAbility ability, OpenResolution cast, CastTracker.PendingCast started) {
        if (cast == null || cast.linkId == null) {
            return;
        }
        castTracker.writeCostHalf(cast, started);
        if (ability.isActivatedAbility() && cast.costs.paidNothingKept()) {
            // Until it resolves, nobody knows whether it changes anything, and
            // the random seat, which keeps priority, would stack another.
            fruitless.pending(ability, cast.keys.key());
        }
    }

    /**
     * A resolution closes the bracket and writes or holds its effect half,
     * which carries the item's {@code outcome} (FR-042b).
     *
     * <p>A fizzled or declined item did nothing: its effect half is written
     * now, with no events. Any other is held until the deaths it causes have
     * happened ({@link HeldResolutionWindow}), and a spell's half takes its
     * card's move off the stack, which comes after the resolution is
     * published. What a static does because of the last clause (an Equip's
     * +2/+0) is already in the bracket: the patched branch publishes the
     * resolution after the static check that follows it.
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
        router.setStepTakenOver(false);
        // Another resolution has run since the held one ended, so the held
        // one's state-based check is over: what dies after this is this
        // resolution's doing, or a mix nobody can untangle.
        onNextAction();
        // An item this collector never saw cast has no bracket, and no
        // effect half.
        OpenResolution resolved = brackets.closeAfterResolution(new AbilityId(abilityId));
        if (resolved == null) {
            return;
        }
        // The last mode's half closes with the resolution, since no next
        // mode's first clause comes to close it.
        resolved.modes.close();
        ActivationOutcome outcome = resolved.outcome(fizzled && !resolvesAsACreature(resolved.ability));
        if (outcome == ActivationOutcome.FIZZLED || outcome == ActivationOutcome.DECLINED) {
            heldResolution.writeEmpty(resolved, outcome, dropped,
                    game == null ? DroppedEvents.NO_PHASE : DroppedEvents.phaseOf(game));
            return;
        }
        // Held rather than written: the deaths this resolution causes have
        // not happened yet. An activation that paid nothing the game keeps is
        // one the random seat could repeat at will (research C9). A
        // resolution that ended the game ("you lose the game") is held too:
        // its card's move off the stack is still to come, and the game's end
        // writes the half (finishGame).
        boolean free = resolved.costs != null && resolved.ability.isActivatedAbility()
                && resolved.costs.paidNothingKept();
        heldResolution.hold(resolved, outcome, free);
    }

    /**
     * Whether an item Forge reports as fizzled still resolved: a bestowed
     * Aura spell or a mutating creature spell whose target became illegal
     * stops being one and resolves as a creature spell (702.103e,
     * 702.140b; {@code MagicStack.resolveStack}), its permanent entering the
     * battlefield. Its effect half is a resolution's, held with its events.
     */
    static boolean resolvesAsACreature(SpellAbility ability) {
        return ability.isBestow() || ability.isMutate();
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
        if (game.getStack().isResolving()) {
            // A step begun by the resolving effect itself: what follows is
            // the step's, not the effect's.
            router.setStepTakenOver(true);
        }
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
        castTracker.forgetStarted();
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
        if (game.getStack().isResolving() && endsTheStep(PatchHooks.currentSubAbility())) {
            // The effect's own check is the last thing it does itself; the
            // step it ends, and the one it skips to, follow.
            router.setStepTakenOver(true);
        }
        onNextAction();
        if (stackUsedInLoop) {
            combatStep.closeWindow();
        }
        stackUsedInLoop = false;
    }

    /** Whether a resolving clause ends the turn or the combat phase. */
    private static boolean endsTheStep(Object clause) {
        return clause instanceof SpellAbility ending && ending.getApi() != null
                && (ending.getApi() == forge.game.ability.ApiType.EndTurn
                        || ending.getApi() == forge.game.ability.ApiType.EndCombatPhase);
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
        castTracker.forgetStarted();
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
            router.route(damage, namedCause, EventRouter.Filing.NON_COMBAT_DAMAGE);
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
            router.route(moved, null, EventRouter.Filing.STATE_BASED_DEATH);
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
     * the rest as deaths. Every other event is routed by {@link EventRouter#classifyNow}, a
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
        fileOutcome(event, namedCause, EventRouter.Filing.OUTCOME);
    }

    /**
     * The game's outcome, {@code player_lost} and {@code player_won}, has a
     * filing of its own and is never an open cast's: a cast whose own cost
     * ended the game (life paid to 0) never resolves, and its bracket is
     * never written. The resolving item's half takes it when a resolution
     * ended the game ({@code LosesGame}, {@code WinsGame}); else the combat
     * record, while a damage step's window is open; else the game-end
     * record ({@link #writeGameEnd}). Never a tap's, and never dropped.
     */
    @Override
    void fileGameOutcome(EffectEvent event) {
        outcomeFiled = true;
        if (router.classifyNow().cause() == EventRouter.Cause.RESOLUTION && resolvingTakesTheOutcome()) {
            brackets.file(EventAttribution.stamp(event, brackets.resolving(), null),
                    DroppedEvents.phaseOf(game));
        } else if (combatStep.takesEventsNow()) {
            combatStep.file(EventAttribution.stampAt(event, null, null, null));
        } else {
            gameEnd.add(EventAttribution.stampAt(event, null, null, null));
        }
    }

    /** Whether a resolution is open to take the outcome as its own, its step not handed on. */
    private boolean resolvingTakesTheOutcome() {
        return brackets.open() != null && !router.stepTakenOver();
    }

    /**
     * Whether the game's outcome, if one was filed, is in a record written:
     * the game-end marker is written only then.
     */
    boolean outcomeIsWritten() {
        return !outcomeFiled || context.wroteTheOutcome();
    }

    private void fileOutcome(EffectEvent event, String namedCause, EventRouter.Filing filing) {
        if (detachedByTheHeldCheck(event)) {
            // An Aura or Equipment the state-based check takes off a
            // permanent the held resolution killed or changed: part of that
            // check's moves, as the Aura's own move to the graveyard is,
            // filed with the half that names the permanent it left.
            Object host = event.param(EventParam.DETACHED_FROM);
            heldResolution.fileConsequence(
                    event, host instanceof String left ? left : event.subjects().get(0));
            return;
        }
        router.route(event, namedCause, filing);
    }

    /**
     * Whether {@code event} is an unattachment, naming the permanent left,
     * made by the state-based check while a resolution's half is held. The
     * check is told as fileMove tells it: the loop's signals, or a frozen
     * tracker without them.
     */
    private boolean detachedByTheHeldCheck(EffectEvent event) {
        return event.type() == EventType.UNATTACHED && heldResolution.isHolding()
                && (inStateBasedLoop || game.getTracker().isFrozen())
                && !event.subjects().isEmpty();
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
            fileOutcome(event, null, EventRouter.Filing.CLAUSE);
        }
    }

    /** Ask {@code tap} first, before routing any event ({@link EventRouter#addTap}). */
    void addTap(EventRouter.EventTap tap) {
        router.addTap(tap);
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
        if (router.classifyNow().cause() == EventRouter.Cause.ACTION) {
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
        router.setStepTakenOver(false);
        brackets.finished();
        castsUnderWay.abandonAll();
        // A cast started before priority and never pushed was given up.
        castTracker.forgetStarted();
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
     * clause of a player's action at priority ({@link EventRouter.Cause#ACTION}): a mana
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
        EventRouter.Cause cause = router.classifyNow().cause();
        OpenResolution open = brackets.open();
        if (!cutsTheOpenBracket(open, cause, clause)) {
            return;
        }
        open.modes.begin(option, snapshots.ofResolving(open.ability, open.castTargetsOfMode(option)));
        rebaselineAtSnapshot();
    }

    /** Whether a mode's clause, resolving for {@code cause}, is the open bracket's own. */
    private static boolean cutsTheOpenBracket(
            OpenResolution open, EventRouter.Cause cause, SpellAbility clause) {
        return open != null && EventRouter.fromOpenBracket(cause) && open.hosts(clause);
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
     * clause begins. Filed when it is a consequence of the open bracket, of
     * a clause on its own line ({@link OpenResolution#onItsLine}): a clause
     * another line of the card resolves meanwhile, an enters-the-battlefield
     * replacement's, is not the resolution's; a clause of a mana ability or
     * a special action resolves at an action and is no resolution's.
     */
    void recordClauseCondition(SpellAbility clause, ConditionsMet condition) {
        if (condition == null) {
            return;
        }
        OpenResolution open = openOnTheLineOf(clause);
        if (open != null) {
            open.modes.noteCondition(condition);
        }
    }

    /**
     * A clause's own effect ran: on the open bracket's line, as {@link
     * #recordClauseCondition} reads it, it is the last clause that ran so
     * far, the one what the static check after the resolution's last clause
     * applies is attributed to. A clause whose condition failed or whose
     * unless cost decided against it did nothing, and is not told.
     */
    void recordClauseRan(SpellAbility clause) {
        OpenResolution open = openOnTheLineOf(clause);
        if (open != null) {
            open.noteClauseRan(clause);
        }
    }

    /** The open bracket, when what happens now is its own and the clause is on its line; else null. */
    private OpenResolution openOnTheLineOf(SpellAbility clause) {
        boolean itsOwn = EventRouter.fromOpenBracket(router.classifyNow().cause());
        OpenResolution open = brackets.open();
        if (!itsOwn || open == null) {
            return null;
        }
        return open.onItsLine(clause) ? open : null;
    }

    /** The free activations that changed nothing this turn (FR-040). */
    FruitlessActivations fruitlessActivations() {
        return fruitless;
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
        writeOffRemoved(router.idsOnStack());
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
        brackets.removeAbsent(onStack, game == null ? null : DroppedEvents.phaseOf(game),
                game != null && game.getStack().isResolving());
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
     * <p>The <b>held half</b> of a resolution that ended the game ("you
     * lose the game") is written here, with its card's move off the stack:
     * no next action or state-based loop follows it.
     *
     * <p>The <b>game's outcome</b>, when no resolution or combat took it
     * (the ordinary case: burn, drain, decking out, a life cost paid to 0),
     * is the game-end record's ({@link #writeGameEnd}).
     *
     * <p>An item still on the stack had its cost half written at its cast,
     * and its effect half never comes: nothing is written for it, and what
     * its bracket gathered is counted as dropped.
     */
    @Subscribe
    public void onGameFinished(GameEventGameFinished event) {
        finishGame(router.idsOnStack());
    }

    /**
     * A fork played on has stopped: at its turn's end, or at the game's end,
     * which Forge publishes only from a match and a fork has none (C1).
     *
     * <p>At the game's end it writes what the game's end writes: the held
     * half, the lethal damage step's record and the outcome. At the turn's
     * end, the last step's turn-based actions and the held half, which no
     * next step or action will close.
     */
    void stopPlaying() {
        if (game.isGameOver()) {
            finishGame(router.idsOnStack());
            return;
        }
        turnActions.close();
        onNextAction();
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
        // describes ended before anything the two below hold.
        onNextAction();
        combatStep.flush();
        writeGameEnd();
        writeOffRemoved(onStack);
        // An item still on the stack kept its bracket above: what it gathered
        // (a cast whose cost ended the game) reaches no record either.
        brackets.dropAll(game == null ? null : DroppedEvents.phaseOf(game));
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
    Listener confirmHandler() {
        return PatchHooks.CONFIRM_LISTENER.listener((callback, args) -> {
            // This hook is a JVM static, and PlayerControllerAi.confirmAction
            // is exactly what a fork's own forced resolution asks when it
            // pays a cost or resolves an optional clause (GameSimulator builds
            // its own PlayerControllerAi over the forked Game, but the static
            // listener it calls is this game's). args[0] is declared
            // SpellAbility on Forge's side. A refusal of another game's
            // ability, and one naming no ability while a fork that is not
            // this collector's game runs on this thread, is not this
            // game's: a forced resolution's is its fork-bound recording's.
            if (!Boolean.FALSE.equals(args.get(1))) {
                return;
            }
            SpellAbility offered = args.get(0, SpellAbility.class);
            boolean ours = offered != null ? live.owns(offered) : live.ownsNullAbilityRefusal();
            if (ours) {
                noteDeclined();
            }
        });
    }

    /**
     * Record a refusal against the open bracket, if there is one and the
     * refusal is a consequence of it ({@link EventRouter.Cause#RESOLUTION} or {@link
     * EventRouter.Cause#CONSEQUENCE}). One made paying for something else is that other
     * ability's; one made announcing a cast, or in an action at priority, is
     * no resolution's.
     */
    void noteDeclined() {
        EventRouter.Cause cause = router.classifyNow().cause();
        OpenResolution open = brackets.open();
        if (open != null && EventRouter.fromOpenBracket(cause)) {
            open.markDeclined();
        }
    }

    /**
     * Write the game-end record: the game's outcome that no resolution or
     * combat took ({@link #fileGameOutcome}), and nothing else. Every orphan
     * still kept — an attack's taps, a land played, a stat change since the
     * last cast or resolution — is counted as dropped: none is the outcome's
     * doing, and the board is read after them.
     *
     * <p>{@code ability} is the empty array with no {@code ability_unresolved}:
     * no resolution is open, so the record seeks no acting line. {@code state}
     * is read now, the board the game ended on.
     */
    private void writeGameEnd() {
        brackets.discardOrphans();
        if (gameEnd.isEmpty()) {
            return;
        }
        context.emit(context.newResolution(Moment.RESOLUTION)
                .seeksNoLine()
                .state(snapshots.ofBoard().state())
                .payload(RecordPayloads.resolutionPayload(List.copyOf(gameEnd), List.of(), null)));
        gameEnd.clear();
    }
}
