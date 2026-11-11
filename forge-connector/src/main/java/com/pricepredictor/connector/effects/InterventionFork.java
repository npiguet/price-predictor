package com.pricepredictor.connector.effects;

import com.google.common.eventbus.Subscribe;
import forge.ai.simulation.GameSimulator;
import forge.game.Game;
import forge.game.event.GameEventSpellAbilityCast;
import forge.game.event.GameEventSpellResolved;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

import java.util.List;
import java.util.Random;
import java.util.function.Supplier;

/**
 * An <b>interventional resolution</b>: a fork that forces an ability Forge's
 * AI never chose to use.
 *
 * <p>Some abilities are never cast in any game the AI plays — it judges them
 * bad, or cannot afford them, or never draws them into a board where they do
 * anything — and no amount of observation will produce a record for them. The
 * fork resolves the ability with chosen targets and modes, routing an
 * unaffordable candidate through the play-without-paying path, and writes the
 * <b>effect half only</b>: there was no real activation, so there is no cost
 * half to link to.
 */
final class InterventionFork {

    private final Game game;
    private final CollectorContext context;
    private final ForkBudget budget;
    /** The draws a forced resolution makes, from the game's seed. */
    private final RandomChoices choices;

    InterventionFork(Game game, CollectorContext context, ForkBudget budget, long seed) {
        this.game = game;
        this.context = context;
        this.budget = budget;
        // Seeded so a rerun of a game forces the same choices. Scrambled as
        // the sampler's seed is: games are seeded consecutively, and the first
        // draw here is a mode or target count, which an unscrambled seed would
        // make nearly the same in every game.
        this.choices = new RandomChoices(new Random(GameRecording.scramble(seed)));
    }

    /** Whether the budget lets {@link #intervene} take another fork of this resolution. */
    boolean mayIntervene(int realResolutionId) {
        return budget.mayIntervene(realResolutionId);
    }

    /**
     * Fork the game, force {@code ability} to resolve, and record the effect.
     *
     * @param realResolutionId the real resolution this fork varies, for the
     *                         per-resolution cap
     * @return whether a record was written
     */
    boolean intervene(SpellAbility ability, int realResolutionId) {
        return intervene(ability, realResolutionId, () -> ability.getActivatingPlayer() != null
                ? ability.getActivatingPlayer()
                : game.getPhaseHandler().getPlayerTurn());
    }

    /**
     * The same, the ability resolved for {@code holder}: the player whose hand
     * holds it, who need not be the player whose turn it is (FR-057).
     */
    boolean intervene(SpellAbility ability, int realResolutionId, Player holder) {
        return intervene(ability, realResolutionId, () -> holder);
    }

    private boolean intervene(SpellAbility ability, int realResolutionId, Supplier<Player> perspective) {
        if (!budget.mayIntervene(realResolutionId)) {
            return false;
        }
        Forks.ForkRun run = new Forks.ForkRun(
                game, () -> budget.spendIntervention(realResolutionId), perspective);
        Boolean written = run.take((verified, snapshots) -> {
            SpellAbility forked = Forks.mappedAbility(verified.copier(), ability);
            Player actor = forked == null
                    ? null : Forks.mappedPlayer(verified.copier(), verified.perspective());
            if (forked == null || actor == null) {
                return null;
            }
            ForcedResolution resolution = forceResolution(verified.fork(), forked, actor, snapshots);
            // An empty event list is not an observation. "The ability resolved
            // and did nothing" and "the ability never got to act" render the
            // same record, and nothing in it tells them apart. A
            // counterfactual nobody can read is worse than one that was never
            // written, so this drops it.
            return resolution == null || resolution.isEmpty() ? null : resolution;
        }, (verified, resolution) -> {
            write(resolution, ability, verified.perspective());
            return Boolean.TRUE;
        });
        return Boolean.TRUE.equals(written);
    }

    /**
     * Write a forced resolution's effect halves.
     *
     * @param perspective the live player the fork acted for
     */
    private void write(ForcedResolution resolution, SpellAbility ability, Player perspective) {
        // The *live* ability's keys, not the fork's. A provenance key
        // names a printed line, which the copy shares — but keying off
        // the copy would make the record's identity depend on a game
        // that no longer exists.
        ProvenanceKey.Resolved keys = ProvenanceResolver.resolve(ability);
        String actorId = Refs.of(perspective);
        // A modal resolution writes one half per chosen mode, by the same
        // rule the live bracket applies (FR-029f), each through the mode's
        // option line and with the board as it stood when that mode's
        // first clause began. Events before the first mode's clause — the
        // charm's own, of which there are none — ride with the first.
        for (HeldHalves.Half<ModeSplit.ModeHalf> half : resolution.halves()) {
            context.emit(forkedHalf(
                    actorId, keys.withOption(half.about().mode().option()),
                    resolution.render(half.about().mode().state(), half.events()),
                    half.events(), half.about().conditions(), resolution.outcome()));
        }
    }

    /**
     * One interventional effect half. No {@code link_id}: it is written alone,
     * because there was no real activation to pair it with.
     */
    private EffectRecord forkedHalf(
            String actorId, ProvenanceKey.Resolved keys, String state,
            List<EffectEvent> events, List<ConditionsMet> conditions,
            ActivationOutcome outcome) {
        return context.newResolution(Moment.RESOLUTION)
                .interventional(true)
                .fork(true)
                .actor(actorId)
                .ability(keys)
                .state(state)
                .payload(RecordPayloads.resolutionPayload(events, conditions, outcome));
    }

    /**
     * What a forced resolution produced: its effect halves, one per chosen
     * mode or one for the whole resolution, each with the board it is read
     * against and the deaths it left behind ({@link ForkEventSink#halves}).
     *
     * @param refs    the refs block read as the forced ability finished
     *                resolving, where the live bracket re-reads it, or null
     *                when the fork never saw it finish
     * @param outcome what became of it, as a live effect half says
     *                (FR-042b): resolved or partially fizzled; a fizzled
     *                one did nothing and is never written
     */
    private record ForcedResolution(
            String refs, List<HeldHalves.Half<ModeSplit.ModeHalf>> halves,
            ActivationOutcome outcome) {

        /**
         * A board this resolution read, paired with its refs as the live
         * bracket pairs them: the half's random results in the refs, and the
         * library cards its events name among the entities.
         */
        String render(Snapshot board, List<EffectEvent> events) {
            Snapshot read = refs == null ? board : board.withRefs(RefsJson.withRandom(refs, events));
            return read.withEventCards(events).render();
        }

        /** Whether no half holds an event: the forced ability did nothing observable. */
        boolean isEmpty() {
            for (HeldHalves.Half<ModeSplit.ModeHalf> half : halves) {
                if (!half.events().isEmpty()) {
                    return false;
                }
            }
            return true;
        }
    }

    /**
     * Put the ability on the fork's stack and resolve it, collecting what it did.
     *
     * <p>Placed on the stack directly rather than played through the AI's cost
     * machinery, because <b>the cost is the reason the record is missing</b>. An
     * ability the AI never used is usually one it could never afford, and
     * routing the fork through cost payment fails on exactly the population the
     * intervention exists to reach — leaving lands and other free plays, which
     * observation already covers. The corpus wants what the effect does, and
     * the cost half is recorded from real activations elsewhere.
     *
     * <p>Placed through {@code MagicStack.addAndUnfreeze}, the call a real cast
     * ends in, so a spell's card moves to the stack zone first, as casting
     * moves it before any cost is paid. Its resolution then moves the card
     * from the stack, as an observed one does, rather than from the hand.
     *
     * <p>The state is read where the live bracket reads an observed one's
     * ({@link ForcedCast}): the board as the forced ability's resolution
     * starts (FR-042b), told by the resolution hook, or the board at its
     * cast event where no hook tells it; and the refs block re-read as it
     * finishes resolving. Two differences remain, both from the costs the fork does
     * not pay. The board still holds what a cost would have used up: the
     * mana sources a payment would have tapped, the mana it would have
     * spent from the pool, and anything sacrificed, discarded, exiled or
     * tapped as a cost. And nothing was paid into the ability, so a
     * resolution that reads what was paid reads nothing.
     *
     * <p>Resolution drains the whole stack: the cast's own triggers above the
     * forced ability, and the triggers its effect causes after it. Neither is
     * the forced ability's record live, where each has a bracket of its own,
     * so the sink files what live would put in the forced ability's effect
     * half ({@link ForkEventSink}): its resolution, its spell card's move off
     * the stack, and the deaths of the check that follows it.
     *
     * <p>The modes, targets and X are drawn at random from the legal options,
     * out of the fork's own seeded source, rather than through the AI. That
     * is the same decision the cost bypass makes and for the same reason: the
     * corpus wants what the effect <b>does</b>, and asking the AI which mode
     * is good would reproduce exactly the selection bias the intervention
     * exists to escape. Seeded so both branches of a probe and a rerun of a
     * game make the same choice. Choosing none at all is not an option: a
     * Lava Axe pushed with no target resolves into nothing.
     *
     * @return what resolved, one half per chosen mode; every half empty if the
     *         ability resolved and did nothing observable; null if it could
     *         not be resolved at all
     */
    private ForcedResolution forceResolution(
            Game fork, SpellAbility ability, Player actor, SnapshotBuilder snapshots) {
        // Not guarded here: a forced resolution reaches states ordinary play
        // does not, and Forks.ForkRun discards the fork when a card throws.
        if (!drawForcedChoices(ability, actor)) {
            return null;
        }
        fork.copyLastState();
        ForkEventSink sink = ForkEventSink.forCast(fork, ability);
        ForcedCast cast = new ForcedCast(fork, snapshots, sink::rebaselineStats);
        SpellAbility onStack = pushForcedCast(fork, ability, cast, sink);
        if (onStack == null) {
            // Nothing reached the stack. Mostly a special action:
            // foretell, plot and suspend are AbilityStatic, which
            // MagicStack.add resolves on the spot without a push, and
            // live opens no bracket for one either, so there is no
            // resolution record to write. Otherwise a frozen stack parked
            // it or its targeting was refused, and nothing will resolve.
            return null;
        }
        // The ability on the stack is the sink's root from here, as the
        // live bracket's root is the stack's: an event's attributed_to is
        // read against the ability the bracket belongs to, and a fork's
        // bracket is exactly this one forced resolution.
        sink.castAs(onStack, cast.state());
        // The forced ability's own moments, told by the hooks the live
        // handlers hand a fork's callbacks on to (Forks.forkHooks): its
        // resolution's start, where its board is read (FR-042b); a mode's
        // first clause, the boundary between per-mode halves and the moment
        // its snapshot is taken (FR-029b, FR-029f); each conditional clause's
        // answer (FR-034); and the mana its own effect makes (C3).
        //
        // Only this ability's own modes and clauses count: the live clause
        // handler forwards every clause it drops, and anything else resolving
        // on the fork's stack may be modal too. Neither the host card nor the
        // key can tell them apart: a creature spell's own modal enters
        // trigger shares its host, and a copy of the forced charm a cast
        // trigger made (Double Vision) shares its key too, so its modes
        // would cut halves the forced charm's events never reach. The mode's
        // charm must be the very ability on the stack, the one this record
        // is about, and a clause or a producer its chain.
        boolean[] fizzled = {false};
        Forks.ForkHooks hooks = new Forks.ForkHooks() {
            @Override
            public void resolutionStarting(SpellAbility item, boolean itemFizzled) {
                if (item.getId() == onStack.getId()) {
                    fizzled[0] = itemFizzled;
                    sink.resolutionStarted(snapshots.ofResolving(onStack));
                }
            }

            @Override
            public void clauseResolving(SpellAbility clause) {
                CharmModes.ModeStart start = CharmModes.modeStartOf(clause);
                if (start != null && start.charm() == onStack) {
                    sink.beginMode(start.option(), snapshots.ofResolving(onStack));
                }
            }

            @Override
            public void clauseResolved(SpellAbility clause, String run) {
                if (clause.getRootAbility() == onStack) {
                    sink.noteCondition(ConditionsMet.of(clause, run));
                }
            }

            @Override
            public void manaProduced(SpellAbility producer, EffectEvent produced) {
                if (producer.getRootAbility() == onStack) {
                    sink.manaProduced(produced);
                }
            }
        };
        Forks.listeningForHooks(hooks, () -> {
            GameSimulator.resolveStack(fork, actor.getWeakestOpponent());
            return null;
        });
        ActivationOutcome outcome = fizzled[0] ? ActivationOutcome.FIZZLED
                : OpenResolution.outcomeOf(cast.targetsAtCast(), onStack);
        return new ForcedResolution(cast.refs(), sink.halves(), outcome);
    }

    /**
     * Make the forced ability's choices for {@code actor}: its modes, its
     * targets and its X, drawn at random from the legal options.
     *
     * @return whether a mode and a target could be chosen wherever the
     *         ability needs one
     */
    private boolean drawForcedChoices(SpellAbility ability, Player actor) {
        ability.setActivatingPlayer(actor);
        if (!choices.chooseModes(ability) || !choices.chooseTargets(ability)) {
            return false;
        }
        // Floor one: an X of zero resolves into the same silence a
        // missing target does, and the fork never pays, so the leftover mana of
        // the forked actor is what bounds the draw above it.
        choices.announceX(ability, actor, 1);
        return true;
    }

    /**
     * Put the forced ability on the fork's stack, with {@code cast} and
     * {@code sink} listening from before it moves.
     *
     * <p>Listening from before the cast: what the card's move onto the stack
     * and the push set off is the cast's own, as live gives it to the cast's
     * bracket ({@code ForkEventSink}'s {@code PRE_CAST}). The sink was primed
     * when it was built, so a stat change the move makes has a reading to
     * differ from.
     *
     * @return the ability the cast put on the stack, or null when nothing
     *         reached it
     */
    private static SpellAbility pushForcedCast(
            Game fork, SpellAbility ability, ForcedCast cast, ForkEventSink sink) {
        fork.subscribeToEvents(cast);
        fork.subscribeToEvents(sink);
        fork.getStack().addAndUnfreeze(ability);
        return cast.onStack();
    }

    /**
     * The forced ability's two moments on the fork's bus: its cast, which puts
     * it on the stack, and the end of its resolution.
     *
     * <p>The board read at the cast is the effect half's only where no
     * resolution hook tells the resolution's start, which is where the live
     * bracket reads its board (FR-042b). The live bracket re-reads the refs block when
     * the ability finishes resolving, because modes, X and a named card are
     * set during the resolution. A state read anywhere else differs from an
     * observed one in exactly those fields, every time, and a difference that
     * regular would let the model tell a forced record from an observed one
     * although {@code interventional} never reaches it.
     */
    private static final class ForcedCast {
        private final Game fork;
        private final SnapshotBuilder snapshots;
        private SpellAbility onStack;
        private Snapshot state;
        private String refs;
        /** The X the forced ability announced, read at its cast. */
        private Integer x;
        private int targetsAtCast;
        /**
         * Re-reads the sink's stat differ once the board is taken, so the
         * forced bracket's stat changes are the ones after its snapshot, as
         * the live bracket's are.
         */
        private final Runnable atSnapshot;

        ForcedCast(Game fork, SnapshotBuilder snapshots, Runnable atSnapshot) {
            this.fork = fork;
            this.snapshots = snapshots;
            this.atSnapshot = atSnapshot;
        }

        /**
         * The first cast onto the fork's stack is the forced one: the copy is
         * taken at a phase boundary with nothing in flight, and anything the
         * cast triggers reaches the stack only once the fork resolves.
         */
        @Subscribe
        public void onCast(GameEventSpellAbilityCast event) {
            if (onStack != null) {
                return;
            }
            // Read off the stack, as the live bracket reads it: an activated
            // ability goes on as a copy of the line it was handed.
            onStack = StackActions.topAbility(fork.getStack());
            if (onStack != null) {
                // Read once, cast and paid for, as the live bracket reads it (B3).
                x = RefsJson.announcedX(onStack);
                state = snapshots.ofResolving(onStack);
                targetsAtCast = OpenResolution.targetCount(onStack);
                atSnapshot.run();
            }
        }

        @Subscribe
        public void onResolved(GameEventSpellResolved event) {
            if (onStack != null && refs == null && event.spell() != null
                    && event.spell().getId() == onStack.getId()) {
                refs = RefsJson.renderWithX(onStack, x);
            }
        }

        /** The forced ability as the fork's stack holds it, or null if it never got there. */
        SpellAbility onStack() {
            return onStack;
        }

        Snapshot state() {
            return state;
        }

        /** How many targets the forced ability's root line had when it was pushed. */
        int targetsAtCast() {
            return targetsAtCast;
        }

        /** The refs read as the forced ability finished resolving, or null. */
        String refs() {
            return refs;
        }
    }
}
