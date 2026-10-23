package com.pricepredictor.connector.effects;

import com.google.common.eventbus.Subscribe;
import forge.ai.simulation.GameSimulator;
import forge.game.Game;
import forge.game.event.GameEventSpellAbilityCast;
import forge.game.event.GameEventSpellResolved;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

import java.util.List;
import java.util.Optional;
import java.util.Random;
import java.util.function.Consumer;

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
        if (!budget.mayIntervene(realResolutionId)) {
            return false;
        }
        // The whole method is the fork, not just the forced resolution: see
        // Forks.FORK_RUNNING for what the narrower scope left uncovered.
        return Forks.asFork(() -> {
            // Counted before the score check, so a discarded fork still spends its
            // budget.
            budget.spendIntervention(realResolutionId);

            Optional<Forks.VerifiedFork> verified = Forks.copyAndVerify(
                    game,
                    () -> ability.getActivatingPlayer() != null
                            ? ability.getActivatingPlayer()
                            : game.getPhaseHandler().getPlayerTurn());
            if (verified.isEmpty()) {
                return false;
            }
            Game fork = verified.get().fork();
            Player perspective = verified.get().perspective();
            SpellAbility forked = Forks.findInFork(verified.get().copier(), ability);
            Player actor = forked == null
                    ? null : Forks.mappedPlayer(verified.get().copier(), perspective);
            if (forked == null || actor == null) {
                return false;
            }

            // The fork's own state, not the live game's.
            SnapshotBuilder snapshots = new SnapshotBuilder(fork);
            ForcedResolution resolution = forceResolution(fork, forked, actor, snapshots);
            if (resolution == null || resolution.isEmpty()) {
                // An empty event list is not an observation. "The ability resolved
                // and did nothing" and "the ability never got to act" render the
                // same record, and nothing in it tells them apart. A
                // counterfactual nobody can read is worse than one that was never
                // written, so this drops it.
                return false;
            }

            // The *live* ability's keys, not the fork's. A provenance key
            // names a printed line, which the copy shares — but keying off
            // the copy would make the record's identity depend on a game
            // that no longer exists.
            ProvenanceKey.Resolved keys = ProvenanceResolver.resolve(ability);
            String actorId = SnapshotBuilder.playerId(perspective);
            // A modal resolution writes one half per chosen mode, by the same
            // rule the live bracket applies (FR-029f), each through the mode's
            // option line and with the board as it stood when that mode's
            // first clause began. Events before the first mode's clause — the
            // charm's own, of which there are none — ride with the first.
            for (HeldHalves.Half<ModeSplit.Mode> half : resolution.halves()) {
                context.emit(forkedHalf(
                        actorId, keys.withOption(half.about().option()),
                        resolution.render(half.about().state()), half.events()));
            }
            return true;
        });
    }

    /**
     * One interventional effect half. No {@code link_id}: it is written alone,
     * because there was no real activation to pair it with.
     */
    private EffectRecord forkedHalf(
            String actorId, ProvenanceKey.Resolved keys, String state,
            List<EffectEvent> events) {
        return context.newResolution(Moment.RESOLUTION)
                .interventional(true)
                .fork(true)
                .actor(actorId)
                .ability(keys)
                .state(state)
                .payload(RecordPayloads.eventsPayload(events));
    }

    /**
     * What a forced resolution produced: its effect halves, one per chosen
     * mode or one for the whole resolution, each with the board it is read
     * against and the deaths it left behind ({@link ForkEventSink#halves}).
     *
     * @param refs the refs block read as the forced ability finished
     *             resolving, where the live bracket re-reads it, or null
     *             when the fork never saw it finish
     */
    private record ForcedResolution(String refs, List<HeldHalves.Half<ModeSplit.Mode>> halves) {

        /** A board this resolution read, paired with its refs as the live bracket pairs them. */
        String render(Snapshot board) {
            return (refs == null ? board : board.withRefs(refs)).render();
        }

        /** Whether no half holds an event: the forced ability did nothing observable. */
        boolean isEmpty() {
            for (HeldHalves.Half<ModeSplit.Mode> half : halves) {
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
     * ({@link ForcedCast}): the board at the cast event, the forced ability
     * on top of the fork's stack, and the refs block re-read as it finishes
     * resolving. Two differences remain, both from the costs the fork does
     * not pay. The board still holds what a cost would have used up: the
     * mana sources a payment would have tapped, the mana it would have
     * spent from the pool, and anything sacrificed, discarded, exiled or
     * tapped as a cost. And nothing was paid into the ability, so a
     * resolution that reads what was paid reads nothing.
     *
     * <p>Resolution drains the whole stack: the cast's own triggers above the
     * forced ability, and the triggers its effect causes after it. Neither is
     * the forced ability's record live, where each has a bracket of its own,
     * so the sink files what live would put in the forced ability's own
     * bracket ({@link ForkEventSink}): what happens while it is the top of
     * the stack before it resolves, except an item above's aftermath; its
     * resolution; and the deaths of the check that follows it.
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
        try {
            if (!drawForcedChoices(ability, actor)) {
                return null;
            }
            fork.copyLastState();
            ForcedCast cast = new ForcedCast(fork, snapshots);
            ForkEventSink sink = ForkEventSink.forCast(fork, ability);
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
            // A mode's first clause is the boundary between per-mode halves, and
            // the moment its snapshot is taken (FR-029b, FR-029f). Only this
            // ability's own modes count: the live clause handler forwards every
            // clause it drops, and anything else resolving on the fork's stack
            // may be modal too. Neither the host card nor the key can tell them
            // apart: a creature spell's own modal enters trigger shares its
            // host, and a copy of the forced charm a cast trigger made (Double
            // Vision) shares its key too, so its modes would cut halves the
            // forced charm's events never reach. The mode's charm must be the
            // very ability on the stack, the one this record is about.
            Consumer<SpellAbility> modeStarts = clause -> {
                CharmModes.ModeStart start = CharmModes.modeStartOf(clause);
                if (start != null && start.charm() == onStack) {
                    sink.beginMode(start.option(), snapshots.ofResolving(onStack));
                }
            };
            Forks.listeningForClauses(modeStarts, () -> {
                GameSimulator.resolveStack(fork, actor.getWeakestOpponent());
                return null;
            });
            return new ForcedResolution(cast.refs(), sink.halves());
        } catch (RuntimeException | StackOverflowError e) {
            // A forced resolution reaches states ordinary play does not — an
            // ability resolving with no legal target, a cost that was never
            // paid — and a card that throws is one this fork cannot describe.
            // Discarding is right; failing the worker is not.
            return null;
        }
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
     * The forced ability's two moments on the fork's bus, each read where the
     * live bracket reads an observed resolution's.
     *
     * <p>The live bracket takes its board at the cast event ({@code
     * BusBracketCollector.beginBracket}): the ability on top of the stack, a
     * spell's card in the stack zone, priority with the caster, and the cast
     * not yet counted among the turn's spells. It re-reads the refs block when
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

        ForcedCast(Game fork, SnapshotBuilder snapshots) {
            this.fork = fork;
            this.snapshots = snapshots;
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
            var top = fork.getStack().peek();
            onStack = top == null ? null : top.getSpellAbility();
            if (onStack != null) {
                state = snapshots.ofResolving(onStack);
            }
        }

        @Subscribe
        public void onResolved(GameEventSpellResolved event) {
            if (onStack != null && refs == null && event.spell() != null
                    && event.spell().getId() == onStack.getId()) {
                refs = snapshots.refsJson(onStack);
            }
        }

        /** The forced ability as the fork's stack holds it, or null if it never got there. */
        SpellAbility onStack() {
            return onStack;
        }

        Snapshot state() {
            return state;
        }

        /** The refs read as the forced ability finished resolving, or null. */
        String refs() {
            return refs;
        }
    }
}
