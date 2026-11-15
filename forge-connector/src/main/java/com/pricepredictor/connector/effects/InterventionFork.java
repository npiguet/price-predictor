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
final class InterventionFork implements InterventionScheduler.Forcing {

    /** A refusal's reason: the forced ability never reached the copy's stack (a special action). */
    static final String NOT_ON_THE_STACK = "nothing reached the stack";

    /** A refusal's reason: the forced ability reached the stack, never began to resolve and did nothing. */
    static final String NEVER_RESOLVED = "never resolved";

    /**
     * A refusal's reason: the line's resolution reads, unguarded, a cost the
     * fork does not pay ({@link #readsAnUnpaidCost}).
     */
    static final String READS_AN_UNPAID_COST = "reads a cost it never paid";

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

    /**
     * Whether {@code line}'s resolution reads what its cost paid where Forge
     * reads it without a guard, so a forced line, which pays nothing, throws
     * there: ninjutsu and a sneak cast take the attacker returned to hand
     * ({@code getPaidList("Returned")}) and the defender it attacked; crewing
     * or saddling a vehicle or mount already crewed or saddled this turn
     * adds the creatures tapped to pay to a list it holds. Refused before the
     * fork is taken, as a line the engine cannot resolve off its normal path.
     */
    static boolean readsAnUnpaidCost(SpellAbility line) {
        if (line == null) {
            return false;
        }
        SpellAbility root = line.getRootAbility();
        if (root.isNinjutsu() || root.isSneak()) {
            return true;
        }
        forge.game.card.Card host = root.getHostCard();
        if (host == null) {
            return false;
        }
        return root.isCrew() && host.getCrewedByThisTurn() != null
                || root.isKeyword(forge.game.keyword.Keyword.SADDLE) && host.getSaddledByThisTurn() != null;
    }

    /** Whether the budget lets {@link #intervene} take another fork of this resolution. */
    @Override
    public boolean mayIntervene(int realResolutionId) {
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
    @Override
    public boolean intervene(SpellAbility ability, int realResolutionId, Player holder) {
        return intervene(ability, realResolutionId, () -> holder);
    }

    private boolean intervene(SpellAbility ability, int realResolutionId, Supplier<Player> perspective) {
        if (!budget.mayIntervene(realResolutionId)) {
            return false;
        }
        if (readsAnUnpaidCost(ability)) {
            // Before the fork is taken: it would spend the budget and a copy
            // on a line whose forced resolution throws.
            ForkRefusals.count("intervention", READS_AN_UNPAID_COST);
            return false;
        }
        Forks.ForkRun run = new Forks.ForkRun(
                "intervention", game, () -> budget.spendIntervention(realResolutionId), perspective);
        Boolean written = run.take((verified, snapshots) -> {
            SpellAbility forked = Forks.mappedAbility(verified.copier(), ability);
            Player actor = forked == null
                    ? null : Forks.mappedPlayer(verified.copier(), verified.perspective());
            if (forked == null || actor == null) {
                throw new ForkRefusal(ForkRefusal.NO_COUNTERPART);
            }
            ForcedResolution resolution = forceResolution(verified.fork(), forked, actor, snapshots);
            // "Never got to act" and "resolved and did nothing" both leave no
            // event; the resolution hook tells them apart. One whose
            // resolution started is written as a live one is, its halves
            // empty ([], resolved or fizzled); one that never started is
            // nothing to write.
            if (resolution.isEmpty() && !resolution.started()) {
                throw new ForkRefusal(NEVER_RESOLVED);
            }
            return resolution;
        }, (verified, resolution) -> {
            write(resolution, ability, verified);
            return Boolean.TRUE;
        });
        return Boolean.TRUE.equals(written);
    }

    /**
     * Write a forced resolution's effect halves.
     *
     * @param verified the fork: the live player it acted for, and what its
     *                 copy left out, which every half carries
     */
    private void write(ForcedResolution resolution, SpellAbility ability, Forks.VerifiedFork verified) {
        // The *live* ability's keys, not the fork's. A provenance key
        // names a printed line, which the copy shares — but keying off
        // the copy would make the record's identity depend on a game
        // that no longer exists.
        ProvenanceKey.Resolved keys = ProvenanceResolver.resolve(ability);
        String actorId = Refs.of(verified.perspective());
        // A modal resolution writes one half per chosen mode, by the same
        // rule the live bracket applies (FR-029f), each through the mode's
        // option line and with the board as it stood when that mode's
        // first clause began. Events before the first mode's clause — the
        // charm's own, of which there are none — ride with the first.
        List<HeldHalves.Half<ModeSplit.ModeHalf>> halves = resolution.halves();
        for (int index = 0; index < halves.size(); index++) {
            HeldHalves.Half<ModeSplit.ModeHalf> half = halves.get(index);
            context.emit(verified.stamp(forkedHalf(
                    actorId, keys.withOption(half.about().mode().option()),
                    resolution.render(half.about().mode(), half.events(), index, halves.size()),
                    half.events(), half.about().conditions(), resolution.outcome())));
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
     *                (FR-042b): resolved, partially fizzled or fizzled; a
     *                fizzled one holds its spell card's move off the stack
     *                and nothing else, an ability's nothing at all
     * @param started whether the resolution hook told the forced item's
     *                resolution starting: a resolution that started and did
     *                nothing is written with empty halves, one that never
     *                started is not
     */
    private record ForcedResolution(
            String refs, List<HeldHalves.Half<ModeSplit.ModeHalf>> halves,
            ActivationOutcome outcome, ChainTargets resolving, ChainTargets atCast,
            boolean started) {

        /**
         * A half's board, paired with its refs as the live bracket pairs
         * them: the half's random results and its own links' targets, as
         * its resolution started and as cast, in the refs, and the library
         * cards its events name among the entities.
         */
        String render(ModeSplit.Mode mode, List<EffectEvent> events, int index, int count) {
            Snapshot board = mode.state();
            if (refs == null) {
                return board.withEventCards(events).render();
            }
            String read = RefsJson.withRandom(refs, events);
            if (resolving != null) {
                List<String> now = resolving.ofHalf(index, count, mode.option());
                List<String> cast = atCast == null ? null : atCast.ofHalf(index, count, mode.option());
                List<String> written = new java.util.ArrayList<>(now);
                if (cast != null) {
                    written.addAll(cast);
                }
                read = RefsJson.withTargets(read, now, cast,
                        ChainTargets.onStackOf(written, resolving, atCast));
            }
            return board.withRefs(read).withEventCards(events).render();
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
     *         ability resolved and did nothing observable, or never began to
     *         resolve ({@link ForcedResolution#started} tells which)
     * @throws ForkRefusal when its choices cannot be drawn or nothing
     *         reached the stack
     */
    private ForcedResolution forceResolution(
            Game fork, SpellAbility ability, Player actor, SnapshotBuilder snapshots) {
        // Not guarded here: a forced resolution reaches states ordinary play
        // does not, and Forks.ForkRun discards the fork when a card throws.
        if (!drawForcedChoices(ability, actor)) {
            throw new ForkRefusal(ForkRefusal.NO_CHOICES);
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
            throw new ForkRefusal(NOT_ON_THE_STACK);
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
        // answer (FR-034); the mana its own effect makes (C3); and the events
        // its clauses promise and its effects report (a token made, a coin
        // flipped, damage prevented), which the live bracket hears from the
        // hooks too.
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
        boolean[] started = {false};
        // Whether the forced item's controller turned down a "you may" while
        // it resolved, told by the confirm hook as the live bracket is told.
        boolean[] declined = {false};
        // The forced item's targets as its resolution started and as cast,
        // read as the live bracket reads them.
        ChainTargets[] targets = {null, null};
        Forks.ForkHooks hooks = new Forks.ForkHooks() {
            @Override
            public void resolutionStarting(SpellAbility item, boolean itemFizzled) {
                resolutionStarting(item, itemFizzled, null);
            }

            @Override
            public void resolutionStarting(
                    SpellAbility item, boolean itemFizzled, Object targetsAtCast) {
                if (item.getId() == onStack.getId()) {
                    started[0] = true;
                    // A bestowed Aura or a mutating spell whose target went
                    // resolves as a creature spell, as the live bracket reads it.
                    fizzled[0] = itemFizzled && !BusBracketCollector.resolvesAsACreature(item);
                    targets[0] = ChainTargets.now(item);
                    targets[1] = ChainTargets.atCast(item, targetsAtCast);
                    sink.resolutionStarted(snapshots.ofResolving(onStack, targets[1]));
                }
            }

            @Override
            public void clauseResolving(SpellAbility clause) {
                CharmModes.ModeStart start = CharmModes.modeStartOf(clause);
                if (start != null && start.charm() == onStack) {
                    sink.beginMode(start.option(), snapshots.ofResolving(onStack, targets[1]));
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

            // What the live bracket takes from the hooks rather than the
            // bus: filed as the live router files it, where the forced
            // ability stands deciding whether it is the forced ability's.
            @Override
            public void clauseEvent(SpellAbility clause, EffectEvent event) {
                sink.fileOutcome(event, null);
            }

            @Override
            public void outcome(SpellAbility ability, EffectEvent event) {
                sink.fileOutcome(event, null);
            }

            // While the forced item resolves: its start told, its end not yet
            // (the refs are read at its end), as the live bracket scopes a
            // refusal to the bracket that is open.
            @Override
            public void declined(SpellAbility offered) {
                if (started[0] && cast.refs() == null
                        && (offered == null || offered.getRootAbility() == onStack
                                || offered.getId() == onStack.getId())) {
                    declined[0] = true;
                }
            }
        };
        // A belt (FR-010d): the copier copies no stack, so the copy's holds
        // only the forced line, never a restart (InterventionScheduler), and
        // what its cast triggered; Forge's one restart is Karn Liberated's
        // loyalty ability.
        Forks.refuseARestart(fork);
        Forks.listeningForHooks(hooks, () -> {
            GameSimulator.resolveStack(fork, actor.getWeakestOpponent());
            return null;
        });
        // Counted as the live bracket counts: the hook's targets as cast
        // against those left when the resolution started, every link's.
        List<HeldHalves.Half<ModeSplit.ModeHalf>> halves = sink.halves();
        // A refusal is the outcome only where nothing happened but the
        // spell's card leaving the stack, as live (OpenResolution.wasDeclined).
        ActivationOutcome outcome = fizzled[0] ? ActivationOutcome.FIZZLED
                : declined[0] && nothingButTheMoveOffTheStack(halves, onStack)
                        ? ActivationOutcome.DECLINED
                : targets[0] != null && targets[1] != null
                        ? OpenResolution.outcomeOf(targets[1].count(), targets[0].count())
                        : OpenResolution.outcomeOf(cast.targetsAtCast(), onStack);
        if (outcome == ActivationOutcome.FIZZLED) {
            halves = onlyMoveOffTheStack(halves, onStack);
        }
        return new ForcedResolution(cast.refs(), halves, outcome, targets[0], targets[1], started[0]);
    }

    /** Whether no half holds anything but the forced spell's own card leaving the stack. */
    private static boolean nothingButTheMoveOffTheStack(
            List<HeldHalves.Half<ModeSplit.ModeHalf>> halves, SpellAbility onStack) {
        String card = onStack.isSpell() && onStack.getHostCard() != null
                ? Refs.of(onStack.getHostCard()) : null;
        for (HeldHalves.Half<ModeSplit.ModeHalf> half : halves) {
            for (EffectEvent event : half.events()) {
                boolean move = card != null && event.type() == EventType.ZONE_CHANGE
                        && List.of(card).equals(event.subjects())
                        && "stack".equals(event.params().get("from_zone"));
                if (!move) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * A fizzled forced item's halves as a live one's (608.2b): nothing but a
     * spell's own card's move off the stack, in the last half; nothing for
     * an ability, or a copy, which leaves no card behind.
     */
    private static List<HeldHalves.Half<ModeSplit.ModeHalf>> onlyMoveOffTheStack(
            List<HeldHalves.Half<ModeSplit.ModeHalf>> halves, SpellAbility onStack) {
        String card = onStack.isSpell() && onStack.getHostCard() != null
                ? Refs.of(onStack.getHostCard()) : null;
        HeldHalves<ModeSplit.ModeHalf> kept = new HeldHalves<>();
        for (HeldHalves.Half<ModeSplit.ModeHalf> half : halves) {
            List<EffectEvent> move = half.events().stream()
                    .filter(event -> card != null && event.type() == EventType.ZONE_CHANGE
                            && List.of(card).equals(event.subjects())
                            && "stack".equals(event.params().get("from_zone")))
                    .limit(1).toList();
            kept.add(half.about(), move);
        }
        return kept.halves();
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
        return drawChoices(choices, ability, actor);
    }

    /**
     * Modes, then X, then targets, the order a cast takes them (601.2b-c):
     * a clause dividing X among its targets splits the X just announced, and
     * one drawn before it split nothing.
     *
     * @param payer whose leftover mana bounds X, or null to announce the floor
     */
    private static boolean drawChoices(RandomChoices from, SpellAbility ability, Player payer) {
        if (!from.chooseModes(ability)) {
            return false;
        }
        // Floor one: an X of zero resolves into the same silence a
        // missing target does, and the fork never pays, so the leftover mana of
        // the forked actor is what bounds the draw above it.
        from.announceX(ability, payer, 1);
        return from.chooseTargets(ability);
    }

    /**
     * Whether {@code ability} has a mode and targets for {@code holder} on
     * the live board: the forced choices tried on a copy of it, with draws
     * of their own, so neither the live ability nor any stream is touched.
     * A copy that cannot be tried is let through, to the fork's own guard.
     */
    @Override
    public boolean hasChoices(SpellAbility ability, Player holder) {
        try {
            SpellAbility trial = ability.copy(ability.getHostCard(), holder, true);
            trial.setActivatingPlayer(holder);
            return drawChoices(new RandomChoices(new Random(0L)), trial, null);
        } catch (RuntimeException e) {
            return true;
        }
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
