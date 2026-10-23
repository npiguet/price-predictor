package com.pricepredictor.connector.effects;

import com.google.common.eventbus.Subscribe;
import forge.ai.simulation.GameCopier;
import forge.ai.simulation.GameSimulator;
import forge.ai.simulation.GameStateEvaluator;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.event.GameEventSpellAbilityCast;
import forge.game.event.GameEventSpellResolved;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import java.util.Set;
import java.util.function.Supplier;

/**
 * Stage three: records for what observation cannot reach.
 *
 * <p>Two kinds of fork, and the flags tell them apart on their own.
 *
 * <p>An <b>interventional resolution</b> forces an ability Forge's AI never
 * chose to use. Some abilities are never cast in any game the AI plays — it
 * judges them bad, or cannot afford them, or never draws them into a board
 * where they do anything — and no amount of observation will produce a record
 * for them. The fork resolves the ability with chosen targets and modes,
 * routing an unaffordable candidate through the play-without-paying path, and
 * writes the <b>effect half only</b>: there was no real activation, so there is
 * no cost half to link to.
 *
 * <p>A <b>damage-step probe</b> re-runs one combat with a keyword stripped, to
 * isolate what that keyword did. It is built only for keywords gate 2 routed to
 * it, and taken only when {@code --probe-keywords} names one — the flag is the
 * runtime switch, separate from the build decision. It names all eight
 * damage-step keywords unless a run narrows or empties it.
 *
 * <p><b>Every fork is score-checked against the live game before any
 * perturbation.</b> A copy that already disagrees with the game it came from is
 * measuring the copier rather than the ability, and a discarded fork still
 * counts against its budget — otherwise a systematically-failing copy would
 * retry forever.
 */
public final class ForkCollector {

    /** At most this many forks may target one real resolution (FR-040). */
    public static final int MAX_FORKS_PER_RESOLUTION = 2;

    /**
     * No turn could be read, so the pairing guard stands down.
     *
     * <p>A collector with no live game — the unit tests' shape — has no turn
     * to compare against, and refusing to write there would test the guard
     * rather than the pairing.
     */
    public static final int TURN_UNKNOWN = -1;

    /**
     * Marks this thread as running one of this class's two forks -- an
     * {@link #intervene interventional resolution} or a {@link #probe
     * damage-step probe} -- for {@code
     * LiveGame.ownsNullAbilityOutcome()}.
     *
     * <p>It covers both methods' <em>entire</em> bodies, not just the forced
     * resolution or {@link #probe}'s {@code Combat.dealAssignedDamage}: the
     * resolving-clause pointer is null for every segment of a fork outside
     * {@code AbilityUtils.resolve} -- the copy, the score check and
     * {@code findInFork}, {@link #forceResolution}'s mode, target and X
     * draws, and {@code GameSimulator.resolveStack}'s own state-based check
     * and trigger pass. A null-ability outcome fired in any of them would
     * otherwise be indistinguishable from the live game's own turn-based
     * actions. So "a fork is running on this thread" is a sufficient answer
     * on its own, and {@code ownsNullAbilityOutcome} checks it first, with
     * the pointer behind it as a second, independent signal.
     *
     * <p>Saved and restored rather than set and cleared, as {@code
     * AbilityUtils}' {@code EFFECT_RECORD_SUB_ABILITY} is: correct without
     * relying on the two methods never nesting.
     */
    private static final ThreadLocal<Boolean> FORK_RUNNING =
            ThreadLocal.withInitial(() -> Boolean.FALSE);

    /** Whether an {@link #intervene} or a {@link #probe} is running on this thread. */
    static boolean isForkRunningOnThisThread() {
        return FORK_RUNNING.get();
    }

    /** Run {@code action} with this thread marked as running a fork; see {@link #FORK_RUNNING}. */
    private static <T> T asFork(Supplier<T> action) {
        boolean was = FORK_RUNNING.get();
        FORK_RUNNING.set(true);
        try {
            return action.get();
        } finally {
            FORK_RUNNING.set(was);
        }
    }

    /**
     * Where a forced resolution's clauses report in, while one is running.
     *
     * <p>The clause hook is a JVM static that {@code ClauseOutcomeRouter} owns,
     * and the live collector drops a fork's clauses on game identity — rightly,
     * for its own bracket. A fork that splits a modal resolution into per-mode
     * halves (FR-029f) needs those same clauses, so the live handler hands a
     * clause it is not keeping to whatever fork is running on its thread, and
     * {@link #forceResolution} installs the consumer for exactly the span of
     * its own resolution. Thread-local like {@link #FORK_RUNNING}, and for the
     * same reason: the engine runs games on more than one thread.
     */
    private static final ThreadLocal<java.util.function.Consumer<SpellAbility>> CLAUSE_LISTENER =
            new ThreadLocal<>();

    /** A clause of some other game is resolving; the running fork may want it. */
    static void noteClauseResolving(SpellAbility clause) {
        java.util.function.Consumer<SpellAbility> listener = CLAUSE_LISTENER.get();
        if (listener != null) {
            listener.accept(clause);
        }
    }

    private final Game game;
    private final CollectorContext context;
    private final int interventionsPerGame;
    private final int probesPerGame;
    private final List<String> probeKeywords;
    /** The draws a forced resolution makes, from the game's seed. */
    private final RandomChoices choices;

    private int interventionsUsed;
    private int probesUsed;
    private final Map<Integer, Integer> forksPerResolution = new HashMap<>();

    ForkCollector(Game game, CollectorContext context, CollectionCaps caps, long seed) {
        this.game = game;
        this.context = context;
        this.interventionsPerGame = caps.interventionsPerGame();
        this.probesPerGame = caps.probesPerGame();
        this.probeKeywords = List.copyOf(caps.probeKeywords());
        // Seeded so a rerun of a game forces the same choices. Scrambled as
        // the sampler's seed is: games are seeded consecutively, and the first
        // draw here is a mode or target count, which an unscrambled seed would
        // make nearly the same in every game.
        this.choices = new RandomChoices(new Random(GameRecording.scramble(seed)));
    }

    // ── budgets ─────────────────────────────────────────────────────────

    /**
     * Whether another interventional fork may be taken.
     *
     * <p>Two budgets, both binding: the per-game one from
     * {@code --interventions-per-game}, and a fixed cap of
     * {@value #MAX_FORKS_PER_RESOLUTION} forks per real resolution that no flag
     * can raise. The second exists because forking the same resolution twice
     * already gives both counterfactuals worth having; a third would cost a
     * game's simulation for a third variation on one board.
     */
    public boolean mayIntervene(int realResolutionId) {
        if (interventionsUsed >= interventionsPerGame) {
            return false;
        }
        return forksPerResolution.getOrDefault(realResolutionId, 0)
                < MAX_FORKS_PER_RESOLUTION;
    }

    /**
     * Whether another probe fork may be taken.
     *
     * <p>With {@code --probe-keywords} empty, never — whatever the build state.
     * The flag is the runtime switch; whether the probe code exists at all is
     * the build decision gate 2 drives, and the two are deliberately separate.
     * A worker started with no {@code effect.probe.keywords} property reads
     * the list as empty; every collecting supervisor sets it, naming all eight
     * damage-step keywords unless the run narrows them.
     */
    public boolean mayProbe(String keyword) {
        // Normalized before the check: the flag names keywords the way the
        // corpus spells them (first_strike) and a caller holding a card's
        // keyword has Forge's (First Strike). Compared directly, every probe
        // of a keyword spelled differently would silently decline here.
        if (keyword == null || probeKeywords.isEmpty()
                || !probeKeywords.contains(
                        ProbeScheduler.normalizeKeyword(keyword))) {
            return false;
        }
        return probesUsed < probesPerGame;
    }

    // ── the score check ─────────────────────────────────────────────────

    /**
     * Verify a fresh copy scores the same as the game it came from.
     *
     * <p>Run <b>before any perturbation</b>: a copy that already disagrees is
     * measuring the copier, and a record from it would be noise dressed as a
     * counterfactual. A mismatch logs a warning, discards the fork, and
     * <b>still counts against the budget</b> — otherwise a systematically
     * failing copy would retry until the game ended.
     */
    public boolean scoreCheck(Game fork, Player perspective) {
        try {
            GameStateEvaluator evaluator = new GameStateEvaluator();
            int original = evaluator.getScoreForGameState(game, perspective).value;
            Player forkPerspective = matchingPlayer(fork, perspective);
            if (forkPerspective == null) {
                return false;
            }
            int copied = evaluator
                    .getScoreForGameState(fork, forkPerspective).value;
            if (original != copied) {
                System.err.println(
                        "Effect records: discarding a fork whose score "
                                + copied + " disagrees with the live game's "
                                + original + "; it would measure the copier "
                                + "rather than the ability");
                return false;
            }
            return true;
        } catch (RuntimeException e) {
            System.err.println(
                    "Effect records: fork score check failed (" + e.getMessage()
                            + "); discarding");
            return false;
        }
    }

    /**
     * A copy of the live game that scored as the original does.
     *
     * @param copier      maps the live game's objects to the copy's
     * @param perspective the live player the score was checked from
     */
    private record VerifiedFork(GameCopier copier, Game fork, Player perspective) {
    }

    /**
     * Copy the live game and score-check the copy, empty when either fails.
     *
     * @param perspective read after the copy, because a game the copier cannot
     *                    read may not answer it either
     */
    private Optional<VerifiedFork> copyAndVerify(Supplier<Player> perspective) {
        GameCopier copier;
        Game fork;
        try {
            // Both inside the guard: the copier's own constructor reads the
            // game, so a game it cannot read fails here rather than at makeCopy.
            copier = new GameCopier(game);
            fork = copier.makeCopy();
        } catch (RuntimeException e) {
            return Optional.empty();
        }
        Player from = perspective.get();
        if (from == null || !scoreCheck(fork, from)) {
            return Optional.empty();
        }
        return Optional.of(new VerifiedFork(copier, fork, from));
    }

    private static Player matchingPlayer(Game fork, Player original) {
        for (Player candidate : fork.getPlayers()) {
            if (candidate.getId() == original.getId()) {
                return candidate;
            }
        }
        return null;
    }

    // ── interventional resolutions ──────────────────────────────────────

    /**
     * Fork the game, force {@code ability} to resolve, and record the effect.
     *
     * @param realResolutionId the real resolution this fork varies, for the
     *                         per-resolution cap
     * @return whether a record was written
     */
    public boolean intervene(SpellAbility ability, int realResolutionId) {
        if (!mayIntervene(realResolutionId)) {
            return false;
        }
        // The whole method is the fork, not just the forced resolution: see
        // FORK_RUNNING for what the narrower scope left uncovered.
        return asFork(() -> {
            // Counted before the score check, so a discarded fork still spends its
            // budget.
            interventionsUsed++;
            forksPerResolution.merge(realResolutionId, 1, Integer::sum);

            Optional<VerifiedFork> verified = copyAndVerify(
                    () -> ability.getActivatingPlayer() != null
                            ? ability.getActivatingPlayer()
                            : game.getPhaseHandler().getPlayerTurn());
            if (verified.isEmpty()) {
                return false;
            }
            Game fork = verified.get().fork();
            Player perspective = verified.get().perspective();
            SpellAbility forked = findInFork(verified.get().copier(), fork, ability);
            Player actor = forked == null
                    ? null : mappedPlayer(verified.get().copier(), perspective);
            if (forked == null || actor == null) {
                return false;
            }

            // The fork's own state, not the live game's.
            SnapshotBuilder snapshots = new SnapshotBuilder(fork);
            ForcedResolution resolution = forceResolution(fork, forked, actor, snapshots);
            if (resolution == null || resolution.events().isEmpty()) {
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
            if (resolution.modes().isEmpty()) {
                context.emit(forkedHalf(
                        actorId, keys, resolution.render(resolution.state()),
                        resolution.halves().get(0)));
            } else {
                emitModeHalves(actorId, keys, resolution);
            }
            return true;
        });
    }

    /**
     * A modal resolution's effect halves: one per chosen mode, by the same rule
     * the live bracket applies (FR-029f), each through the mode's option line
     * and with the board as it stood when that mode's first clause began.
     * Events before the first mode's clause — the charm's own, of which there
     * are none — ride with the first.
     */
    private void emitModeHalves(
            String actorId, ProvenanceKey.Resolved keys, ForcedResolution resolution) {
        List<ModeBoundary> modes = resolution.modes();
        List<List<EffectEvent>> halves = resolution.halves();
        for (int i = 0; i < modes.size(); i++) {
            ModeBoundary boundary = modes.get(i);
            context.emit(forkedHalf(
                    actorId, keys.withOption(boundary.option()),
                    resolution.render(boundary.state()), halves.get(i)));
        }
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
                .payload(EffectRecord.eventsPayload(events));
    }

    /**
     * Where one chosen mode began in a forced resolution's event stream.
     *
     * @param option     the mode's {@code Choices$} position, or null
     * @param firstEvent the index of the first event filed after its first
     *                   clause began resolving
     * @param state      the fork's board at that clause (FR-029b), read
     *                   against the forced ability on the fork's stack as
     *                   the live bracket's {@code beginMode} reads it
     */
    private record ModeBoundary(Integer option, int firstEvent, Snapshot state) {
    }

    /**
     * What a forced resolution produced, where its modes began, and which of
     * its events are permanents leaving the battlefield in a state-based check.
     *
     * @param state the fork's board at the forced cast, the moment the live
     *              bracket reads an observed cast's: a non-modal record's
     *              state
     * @param refs  the refs block read as the forced ability finished
     *              resolving, where the live bracket re-reads it, or null
     *              when the fork never saw it finish
     */
    private record ForcedResolution(
            Snapshot state, String refs, List<EffectEvent> events, List<ModeBoundary> modes,
            Set<Integer> stateBasedLeaves) {

        /** A board this resolution read, paired with its refs as the live bracket pairs them. */
        String render(Snapshot board) {
            return (refs == null ? board : board.withRefs(refs)).render();
        }

        /** The events, one list per chosen mode, or one list when no mode began. */
        List<List<EffectEvent>> halves() {
            List<Integer> starts = new ArrayList<>();
            for (ModeBoundary mode : modes) {
                starts.add(mode.firstEvent());
            }
            return splitByMode(events, starts, stateBasedLeaves);
        }
    }

    /**
     * A forced resolution's events, one list per chosen mode, or one list
     * when no mode began.
     *
     * <p>Every event belongs to the mode in whose span it was filed, except a
     * state-based death. {@code GameSimulator.resolveStack} runs the check
     * only after the whole charm has resolved, so by position every death
     * would fall into the last mode's half, whichever mode killed. Those are
     * filed by the live collector's rule instead ({@link
     * BusBracketCollector#halfForDeath}): the last half whose own events named
     * the permanent, or the last half when none did. Order is kept within each
     * half, so a death still follows the events that caused it.
     *
     * <p>A death is checked for a repeat against the half it joins, as the
     * live collector checks it against the held half it files it into: the
     * sink filed it unchecked ({@link ForkEventSink}), since until now no one
     * knew which half that is.
     *
     * @param modeStarts       the position in {@code events} where each chosen
     *                         mode's span begins, in order; the first span
     *                         also takes what precedes it
     * @param stateBasedLeaves positions in {@code events} of moves off the
     *                         battlefield during a state-based check
     */
    static List<List<EffectEvent>> splitByMode(
            List<EffectEvent> events, List<Integer> modeStarts, Set<Integer> stateBasedLeaves) {
        int halfCount = Math.max(1, modeStarts.size());
        int[] spanOf = new int[events.size()];
        List<Set<String>> touched = new ArrayList<>();
        for (int i = 0; i < halfCount; i++) {
            int from = i == 0 ? 0 : modeStarts.get(i);
            int to = i + 1 < modeStarts.size() ? modeStarts.get(i + 1) : events.size();
            List<EffectEvent> span = new ArrayList<>();
            for (int at = from; at < to; at++) {
                spanOf[at] = i;
                if (!stateBasedLeaves.contains(at)) {
                    span.add(events.get(at));
                }
            }
            touched.add(BusBracketCollector.touchedBy(span));
        }
        List<EventBuffer> halves = new ArrayList<>();
        for (int i = 0; i < halfCount; i++) {
            halves.add(new EventBuffer());
        }
        for (int at = 0; at < events.size(); at++) {
            EffectEvent event = events.get(at);
            int half = stateBasedLeaves.contains(at) && !event.subjects().isEmpty()
                    ? BusBracketCollector.halfForDeath(touched, event.subjects().get(0))
                    : spanOf[at];
            halves.get(half).file(event);
        }
        List<List<EffectEvent>> lists = new ArrayList<>();
        for (EventBuffer half : halves) {
            lists.add(List.copyOf(half.events()));
        }
        return lists;
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
     * <p>Resolution drains the whole stack, so a trigger the effect put there
     * is part of what the effect did.
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
     * @return what resolved and where its modes began; events empty if the
     *         ability resolved and did nothing observable; null if it could
     *         not be resolved at all
     */
    private ForcedResolution forceResolution(
            Game fork, SpellAbility ability, Player actor, SnapshotBuilder snapshots) {
        java.util.function.Consumer<SpellAbility> previousListener = CLAUSE_LISTENER.get();
        try {
            ability.setActivatingPlayer(actor);
            if (!choices.chooseModes(ability) || !choices.chooseTargets(ability)) {
                return null;
            }
            // Floor one: an X of zero resolves into the same silence a
            // missing target does, and the fork never pays, so the leftover mana of
            // the forked actor is what bounds the draw above it.
            choices.announceX(ability, actor, 1);
            fork.copyLastState();
            ForcedCast cast = new ForcedCast(fork, snapshots);
            fork.subscribeToEvents(cast);
            fork.getStack().addAndUnfreeze(ability);
            SpellAbility onStack = cast.onStack();
            if (onStack == null) {
                // Nothing reached the stack. Mostly a special action:
                // foretell, plot and suspend are AbilityStatic, which
                // MagicStack.add resolves on the spot without a push, and
                // live opens no bracket for one either, so there is no
                // resolution record to write. Otherwise a frozen stack parked
                // it or its targeting was refused, and nothing will resolve.
                return null;
            }
            // Listening from here, once the cast is over: the card's move to
            // the stack and the cast's own bookkeeping happen before the live
            // bracket opens, so they are no resolution's events. The ability
            // on the stack is the sink's root, as the live bracket's root is
            // the stack's: an event's attributed_to is read against the
            // ability the bracket belongs to, and a fork's bracket is exactly
            // this one forced resolution.
            ForkEventSink sink = ForkEventSink.forIntervention(fork, onStack);
            fork.subscribeToEvents(sink);
            List<ModeBoundary> modes = new ArrayList<>();
            // A mode's first clause is the boundary between per-mode halves, and
            // the moment its snapshot is taken (FR-029b, FR-029f). Only this
            // ability's own modes count: the live clause handler forwards every
            // clause it drops, and a trigger the forced resolution put on the
            // fork's stack may be modal too. The host card cannot tell them
            // apart: a creature spell's own modal enters trigger shares its host,
            // and its modes filed under the spell's key would name spell[0] with
            // an option, a line no sidecar has. The charm's own key can.
            ProvenanceKey forcedKey = ProvenanceResolver.resolve(onStack).key();
            CLAUSE_LISTENER.set(clause -> {
                CharmModes.ModeStart start = CharmModes.modeStartOf(clause);
                if (start != null && forcedKey != null
                        && forcedKey.equals(ProvenanceResolver.resolve(start.charm()).key())) {
                    modes.add(new ModeBoundary(
                            start.option(), sink.beginMode(), snapshots.ofResolving(onStack)));
                }
            });
            GameSimulator.resolveStack(fork, actor.getWeakestOpponent());
            return new ForcedResolution(
                    cast.state(), cast.refs(), sink.events(), List.copyOf(modes),
                    sink.stateBasedLeaves());
        } catch (RuntimeException | StackOverflowError e) {
            // A forced resolution reaches states ordinary play does not — an
            // ability resolving with no legal target, a cost that was never
            // paid — and a card that throws is one this fork cannot describe.
            // Discarding is right; failing the worker is not.
            return null;
        } finally {
            CLAUSE_LISTENER.set(previousListener);
        }
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

    /**
     * The fork's copy of a live ability, or null when it cannot be matched.
     *
     * <p>The host card maps directly through the copier. The ability itself
     * does not, so it is matched by description among the copy's — the same
     * approach Forge's own simulator takes.
     */
    private static SpellAbility findInFork(
            GameCopier copier, Game fork, SpellAbility ability) {
        Card host;
        try {
            host = copier.find(ability.getHostCard());
        } catch (RuntimeException e) {
            return null;
        }
        if (host == null) {
            return null;
        }
        String description = ability.getDescription();
        for (SpellAbility candidate : host.getSpellAbilities()) {
            if (candidate.getDescription().equals(description)) {
                return candidate;
            }
        }
        return null;
    }

    private static Player mappedPlayer(GameCopier copier, Player player) {
        try {
            return copier.find(player);
        } catch (RuntimeException e) {
            return null;
        }
    }

    private static Card mappedCard(GameCopier copier, Card card) {
        try {
            return copier.find(card);
        } catch (RuntimeException e) {
            return null;
        }
    }

    // ── damage-step probes ──────────────────────────────────────────────

    /**
     * Re-run this damage step without one creature's keyword.
     *
     * <p>Called <b>before</b> the real step, which is the only moment the fork
     * can diverge: the phase event fires ahead of the turn-based action that
     * deals the damage, so the copy is taken from a board where nothing has
     * been dealt yet. The fork then assigns and deals its own damage with the
     * keyword gone, and what the bus publishes is what that keyword was worth.
     *
     * <p>The branch is <b>held rather than written</b>. It mirrors the real
     * combat record, and that record does not exist until the step it forked
     * ahead of has finished — so the caller completes the pairing once the id
     * is known. Reconstructing the pairing from board state instead would break
     * on two identical-looking combats in one turn.
     *
     * @param step the damage step about to be dealt, which the fork deals too
     * @return the held branch, or null when the fork could not be taken
     */
    public HeldProbe probe(String keyword, Card carrier, DamageStep step) {
        if (!mayProbe(keyword) || carrier == null) {
            return null;
        }
        // The whole method is the fork, not just the damage it deals: see
        // FORK_RUNNING for what the narrower scope left uncovered.
        return asFork(() -> {
            // Counted before the score check, like an intervention: a copy that
            // systematically fails must not retry until the game ends.
            probesUsed++;

            Optional<VerifiedFork> verified = copyAndVerify(
                    () -> game.getPhaseHandler() == null
                            ? null : game.getPhaseHandler().getPlayerTurn());
            if (verified.isEmpty()) {
                return null;
            }
            Game fork = verified.get().fork();
            Player perspective = verified.get().perspective();
            Card forkCarrier = mappedCard(verified.get().copier(), carrier);
            if (forkCarrier == null || fork.getCombat() == null) {
                return null;
            }

            SnapshotBuilder snapshots = new SnapshotBuilder(fork);
            String state;
            ProbedStep dealt;
            try {
                stripKeyword(fork, forkCarrier, keyword);
                // The state is the board as the step began, with the keyword
                // already gone: that is the input the counterfactual answers for.
                state = snapshots.ofBoard().render();
                dealt = runDamageStep(fork, step == DamageStep.FIRST_STRIKE);
            } catch (RuntimeException | StackOverflowError e) {
                // A stripped keyword reaches combat states ordinary play does not.
                // Discarding is right; failing the worker is not.
                return null;
            }
            return new HeldProbe(
                    ProbeScheduler.normalizeKeyword(keyword),
                    SnapshotBuilder.entityId(carrier), state, dealt.events(),
                    SnapshotBuilder.playerId(perspective), dealt.combatFields(),
                    step,
                    // The *live* game's turn, which is also the fork's: the copier
                    // carries the counter across (GameCopier hands it to
                    // devModeSet). Read here so the branch remembers the step it
                    // belongs to.
                    liveTurn());
        });
    }

    /**
     * Take {@code keyword} away from the fork's copy of the carrier.
     *
     * <p>Through the layer system rather than off the printed list: the
     * keyword may be printed, equipped or granted until end of turn, and a
     * removal layer takes it away in all three cases the way the rules would.
     */
    private static void stripKeyword(Game fork, Card forkCarrier, String keyword) {
        forkCarrier.addChangedCardKeywords(
                null, List.of(keyword), false,
                fork.getNextTimestamp(), null);
    }

    /**
     * What a probe's damage step did: the fork's combat, read as the observed
     * record reads it, and the events the bus published.
     */
    private record ProbedStep(Map<String, Object> combatFields, List<EffectEvent> events) {
    }

    /**
     * Assign and deal the fork's damage for one step, then let the deaths
     * happen, listening throughout.
     *
     * @param firstStrike whether this is the first-strike damage step
     */
    private static ProbedStep runDamageStep(Game fork, boolean firstStrike) {
        // Listening from here, once the keyword is gone: the stripped
        // keyword is the branch's input, and a sink primed before it
        // would report its removal as something the step did. No root:
        // a damage step is a turn-based action, not a resolution, so
        // this branch's events attribute the way the observed combat
        // record's do -- from the resolving-clause pointer alone, which
        // names nothing.
        ForkEventSink sink = ForkEventSink.forProbe(fork);
        fork.subscribeToEvents(sink);
        fork.getCombat().removeAbsentCombatants();
        // Who is blocking whom, before the damage removes any of them; the
        // assignment only exists after. Both are read as the observed
        // record reads them, so the branch and the record it mirrors agree.
        CombatShape shape = CombatShape.of(fork.getCombat());
        if (fork.getCombat().assignCombatDamage(firstStrike)) {
            shape.addAssignment(fork.getCombat());
            fork.getCombat().dealAssignedDamage();
            // And then let the deaths happen. Damage alone kills nothing;
            // a creature with lethal damage on it leaves the battlefield
            // during state-based actions, and stopping before them would make
            // the branch systematically miss every death the real step had —
            // biasing the difference gate 2 computes towards "the keyword
            // changed nothing". The same call Forge's own simulator makes
            // after resolving a stack.
            fork.getAction().checkStateEffects(false, new HashSet<>());
        }
        return new ProbedStep(shape.fields(), sink.events());
    }

    /**
     * A probe branch waiting for the record it mirrors.
     *
     * <p>Holds what the fork observed until the real combat record has an id.
     * The probe's own budget was already spent taking the fork, so a branch
     * that is never completed still counted — the cost was the simulation, not
     * the write.
     *
     * <p>{@code substep} names the damage step this branched from, the one
     * the bracket collector's record names. Without it a branch taken at the
     * first-strike step would be completed against whichever combat record
     * came next, the <b>regular</b> step's, and gate 2 would read a
     * counterfactual for a step that did not happen against a real record for
     * a different one.
     */
    public record HeldProbe(
            String keyword, String carrier, String state,
            List<EffectEvent> events, String actor, Map<String, Object> combatFields,
            DamageStep substep, int turn) {

        /**
         * The branch's payload, naming what was perturbed.
         *
         * <p>Both the keyword and the creature it came from: a board with two
         * tramplers gives the keyword alone two readings, and the evaluator
         * reproducing this perturbation model-side would strip the wrong one.
         *
         * <p>{@code combatFields} is the fork's own combat: a view of the
         * {@link CombatShape} the probe's damage step filled in, rendered
         * here rather than copied when the branch was taken. Nothing writes to
         * that shape once the step has run, so it renders as the step left it.
         * It is the counterfactual's half of what gate 2
         * compares: the observed record says who blocked whom and how the
         * damage was split with the keyword, and this says the same about the
         * board without it.
         */
        String payload() {
            return EffectRecord.probePayload(combatFields, events, keyword, carrier);
        }
    }

    /**
     * Write a held branch, now that the record it mirrors has an id.
     *
     * <p>As an ordinary combat record with {@code fork = true} and {@code
     * interventional = false} — which is what distinguishes a probe from an
     * intervention by flags alone — and {@code mirror_of} naming the real
     * record it varies. The real-versus-fork <b>difference is never computed
     * here</b> and never becomes a training target: it is a diagnostic about
     * the model, computed at evaluation time, and training on it would teach
     * the model to reproduce its own errors.
     *
     * <p>Only if the two describe the same moment. A branch is completed
     * against the next combat record of its own substep, and a damage step that
     * wrote no real record leaves its branch waiting for one, which can then
     * be completed against a mirror turns later.
     *
     * <p>Dropped rather than relabelled, which is the choice worth stating.
     * The fork's turn is not stale: the copier carries the counter across, and
     * the branch's snapshot honestly describes the board of the turn it was
     * taken on. Stamping the mirror's turn over it would make the state block
     * lie about a board that is genuinely several turns old, and gate 2 would
     * read the difference between two unrelated combats as something the
     * keyword did. The branch's budget was spent taking the fork either way;
     * the only question is whether an unreadable counterfactual is written,
     * and it is not — the same policy {@code discardStaleProbes} applies to a
     * branch of the wrong substep.
     */
    public boolean writeHeldProbe(HeldProbe held, String mirrorOfRecordId) {
        if (held == null || mirrorOfRecordId == null) {
            return false;
        }
        int now = liveTurn();
        if (held.turn() != TURN_UNKNOWN && now != TURN_UNKNOWN
                && held.turn() != now) {
            System.err.println(
                    "Effect records: dropping a probe branch taken on turn "
                            + held.turn() + " that would mirror "
                            + mirrorOfRecordId + " from turn " + now
                            + "; its own damage step wrote no record");
            return false;
        }
        context.emit(context.newRecord(RecordKind.COMBAT)
                .fork(true)
                .interventional(false)
                .mirrorOf(mirrorOfRecordId)
                .actor(held.actor())
                .state(held.state())
                .payload(held.payload()));
        return true;
    }

    // ── plumbing ────────────────────────────────────────────────────────

    /**
     * The live game's turn, or {@link #TURN_UNKNOWN}.
     *
     * <p>Read from the game this collector forks from rather than from a fork,
     * so a branch and the record it will mirror are timed by the same clock:
     * the mirror's snapshot reads this same counter at the instant the branch
     * is completed.
     */
    private int liveTurn() {
        if (game == null || game.getPhaseHandler() == null) {
            return TURN_UNKNOWN;
        }
        return game.getPhaseHandler().getTurn();
    }
}
