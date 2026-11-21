package com.pricepredictor.connector.effects;

import forge.game.CardTraitBase;
import forge.game.spellability.SpellAbility;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.function.BiPredicate;

/**
 * One {@code rewrite} record per replacement, carrying the event it received,
 * the one it produced, which of the five results it returned and what ran
 * instead.
 *
 * <p>Two replacements stacked on one event each get their own record, and the
 * second's incoming is the first's outgoing — which only holds because the
 * hook deep-copies the parameter map before the call.
 *
 * <p>The record's board is read before the replacement runs ({@code
 * onReplacing}, H1-2, root § 8.14): by {@code onReplacement} its {@code
 * ReplaceWith$} ability has already run, so the land that enters tapped would
 * already be tapped and the card exiled instead already in exile. The engine
 * pairs each {@code onReplacing} with exactly one {@code onReplacement}, the
 * pairs nested strictly — an exile-instead runs its own zone change, which
 * other replacements see — so the boards wait on a stack, the innermost open
 * one closing first.
 *
 * <p>A damage replacement may set an ability aside to run once every
 * replacement of the damage event has applied (Swans of Bryn Argoll's draw,
 * Rock Hydra's counter removal, a {@code Prevent$ True} rider): Forge hands
 * it as {@code onReplacement}'s sixth argument, {@code buffered}, never as
 * the ability that ran instead. Such a record names it in {@code buffered}
 * and waits for its run, which Forge brackets between
 * {@code onBufferedAbilityStarting} and {@code onBufferedAbilityFinished}:
 * every event published between the two is the run's, taken from the
 * bracket into the waiting record's {@code buffered_events}. Forge sums the
 * damage of several replacements naming one ability into one run, so the
 * oldest record waiting on that ability takes the run's events and the rest,
 * written when the state-based loop next begins, say {@code buffered_ran}
 * with no events; a record whose ability never ran says it did not.
 *
 * <p>An {@code AddCounter} replacement over a map of several counter types
 * writes one record per type whose count it changed, none for a type it left
 * as it came ({@link RunParamNormalizer#changedCounterTypes}): each carries
 * that type's counts as its {@code incoming} and {@code outgoing}, the
 * call's result and the ability that ran, and the board read before the
 * call with that {@code incoming} as its pending event.
 */
final class RewriteChannel implements EventRouter.EventTap {

    /** The worker's rewrite tally, shared by every game it plays. */
    static final RewriteTally JVM_REWRITES = new RewriteTally();

    private final LiveGame live;
    private final CollectorContext context;
    private final RecordDedup dedup;

    /** The boards read at each open {@code onReplacing}, innermost on top. */
    private final Deque<Pending> replacing = new ArrayDeque<>();

    /** A replacement's board, read before it ran, and the trait it was read for. */
    private record Pending(Object trait, Snapshot board) {
    }

    /** The records waiting on their buffered ability's run, oldest first. */
    private final List<Waiting> waiting = new ArrayList<>();
    /** The buffered abilities running now, innermost on top. */
    private final Deque<Run> running = new ArrayDeque<>();
    /** The (effect, ability) pairs that ran since the waiting records were last written. */
    private final List<Waiting> ranPairs = new ArrayList<>();

    /**
     * A replacement effect and the buffered ability it set aside, which Forge
     * names in the bracket of the run; matched by identity, not equality.
     */
    private record BufferedPair(Object effect, Object ability) {
        boolean sameAs(BufferedPair other) {
            return effect == other.effect && ability == other.ability;
        }
    }

    /** What {@code onReplacement} reported: the effect, the event before and after, the call's outcome. */
    private record Reported(
            Object trait, Object before, Object after, Object result, Object replacedBy,
            Object buffered) {
    }

    /** A replacement's record, built but for its payload, waiting on its buffered ability. */
    private record Waiting(BufferedPair pair, EffectRecord record, Rewrite rewrite) {
        boolean waitsOn(BufferedPair other) {
            return pair.sameAs(other);
        }
    }

    /**
     * What a rewrite record's payload says, rendered when it is written.
     *
     * @param buffered the buffered ability's keys, or null when none was set aside
     */
    private record Rewrite(
            String mode, EffectEvent incoming, EffectEvent outgoing, boolean incomingUnknown,
            String result, List<ProvenanceKey> replacedBy, List<ProvenanceKey> buffered,
            Object before, Object after) {
    }

    /** One buffered ability's run, and the record that takes its events, or none. */
    private record Run(BufferedPair pair, Waiting taker, List<EffectEvent> events) {
    }

    RewriteChannel(LiveGame live, CollectorContext context, RecordDedup dedup) {
        this.live = live;
        this.context = context;
        this.dedup = dedup;
    }

    /**
     * The replacement hook's listener.
     *
     * <p>The last two arguments are read by length rather than assumed present.
     * The listener is installed reflectively over whatever interface the patched
     * checkout declares, so its arity is a runtime fact: against a Forge jar
     * built before the result and the substituted ability were added, this is
     * still a three-argument call and the record honestly says it learnt
     * neither. Indexing {@code args[3]} unguarded would instead throw inside a
     * dynamic proxy, and the engine would surface that in the middle of a
     * replacement as an UndeclaredThrowableException.
     *
     * <p>Package-private: that argument order is a runtime contract with no
     * compiler behind it, so {@code RewriteContractTest} calls this with a
     * synthesized argument array in the compiler's place.
     */
    Listener handler() {
        return PatchHooks.REPLACEMENT_LISTENER.listener((callback, args) -> {
            // A fork's own replacements resolve through this same JVM-static
            // hook (InterventionFork.forceResolution
            // runs the fork's forced ability through the real resolution
            // pipeline, which is where replacements apply). args[0] is
            // declared ReplacementEffect on Forge's side -- always a
            // CardTraitBase with a real host -- so a positively-identified
            // other game is dropped.
            if (live.foreign(args.get(0))) {
                return;
            }
            switch (callback) {
                case ON_REPLACING -> replacing.push(new Pending(
                        args.get(0), board(args.get(0), args.get(1))));
                case ON_REPLACEMENT -> replaced(
                        new Reported(
                                args.get(0), args.get(1), args.get(2), args.optional(3),
                                args.optional(4), args.optional(5)),
                        boardBefore(args.get(0)));
                case ON_BUFFERED_ABILITY_STARTING ->
                        bufferedStarting(new BufferedPair(args.get(0), args.get(1)));
                case ON_BUFFERED_ABILITY_FINISHED ->
                        bufferedFinished(new BufferedPair(args.get(0), args.get(1)));
                default -> { }
            }
        });
    }

    /**
     * One replacement reported: its record, or one per counter type it
     * changed, each written now or, when the replacement set an ability
     * aside, held until that ability runs.
     *
     * @param board the board read at {@code onReplacing}, or null to read it now
     */
    private void replaced(Reported reported, Snapshot board) {
        List<String> types = RunParamNormalizer.changedCounterTypes(
                reported.trait(), reported.before(), reported.after());
        if (types.isEmpty()) {
            replaced(reported, rewrite(reported, null), board == null ? null : board.state());
            return;
        }
        for (String type : types) {
            Rewrite rewrite = rewrite(reported, type);
            // The board this type's event would have been read with: the
            // same subjects and last-known card as the whole map's.
            replaced(reported, rewrite,
                    board == null ? null : board.withPendingEvent(rewrite.incoming()).state());
        }
    }

    /** One record of a replacement, on {@code state}, or on the board now. */
    private void replaced(Reported reported, Rewrite rewrite, RecordState state) {
        EffectRecord record = shell(reported.trait(), rewrite.incoming(), state);
        if (rewrite.buffered() == null) {
            write(record, rewrite, false, List.of());
            return;
        }
        waiting.add(new Waiting(
                new BufferedPair(reported.trait(), reported.buffered()), record, rewrite));
    }

    /** A buffered ability starts its run: the oldest record waiting on it takes its events. */
    private void bufferedStarting(BufferedPair started) {
        Waiting taker = null;
        for (Waiting candidate : waiting) {
            if (candidate.waitsOn(started)) {
                taker = candidate;
                break;
            }
        }
        running.push(new Run(started, taker, new ArrayList<>()));
    }

    /** A buffered ability's run is over: its taker is written with the run's events. */
    private void bufferedFinished(BufferedPair finished) {
        Run run = running.peek();
        if (run == null || !run.pair().sameAs(finished)) {
            return;
        }
        running.pop();
        if (run.taker() == null) {
            return;
        }
        waiting.remove(run.taker());
        ranPairs.add(run.taker());
        write(run.taker().record(), run.taker().rewrite(), true, run.events());
    }

    /**
     * An event published while a buffered ability runs is its run's, when a
     * record waits to take it; the bracket routes it otherwise.
     */
    @Override
    public boolean take(EffectEvent event, String namedCause) {
        Run run = running.peek();
        if (run == null || run.taker() == null) {
            return false;
        }
        run.events().add(EventAttribution.stamp(
                event, run.pair().ability() instanceof SpellAbility sa ? sa : null, namedCause));
        return true;
    }

    /**
     * Write the records still waiting: one whose ability ran (its run went to
     * an older record, Forge having summed the damage) says it ran, with no
     * events; one whose ability never ran says it did not. At the state-based
     * loop's start, which no buffered run straddles, under the per-game cap.
     */
    void writeWaiting() {
        writeWaiting(context::emitDistinct);
    }

    /**
     * {@link #writeWaiting()} at the game's end: an end-of-game write, which
     * the per-game cap does not withhold.
     */
    void writeWaitingAtGameEnd() {
        writeWaiting(context::emitDistinctAtGameEnd);
    }

    private void writeWaiting(BiPredicate<EffectRecord, RecordDedup> door) {
        for (Waiting held : waiting) {
            boolean ran = ranPairs.stream()
                    .anyMatch(done -> done.waitsOn(held.pair()));
            fill(held.record(), held.rewrite(), ran, List.of());
            count(held.rewrite(), door.test(held.record(), dedup));
        }
        waiting.clear();
        ranPairs.clear();
    }

    /** The state-based loop's signals: where the waiting records are written. */
    PhaseSignals.Sink phaseSink() {
        return new PhaseSignals.Sink() {
            @Override
            public void stateBasedLoopBegan() {
                writeWaiting();
            }
        };
    }

    /** Write a record with its payload, under the per-game cap, and count it if it got in. */
    private void write(
            EffectRecord record, Rewrite rewrite, boolean bufferedRan,
            List<EffectEvent> bufferedEvents) {
        fill(record, rewrite, bufferedRan, bufferedEvents);
        count(rewrite, context.emitDistinct(record, dedup));
    }

    /** Give a record its payload. */
    private void fill(
            EffectRecord record, Rewrite rewrite, boolean bufferedRan,
            List<EffectEvent> bufferedEvents) {
        record.payload(RecordPayloads.rewritePayload(
                rewritten(rewrite),
                rewrite.buffered() == null
                        ? null
                        : new RecordPayloads.Buffered(
                                rewrite.buffered(), bufferedRan, bufferedEvents)));
    }

    /**
     * Count a record in the worker's tally when it reached the shard of a
     * live game, never a fork's (C5 #16): one the cap withheld is not in the
     * shard, so it is not in the tally.
     */
    private void count(Rewrite rewrite, boolean reachedTheShard) {
        if (reachedTheShard && !context.isFork()) {
            JVM_REWRITES.record(
                    rewrite.mode(), rewrite.result(), rewrite.outgoing() != null,
                    !rewrite.replacedBy().isEmpty(), rewrite.before(), rewrite.after());
        }
    }

    /**
     * The board a replacement is about to change, with the event as it
     * arrived: read at {@code onReplacing}, before anything ran.
     */
    private Snapshot board(Object trait, Object runParams) {
        return live.traitSnapshot(trait, RunParamNormalizer.describeParams(trait, runParams));
    }

    /**
     * The board read when the closing replacement started, or null where
     * none was read for it: a jar whose listener never says {@code
     * onReplacing}, which the record then reads as before, now.
     */
    private Snapshot boardBefore(Object trait) {
        Pending open = replacing.peek();
        if (open == null || open.trait() != trait) {
            return null;
        }
        replacing.pop();
        return open.board();
    }

    /**
     * One replacement's record, built from the effect the hook hands over.
     *
     * <p>Package-private rather than inlined into the handler: what the record
     * says about the acting line is the part worth testing, and a dynamic proxy
     * over an interface that only exists on a patched checkout is not a thing a
     * unit test can call.
     *
     * <p>The host is taken as delivered, deliberately not re-resolved through
     * {@code game.getCardState(host)} the way {@code executeReplacementUnobserved}
     * does: that answers an alternate-state card, and the entity id in
     * {@code state.entities} is the one this object carries.
     *
     * <p><b>Every replacement writes a record, including the ones that declined.</b>
     * A substitution leaves the map untouched and so looks exactly like a
     * replacement that did nothing. The result argument is what tells those
     * apart, so no rewrite record is uninformative: a {@code not_replaced} is
     * this channel's negative, the same role a non-fired evaluation plays on
     * the trigger channel, and it is what teaches when a replacement applies
     * rather than only what it does when it has.
     *
     * @param result     Forge's {@code ReplacementResult}, or null where the
     *                   replacement threw or the listener predates the argument
     * @param replacedBy the {@code ReplaceWith$} ability that stood in for the
     *                   event, or null where none ran
     */
    EffectRecord record(
            Object trait, Object before, Object after, Object result,
            Object replacedBy) {
        return record(trait, before, after, result, replacedBy, null);
    }

    /**
     * The same, on the board read before the replacement ran.
     *
     * @param state the board read at {@code onReplacing}, or null to read it now
     */
    EffectRecord record(
            Object trait, Object before, Object after, Object result,
            Object replacedBy, RecordState state) {
        Rewrite rewrite = rewrite(
                new Reported(trait, before, after, result, replacedBy, null), null);
        return shell(trait, rewrite.incoming(), state)
                .payload(RecordPayloads.rewritePayload(rewritten(rewrite), null));
    }

    /** What the rewrite says of the replacement itself. */
    private static RecordPayloads.Rewritten rewritten(Rewrite rewrite) {
        return new RecordPayloads.Rewritten(
                rewrite.incoming(), rewrite.outgoing(), rewrite.result(), rewrite.replacedBy(),
                rewrite.incomingUnknown());
    }

    /** A rewrite record but for its payload, on {@code state}, or on the board now. */
    private EffectRecord shell(Object trait, EffectEvent incoming, RecordState state) {
        return state == null
                ? live.traitRecord(context, RecordKind.REWRITE, trait, incoming)
                : live.traitRecordOn(
                        new LiveGame.ActingTrait(
                                context, RecordKind.REWRITE, trait, LiveGame.lineOf(trait)),
                        state);
    }

    /**
     * What one replacement's record says.
     *
     * <p>A null {@code before} means Forge could not copy the run parameters:
     * the event as it arrived is unknown, so nothing is compared against it.
     * {@code incoming} is then the event as it left, {@code outgoing} null and
     * {@code incoming_unknown} true.
     *
     * @param counterType the one counter type this record is about, or null
     *                    for the whole event
     */
    private Rewrite rewrite(Reported reported, String counterType) {
        Object trait = reported.trait();
        Object before = reported.before();
        Object after = reported.after();
        boolean unknown = before == null;
        EffectEvent rendered = RunParamNormalizer.describeParams(trait, after, counterType);
        EffectEvent incoming = unknown
                ? rendered : RunParamNormalizer.describeParams(trait, before, counterType);
        // Null rather than a copy, and this is the whole of the contract. A
        // substitution does not edit the map -- it runs another ability and
        // returns a result -- so a copy here says "changed nothing observable"
        // about a replacement that exiled the card instead of killing it. What
        // survives the comparison is the genuine in-place edit:
        // ReplaceCounterEffect.setCount, a halved draw, Orim's Cure taking five
        // combat damage down to one.
        EffectEvent outgoing = unknown || incoming.toJson().equals(rendered.toJson())
                ? null : rendered;
        CardTraitBase ran = reported.replacedBy() instanceof CardTraitBase ctb ? ctb : null;
        // Keyed through the same resolver as the record's own acting line, so
        // both join the provenance sidecar identically. Usually it answers that
        // same printed line, because ReplaceWith$ names an SVar on the card the
        // replacement is printed on -- what the field carries that `ability`
        // does not is that an ability ran at all, and the rarer case where the
        // overriding ability was granted from somewhere else.
        List<ProvenanceKey> ranKeys = ran == null ? List.of() : ProvenanceResolver.keysOf(ran);
        List<ProvenanceKey> bufferedKeys = reported.buffered() instanceof CardTraitBase set
                ? ProvenanceResolver.keysOf(set) : null;
        return new Rewrite(
                RunParamNormalizer.modeOf(trait), incoming, outgoing, unknown,
                RecordPayloads.rewriteResult(reported.result()), ranKeys, bufferedKeys,
                unknown ? after : before, after);
    }
}
