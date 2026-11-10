package com.pricepredictor.connector.effects;

import forge.game.CardTraitBase;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.List;

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
 */
final class RewriteChannel {

    /** The worker's rewrite tally, shared by every game it plays. */
    static final RewriteTally JVM_REWRITES = new RewriteTally();

    private final LiveGame live;
    private final CollectorContext context;
    private final RecordDedup dedup;

    /** The boards read at each open {@code onReplacing}, innermost on top. */
    private final Deque<Pending> replacing = new ArrayDeque<>();

    /** A replacement's board, read before it ran, and the trait it was read for. */
    private record Pending(Object trait, String state) {
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
    PatchHooks.Listener handler() {
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
                case ON_REPLACEMENT -> context.emitDistinct(record(
                        args.get(0), args.get(1), args.get(2), args.optional(3),
                        args.optional(4), boardBefore(args.get(0))),
                        dedup);
                default -> { }
            }
        });
    }

    /**
     * The board a replacement is about to change, with the event as it
     * arrived: read at {@code onReplacing}, before anything ran.
     */
    private String board(Object trait, Object runParams) {
        return live.traitState(trait, RunParamNormalizer.describeParams(trait, runParams));
    }

    /**
     * The board read when the closing replacement started, or null where
     * none was read for it: a jar whose listener never says {@code
     * onReplacing}, which the record then reads as before, now.
     */
    private String boardBefore(Object trait) {
        Pending open = replacing.peek();
        if (open == null || open.trait() != trait) {
            return null;
        }
        replacing.pop();
        return open.state();
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
            Object replacedBy, String state) {
        EffectEvent incoming = RunParamNormalizer.describeParams(trait, before);
        EffectEvent rendered = RunParamNormalizer.describeParams(trait, after);
        // Null rather than a copy, and this is the whole of the contract. A
        // substitution does not edit the map -- it runs another ability and
        // returns a result -- so a copy here says "changed nothing observable"
        // about a replacement that exiled the card instead of killing it. What
        // survives the comparison is the genuine in-place edit:
        // ReplaceCounterEffect.setCount, a halved draw, Orim's Cure taking five
        // combat damage down to one.
        EffectEvent outgoing =
                incoming.toJson().equals(rendered.toJson()) ? null : rendered;
        CardTraitBase ran = replacedBy instanceof CardTraitBase ctb ? ctb : null;
        // Keyed through the same resolver as the record's own acting line, so
        // both join the provenance sidecar identically. Usually it answers that
        // same printed line, because ReplaceWith$ names an SVar on the card the
        // replacement is printed on -- what the field carries that `ability`
        // does not is that an ability ran at all, and the rarer case where the
        // overriding ability was granted from somewhere else.
        List<ProvenanceKey> ranKeys = ran == null ? List.of() : ProvenanceResolver.keysOf(ran);
        String wireResult = RecordPayloads.rewriteResult(result);
        JVM_REWRITES.record(
                RunParamNormalizer.modeOf(trait), wireResult, outgoing != null,
                !ranKeys.isEmpty(), before, after);
        EffectRecord record = state == null
                ? live.traitRecord(context, RecordKind.REWRITE, trait, incoming)
                : live.traitRecordOn(
                        context, RecordKind.REWRITE, trait, LiveGame.lineOf(trait), state);
        return record.payload(RecordPayloads.rewritePayload(
                incoming, outgoing, wireResult, ranKeys));
    }
}
