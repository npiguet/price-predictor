package com.pricepredictor.connector.effects;

import forge.game.CardTraitBase;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.replacement.ReplacementEffect;

import java.lang.reflect.InvocationHandler;
import java.util.List;
import java.util.function.Supplier;

/**
 * The four record kinds the engine patch unlocks.
 *
 * <p>{@code rewrite}, {@code trigger}, {@code playability} and {@code mana}
 * records all need a hook at a point the bus never publishes: what a replacement
 * received before it rewrote it, a trigger condition that evaluated to false, a
 * legality verdict the rules computed and discarded, a mana ability that resolved
 * on the inline path. Each is installed reflectively and simply does not run on
 * an unpatched checkout.
 *
 * <p>Two caps live here rather than in the supervisor, because both are
 * per-worker-process quantities the Python side cannot see: {@code --mana-cap}
 * counts resolution records per unique mana-ability text in this JVM, and
 * {@code --playability-rate} samples the {@code decision}-subkind logging points
 * (the {@code attackers} and {@code blockers} ones are always logged, because
 * they fire once per combat rather than once per candidate).
 *
 * <p>{@code continuous} records take no cap at all: coalescing per stable board
 * is itself the cap. A static that applies on every recompute would otherwise
 * write a record per priority pass, and none of them would say anything the
 * first did not.
 */
public final class PatchedCollectors implements AutoCloseable {

    private final LiveGame live;
    private final CollectorContext context;
    private final SnapshotBuilder snapshots;
    private final AttributionMode mode;

    /** The records this game has written, shared by every channel that deduplicates. */
    private final RecordDedup dedup = new RecordDedup();
    private final java.util.Random sampler;
    private final TriggerChannel triggers;
    private final PlayabilityChannel playability;
    private final ContinuousChannel continuous;
    private final ManaChannel mana;
    private final ClauseOutcomeRouter clauses;

    /** A hook this collector installs a listener on. */
    private record Listener(PatchHooks.Hook hook, Supplier<InvocationHandler> handler) {
    }

    /** Every listener this collector installs, in the order it installs them. */
    private final List<Listener> listeners;


    /** @param bracket where the clause and outcome hooks file their events */
    PatchedCollectors(
            Game game, CollectorContext context, CollectionCaps caps, long seed,
            BusBracketCollector bracket) {
        this.live = new LiveGame(game);
        this.context = context;
        this.snapshots = live.snapshots();
        this.mode = context.mode();
        this.sampler = new java.util.Random(GameRecording.scramble(seed));
        this.triggers = new TriggerChannel(
                live, context, dedup, new TriggerNegativeSampler(sampler));
        this.playability = new PlayabilityChannel(
                live, context, dedup, sampler, caps.playabilityRate());
        this.continuous = new ContinuousChannel(live, context, dedup);
        this.mana = new ManaChannel(
                live, context, new ManaReservoir(caps.manaCap(), sampler));
        this.clauses = new ClauseOutcomeRouter(live, bracket);
        this.listeners = List.of(
                new Listener(PatchHooks.REPLACEMENT_LISTENER, this::rewriteHandler),
                new Listener(PatchHooks.TRIGGER_LISTENER, triggers::handler),
                new Listener(PatchHooks.PLAYABILITY_LISTENER, playability::decisionHandler),
                new Listener(PatchHooks.MANA_LISTENER, mana::handler),
                new Listener(PatchHooks.COMBAT_LISTENER, playability::legalityHandler),
                new Listener(PatchHooks.CLAUSE_LISTENER, clauses::clauseHandler),
                new Listener(PatchHooks.OUTCOME_LISTENER, clauses::outcomeHandler));
    }

    /** The trigger-condition channel. */
    TriggerChannel triggers() {
        return triggers;
    }

    /** The decision and legality channel, which the random seat also writes through. */
    PlayabilityChannel playability() {
        return playability;
    }

    /** The continuous channel, collected at each phase boundary rather than on a hook. */
    ContinuousChannel continuous() {
        return continuous;
    }

    /** The mana channel. */
    ManaChannel mana() {
        return mana;
    }

    /** The clause and outcome router. */
    ClauseOutcomeRouter clauses() {
        return clauses;
    }

    /**
     * Install every hook the checkout offers.
     *
     * @return how many were found
     */
    int install() {
        if (mode == AttributionMode.DEGRADED) {
            return 0;
        }
        int installed = 0;
        for (Listener listener : listeners) {
            if (listener.hook().install(listener.handler().get())) {
                installed++;
            }
        }
        return installed;
    }

    @Override
    public void close() {
        // Before the hooks come down: this is the last moment the game can be
        // said to have ended, and the mana reservoir is only decided then.
        mana.flush();
        for (Listener listener : listeners) {
            listener.hook().uninstall();
        }
        // Last, and outside the hook teardown: the question the tally answers
        // is about the run, so it is asked once per game whatever the game did.
        JVM_REWRITES.endOfGame();
    }

    // ── rewrite records ─────────────────────────────────────────────────

    /**
     * One record per replacement, carrying the event it received, the one it
     * produced, which of the five results it returned and what ran instead.
     *
     * <p>Two replacements stacked on one event each get their own record, and
     * the second's incoming is the first's outgoing — which only holds because
     * the hook deep-copies the parameter map before the call.
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
     * <p>Package-private rather than private, which none of the other handlers
     * needs to be: that argument order is a runtime contract with no compiler
     * behind it, so {@code RewriteContractTest} calls this with a synthesized
     * argument array in the compiler's place.
     */
    InvocationHandler rewriteHandler() {
        return PatchHooks.REPLACEMENT_LISTENER.listener((callback, args) -> {
            // A fork's own replacements resolve through this same JVM-static
            // hook (ForkCollector.forceResolution
            // runs the fork's forced ability through the real resolution
            // pipeline, which is where replacements apply). args[0] is
            // declared ReplacementEffect on Forge's side -- always a
            // CardTraitBase with a real host -- so a positively-identified
            // other game is dropped; anything this cannot identify is left
            // alone rather than newly dropped.
            CardTraitBase trait = args.get(0, CardTraitBase.class);
            if (trait != null && !live.owns(trait)) {
                return;
            }
            EffectRecord record = rewriteRecord(
                    args.get(0), args.get(1), args.get(2), args.optional(3), args.optional(4));
            if (!dedup.allowDistinct(
                    RecordKind.REWRITE.wireName(), record.payloadJson(), record.stateJson())) {
                return;
            }
            emit(record);
        });
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
     * There used to be a rule here dropping a pair whose halves read alike, and
     * it took 87% of them, because a substitution leaves the map untouched and
     * looked exactly like a replacement that did nothing. The result argument is
     * what tells those apart, so there is no longer such a thing as an
     * uninformative rewrite record: a {@code not_replaced} is this channel's
     * negative, the same role a non-fired evaluation plays on the trigger
     * channel, and it is what teaches when a replacement applies rather than
     * only what it does when it has.
     *
     * @param result     Forge's {@code ReplacementResult}, or null where the
     *                   replacement threw or the listener predates the argument
     * @param replacedBy the {@code ReplaceWith$} ability that stood in for the
     *                   event, or null where none ran
     */
    EffectRecord rewriteRecord(
            Object trait, Object before, Object after, Object result,
            Object replacedBy) {
        CardTraitBase acting = trait instanceof CardTraitBase ctb ? ctb : null;
        Card source = acting == null ? null : acting.getHostCard();
        EffectEvent incoming = RunParamNormalizer.describeParams(trait, before);
        EffectEvent rendered = RunParamNormalizer.describeParams(trait, after);
        // Null rather than a copy, and this is the whole of the contract. The
        // two halves were byte-identical on 87% of the candidates because a
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
        List<ProvenanceKey> ranKeys = ran == null ? List.of() : ProvenanceKey.keysOf(ran);
        String wireResult = EffectRecord.rewriteResult(result);
        JVM_REWRITES.record(
                RunParamNormalizer.modeOf(trait), wireResult, outgoing != null,
                !ranKeys.isEmpty(), before, after);
        return context.newRecord(RecordKind.REWRITE)
                .actor(live.controllerOf(source))
                .ability(ProvenanceKey.resolve(acting))
                .state(snapshots.ofTrait(source, incoming).render())
                .payload(EffectRecord.rewritePayload(
                        incoming, outgoing, wireResult, ranKeys));
    }

    /** The worker's rewrite tally, shared by every game it plays. */
    static final RewriteTally JVM_REWRITES = new RewriteTally();

    private void emit(EffectRecord record) {
        context.emit(record);
    }
}
