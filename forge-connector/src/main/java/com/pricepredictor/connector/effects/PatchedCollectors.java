package com.pricepredictor.connector.effects;

import com.google.common.collect.Table;
import forge.card.CardChangedType;
import forge.card.CardTypeView;
import forge.card.ColorSet;
import forge.card.RemoveType;
import forge.card.StateChangedType;
import forge.card.WordChangedType;
import forge.game.CardTraitBase;
import forge.game.Game;
import forge.game.GameEntity;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.keyword.KeywordInterface;
import forge.game.keyword.KeywordsChange;
import forge.game.player.Player;
import forge.game.replacement.ReplacementEffect;
import forge.game.spellability.SpellAbility;
import forge.game.staticability.StaticAbility;
import forge.game.trigger.Trigger;
import forge.game.zone.ZoneType;
import org.apache.commons.lang3.tuple.Pair;

import java.lang.reflect.InvocationHandler;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.StringJoiner;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Function;
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

    private final Game game;
    private final CollectorContext context;
    private final SnapshotBuilder snapshots;
    private final AttributionMode mode;
    private final String gameId;
    private final CollectionCaps caps;

    /** The sampled mana records, held until the game ends. */
    private final ManaReservoir manaReservoir;
    /** Lines already offered an intervention, so a redraw does not repeat one. */
    private final Set<String> interveneSeen = new LinkedHashSet<>();
    /** Stage three's forks, or null when the run takes none. */
    private ForkCollector forks;
    /**
     * The bracket this game's clause events are filed into, or null when none
     * is wired: {@code recordClauseEvent} is the door, and only the bracket
     * collector for this same game knows which resolution is open.
     */
    private BusBracketCollector bracket;
    /** Probe branches waiting for the combat record they mirror. */
    private final ProbeQueue heldProbes = new ProbeQueue();
    /** Boards a continuous static has already been recorded on, for coalescing. */
    private final Set<String> coalescedBoards = new LinkedHashSet<>();
    /** Records already written this game, by a hash of what they say. */
    private final Set<Long> coalescedRecords = new java.util.HashSet<>();
    /** Each trigger mode's written evaluations and pending offers, this game. */
    private final Map<String, ModeBalance> triggerBalance = new HashMap<>();
    /** Records written per acting line and verdict, for the flood cap. */
    private final Map<LineVerdict, Integer> triggerPerLine = new HashMap<>();
    /** Static ability by layer-table id, resolved once per game. */
    private final Map<Long, StaticAbility> staticsById = new HashMap<>();
    private final java.util.Random sampler;

    /**
     * Whether the engine can say what a card would be without one static.
     *
     * <p>Separate from {@link AttributionMode}, which probes the sub-ability
     * hook. These three arrived later, so a checkout patched against an older
     * revision of the patch reports {@code patched} and still cannot suppress.
     */
    static final boolean CAN_SUPPRESS = PatchHooks.TYPE_WITHOUT.present()
            && PatchHooks.COLOR_WITHOUT.present()
            && PatchHooks.KEYWORDS_WITHOUT.present();

    /** A hook this collector installs a listener on. */
    private record Listener(PatchHooks.Hook hook, Supplier<InvocationHandler> handler) {
    }

    /** Every listener this collector installs, in the order it installs them. */
    private final List<Listener> listeners = List.of(
            new Listener(PatchHooks.REPLACEMENT_LISTENER, this::rewriteHandler),
            new Listener(PatchHooks.TRIGGER_LISTENER, this::triggerFireHandler),
            new Listener(PatchHooks.PLAYABILITY_LISTENER, this::playabilityHandler),
            new Listener(PatchHooks.MANA_LISTENER, this::manaHandler),
            new Listener(PatchHooks.COMBAT_LISTENER, this::combatLegalityHandler),
            new Listener(PatchHooks.CLAUSE_LISTENER, this::clauseHandler),
            new Listener(PatchHooks.OUTCOME_LISTENER, this::outcomeHandler));

    public PatchedCollectors(
            Game game, RecordShardWriter writer, String gameId,
            CollectionCaps caps, long seed) {
        this.game = game;
        this.context = new CollectorContext(writer, gameId);
        this.gameId = gameId;
        this.caps = caps;
        this.snapshots = new SnapshotBuilder(game);
        this.mode = context.mode();
        this.sampler = new java.util.Random(scramble(seed));
        this.manaReservoir = new ManaReservoir(caps.manaCap(), sampler);
    }

    /**
     * Spread out a seed before it reaches {@link java.util.Random}.
     *
     * <p>Games are seeded consecutively, and a linear congruential generator's
     * first output is close to a linear function of its seed — so consecutive
     * seeds make consecutive games take nearly the same first sampling
     * decisions. That would correlate the mana reservoir and the playability
     * sample across a whole run, in a way no test of one game could see.
     *
     * <p>SplitMix64's finalizer, which is the standard mixer for exactly this.
     */
    public static long scramble(long seed) {
        long z = seed + 0x9E3779B97F4A7C15L;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    /**
     * Attach stage three's fork collector.
     *
     * <p>Optional and off by default: with none attached, or with
     * {@code --interventions-per-game 0}, no game is ever copied. Separate from
     * the constructor because forking is a build decision gate 2 informs, not
     * something every collecting run should pay for.
     */
    public void withForks(ForkCollector forks) {
        this.forks = forks;
        if (forks != null) {
            forks.setRandomSeatPlayerId(context.randomSeatPlayerId());
        }
    }

    /**
     * Give the clause-event hook somewhere to file what it builds.
     *
     * <p>Optional in the same sense {@link #withForks} is: on a checkout
     * without the hook, or a caller that never wires this, {@code install()}
     * still installs the listener where the patch offers it, and the handler's
     * null check just means nothing is ever filed.
     */
    public void withBracket(BusBracketCollector bracket) {
        this.bracket = bracket;
    }

    /**
     * Subscribes {@link #collectContinuous()} to the event bus.
     *
     * <p>A separate object rather than {@code @Subscribe} on the collector
     * itself, because Forge's bus reflects over every public method of a
     * subscriber and the collector's are the handler factories and the caps.
     */
    public final class PhaseBridge {
        @com.google.common.eventbus.Subscribe
        public void onPhase(forge.game.event.GameEventTurnPhase event) {
            collectContinuous();
            collectInterventions();
            // Before the turn-based action that deals the damage: the phase
            // event fires ahead of it, which is the only moment a probe can
            // fork from a board where nothing has been dealt yet.
            collectProbes();
        }
    }

    /** A bus subscriber that records continuous effects at each phase. */
    public PhaseBridge phaseBridge() {
        return new PhaseBridge();
    }

    /**
     * Install every hook the checkout offers.
     *
     * @return how many were found
     */
    public int install() {
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
        flushMana();
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
            if (trait != null && !belongsToLiveGame(trait)) {
                return;
            }
            EffectRecord record = rewriteRecord(
                    args.get(0), args.get(1), args.get(2), args.optional(3), args.optional(4));
            if (!allowDistinctRecord(
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
        List<ProvenanceKey> ranKeys = ran == null ? List.of() : keysOf(ran);
        String wireResult = EffectRecord.rewriteResult(result);
        JVM_REWRITES.record(
                RunParamNormalizer.modeOf(trait), wireResult, outgoing != null,
                !ranKeys.isEmpty(), before, after);
        return context.newRecord(RecordKind.REWRITE)
                .actor(controllerOf(source))
                .ability(ProvenanceKey.resolve(acting))
                .state(snapshots.toJsonForTrait(source, referencedOf(source), incoming))
                .payload(EffectRecord.rewritePayload(
                        incoming, outgoing, wireResult, ranKeys));
    }

    /** The worker's rewrite tally, shared by every game it plays. */
    static final RewriteTally JVM_REWRITES = new RewriteTally();

    // ── trigger-fire records ────────────────────────────────────────────

    /**
     * One record per evaluated trigger condition, fired or not.
     *
     * <p>Non-fired negatives are what teach the condition; without them the
     * model would learn the base rate of a trigger type firing and nothing
     * about when. Drawn at roughly 1:1 by sampling the negatives, since a turn
     * evaluates far more conditions than it fires.
     *
     * <p>Package-private rather than private, the same reason {@link
     * #rewriteHandler()} is: the argument order has no
     * compiler behind it, so this hook's own game-identity test calls it with
     * a synthesized argument array.
     */
    InvocationHandler triggerFireHandler() {
        return PatchHooks.TRIGGER_LISTENER.listener((callback, args) -> {
            // Checked first, before this evaluation can spend any of the mode's negative-sampling budget (offer/keep
            // below) -- a fork's trigger evaluations must not skew the live
            // game's ratio bookkeeping any more than they must reach its
            // shard. args[0] is declared Trigger on Forge's side, always a
            // CardTraitBase with a real host; anything this cannot identify
            // is left alone rather than newly dropped.
            CardTraitBase trait = args.get(0, CardTraitBase.class);
            if (trait != null && !belongsToLiveGame(trait)) {
                return;
            }
            boolean fired = Boolean.TRUE.equals(args.get(2));
            String mode = RunParamNormalizer.modeOf(args.get(0));
            if (mode == null) {
                mode = RunParamNormalizer.UNKNOWN_MODE;
            }
            if (!offerTriggerEvaluation(mode, fired)) {
                return;
            }
            EffectRecord record = triggerRecord(args.get(0), args.get(1), fired);
            if (!allowDistinctRecord(
                    RecordKind.TRIGGER.wireName(), record.payloadJson(), record.stateJson())
                    || !allowTriggerLine(record.abilityJson(), fired)) {
                return;
            }
            // Counted only for a record that is actually written, so a
            // suppressed duplicate does not spend the negatives' quota and the
            // ratio holds over what reaches the shard.
            keepTriggerEvaluation(mode, fired);
            emit(record);
        });
    }

    /** Negatives kept per positive kept, per trigger mode. */
    private static final double TARGET_NEGATIVE_RATIO = 1.0;
    /**
     * Negatives a mode may be offered at full rate before it backs off.
     *
     * <p>Only reached by a mode whose offered negatives are being refused
     * downstream -- by the duplicate check or the per-line flood cap -- so it
     * is a bound on wasted snapshot building, not a sampling parameter.
     */
    private static final int NEGATIVE_OFFERS_AT_FULL_RATE = 4;
    /** No negative is ever kept with probability below this. */
    private static final double MIN_NEGATIVE_RATE = 1.0 / 4096.0;
    /** Records one acting line may write per verdict in one game. */
    private static final int TRIGGER_RECORDS_PER_LINE = 24;

    /**
     * Whether this evaluation should be offered to the writer.
     *
     * <p>A fired evaluation always is. A non-fired one is offered whenever this
     * mode's kept negatives are behind its kept positives -- the deficit is the
     * whole rule, with no rate in front of it.
     *
     * <p>Three designs, and why this is the third. A fixed 2% rate landed at
     * 74:26 instead of 1:1, because <b>a fixed rate cannot hold a ratio</b>:
     * the population ratio varies by mode, board size and turn, and it measured
     * 1:17.7 over that run. A deficit rule with a probability in front of it --
     * keep a negative with probability {@code deficit / estimated run length},
     * so the kept ones spread through the run rather than bunching behind the
     * last firing -- landed at 58:42. That rate is a <b>lagging</b> controller:
     * over a run of the estimated length it catches a negative only about
     * 1 - 1/e of the time, and one game gives a mode too few firings for the
     * carried deficit ever to be repaid. Offering on the deficit alone removes
     * the lag, which is the whole of the remaining eight points.
     *
     * <p>Bunching behind the firing is not a loss worth a controller. The
     * negatives that reach the shard are spread by
     * {@link #allowDistinctRecord} instead: two consecutive non-fired
     * evaluations on an unmoved board render the same record and the second is
     * refused, so this walks forward until it finds one that differs. The
     * negative it lands on is then the one closest in time to the positive it
     * balances, which is the more useful contrast rather than the less.
     *
     * <p>What the deterministic rule needs in exchange is a bound on wasted
     * work, because a refusal downstream leaves the deficit standing and the
     * next evaluation is offered again. A mode whose negatives are all being
     * refused -- every acting line at its per-line cap, say -- would otherwise
     * build a snapshot per evaluation of the busiest hook in the collector.
     * {@link ModeBalance#offersSinceWrite} counts offers since that mode last
     * wrote anything and decays the rate harmonically past
     * {@link #NEGATIVE_OFFERS_AT_FULL_RATE}, so an unfillable deficit costs a
     * few offers rather than all of them.
     *
     * <p>Per mode, and per game -- the balances live on this collector, which is
     * built per game. A mode that never fires contributes no negatives, which
     * is the intended reading of "negatives drawn from same-event-type
     * evaluations". The aggregate ratio is therefore near 1:1 rather than
     * exactly it, and the per-mode ratios are the ones to validate.
     */
    boolean offerTriggerEvaluation(String mode, boolean fired) {
        if (fired) {
            return true;
        }
        ModeBalance balance = triggerBalance.computeIfAbsent(mode, m -> new ModeBalance());
        if (balance.fired * TARGET_NEGATIVE_RATIO - balance.notFired <= 0.0) {
            return false;
        }
        balance.offersSinceWrite++;
        if (balance.offersSinceWrite <= NEGATIVE_OFFERS_AT_FULL_RATE) {
            return true;
        }
        double rate = Math.max(
                MIN_NEGATIVE_RATE,
                (double) NEGATIVE_OFFERS_AT_FULL_RATE / balance.offersSinceWrite);
        return sampler.nextDouble() < rate;
    }

    /**
     * Count a written evaluation against its mode's balance.
     *
     * <p>Called only for a record that reached the shard, which is what makes
     * the ratio hold over the corpus rather than over the offers: a negative
     * refused as a duplicate has not been drawn, and the deficit it leaves
     * standing is repaid by the next evaluation of its mode.
     */
    void keepTriggerEvaluation(String mode, boolean fired) {
        ModeBalance balance = triggerBalance.computeIfAbsent(mode, m -> new ModeBalance());
        if (fired) {
            balance.fired++;
        } else {
            balance.notFired++;
        }
        // A written record of either verdict opens a fresh window: a positive
        // has just created a debt and a negative has just paid one, and either
        // way the offers spent finding this record are spent.
        balance.offersSinceWrite = 0;
    }

    /** One trigger mode's written evaluations, and the negatives offered since. */
    private static final class ModeBalance {
        private int fired;
        private int notFired;
        /** Negatives offered since this mode last wrote a record of either verdict. */
        private int offersSinceWrite;
    }

    /** One acting line's verdict, the key the flood cap counts by. */
    private record LineVerdict(String abilityJson, boolean fired) {
    }

    /**
     * A flood cap per acting line, beside the duplicate check.
     *
     * <p>The duplicate check collapses records that say the same thing; this
     * bounds the ones that differ only in a board that moved a little. One
     * line's condition is worth a couple of dozen looks in a game, not the
     * hundreds an aura on a creature that keeps being pumped would write.
     */
    boolean allowTriggerLine(String abilityJson, boolean fired) {
        LineVerdict line = new LineVerdict(abilityJson, fired);
        int written = triggerPerLine.getOrDefault(line, 0);
        if (written >= TRIGGER_RECORDS_PER_LINE) {
            return false;
        }
        triggerPerLine.put(line, written + 1);
        return true;
    }

    /**
     * One evaluated condition's record, built from the trigger itself.
     *
     * <p>Built after the sampling gate, so a non-fired evaluation that will not
     * be written never pays for resolving its key — which matters, because this
     * is the hook that fires most often in the whole collector.
     */
    EffectRecord triggerRecord(Object trait, Object runParams, boolean fired) {
        CardTraitBase acting = trait instanceof CardTraitBase ctb ? ctb : null;
        Card source = acting == null ? null : acting.getHostCard();
        EffectEvent event = RunParamNormalizer.describeParams(trait, runParams);
        return context.newRecord(RecordKind.TRIGGER)
                .actor(controllerOf(source))
                .ability(ProvenanceKey.resolve(acting))
                .state(snapshots.toJsonForTrait(source, referencedOf(source), event))
                .payload("{\"event\":" + event.toJson()
                        + ",\"fired\":" + fired + "}");
    }

    // ── playability records ─────────────────────────────────────────────

    /**
     * Rules-level verdicts only.
     *
     * <p>The AI's policy judgments never reach a record: they are its opinion
     * rather than the game's rules, and training on them would teach the model
     * Forge's play style instead of Magic. So the hook hands over only the
     * candidate, and the verdict is {@link PlayabilityVerdict}'s, computed here
     * after the sampler draw. A patch that still passes three verdict booleans
     * after the candidate has them ignored: they were computed by the AI's own
     * cost check, which is policy.
     *
     * <p>Sampled at {@code --playability-rate} because the AI evaluates every
     * candidate at every priority, which would otherwise swamp the corpus. The
     * {@code attackers} and {@code blockers} subkinds are always logged: they
     * fire once per combat, not once per candidate.
     *
     * <p>Package-private rather than private, the same reason {@link
     * #rewriteHandler()} is: this hook's own game-identity test calls it with
     * a synthesized argument array.
     */
    InvocationHandler playabilityHandler() {
        return PatchHooks.PLAYABILITY_LISTENER.listener((callback, args) -> {
            // Checked before the sampler draw below, not after -- a fork's own candidates are
            // evaluated for playability too (ForkCollector.forceResolution's
            // chooseTargets, and GameSimulator's own AI decisions, both ask the
            // same AiController this hook is installed on), and letting one
            // consume from the shared `sampler` before being dropped would
            // re-randomise the live sample stream for every other subkind that
            // draws from it (trigger negatives, the mana reservoir) --
            // exactly the cost triggerFireHandler's own identity check already
            // avoids by running first. A null candidate is let through: its
            // record names no ability either way.
            SpellAbility candidate = args.get(0, SpellAbility.class);
            if (candidate != null && !belongsToLiveGame(candidate)) {
                return;
            }
            recordCandidate(deciderOf(candidate), candidate,
                    () -> PlayabilityVerdict.of(candidate));
        });
    }

    /**
     * The player a candidate ability belongs to (FR-030a).
     *
     * <p>The activator, which the AI sets before it evaluates; failing that
     * the host's controller. Null for a candidate with neither, which the
     * record then spells as the active player, the way gen-1 spelled every
     * decision record.
     */
    private static Player deciderOf(SpellAbility candidate) {
        if (candidate == null) {
            return null;
        }
        if (candidate.getActivatingPlayer() != null) {
            return candidate.getActivatingPlayer();
        }
        Card host = candidate.getHostCard();
        return host == null ? null : host.getController();
    }

    /**
     * One {@code decision} record, for a candidate someone evaluated.
     *
     * <p>Public because the random seat writes its own (FR-022d): when its
     * draw succeeds the AI's evaluation never runs, so the hook that would
     * have written this never fires, and the seat reports the candidates it
     * drew from through the same door. Sampled at {@code --playability-rate}
     * and de-duplicated here, so both callers get the same treatment.
     *
     * <p>The verdict is asked for only once the candidate is sampled in, so a
     * candidate the sampler drops costs no rules check. A candidate whose
     * check throws is one the engine cannot evaluate on this board, the way
     * the random seat treats it, and is not recorded.
     *
     * @param decider the candidate's controller, written as {@code actor_player}
     */
    public void recordCandidate(
            Player decider, SpellAbility candidate, Supplier<PlayabilityVerdict> verdictOf) {
        if (sampler.nextDouble() > caps.playabilityRate()) {
            return;
        }
        PlayabilityVerdict verdict;
        try {
            verdict = verdictOf.get();
        } catch (RuntimeException e) {
            return;
        }
        // Snapshot defensively: a verdict can be abandoned mid-evaluation,
        // and the legality check mutates the checked ability's targets, so
        // nothing here may keep a reference to the ability object. Read now,
        // into values: what the ability says at this instant is safe to
        // keep, and without it the record names no ability at all.
        List<ProvenanceKey> candidateKeys = List.of();
        StringJoiner legalTargets = new StringJoiner(",", "[", "]");
        String manaCost = null;
        if (candidate != null) {
            candidateKeys = keysOf(candidate);
            for (String id : legalTargetsOf(candidate)) {
                legalTargets.add(Json.string(id));
            }
            if (candidate.getPayCosts() != null) {
                manaCost = String.valueOf(
                        candidate.getPayCosts().getTotalMana());
            }
        }
        // The candidate goes into the snapshot: refs.source was null on all
        // 11.2 million playability records, so a decision record did not say
        // which card the candidate was even on except through its key.
        String state = snapshots.toJson(
                candidate,
                candidate == null || candidate.getHostCard() == null
                        ? List.of() : List.of(candidate.getHostCard()));
        String payload = "{\"candidates\":[{"
                        + "\"ability\":" + ProvenanceKey.toJsonArray(candidateKeys)
                        + ",\"verdict\":{\"can_play\":" + verdict.canPlay()
                        + ",\"affordable\":" + verdict.affordable()
                        + ",\"has_legal_target\":" + verdict.hasLegalTarget() + "}"
                        + ",\"legal_targets\":" + legalTargets
                        + ",\"cost_after_adjustment\":"
                        + (manaCost == null
                                ? "{}"
                                : "{\"mana\":" + Json.string(manaCost) + "}")
                        // responsible_static needs a cantBeCastStatic hook
                        // that does not exist; the attacker and blocker
                        // subkinds carry theirs, this one does not.
                        + ",\"responsible_static\":[]}]}";
        // The decision path had no dedup at all, and every duplicate
        // playability record measured in the first corpus was this subkind:
        // the AI re-asks the same question about the same board, and the
        // answer repeats verbatim.
        if (!allowDistinctRecord(PlayabilitySubkind.DECISION.wireName(), payload, state)) {
            return;
        }
        emit(context.newPlayability(PlayabilitySubkind.DECISION)
                .actor(playerIdOrActive(decider))
                .state(state)
                .payload(payload));
    }

    /** Legal-target refs one decision may carry before the payload is capped. */
    private static final int MAX_LEGAL_TARGETS = 32;

    /**
     * What the rules allowed this candidate to target, not what the AI picked.
     *
     * <p>The field was read off {@code getTargets()}, which is the chosen set --
     * so it was present on 1.8% of candidates and, where it was present, meant
     * the wrong thing. The per-entity target-legality head is specified to train
     * on the legal set, and an AI's choice is a policy judgment the spec keeps
     * out of the corpus.
     *
     * <p>Capped, because a wide board can make this list longer than the record
     * around it.
     */
    private static List<String> legalTargetsOf(SpellAbility candidate) {
        if (!candidate.usesTargeting() || candidate.getTargetRestrictions() == null) {
            return List.of();
        }
        List<String> ids = new ArrayList<>();
        for (GameEntity entity
                : candidate.getTargetRestrictions().getAllCandidates(candidate)) {
            Refs.ofAny(entity).ifPresent(ids::add);
            if (ids.size() >= MAX_LEGAL_TARGETS) {
                break;
            }
        }
        return ids;
    }

    // ── playability records: the legality subkinds ──────────────────────

    /**
     * The legal attacker and blocker sets, and what keeps the rest out.
     *
     * <p>The eighth sampling class, and the only one that says what the rules
     * *forbid* rather than what happened.
     *
     * <p>Asked by the engine's turn-based actions just before a player
     * declares, once per defender and once per attacker coming at the
     * defending player, so every record is a real declaration whoever
     * controls the seat — the AI, the random seat or a human — and is stamped
     * {@code what_if=false}. The engine names the deciding player, every
     * creature that player controls is a candidate, and the legal set comes
     * from the rules alone, against a combat holding the attackers and no
     * blocks yet.
     *
     * <p>Deduplicated on payload and snapshot (FR-028): a board on which
     * nothing moved between two declarations answers the same question the
     * same way, and two records that would be byte-identical carry one
     * observation once.
     *
     * <p>Package-private rather than private, the same reason
     * {@link #rewriteHandler()} already is: this hook's own game-identity
     * test calls it with a synthesized argument array.
     */
    InvocationHandler combatLegalityHandler() {
        return PatchHooks.COMBAT_LISTENER.listener((callback, args) -> {
            // No CardTraitBase here: the defender or attacker is the
            // GameEntity itself, so identity comes straight off it rather
            // than through a host card. A wrong-typed or null entity is left
            // to recordAttackers/recordBlockers, which tolerate both.
            Player decider = args.get(0, Player.class);
            GameEntity declaredAgainst = args.get(1, GameEntity.class);
            if (declaredAgainst != null && !belongsToLiveGame(declaredAgainst)) {
                return;
            }
            switch (callback) {
                case ON_ATTACKERS_DECLARABLE -> recordAttackers(
                        decider, declaredAgainst, asCards(args.get(2)), asCards(args.get(3)));
                case ON_BLOCKERS_DECLARABLE -> {
                    Integer min = args.get(4, Integer.class);
                    recordBlockers(
                            decider, args.get(1, Card.class),
                            asCards(args.get(2)), asCards(args.get(3)),
                            min == null ? 0 : min);
                }
            }
        });
    }

    /**
     * One {@code attackers} record: who may attack this defender, and what
     * keeps the rest out.
     *
     * <p>De-duplicated by FR-028, keyed on the snapshot as well as the
     * answer, so the same answer on a board that moved is kept.
     *
     * @param decider the attacking player the engine names, written as
     *                {@code actor_player}
     */
    void recordAttackers(
            Player decider, GameEntity defender, List<Card> candidates,
            List<Card> legal) {
        if (candidates == null || candidates.isEmpty()) {
            return;
        }
        StringJoiner attackers = new StringJoiner(",", "[", "]");
        for (Card card : legal) {
            attackers.add(Json.string(SnapshotBuilder.entityId(card)));
        }
        StringJoiner forbidden = new StringJoiner(",", "[", "]");
        for (Card card : candidates) {
            if (legal.contains(card)) {
                continue;
            }
            forbidden.add(forbiddenJson(
                    card, defender == null ? null : cantAttackStatic(card, defender)));
        }
        String payload = "{\"legal_attackers\":" + attackers
                + ",\"forbidden\":" + forbidden + "}";
        emitLegality(PlayabilitySubkind.ATTACKERS, decider, payload);
    }

    /**
     * One {@code blockers} record: who may block this attacker, how many it
     * takes, and what keeps the rest out. The sibling of
     * {@link #recordAttackers} in every respect.
     *
     * @param decider the defending player the engine names (FR-030a) —
     *                gen-1 wrote the attacking player here, and a gen-2 shard
     *                is what tells a reader which meaning it holds
     */
    void recordBlockers(
            Player decider, Card attacker, List<Card> candidates, List<Card> legal,
            int minBlockers) {
        if (attacker == null || candidates == null || candidates.isEmpty()) {
            return;
        }
        StringJoiner blockers = new StringJoiner(",", "[", "]");
        for (Card card : legal) {
            blockers.add(Json.string(SnapshotBuilder.entityId(card)));
        }
        StringJoiner forbidden = new StringJoiner(",", "[", "]");
        for (Card card : candidates) {
            if (legal.contains(card)) {
                continue;
            }
            forbidden.add(forbiddenJson(card, cantBlockByStatic(attacker, card)));
        }
        String payload = "{\"anchor_attacker\":"
                + Json.string(SnapshotBuilder.entityId(attacker))
                + ",\"legal_blockers\":" + blockers
                + ",\"forbidden\":" + forbidden
                + ",\"min_blockers\":" + minBlockers + "}";
        emitLegality(PlayabilitySubkind.BLOCKERS, decider, payload);
    }

    private void emitLegality(PlayabilitySubkind subkind, Player decider, String payload) {
        String state = snapshots.toJson(null, List.of());
        // Keyed on the snapshot as well as the payload (FR-028): the same
        // answer on a board that moved is a second observation, which a
        // payload-only key would drop as a repeat of the first.
        if (!allowDistinctRecord(subkind.wireName(), payload, state)) {
            return;
        }
        emit(context.newPlayability(subkind)
                .actor(playerIdOrActive(decider))
                .whatIf(false)
                .state(state)
                .payload(payload));
    }

    /** The deciding player's id, or the active player's where none is known. */
    private String playerIdOrActive(Player decider) {
        return decider == null ? activePlayerId() : SnapshotBuilder.playerId(decider);
    }

    /** Distinct rendered records this game may hold before dedup stops. */
    private static final int COALESCE_MAX_KEYS = 200_000;

    /**
     * Whether a record saying exactly this has already been written this game.
     *
     * <p>The generalisation of the legality cap to the kinds that had none.
     * 624,679 records of the first corpus — 4.3% of it — were byte-identical to
     * another record of the same game, and dedup existed only for the attacker
     * and blocker subkinds: every duplicate playability record measured was the
     * {@code decision} subkind, which is exactly what an unguarded path
     * predicts. Two records that would be byte-identical carry one observation
     * once, so collapsing them loses nothing.
     *
     * <p>A 64-bit hash rather than the 32-bit {@code String.hashCode} the
     * legality cap used: over the hundred thousand keys one game can produce,
     * a 32-bit space has a real chance of colliding, and a collision here
     * silently drops a record that was not a duplicate.
     *
     * <p>Bounded: past the cap it stops deduplicating rather than growing, so a
     * pathological game reverts to writing duplicates instead of exhausting a
     * 1200 MB heap. The duplicate rate of a run is therefore something to
     * measure rather than assume.
     */
    public boolean allowDistinctRecord(String tag, String payload, String state) {
        if (coalescedRecords.size() >= COALESCE_MAX_KEYS) {
            return true;
        }
        long digest = com.google.common.hash.Hashing.murmur3_128().newHasher()
                .putUnencodedChars(tag).putChar('@')
                .putUnencodedChars(payload == null ? "" : payload).putChar('#')
                .putUnencodedChars(state == null ? "" : state)
                .hash().asLong();
        return coalescedRecords.add(digest);
    }

    /**
     * One entity kept out of a legal set, and the static responsible.
     *
     * <p>The static is often absent, and that is a real answer rather than a
     * gap: a creature that is tapped, or summoning sick, or has "can't attack"
     * printed on it, is stopped by the rules themselves and no static ability
     * is to blame.
     */
    private static String forbiddenJson(Card card, StaticAbility responsible) {
        ProvenanceKey key = responsible == null ? null : ProvenanceKey.of(responsible);
        return "{\"entity\":" + Json.string(SnapshotBuilder.entityId(card))
                + ",\"responsible_static\":["
                + (key == null ? "" : key.toJson()) + "]}";
    }

    /** Reflective, because the engine exposes these only on a patched checkout. */
    private static StaticAbility cantAttackStatic(Card attacker, GameEntity defender) {
        return (StaticAbility) PatchHooks.CANT_ATTACK_STATIC.invoke(
                new Class<?>[]{Card.class, GameEntity.class}, attacker, defender);
    }

    private static StaticAbility cantBlockByStatic(Card attacker, Card blocker) {
        return (StaticAbility) PatchHooks.CANT_BLOCK_BY_STATIC.invoke(
                new Class<?>[]{Card.class, Card.class}, attacker, blocker);
    }

    @SuppressWarnings("unchecked")
    private static List<Card> asCards(Object value) {
        return value instanceof List<?> list ? (List<Card>) list : List.of();
    }

    // ── interventional resolutions (stage three) ────────────────────────

    /**
     * Force an ability nobody played, at a phase boundary.
     *
     * <p>Some abilities appear in no record however long collection runs: the
     * AI judges them bad, cannot afford them, or never draws them into a board
     * where they do anything. Observation cannot reach those, so the corpus
     * forks the game and makes one happen.
     *
     * <p>Driven from a phase boundary rather than from the playability hook,
     * even though that hook knows exactly which candidates were declined. The
     * hook fires inside the AI's own evaluation, and forking the game there
     * means copying a game that is mid-decision and re-entering the rules
     * engine underneath it. A phase boundary is the same board with nothing in
     * flight.
     *
     * <p>Costs a game copy and a stack resolution each, which is why
     * {@code --interventions-per-game} defaults low and why nothing happens at
     * all when it is zero.
     */
    public void collectInterventions() {
        if (forks == null || mode == AttributionMode.DEGRADED) {
            return;
        }
        Player actor = game.getPhaseHandler() == null
                ? null : game.getPhaseHandler().getPlayerTurn();
        if (actor == null) {
            return;
        }
        for (Card card : actor.getCardsIn(ZoneType.Hand)) {
            for (SpellAbility candidate : card.getSpellAbilities()) {
                if (!worthIntervening(candidate)) {
                    continue;
                }
                if (!forks.mayIntervene(candidate.getId())) {
                    return;
                }
                interveneSeen.add(manaAbilityText(candidate, ""));
                forks.intervene(candidate, candidate.getId());
            }
        }
    }

    /**
     * Is this line one observation cannot reach anyway?
     *
     * <p>Lands and mana abilities are excluded because every game plays them —
     * forking to force a Mountain onto the battlefield spends a game copy to
     * observe the most common event in the corpus. What is left is the spells
     * and activated abilities the AI passed over, which is the population the
     * intervention exists for.
     *
     * <p>Once per line per game: a card redrawn or returned to hand is the same
     * line, and forcing it twice buys the second copy of one observation.
     */
    private boolean worthIntervening(SpellAbility candidate) {
        if (candidate.isLandAbility() || candidate.isManaAbility()) {
            return false;
        }
        return !interveneSeen.contains(manaAbilityText(candidate, ""));
    }

    // ── damage-step probes (stage three) ────────────────────────────────

    /**
     * Fork this damage step for each probed keyword a combatant carries.
     *
     * <p>Gate 2 perturbs the keyword <b>model-side</b> and asks whether the
     * prediction moves the right way. A probe answers the same question against
     * the engine: the same combat, actually re-run without the keyword, so the
     * model's response has a ground truth to be checked against. Every
     * collecting supervisor names all eight damage-step keywords unless a run
     * narrows {@code --probe-keywords}; a worker started without the property
     * probes none.
     *
     * <p>The branches are held rather than written. Each mirrors the real combat
     * record, which does not exist until the step this ran ahead of has
     * finished.
     */
    public void collectProbes() {
        if (forks == null || !caps.probesEnabled() || mode == AttributionMode.DEGRADED) {
            return;
        }
        DamageStep step = DamageStep.of(game.getPhaseHandler());
        if (step == null) {
            return;
        }
        boolean firstStrike = step == DamageStep.FIRST_STRIKE;
        if (firstStrike && !anyFirstStriker()) {
            // Forge always enters the first-strike step and skips it when no
            // combatant has first or double strike, so the phase event alone is
            // no evidence that a step will happen. Probing anyway spent half the
            // budget on forks that assigned no damage: every empty probe fork in
            // the sampled corpus was a first-strike-step fork, and every
            // first-strike-step fork was empty. Worse, the real game writes no
            // combat record for a step that assigns nothing, so those branches
            // survived and were completed against the *regular* step's record.
            return;
        }
        discardStaleProbes(step);
        // One keyword per game rather than whichever the battlefield order
        // reaches first. With eight probed keywords and a budget of two, the
        // first corpus probed lifelink and trample eight times each and
        // double strike, indestructible, wither and infect not at all -- and
        // gate 2 is a per-keyword decision, so an unprobed keyword gets no
        // check whatever.
        String focus = probeFocus();
        if (probeCombatants(focus, step) == 0) {
            // The focus is a rotation, not a restriction: a game whose combat
            // carries none of it still spends its budget rather than saving it
            // for a keyword that is not there.
            probeCombatants(null, step);
        }
    }

    /**
     * Hold a branch until the record it mirrors exists.
     *
     * <p>The one place a branch enters the queue, which is also what lets the
     * pairing rules below be exercised without a live combat to fork.
     */
    void hold(ForkCollector.HeldProbe held) {
        heldProbes.hold(held);
    }

    /** This game's turn in the probe-keyword rotation, or null when none. */
    private String probeFocus() {
        List<String> keywords = caps.probeKeywords();
        if (keywords.isEmpty()) {
            return null;
        }
        return keywords.get(
                Math.floorMod(String.valueOf(gameId).hashCode(), keywords.size()));
    }

    /**
     * Take a probe for each combatant carrying a probed keyword.
     *
     * @param focus the one keyword to spend on, or null for any probed one
     * @return how many branches were taken
     */
    private int probeCombatants(String focus, DamageStep step) {
        int taken = 0;
        for (Card card : game.getCardsIn(ZoneType.Battlefield)) {
            if (!inCombat(card)) {
                continue;
            }
            for (KeywordInterface keyword : card.getKeywords()) {
                String original = keyword.getOriginal();
                if (original == null || original.isEmpty()) {
                    continue;
                }
                // The flag names keywords in the corpus's spelling
                // (first_strike); the card carries Forge's (First Strike).
                // Matched on the normalized form and stripped by the card's own,
                // so neither side has to guess the other's.
                String normalized = normalizeKeyword(original);
                if (!caps.probeKeywords().contains(normalized)
                        || (focus != null && !focus.equals(normalized))) {
                    continue;
                }
                ForkCollector.HeldProbe held =
                        forks.probe(original, card, step);
                if (held != null) {
                    hold(held);
                    taken++;
                }
            }
        }
        return taken;
    }

    /** Is anything in this combat going to deal first-strike damage? */
    private boolean anyFirstStriker() {
        for (Card card : game.getCardsIn(ZoneType.Battlefield)) {
            if (inCombat(card) && (card.hasFirstStrike() || card.hasDoubleStrike())) {
                return true;
            }
        }
        return false;
    }

    /** Is this creature in the combat about to deal damage? */
    private boolean inCombat(Card card) {
        var combat = game.getCombat();
        return combat != null
                && (combat.isAttacking(card) || combat.isBlocking(card));
    }

    /**
     * Forge's spelling of a keyword in the one every reader uses.
     *
     * <p>Mirrors {@code effects.domain.state_snapshot.normalize_keyword}: the
     * parameter after {@code :} goes, spaces become underscores. Two spellings
     * of first strike is exactly the mismatch that made gate 2 report zero
     * observations of it.
     */
    static String normalizeKeyword(String original) {
        int parameter = original.indexOf(':');
        String base = parameter < 0 ? original : original.substring(0, parameter);
        return base.trim().toLowerCase(java.util.Locale.ROOT).replace(' ', '_');
    }

    /**
     * Complete the branches this combat record is the counterfactual for.
     *
     * <p>The caller does not say which damage step it just wrote, so the oldest
     * held substep is taken to be the one: branches are held in the order the
     * steps happen, and a first-strike branch is therefore completed against the
     * first-strike record and not against the regular one that follows it.
     * Naming the substep outright is one parameter better and is what
     * {@link #writeHeldProbes(String, DamageStep)} is for.
     */
    public void writeHeldProbes(String combatRecordId) {
        if (forks == null || heldProbes.isEmpty()) {
            return;
        }
        writeHeldProbes(combatRecordId, heldProbes.oldestSubstep());
    }

    /**
     * Complete only the branches taken at this damage step.
     *
     * <p>A branch and the record it mirrors have to describe the same step. In
     * the first corpus they did not: a branch forked at the first-strike step —
     * where nothing was ever assigned — outlived that step and was written
     * against the regular step's record, so gate 2's real-versus-fork
     * difference compared a step that did not happen with one that did.
     */
    public void writeHeldProbes(String combatRecordId, DamageStep substep) {
        if (forks == null || heldProbes.isEmpty()) {
            return;
        }
        for (ForkCollector.HeldProbe held : heldProbes.take(substep)) {
            forks.writeHeldProbe(held, combatRecordId);
        }
    }

    /**
     * Drop branches from a step that has been left behind.
     *
     * <p>A branch whose step produced no real record has nothing to mirror, and
     * writing it against the next record is worse than not writing it: the
     * fork's budget is already spent either way, and a mispaired counterfactual
     * is read as a difference the keyword caused.
     */
    public int discardStaleProbes(DamageStep substep) {
        return heldProbes.discardOtherThan(substep);
    }

    // ── clause events (an API's own parameters) ─────────────────────────

    /**
     * The events an effect API promises, taken from the clause that ran.
     *
     * <p>The memo is kept per thread and not per collector: a clause can resolve
     * another, and the engine runs games on more than one thread. Each memo is
     * wrapped because a deque holds no null, and a clause with no memo still
     * pushes one so that every pop meets its own push.
     */
    private final ThreadLocal<Deque<java.util.Optional<Object>>> clauseMemos =
            ThreadLocal.withInitial(ArrayDeque::new);

    /**
     * Whether a clause, replacement, trigger or mana ability belongs to the
     * game this collector is installed for.
     *
     * <p>{@link #clauseHandler()} and {@link #outcomeHandler()} are wired to
     * Forge's clause and outcome hooks, which are plain JVM statics with no
     * {@code Game} reference of their own -- whichever resolution fires them
     * last wins. {@code ForkCollector.forceResolution} fires them too:
     * {@code GameSimulator.resolveStack} builds its own
     * {@code PlayerControllerAi} and unconditionally calls
     * {@code setUseSimulation(AIOption.USE_FULL_SIMULATION)}, regardless of
     * how this game's own two lobby seats were registered, so a forked
     * ability resolves through the exact same
     * {@code AbilityUtils.resolveApiAbility} these hooks are installed
     * against. {@link ForkEventSink}
     * keeps a fork's own <em>bus</em> events out of this game's records by
     * subscribing to the fork's own {@code Game} instead of this one, but
     * these hooks are not bus subscriptions, so that separation never
     * reaches them -- without this check, a fork's {@code vote_taken} or
     * {@code coin_flipped} lands in the live game's bracket with nothing
     * marking it as hypothetical.
     *
     * <p>Game identity is the check because it needs no cooperation from
     * {@code ForkCollector} or {@code GameSimulator}, neither of which know
     * this class exists: {@code GameCopier} builds every copied {@code Card}
     * against a new {@code Game} object ({@code new Card(id, paperCard,
     * newGame)}), so a fork's clause's own host card already carries the
     * answer. Compared by reference, the same way
     * {@code SpellAbility.setActivatingPlayer} itself already compares
     * players ("don't use equals because player might be from simulation")
     * -- a copied {@code Game} is never {@code equals} to the live one by
     * accident, only by being the same object. A clause with no identifiable
     * host is treated as not belonging here rather than let through by
     * default: on the real resolution path a host card is always present
     * (the same assumption {@code AbilityUtils.resolveApiAbilityUnobserved}
     * already makes, unconditionally dereferencing it), so this branch is
     * never live in production and only ever chooses the safer of two
     * defaults for a state that should not occur.
     *
     * <p>Checked in the handlers themselves, immediately before each one's
     * existing {@code bracket != null} (or, for {@link #rewriteHandler()} and
     * {@link #triggerFireHandler()}, sampling/dedup) delivery gate, rather
     * than inside {@link BusBracketCollector#recordClauseEvent}: that method
     * only ever receives the already-built {@code EffectEvent}, not the
     * {@code SpellAbility} that produced it, so checking there would mean
     * widening a documented, two-call-site contract just to re-derive
     * something both callers already have in scope. Both hooks' push/pop
     * bookkeeping in {@link #clauseMemos} is untouched by this: a fork's
     * clause still pushes and pops its own memo exactly as a live one does,
     * so this check cannot desynchronize that deque.
     *
     * <p>Typed {@link CardTraitBase} rather than {@link SpellAbility}:
     * {@code SpellAbility}, {@code ReplacementEffect} and
     * {@code Trigger} all extend it and all declare {@code getHostCard()} on
     * it, so {@link #rewriteHandler()} and {@link #triggerFireHandler()} --
     * whose acting argument is a {@code ReplacementEffect}/{@code Trigger},
     * never a {@code SpellAbility} -- reuse this same check rather than a
     * near-duplicate.
     */
    private boolean belongsToLiveGame(CardTraitBase ability) {
        if (ability == null) {
            return false;
        }
        Card host = ability.getHostCard();
        return host != null && host.getGame() == game;
    }

    /**
     * As {@link #belongsToLiveGame(CardTraitBase)}, for {@link
     * #combatLegalityHandler()}: the attacker/defender Forge hands that hook
     * is a {@code Card} or {@code Player} directly, never a trait with a
     * host, so the game comes straight off the {@code GameEntity} itself.
     */
    private boolean belongsToLiveGame(GameEntity entity) {
        return entity != null && entity.getGame() == game;
    }

    /**
     * Whether a null-ability outcome belongs to the live game.
     *
     * <p>{@link #outcomeHandler()}'s two production null-ability sources --
     * {@code GameAction.reveal}'s six-argument funnel and
     * {@code ReplacementHandler.runSingleReplaceDamageEffect}'s two
     * {@code damage_prevented} sites -- call
     * {@code EffectRecordOutcomes.note} with a literal {@code null} ability,
     * always, on the live game exactly as on a fork: {@link
     * #belongsToLiveGame(CardTraitBase)} has nothing to compare, and simply
     * keeping every null-ability outcome would readmit exactly what that
     * method exists to keep out.
     *
     * <p>Two signals, checked in this order:
     *
     * <ol>
     *   <li>{@link ForkCollector#isForkRunningOnThisThread()}. Set around the
     *   whole of {@code ForkCollector.intervene} and {@code
     *   ForkCollector.probe} -- both this class's only two fork mechanisms --
     *   so "a fork is running on this thread" is by construction true for
     *   every null-ability outcome a fork can produce today, whichever of the
     *   two produces it and whichever segment of either it fires from. Tried
     *   first because, unlike the pointer below, it does not depend on the
     *   call happening to be nested inside an ability resolution.
     *   <li>The resolving-clause pointer ({@code
     *   AbilityUtils.getEffectRecordSubAbility}, read via {@link
     *   PatchHooks#currentSubAbility()}), when the flag is false. Every
     *   ability resolves through {@code AbilityUtils.resolve}/{@code
     *   resolveApiAbility}, live or forked alike, and both set this pointer to
     *   the ability currently resolving before running its body. Kept as a
     *   second, independent signal although the flag alone is sufficient for
     *   both of this class's current fork mechanisms:
     *   insurance against a future one that resolves an ability without going
     *   through {@code intervene}/{@code probe}. When the flag is true this
     *   branch is redundant with it (both cover the resolving case), not a
     *   second opinion; it only does independent work when the flag is false.
     * </ol>
     *
     * <p>Neither signal fires for a real combat-damage prevention on the live
     * game -- {@code Combat.dealAssignedDamage} is a turn-based action, never
     * inside an ability resolution and never inside a fork -- which is the
     * intended "belongs to the live game" default this method falls through
     * to.
     *
     * <p><b>Why not also use {@code damage_prevented}'s own {@code "source"}
     * param</b> (a raw {@code Card}, held at {@link #outcomeRefOf} before it is
     * converted to a ref): with the flag covering the entirety of
     * both fork mechanisms, there is no remaining path where {@code source}
     * would catch a fork this method's two signals miss -- it would be a
     * third check for a gap that is already closed, not a narrower one. It is
     * also not general: {@code card_revealed}'s params carry no card at all
     * (just a count and a zone name), so a {@code source}-keyed check could
     * only ever cover one of the two known null-ability types, where the flag
     * and the pointer both cover either.
     */
    private boolean nullAbilityOutcomeBelongsToLiveGame() {
        if (ForkCollector.isForkRunningOnThisThread()) {
            return false;
        }
        if (PatchHooks.currentSubAbility() instanceof SpellAbility resolving) {
            return belongsToLiveGame(resolving);
        }
        return true;
    }

    /**
     * Dispatches by method name on a listener this side never compiles
     * against, so the argument order below is a runtime contract rather than
     * one the compiler enforces: {@code onClauseResolving} is called with
     * {@code args[0]} the clause; {@code onClauseResolved} with
     * {@code args[0]} the same clause and {@code args[1]} whether it threw.
     *
     * <p>Package-private rather than private, which none of the other
     * handlers but {@link #rewriteHandler} needs to be: that argument order
     * has no compiler behind it either, so {@code ClauseContractTest} calls
     * this with a synthesized argument array in the compiler's place.
     */
    InvocationHandler clauseHandler() {
        return clauseHandler(ApiEvents::ruleFor);
    }

    /**
     * The same, building each clause's event from the rules {@code rules}
     * names, so a test can stand a rule that throws in for one API.
     */
    InvocationHandler clauseHandler(Function<ApiType, ApiEvents.Rule> rules) {
        return PatchHooks.CLAUSE_LISTENER.listener((callback, args) -> {
            SpellAbility clause = args.get(0, SpellAbility.class);
            if (clause == null) {
                return;
            }
            switch (callback) {
                case ON_CLAUSE_RESOLVING -> clauseResolving(clause, rules);
                case ON_CLAUSE_RESOLVED -> clauseResolved(
                        clause, Boolean.TRUE.equals(args.optional(1)), rules);
            }
        });
    }

    private void clauseResolving(SpellAbility clause, Function<ApiType, ApiEvents.Rule> rules) {
        clauseMemos.get().push(
                java.util.Optional.ofNullable(ApiEvents.before(clause, rules)));
        // A chosen mode's first clause is where a modal resolution is
        // cut into per-mode halves (FR-029a): the live bracket opens
        // the next half here, and a fork running on this thread marks
        // the same boundary in its own event stream. Nested, so
        // "resolved" cannot delimit modes; "resolving" of the first
        // clause is the one moment FR-029b's snapshot has to be taken.
        if (belongsToLiveGame(clause)) {
            if (bracket != null) {
                CharmModes.ModeStart start = CharmModes.modeStartOf(clause);
                if (start != null) {
                    bracket.beginMode(clause, start.option());
                }
            }
        } else {
            ForkCollector.noteClauseResolving(clause);
        }
    }

    private void clauseResolved(
            SpellAbility clause, boolean threw, Function<ApiType, ApiEvents.Rule> rules) {
        Deque<java.util.Optional<Object>> memos = clauseMemos.get();
        Object memo = memos.isEmpty() ? null : memos.pop().orElse(null);
        if (threw) {
            // A clause that threw did not finish, and an event claiming it
            // did would be a fact the game never contained.
            return;
        }
        if (bracket != null && belongsToLiveGame(clause)) {
            bracket.recordClauseEvent(ApiEvents.after(clause, memo, rules));
        }
    }

    // ── outcome records (what an effect computed and discarded) ─────────

    /**
     * The outcome types an effect is allowed to name, built from the
     * {@code EffectEvent} constants themselves rather than listed in prose.
     *
     * <p>{@code test_event_schema_completeness.py}'s
     * {@code _emitted_event_types()} is a <em>textual</em> scan for
     * {@code EffectEvent.<NAME>} in this package's source, and a javadoc
     * listing satisfies that scan exactly as well as this set does. Were the
     * types referenced only inside a comment, deleting an effect's {@code
     * note(...)} call, or misspelling the event name, would go unnoticed by
     * the guard built specifically to catch that. This set is what
     * {@link #outcomeHandler()} actually validates incoming types against,
     * so the reference is load-bearing code, not prose a rename can silently
     * strand.
     *
     * <p>{@code CARD_REVEALED} and {@code DAMAGE_PREVENTED} come from the two
     * engine choke points ({@code GameAction.reveal},
     * {@code ReplacementHandler.runSingleReplaceDamageEffect}); the other
     * twelve, from an effect directly. {@code SPELL_COUNTERED} ({@code
     * CounterEffect}), {@code TOKEN_CREATED} ({@code TokenEffectBase}, shared
     * by the six token-making effects) and {@code DICE_ROLLED} ({@code
     * RollDiceEffect}) are the same; {@code PLAYER_WON} is a bus
     * subscription (see {@code BusEventReader#onGameOutcome}), not an
     * outcome-hook type, and does not belong here. {@code TURN_ENDED}
     * ({@code EndTurnEffect}) and {@code TEXT_CHANGE} ({@code
     * TextBoxExchangeEffect} -- {@code ChangeText}'s own contribution to
     * {@code TEXT_CHANGE} is an {@code ApiEvents.RULES} entry, not this hook)
     * are noted by their effects too; the types an {@code ApiEvents.RULES}
     * entry emits never reach this set at all.
     */
    private static final Set<String> KNOWN_OUTCOME_TYPES = Set.of(
            EffectEvent.COIN_FLIPPED, EffectEvent.CLASH_RESOLVED, EffectEvent.VOTE_TAKEN,
            EffectEvent.PILES_MADE, EffectEvent.DUNGEON_VENTURED, EffectEvent.SPELL_COPIED,
            EffectEvent.PERMANENT_COPIED, EffectEvent.CARD_MADE, EffectEvent.RESTRICTION_CHANGE,
            EffectEvent.CARD_REVEALED, EffectEvent.DAMAGE_PREVENTED, EffectEvent.SPELL_COUNTERED,
            EffectEvent.TOKEN_CREATED, EffectEvent.DICE_ROLLED, EffectEvent.TURN_ENDED,
            EffectEvent.TEXT_CHANGE);

    /** Set the first time {@link #outcomeHandler()} sees a type outside {@link #KNOWN_OUTCOME_TYPES}. */
    private static final AtomicBoolean UNKNOWN_OUTCOME_TYPE_REPORTED = new AtomicBoolean();

    /**
     * Prints one line to stderr the first time {@link #outcomeHandler()}
     * receives a type outside {@link #KNOWN_OUTCOME_TYPES}, then stays quiet
     * for every one after it — the same latched-once treatment
     * {@code ApiEvents.reportEmitterFailure} and
     * {@code AbilityUtils.reportClauseListenerFailure} give their own hooks'
     * failures, for the same reason: a deleted {@code note(...)} call or a
     * misspelled event name must not go dark silently.
     */
    private static void reportUnknownOutcomeType(String type) {
        if (UNKNOWN_OUTCOME_TYPE_REPORTED.compareAndSet(false, true)) {
            System.err.println("PatchedCollectors: outcomeHandler() received "
                    + "an outcome type not in KNOWN_OUTCOME_TYPES (\"" + type
                    + "\"); further unrecognized types will not be logged. A "
                    + "renamed or deleted EffectRecordOutcomes.note(...) call "
                    + "in Forge would produce exactly this.");
        }
    }

    /**
     * An effect's own account of what it did.
     *
     * <p>Bound by argument position, because {@code PatchHooks} installs a
     * proxy rather than compiling against the interface: 0 is the ability, 1
     * the event type, 2 the parameters. The event type is a plain string the
     * effect names for itself rather than an enum this side switches on --
     * {@code EffectRecordOutcomes}'s own javadoc explains why: "the caller
     * names the event type, which keeps the vocabulary in the collector that
     * consumes it" -- checked against {@link #KNOWN_OUTCOME_TYPES}, the real
     * vocabulary.
     *
     * <p>The event's subjects are exactly what the effect names under
     * {@code "subjects"}, and an outcome that names none has none. The acting
     * ability's host is never a default: a coin flip or a token made is not
     * something done to the card whose ability did it, and Detain's host is
     * the spell that cast the restriction, not one of the cards it landed on.
     *
     * <p>Three params entries are not generic parameters:
     * <ul>
     *   <li>{@code "subjects"}, carrying the raw Card/Player list of who the
     *   outcome happened to (the cards a restriction landed on, the players
     *   whose library a reveal came from, the player or creature prevented
     *   damage would have been dealt to), is pulled out and converted to refs
     *   on {@link EffectEvent#subject} instead -- the schema's per-type rows
     *   do not list it, so a raw {@code subjects} entry left in {@code params}
     *   would fail the per-type allow-list in the Python {@code EventRecord}'s
     *   own construction.
     *   <li>{@code "source"} (currently only {@code damage_prevented}),
     *   carrying a raw Card, is converted to an entity ref the same way --
     *   every other producer of a {@code source} key ({@code damage_dealt},
     *   and the trigger-derived {@code damage_prevented}) renders one, and a
     *   display name here would silently fail a join against those refs and
     *   collide two sources sharing a name.
     *   <li>{@code "sample"} ({@code token_created} only), one token standing
     *   for its batch, becomes {@code token_script_id} and
     *   {@code characteristics}, rendered by {@link SnapshotBuilder}'s own
     *   rules so the description reads the same as the token's entry on the
     *   board: the same script id, and keywords spelled the way an entity's
     *   are. The token itself never reaches the event.
     * </ul>
     *
     * <p>Package-private rather than private, the same reason
     * {@code clauseHandler} is: that argument order has no compiler behind it
     * either, so this hook's own test calls it directly, the way
     * {@code ClauseContractTest} calls {@code clauseHandler}.
     */
    InvocationHandler outcomeHandler() {
        return PatchHooks.OUTCOME_LISTENER.listener((callback, args) -> {
            String type = args.get(1, String.class);
            if (type == null) {
                return;
            }
            if (!KNOWN_OUTCOME_TYPES.contains(type)) {
                reportUnknownOutcomeType(type);
                return;
            }
            EffectEvent event = new EffectEvent(type);
            if (args.get(2) instanceof Map<?, ?> params) {
                for (Map.Entry<?, ?> entry : params.entrySet()) {
                    if ("subjects".equals(entry.getKey())) {
                        addOutcomeSubjects(event, entry.getValue());
                        continue;
                    }
                    if ("source".equals(entry.getKey())) {
                        event.param("source", outcomeRefOf(entry.getValue()));
                        continue;
                    }
                    if ("sample".equals(entry.getKey())) {
                        if (entry.getValue() instanceof Card token) {
                            event.param("token_script_id", SnapshotBuilder.tokenScriptIdOf(token));
                            event.param("characteristics", tokenCharacteristics(token));
                        }
                        continue;
                    }
                    event.param(String.valueOf(entry.getKey()), entry.getValue());
                }
            }
            SpellAbility ability = args.get(0, SpellAbility.class);
            boolean liveGame = ability != null
                    ? belongsToLiveGame(ability)
                    : nullAbilityOutcomeBelongsToLiveGame();
            if (bracket != null && liveGame) {
                bracket.recordClauseEvent(event);
            }
        });
    }

    /**
     * Converts a raw Card/Player list reported under the {@code "subjects"}
     * params key into the event's own subject refs -- see
     * {@link #outcomeHandler()} for why that key is not a generic param.
     */
    private static void addOutcomeSubjects(EffectEvent event, Object value) {
        if (!(value instanceof Iterable<?> subjects)) {
            return;
        }
        for (Object subject : subjects) {
            Refs.ofAny(subject).ifPresent(event::subject);
        }
    }

    /**
     * What a made token is, for a {@code token_created} event: its base
     * power and toughness, its type line and its keywords — see
     * {@link #outcomeHandler()} for why these are rendered here.
     */
    private static Map<String, Object> tokenCharacteristics(Card token) {
        Map<String, Object> traits = new LinkedHashMap<>();
        traits.put("power", token.getBasePower());
        traits.put("toughness", token.getBaseToughness());
        traits.put("types", token.getType().toString());
        traits.put("keywords", SnapshotBuilder.keywordNames(token.getKeywords()));
        return traits;
    }

    /**
     * Converts a raw Card/Player reported under the {@code "source"} params
     * key into an entity/player ref -- see {@link #outcomeHandler()} for why
     * that key is not a generic param. Anything else passes through
     * unconverted, which is what lets a future producer of a differently
     * shaped {@code source} degrade to its own {@code String.valueOf} rather
     * than being silently dropped.
     */
    private static Object outcomeRefOf(Object value) {
        return Refs.ofAny(value).<Object>map(ref -> ref).orElse(value);
    }

    // ── mana records ────────────────────────────────────────────────────

    /**
     * One resolution record per mana ability that produced mana.
     *
     * <p>A mana ability never uses the stack, so the bracket collector sees no
     * cast and no resolution for it and the corpus has no effect half at all.
     * That is what leaves the role-polarity probe unrunnable: paying {@code R}
     * is observable from stage one, producing it is not.
     *
     * <p>Package-private rather than private, the same reason {@link
     * #rewriteHandler()} is: this hook's own game-identity test calls it with
     * a synthesized argument array.
     */
    InvocationHandler manaHandler() {
        return PatchHooks.MANA_LISTENER.listener((callback, args) -> {
            SpellAbility ability = args.get(0, SpellAbility.class);
            String produced = String.valueOf(args.get(2));
            // A fork's own mana abilities can resolve through the inline path
            // too (forceResolution's forced ability,
            // or costs it pays along the way), and this hook is a JVM static
            // like the others.
            if (ability == null || produced.isEmpty() || !belongsToLiveGame(ability)) {
                return;
            }
            String key = manaAbilityText(ability, produced);
            int slot = manaReservoir.slot(key);
            if (slot < 0) {
                return;
            }
            Player producer = args.get(1, Player.class);
            String player = producer != null
                    ? SnapshotBuilder.playerId(producer) : activePlayerId();
            EffectEvent event = new EffectEvent(EffectEvent.MANA_PRODUCED)
                    .subject(player)
                    .param("mana_by_color", RunParamNormalizer.manaByColor(produced));
            // Held rather than written: which activation survives is not known
            // until the game ends. The state is snapshotted here, at the moment
            // the mana was made, so a record that does survive describes the
            // board it was actually produced on.
            EffectRecord record = context.newResolution(Moment.RESOLUTION)
                    .actor(player)
                    .ability(ProvenanceKey.resolve(ability))
                    .state(snapshots.toJson(null, List.of()))
                    .payload(EffectRecord.eventsPayload(List.of(event)));
            manaReservoir.hold(key, slot, record);
        });
    }

    /**
     * The cap's key: the ability, not the instance.
     *
     * <p>Every Mountain shares one text, so counting by text caps the whole
     * class rather than each copy. The produced string rides along because a
     * filter land producing {@code W} and the same land producing {@code U} are
     * different observations of the same line.
     */
    private static String manaAbilityText(SpellAbility ability, String produced) {
        Object api = ability.getApi();
        return (api == null ? "?" : api.toString()) + "|"
                + ability.getHostCard().getName() + "|" + produced;
    }

    /**
     * Write the sampled mana records, once the game can produce no more.
     *
     * <p>They are held until here because reservoir sampling does not know
     * which activation won until the last one has happened. A game whose JVM
     * dies loses its reservoir — at most {@code --mana-cap} records per mana
     * ability, against a crash that already costs the rest of the game.
     */
    private void flushMana() {
        for (EffectRecord record : manaReservoir.drain()) {
            emit(record);
        }
    }

    // ── continuous records ──────────────────────────────────────────────

    /**
     * One record per static ability that is currently changing the board.
     *
     * <p>Called at a phase boundary, which is the cheapest moment the board is
     * stable. The layer tables are keyed by (timestamp, applying static's id),
     * so each entity's share of an effect is a read rather than an inference —
     * an anthem's +1/+1 is recorded against the anthem, on each creature it
     * touches, without diffing anything.
     *
     * <p>The snapshot a continuous record carries has the acting static's own
     * contributions removed, because a model asked to predict the effect must
     * not be handed a board that already contains it. That removal is the
     * record's whole premise, so a checkout that cannot do it collects nothing
     * here rather than a corpus whose inputs contain their own labels.
     */
    public void collectContinuous() {
        if (mode == AttributionMode.DEGRADED || !CAN_SUPPRESS) {
            return;
        }
        Map<Long, Map<String, Contribution>> byStatic = new LinkedHashMap<>();
        for (Card card : game.getCardsIn(ZoneType.Battlefield)) {
            String entity = SnapshotBuilder.entityId(card);
            for (Table.Cell<Long, Long, Pair<Integer, Integer>> cell
                    : card.getPTBoostTable().cellSet()) {
                contribution(byStatic, cell.getColumnKey(), entity)
                        .boost(cell.getValue().getLeft(), cell.getValue().getRight());
            }
            nameContributions(card, byStatic, entity);
            for (Table.Cell<Long, Long, KeywordsChange> cell
                    : card.getChangedCardKeywords().cellSet()) {
                Contribution into = contribution(byStatic, cell.getColumnKey(), entity);
                for (KeywordInterface keyword : cell.getValue().getKeywords()) {
                    into.keyword(keyword.getOriginal());
                }
                // The same channel's other half. A static that takes a keyword
                // away had no path to keywords_lost, which is a head field, so
                // "loses flying" and "does nothing" recorded identically.
                for (String removed : cell.getValue().getRemoveKeywords()) {
                    into.keyword("-" + removed);
                }
            }
            for (Map.Entry<Long, List<Object>> changed
                    : changesByStatic(card, PatchHooks.CHANGED_TYPES_BY_STATIC)
                            .entrySet()) {
                Contribution into = contribution(byStatic, changed.getKey(), entity);
                for (Object change : changed.getValue()) {
                    typeTokens(change, into);
                }
            }
            for (Map.Entry<Long, List<Object>> changed
                    : changesByStatic(card, PatchHooks.CHANGED_COLORS_BY_STATIC)
                            .entrySet()) {
                Contribution into = contribution(byStatic, changed.getKey(), entity);
                for (Object change : changed.getValue()) {
                    colorTokens(change, into);
                }
            }
        }

        String boardHash = boardHash();
        for (Map.Entry<Long, Map<String, Contribution>> entry : byStatic.entrySet()) {
            // Id 0 is Forge's "no static" column: a temporary pump written by a
            // resolving ability, which the resolution bracket already records.
            if (entry.getKey() == 0L) {
                continue;
            }
            StaticAbility source = staticById(entry.getKey());
            if (source == null) {
                continue;
            }
            String staticKey = String.valueOf(entry.getKey());
            if (!allowContinuousRecord(staticKey, boardHash)) {
                continue;
            }
            StringJoiner contributions = new StringJoiner(",", "[", "]");
            for (Contribution contribution : entry.getValue().values()) {
                contributions.add(contribution.toJson());
            }
            ProvenanceKey.Resolved acting = ProvenanceKey.resolve(source);
            String state = snapshots.toJson(null, List.of(), entry.getKey());
            String payload = "{\"contributions\":" + contributions
                    + ",\"board_hash\":" + Json.string(boardHash) + "}";
            // A second gate behind the per-board one, which cannot see this:
            // two *distinct* static ids -- a second copy of the same anthem,
            // two copies of one aura -- resolve to the same key, the same
            // contributions and the same suppressed board, so they render
            // identically and the board hash says they are different statics.
            if (!allowDistinctRecord(RecordKind.CONTINUOUS.wireName(), payload, state)) {
                continue;
            }
            emit(context.newRecord(RecordKind.CONTINUOUS)
                    .actor(SnapshotBuilder.playerId(source.getHostCard().getController()))
                    .ability(acting)
                    .state(state)
                    .payload(payload));
        }
    }

    private static Contribution contribution(
            Map<Long, Map<String, Contribution>> byStatic, Long staticId, String entity) {
        return byStatic
                .computeIfAbsent(staticId, id -> new LinkedHashMap<>())
                .computeIfAbsent(entity, Contribution::new);
    }

    /**
     * A patched layer accessor's answer, grouped by the static that wrote it.
     *
     * <p>Reflective because the accessor is the patch's, and the P/T and keyword
     * accessors beside it are not: those exist in stock Forge, which is what the
     * connector compiles against. An unpatched checkout answers an empty map,
     * though {@link #collectContinuous()} has already returned by then.
     */
    @SuppressWarnings("unchecked")
    private static Map<Long, List<Object>> changesByStatic(Card card, PatchHooks.Hook accessor) {
        Object answer = accessor.read(card);
        return answer instanceof Map ? (Map<Long, List<Object>>) answer : Map.of();
    }

    /**
     * A card's name contribution, filed under the static that wrote it.
     *
     * <p>Reflective because {@code getChangedCardNames} is the patch's own
     * accessor. Unlike the type and colour "ByStatic" readers above, that
     * table is not split across layers -- it is keyed (timestamp, static id)
     * exactly like {@code boostPT} -- so its shape here is the P/T loop's
     * rather than theirs: the raw table, filtered by column key directly
     * instead of pre-grouped on the Forge side.
     */
    static void nameContributions(
            Card card, Map<Long, Map<String, Contribution>> byStatic, String entity) {
        if (!(PatchHooks.CHANGED_NAMES.read(card) instanceof Table<?, ?, ?> names)) {
            return;
        }
        for (Table.Cell<?, ?, ?> cell : names.cellSet()) {
            if (cell.getColumnKey() instanceof Long staticId) {
                nameTokens(cell.getValue(), contribution(byStatic, staticId, entity));
            }
        }
    }

    /**
     * What one type-layer entry does, as tokens.
     *
     * <p>Read off the change object rather than diffed out of the applied type
     * line, for the reason the P/T and keyword channels are: two statics that
     * both grant {@code Creature} are indistinguishable in the result, and the
     * record has to say which one this is.
     *
     * <p>A bare token is a type the static adds, {@code -x} one it removes, and
     * a lone {@code =} marks the rest as a type line the static sets outright
     * rather than adds to. Removing every type of a kind has no name to give, so
     * it gets {@code -all-creature-types} and its siblings.
     */
    static void typeTokens(Object change, Contribution into) {
        if (change instanceof CardChangedType typed) {
            addTypes(into, "", typed.addType());
            addTypes(into, "-", typed.removeType());
            if (typed.addAllCreatureTypes()) {
                into.type("all-creature-types");
            }
            for (RemoveType removed : typed.remove()) {
                into.type("-all-" + kebab(removed.name()));
            }
        } else if (change instanceof StateChangedType state) {
            into.type("=");
            addTypes(into, "", state.type());
        } else if (change instanceof WordChangedType word) {
            // A text change swaps one subtype word for another, which is a
            // removal and an addition in the same entry.
            into.type("-" + SnapshotBuilder.typeName(word.oldWord()));
            into.type(SnapshotBuilder.typeName(word.newWord()));
        }
    }

    private static void addTypes(Contribution into, String prefix, CardTypeView types) {
        if (types == null) {
            return;
        }
        for (var type : types.getCoreTypes()) {
            into.type(prefix + SnapshotBuilder.typeName(type));
        }
        for (var supertype : types.getSupertypes()) {
            into.type(prefix + SnapshotBuilder.typeName(supertype));
        }
        for (String subtype : types.getSubtypes()) {
            into.type(prefix + SnapshotBuilder.typeName(subtype));
        }
    }

    /** {@code CreatureTypes} as {@code creature-types}. */
    private static String kebab(String name) {
        return name.replaceAll("(?<=.)(?=\\p{Upper})", "-")
                .toLowerCase(java.util.Locale.ROOT);
    }

    /**
     * What one colour-layer entry does, as tokens.
     *
     * <p>The letters are the snapshot's, so a static's share of an entity's
     * colours reads the same as the entity's own. A lone {@code =} carries the
     * same meaning it does for types: the static sets the colour rather than
     * adding to it, and a {@code =} with nothing after it turns the entity
     * colourless.
     *
     * <p>Reflective on both components, because the record they belong to is
     * private in the stock Forge this side compiles against; the patch makes it
     * public.
     */
    static void colorTokens(Object change, Contribution into) {
        // Both components or neither: a half-read entry would report a colour
        // change as a replacement, which is the one reading that is never safe
        // to guess at.
        if (!(PatchHooks.COLOR_CHANGE_ADDITIONAL.read(change) instanceof Boolean additional)
                || !(PatchHooks.COLOR_CHANGE_COLORS.read(change) instanceof ColorSet colors)) {
            return;
        }
        List<String> letters = SnapshotBuilder.colorLetters(colors);
        if (!additional) {
            into.color("=");
            if (letters.isEmpty()) {
                // Setting the colour to nothing is what makes a permanent
                // colourless, which the vocabulary spells C rather than as an
                // empty list.
                into.color("C");
            }
        }
        for (String letter : letters) {
            into.color(letter);
        }
    }

    /**
     * What one name-layer entry does: the whole contribution is the name
     * itself, unlike a type or colour token, and a card either has a static
     * overwriting its name or it does not -- {@code newName} is null on
     * every entry that only ever set {@code addNonLegendaryCreatureNames}.
     *
     * <p>Reflective for the reason {@link #colorTokens} is: the record is
     * private in stock Forge.
     */
    static void nameTokens(Object change, Contribution into) {
        if (PatchHooks.NAME_CHANGE_NAME.read(change) instanceof String name) {
            into.name(name);
        }
    }

    /**
     * What the board looks like, for coalescing.
     *
     * <p>Names and computed power/toughness of everything in play. Two boards
     * that hash the same produce the same contributions, so the second record
     * would repeat the first exactly.
     */
    private String boardHash() {
        StringBuilder shape = new StringBuilder();
        for (Card card : game.getCardsIn(ZoneType.Battlefield)) {
            shape.append(card.getId()).append(':').append(card.getName())
                    .append(':').append(card.getNetPower())
                    .append('/').append(card.getNetToughness()).append(';');
        }
        return Integer.toHexString(shape.toString().hashCode());
    }

    /**
     * The static ability an id names, searched once per game and remembered.
     *
     * <p>The layer tables carry the id rather than the object, and nothing in
     * the engine maps one back, so this walks what is in play. Cached because a
     * phase boundary asks for the same handful of ids every time.
     */
    private StaticAbility staticById(Long id) {
        if (staticsById.containsKey(id)) {
            return staticsById.get(id);
        }
        StaticAbility found = null;
        for (Card card : game.getCardsInGame()) {
            for (StaticAbility candidate : card.getStaticAbilities()) {
                if (candidate.getId() == id) {
                    found = candidate;
                    break;
                }
            }
            if (found != null) {
                break;
            }
        }
        staticsById.put(id, found);
        return found;
    }

    /** One entity's share of one static's effect, accumulated across layers. */
    static final class Contribution {
        private final String entity;
        private int power;
        private int toughness;
        private final Set<String> keywords = new LinkedHashSet<>();
        // Sets rather than lists: a static that writes to two type layers can
        // name the same type twice, and the second says nothing the first did
        // not. Insertion-ordered so a "=" marker stays ahead of what it marks.
        private final Set<String> types = new LinkedHashSet<>();
        private final Set<String> colors = new LinkedHashSet<>();
        // Unlike the token sets above, a name contribution is a single value:
        // a static either overwrites the name or it does not, so the last
        // write standing is the answer rather than an accumulation. Null for
        // the statics that are not one of the handful that rename anything --
        // that is the correct, meaningful answer, not a missing one.
        private String name;

        Contribution(String entity) {
            this.entity = entity;
        }

        void boost(int addPower, int addToughness) {
            power += addPower;
            toughness += addToughness;
        }

        void keyword(String keyword) {
            if (keyword != null && !keyword.isEmpty()) {
                keywords.add(keyword.toLowerCase(java.util.Locale.ROOT));
            }
        }

        void type(String token) {
            if (token != null && !token.isEmpty()) {
                types.add(token);
            }
        }

        void color(String token) {
            if (token != null && !token.isEmpty()) {
                colors.add(token);
            }
        }

        void name(String value) {
            if (value != null && !value.isEmpty()) {
                name = value;
            }
        }

        String toJson() {
            return "{\"entity\":" + Json.string(entity)
                    + ",\"pt_boost\":[" + power + "," + toughness + "]"
                    + ",\"keywords\":" + Json.stringArray(keywords)
                    + ",\"types\":" + Json.stringArray(types)
                    + ",\"colors\":" + Json.stringArray(colors)
                    + ",\"name\":" + Json.string(name) + "}";
        }
    }

    /**
     * Whether a continuous record for this static on this board is new.
     *
     * <p>Coalescing per {@code (game, static, board hash)} is the cap: a static
     * applies on every recompute, and a second record for the same board would
     * repeat the first exactly.
     */
    public boolean allowContinuousRecord(String staticKey, String boardHash) {
        return coalescedBoards.add(staticKey + "@" + boardHash);
    }

    // ── plumbing ────────────────────────────────────────────────────────

    /**
     * The printed line a runtime trait came from, as a list of at most one.
     *
     * <p>For the payload fields that carry keys but no reason — a playability
     * candidate, a rewrite's {@code replaced_by}. A record's own acting line
     * goes through {@link ProvenanceKey#resolve} instead, keeping the reason
     * there is no key: an empty {@code ability} with nothing beside it cannot
     * separate "this card has no printed line anywhere" from "the resolver
     * regressed".
     */
    static List<ProvenanceKey> keysOf(CardTraitBase trait) {
        return ProvenanceKey.resolve(trait).keys();
    }

    /**
     * The trait's host, as the snapshot's tier-1 referenced list.
     *
     * <p>Without it a host outside every zone the other tiers sweep — one in
     * exile, say — is in no {@code state.entities}, and {@code refs.source}
     * would name an entity id the reader cannot follow.
     */
    private static List<Card> referencedOf(Card source) {
        return source == null ? List.of() : List.of(source);
    }

    /**
     * The player whose line acted, falling back to the active player.
     *
     * <p>A triggered or replacement ability belongs to its host's controller,
     * which is often not whoever's turn it is: a death trigger fires on the
     * opponent's turn as often as on its own. Forge decides it the same way —
     * {@code TriggerHandler.runSingleTriggerUnobserved} defaults the controller to
     * the host card's.
     */
    private String controllerOf(Card source) {
        if (source == null || source.getController() == null) {
            return activePlayerId();
        }
        return SnapshotBuilder.playerId(source.getController());
    }

    private String activePlayerId() {
        var phase = game == null ? null : game.getPhaseHandler();
        if (phase == null || phase.getPlayerTurn() == null) {
            return null;
        }
        return SnapshotBuilder.playerId(phase.getPlayerTurn());
    }

    /**
     * Name the match's random seat; the fork collector learns it too.
     *
     * <p>The mana reservoir's held records are stamped against it when they are
     * written, like every other record: the seat is fixed for the game.
     */
    public void setRandomSeatPlayerId(String playerId) {
        context.setRandomSeatPlayerId(playerId);
        if (forks != null) {
            forks.setRandomSeatPlayerId(playerId);
        }
    }

    private void emit(EffectRecord record) {
        context.emit(record);
    }
}
