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

    private final LiveGame live;
    private final CollectorContext context;
    private final SnapshotBuilder snapshots;
    private final AttributionMode mode;
    private final CollectionCaps caps;

    /** The sampled mana records, held until the game ends. */
    private final ManaReservoir manaReservoir;
    /**
     * The bracket this game's clause events are filed into:
     * {@code recordClauseEvent} is the door, and only the bracket collector for
     * this same game knows which resolution is open.
     */
    private final BusBracketCollector bracket;
    /** Boards a continuous static has already been recorded on, for coalescing. */
    private final Set<String> coalescedBoards = new LinkedHashSet<>();
    /** The records this game has written, shared by every channel that deduplicates. */
    private final RecordDedup dedup = new RecordDedup();
    /** Static ability by layer-table id, resolved once per game. */
    private final Map<Long, StaticAbility> staticsById = new HashMap<>();
    private final java.util.Random sampler;
    private final TriggerChannel triggers;

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
    private final List<Listener> listeners;


    /** @param bracket where the clause and outcome hooks file their events */
    PatchedCollectors(
            Game game, CollectorContext context, CollectionCaps caps, long seed,
            BusBracketCollector bracket) {
        this.live = new LiveGame(game);
        this.context = context;
        this.caps = caps;
        this.snapshots = live.snapshots();
        this.mode = context.mode();
        this.sampler = new java.util.Random(GameRecording.scramble(seed));
        this.manaReservoir = new ManaReservoir(caps.manaCap(), sampler);
        this.bracket = bracket;
        this.triggers = new TriggerChannel(
                live, context, dedup, new TriggerNegativeSampler(sampler));
        this.listeners = List.of(
                new Listener(PatchHooks.REPLACEMENT_LISTENER, this::rewriteHandler),
                new Listener(PatchHooks.TRIGGER_LISTENER, triggers::handler),
                new Listener(PatchHooks.PLAYABILITY_LISTENER, this::playabilityHandler),
                new Listener(PatchHooks.MANA_LISTENER, this::manaHandler),
                new Listener(PatchHooks.COMBAT_LISTENER, this::combatLegalityHandler),
                new Listener(PatchHooks.CLAUSE_LISTENER, this::clauseHandler),
                new Listener(PatchHooks.OUTCOME_LISTENER, this::outcomeHandler));
    }

    /** The trigger-condition channel. */
    TriggerChannel triggers() {
        return triggers;
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
        List<ProvenanceKey> ranKeys = ran == null ? List.of() : keysOf(ran);
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
            // exactly the cost the trigger hook's own identity check already
            // avoids by running first. A null candidate is let through: its
            // record names no ability either way.
            SpellAbility candidate = args.get(0, SpellAbility.class);
            if (candidate != null && !live.owns(candidate)) {
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
        String state = snapshots.ofAbility(
                candidate,
                candidate == null || candidate.getHostCard() == null
                        ? List.of() : List.of(candidate.getHostCard())).render();
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
        if (!dedup.allowDistinct(PlayabilitySubkind.DECISION.wireName(), payload, state)) {
            return;
        }
        emit(context.newPlayability(PlayabilitySubkind.DECISION)
                .actor(live.playerIdOrActive(decider))
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
            if (declaredAgainst != null && !live.owns(declaredAgainst)) {
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
        String state = snapshots.ofBoard().render();
        // Keyed on the snapshot as well as the payload (FR-028): the same
        // answer on a board that moved is a second observation, which a
        // payload-only key would drop as a repeat of the first.
        if (!dedup.allowDistinct(subkind.wireName(), payload, state)) {
            return;
        }
        emit(context.newPlayability(subkind)
                .actor(live.playerIdOrActive(decider))
                .whatIf(false)
                .state(state)
                .payload(payload));
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
        if (live.owns(clause)) {
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
        if (bracket != null && live.owns(clause)) {
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
                    ? live.owns(ability)
                    : live.ownsNullAbilityOutcome();
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
            if (ability == null || produced.isEmpty() || !live.owns(ability)) {
                return;
            }
            String key = manaAbilityText(ability, produced);
            int slot = manaReservoir.slot(key);
            if (slot < 0) {
                return;
            }
            Player producer = args.get(1, Player.class);
            String player = producer != null
                    ? SnapshotBuilder.playerId(producer) : live.activePlayerId();
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
                    .state(snapshots.ofBoard().render())
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
    static String manaAbilityText(SpellAbility ability, String produced) {
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
        for (Card card : live.game().getCardsIn(ZoneType.Battlefield)) {
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
            String state = snapshots.ofBoardWithout(entry.getKey()).render();
            String payload = "{\"contributions\":" + contributions
                    + ",\"board_hash\":" + Json.string(boardHash) + "}";
            // A second gate behind the per-board one, which cannot see this:
            // two *distinct* static ids -- a second copy of the same anthem,
            // two copies of one aura -- resolve to the same key, the same
            // contributions and the same suppressed board, so they render
            // identically and the board hash says they are different statics.
            if (!dedup.allowDistinct(RecordKind.CONTINUOUS.wireName(), payload, state)) {
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
        for (Card card : live.game().getCardsIn(ZoneType.Battlefield)) {
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
        for (Card card : live.game().getCardsInGame()) {
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

    private void emit(EffectRecord record) {
        context.emit(record);
    }
}
