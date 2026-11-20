package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.json.Json;
import forge.game.GameEntity;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.combat.CombatUtil;
import forge.game.keyword.KeywordInterface;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.staticability.StaticAbility;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.function.Function;

/**
 * {@code playability} records: rules-level verdicts only.
 *
 * <p>The AI's policy judgments never reach a record: they are its opinion
 * rather than the game's rules, and training on them would teach the model
 * Forge's play style instead of Magic. So a {@code decision} record carries
 * {@link PlayabilityVerdict}'s answer, computed from the rules for a
 * candidate {@link PriorityCandidates} enumerated.
 *
 * <p>The {@code decision} subkind is sampled in two stages (FR-033b),
 * because both players' candidates are enumerated at every priority stop,
 * which would otherwise swamp the corpus: stage 1 at
 * {@code --playability-rate} before a candidate is evaluated, stage 2 by its
 * {@link DecisionSample} bucket after. The {@code attackers} and
 * {@code blockers} subkinds are always logged: they fire once per combat, not
 * once per candidate.
 *
 * <p>An {@code attackers} record is anchored to one attacker
 * ({@link AttackDeclaration}) and written once the declaration is made; a
 * {@code blockers} record is anchored to one attacker by the engine itself.
 */
public final class PlayabilityChannel {

    private final LiveGame live;
    private final CollectorContext context;
    private final RecordDedup dedup;
    /** The game's shared sampling stream; see {@link PatchedCollectors}. */
    private final Random sampler;
    private final double playabilityRate;
    private final double zoneCandidateRate;
    private final double plainCandidateRate;
    /** The declaration of attackers being asked about, held until it is made. */
    private final AttackDeclaration declaration;
    /** The (candidate key, failed checks) pairs this recording's kept candidates have written. */
    private final Set<KeptPair> keptWritten = new HashSet<>();

    /** What a kept candidate is written once per recording for: its printed line and the checks it fails. */
    private record KeptPair(List<ProvenanceKey> keys, List<PlayabilityVerdict.FailedCheck> failedChecks) {
    }

    PlayabilityChannel(
            LiveGame live, CollectorContext context, RecordDedup dedup, Random sampler,
            CollectionCaps caps) {
        this.live = live;
        this.context = context;
        this.dedup = dedup;
        this.sampler = sampler;
        this.playabilityRate = caps.playabilityRate();
        this.zoneCandidateRate = caps.zoneCandidateRate();
        this.plainCandidateRate = caps.plainCandidateRate();
        this.declaration = new AttackDeclaration(live,
                this::emitAttackers,
                PlayabilityChannel::cantAttackStatic);
    }

    /**
     * The declaration of attackers, which reads the declaration's taps and
     * its end off the live game's bus; {@code GameRecording} subscribes it.
     */
    AttackDeclaration declaration() {
        return declaration;
    }

    /** The game is over: an attackers declaration still held is written. */
    void close() {
        declaration.close();
    }

    // ── the decision subkind ────────────────────────────────────────────

    /**
     * Stage 1: whether the next decision candidate is evaluated, at
     * {@code --playability-rate}: one draw from the game's shared stream.
     *
     * <p>Drawn before the candidate is evaluated, so a candidate the sampler
     * drops costs no rules check. {@link PriorityCandidates} draws only for
     * the live game, whose signals alone reach it.
     */
    boolean sampled() {
        return sampler.nextDouble() < playabilityRate;
    }

    /**
     * Whether {@code candidate} is one a directed fork keeps (FR-010c): its
     * decision record is written whatever either stage would draw, and
     * neither draws for it.
     */
    boolean kept(SpellAbility candidate) {
        return context.kept().keepsCandidate(candidate);
    }

    /**
     * Whether a kept candidate's read is written: only the first of its
     * (candidate key, failed checks) pair in this recording, a fork's own
     * (FR-010c). A kept candidate is read at every priority stop, and most
     * stops find it failing the same checks: written each time, one game's
     * near-identical records would fill the side its rule lacks.
     */
    boolean firstKept(DecisionCandidate read) {
        return keptWritten.add(new KeptPair(read.keys(), read.verdict().failedChecks()));
    }

    /**
     * Stage 2: whether a candidate stage 1 kept is written, by its bucket.
     *
     * <p>A bulk bucket's rate is a share of every candidate enumerated, so its
     * chance here is that rate over the stage-1 rate. A special candidate,
     * or a bucket whose rate reaches the stage-1 rate, is kept without a
     * draw, which leaves the stream as one draw per candidate when every
     * rate is 1.
     */
    boolean keeps(DecisionSample sample) {
        double rate = switch (sample) {
            case SPECIAL -> playabilityRate;
            case ZONE -> zoneCandidateRate;
            case PLAIN -> plainCandidateRate;
        };
        if (rate >= playabilityRate) {
            return true;
        }
        return sampler.nextDouble() < rate / playabilityRate;
    }

    /**
     * One {@code decision} record, for a candidate {@link PriorityCandidates}
     * evaluated.
     *
     * @param decider the candidate's player, written as {@code actor_player}
     */
    void recordDecision(Player decider, SpellAbility candidate, DecisionCandidate read) {
        // The candidate's host goes into the snapshot: without it refs.source
        // is null on every playability record, and a decision record would
        // not say which card the candidate is even on except through its key.
        RecordState state = live.snapshots().ofCandidate(candidate).state();
        // Deduplicated because the same candidate at the next stop on an
        // unchanged board answers the same question the same way. The record
        // names no acting line of its own; the candidate's key is in the
        // payload, so two candidates on one host stay two records.
        context.emitDistinct(context.newPlayability(PlayabilitySubkind.DECISION)
                .actor(live.playerIdOrActive(decider))
                .state(state)
                .payload(read.payload()), dedup);
    }

    // ── the legality subkinds ───────────────────────────────────────────

    /**
     * The legal attacker and blocker sets, and what keeps the rest out.
     *
     * <p>The eighth sampling class, and the only one that says what the rules
     * *forbid* rather than what happened.
     *
     * <p>Asked by the engine's turn-based actions just before a player
     * declares, once per defender and once per attacker coming at the
     * defending player, so every record is a real declaration whoever
     * controls the seat — the AI, the random seat or a human. The engine
     * names the deciding player, every creature that player controls is a
     * candidate, and the legal set comes from the rules alone, against a
     * combat holding the attackers and no blocks yet. The per-defender
     * answers are pivoted to one record per attacker
     * ({@link AttackDeclaration}), which carries no {@code what_if}: no
     * what-if attackers record exists (FR-037). A blockers record is stamped
     * {@code what_if=false}.
     *
     * <p>Deduplicated on payload and snapshot (FR-028): a board on which
     * nothing moved between two declarations answers the same question the
     * same way, and two records that would be byte-identical carry one
     * observation once.
     *
     * <p>Package-private: the argument order has no compiler behind it, so
     * this hook's own game-identity test calls it with a synthesized argument
     * array.
     */
    PatchHooks.Listener legalityHandler() {
        return PatchHooks.COMBAT_LISTENER.listener((callback, args) -> {
            // No CardTraitBase here: the defender or attacker is the
            // GameEntity itself, so identity comes straight off it rather
            // than through a host card. A wrong-typed or null entity is left
            // to the declaration and recordBlockers, which tolerate both.
            Player decider = args.get(0, Player.class);
            GameEntity declaredAgainst = args.get(1, GameEntity.class);
            if (declaredAgainst != null && !live.owns(declaredAgainst)) {
                return;
            }
            switch (callback) {
                case ON_ATTACKERS_DECLARABLE -> declaration.declarable(
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
     * One {@code blockers} record: who may block this attacker, how many it
     * takes, what keeps the rest out, and which of them a requirement makes
     * block it ({@code must_block}, {@link MustBlock}).
     *
     * @param decider the defending player the engine names (FR-030a) —
     *                gen-1 wrote the attacking player here, and a gen-2 shard
     *                is what tells a reader which meaning it holds
     */
    private void recordBlockers(
            Player decider, Card attacker, List<Card> candidates, List<Card> legal,
            int minBlockers) {
        if (attacker == null || candidates == null || candidates.isEmpty()) {
            return;
        }
        Combat combat = live.game().getPhaseHandler().getCombat();
        LegalSet blockers = LegalSet.of(candidates, legal,
                card -> Responsible.of(card, blockingStatic(attacker, card), ownBlockingKeyword(card)),
                card -> requiredElsewhere(card, attacker, combat));
        List<String> mustBlock = new ArrayList<>();
        for (Card card : candidates) {
            if (MustBlock.required(card, attacker, combat)) {
                mustBlock.add(Refs.of(card));
            }
        }
        String payload = Json.object()
                .value("anchor_attacker", Refs.of(attacker))
                .raw("legal_blockers", blockers.legal())
                .raw("forbidden", blockers.forbidden())
                .raw("must_block", Json.stringArray(mustBlock))
                .value("min_blockers", minBlockers)
                .render();
        emitLegality(context.newPlayability(PlayabilitySubkind.BLOCKERS).whatIf(false),
                decider, live.snapshots().ofBoard().state(), payload);
    }

    /**
     * One declaration's legal set and the entities kept out of it, as the two
     * JSON arrays a legality payload carries.
     */
    private record LegalSet(String legal, String forbidden) {

        /**
         * @param responsible the line keeping a forbidden candidate out, or
         *                    null where the rules themselves do
         */
        static LegalSet of(
                List<Card> candidates, List<Card> legal,
                Function<Card, Responsible> responsible,
                java.util.function.Predicate<Card> requiredElsewhere) {
            List<String> forbidden = new ArrayList<>();
            for (Card card : candidates) {
                if (legal.contains(card)) {
                    continue;
                }
                forbidden.add(forbiddenJson(
                        card, responsible.apply(card), requiredElsewhere.test(card)));
            }
            return new LegalSet(
                    Json.array(legal, card -> Json.string(Refs.of(card))),
                    Json.array(forbidden, Function.identity()));
        }
    }

    /** One attackers record, with no {@code what_if}: only the turn-based declaration writes one. */
    private void emitAttackers(Player decider, RecordState state, String payload) {
        emitLegality(context.newPlayability(PlayabilitySubkind.ATTACKERS), decider, state, payload);
    }

    private void emitLegality(
            EffectRecord record, Player decider, RecordState state, String payload) {
        // Keyed on the snapshot as well as the payload (FR-028): the same
        // answer on a board that moved is a second observation, which a
        // payload-only key would drop as a repeat of the first.
        context.emitDistinct(record
                .actor(live.playerIdOrActive(decider))
                .state(state)
                .payload(RecordPayload.withoutEvents(payload)), dedup);
    }

    /**
     * The line that keeps a creature out of a legal set: a static
     * ability's (Pacifism, a flyer's keyword line through the static it
     * builds), else a keyword line of the creature's own that Forge reads
     * as a keyword and no static ("CARDNAME can't attack or block."), with
     * its key, or the reason it has none ({@link ProvenanceKey.Resolved});
     * and whether that line is the creature's own (a static it hosts, or
     * a keyword it prints) rather than another card's.
     */
    record Responsible(ProvenanceKey.Resolved line, boolean own) {

        /**
         * The static when there is one, else the creature's own keyword
         * line, else null: the rules themselves keep the creature out
         * (tapped, summoning sick), or a hidden keyword with no line does
         * (Panic's "can't block this turn", read off the entity's
         * {@code granted_temporary}).
         */
        static Responsible of(Card card, StaticAbility responsible, KeywordInterface ownLine) {
            if (responsible != null) {
                ProvenanceKey.Resolved line = ProvenanceResolver.resolve(responsible);
                return new Responsible(line == null ? ProvenanceKey.Resolved.UNKNOWN : line,
                        card.equals(responsible.getHostCard()));
            }
            if (ownLine != null) {
                // A keyword a static grants is that static's line, not the creature's.
                return new Responsible(keywordLine(ownLine), ownLine.getStatic() == null);
            }
            return null;
        }

        /** A keyword line's key: printed on the creature, else its granting static's. */
        private static ProvenanceKey.Resolved keywordLine(KeywordInterface keyword) {
            Card host = keyword.getHostCard();
            ProvenanceKey key = ProvenanceResolver.keyOfKeyword(
                    host == null ? null : host.getCurrentState(), keyword);
            if (key != null) {
                return new ProvenanceKey.Resolved(key, null);
            }
            return new ProvenanceKey.Resolved(null, keyword.getStatic() == null
                    ? UnresolvedReason.UNINDEXABLE : UnresolvedReason.GRANTED_KEYWORD);
        }
    }

    /** The keyword lines Forge reads as keywords, not statics, to keep a creature from attacking. */
    private static final List<String> CANT_ATTACK_KEYWORDS = List.of(
            "CARDNAME can't attack or block.", "CARDNAME can't attack.");
    /** The same, from blocking at all ({@code CombatUtil.canBlock(Card, boolean)}). */
    private static final List<String> CANT_BLOCK_KEYWORDS = List.of(
            "CARDNAME can't attack or block.", "CARDNAME can't block.");
    /** The same, from blocking while its controller has no other creature. */
    private static final List<String> CANT_BLOCK_ALONE_KEYWORDS = List.of(
            "CARDNAME can't attack or block alone.", "CARDNAME can't block alone.");

    /**
     * The creature's own keyword line (not a hidden keyword) that keeps it
     * from blocking at all, as {@code CombatUtil.canBlock(Card, boolean)}
     * reads it, or null: "can't attack or block", "can't block", and the
     * "... alone" forms while its controller has no other creature in play.
     */
    static KeywordInterface ownBlockingKeyword(Card blocker) {
        boolean alone = blocker.getController() != null
                && blocker.getController().getCreaturesInPlay().size() < 2;
        for (KeywordInterface keyword : blocker.getKeywords()) {
            String line = keyword.getOriginal();
            if (CANT_BLOCK_KEYWORDS.contains(line)
                    || (alone && CANT_BLOCK_ALONE_KEYWORDS.contains(line))) {
                return keyword;
            }
        }
        return null;
    }

    /**
     * The creature's own keyword line (not a hidden keyword) that keeps it
     * from attacking at all, as {@code StaticAbilityCantAttackBlock.cantAttack}
     * reads it, or null.
     */
    static KeywordInterface ownAttackingKeyword(Card attacker) {
        for (KeywordInterface keyword : attacker.getKeywords()) {
            if (CANT_ATTACK_KEYWORDS.contains(keyword.getOriginal())) {
                return keyword;
            }
        }
        return null;
    }

    /**
     * One entity kept out of a legal set, and the line responsible.
     *
     * <p>The line is often absent, and that is a real answer rather than a
     * gap: a creature that is tapped or summoning sick is stopped by the
     * rules themselves and no line is to blame.
     */
    static String forbiddenJson(Card card, Responsible responsible) {
        return forbiddenJson(card, responsible, false);
    }

    /**
     * The same, for a blocker: {@code required_elsewhere: true}, last,
     * when what keeps it from blocking this attacker is a requirement to
     * block another (a lure, 509.1c), not a prohibition; absent otherwise.
     *
     * <p>{@code responsible_static} holds the responsible line's key: a
     * static's line, or the creature's own keyword line. After it,
     * {@code responsible_own: true} when that line is the creature's own (a
     * static it hosts, a keyword it prints), absent when it is another
     * card's; then {@code responsible_unresolved}, the reason, when the
     * responsible line has no key (as a record's {@code ability_unresolved}
     * does), leaving {@code responsible_static} empty: kept out by
     * something, though nothing names its line.
     */
    static String forbiddenJson(Card card, Responsible responsible, boolean requiredElsewhere) {
        ProvenanceKey.Resolved line = responsible == null ? null : responsible.line();
        var entry = Json.object()
                .value("entity", Refs.of(card))
                .raw("responsible_static", ProvenanceKey.toJsonArray(
                        line == null ? List.of() : line.keys()));
        if (responsible != null && responsible.own()) {
            entry.value("responsible_own", true);
        }
        if (line != null && line.key() == null) {
            entry.value("responsible_unresolved", line.reason().wireName());
        }
        if (requiredElsewhere) {
            entry.value("required_elsewhere", true);
        }
        return entry.render();
    }

    /**
     * Whether a requirement to block another attacker is what keeps
     * {@code blocker} from blocking {@code attacker}, though the
     * restrictions alone would let it (509.1c: requirements bind only
     * within what the restrictions allow). Read off Forge's own check, so
     * its whole requirement set counts (a lure, provoke, "must be blocked
     * by", and an attacker that "must be blocked if able", "by exactly one
     * creature" or "by two or more creatures" while that is unmet:
     * {@code CombatUtil.mustBlockAnAttacker}): every other gate of
     * {@code CombatUtil.canBlock(attacker, blocker, combat)} lets the block
     * through, and that check refuses it. A blocker the restrictions keep
     * from {@code attacker} anyway (it flies, the blocker does not) is not.
     */
    static boolean requiredElsewhere(Card blocker, Card attacker, Combat combat) {
        return combat != null
                && CombatUtil.canBlock(attacker, blocker)
                && CombatUtil.canBlock(blocker, combat)
                && CombatUtil.canBeBlocked(attacker, combat, blocker.getController())
                && !combat.isBlocking(blocker, attacker)
                && !CombatUtil.canBlock(attacker, blocker, combat);
    }

    /**
     * The static keeping {@code blocker} from blocking {@code attacker}: one
     * aimed at the pair, else one keeping it from blocking at all (Pacifism,
     * {@code cantBlockStatic}).
     */
    static StaticAbility blockingStatic(Card attacker, Card blocker) {
        StaticAbility pair = cantBlockByStatic(attacker, blocker);
        return pair != null ? pair : (StaticAbility) PatchHooks.CANT_BLOCK_STATIC.invoke(
                new Class<?>[]{Card.class}, blocker);
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
}
