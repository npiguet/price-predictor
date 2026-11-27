package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.json.Json;
import forge.game.GameEntity;
import forge.game.card.Card;
import forge.game.combat.Combat;
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
     * <p>Deduplicated on payload and snapshot (024 FR-028): a board on which
     * nothing moved between two declarations answers the same question the
     * same way, and two records that would be byte-identical carry one
     * observation once.
     *
     * <p>Package-private: the argument order has no compiler behind it, so
     * this hook's own game-identity test calls it with a synthesized argument
     * array.
     */
    Listener legalityHandler() {
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
                    recordBlockers(new BlockersDeclarable(
                            decider, args.get(1, Card.class),
                            asCards(args.get(2)), asCards(args.get(3)),
                            min == null ? 0 : min));
                }
            }
        });
    }

    /**
     * What the engine asks about one attacker coming at the defending
     * player: who could block it, who may, and how many it takes.
     *
     * @param decider the defending player the engine names (024 FR-030a) —
     *                gen-1 wrote the attacking player here, and a gen-2 shard
     *                is what tells a reader which meaning it holds
     */
    private record BlockersDeclarable(
            Player decider, Card attacker, List<Card> candidates, List<Card> legal,
            int minBlockers) {

        /** Whether the engine named no candidate, or no list of them. */
        boolean noCandidates() {
            return candidates == null || candidates.isEmpty();
        }
    }

    /**
     * One {@code blockers} record: who may block this attacker, how many it
     * takes, what keeps the rest out, and which of them a requirement makes
     * block it ({@code must_block}, {@link MustBlock}).
     */
    private void recordBlockers(BlockersDeclarable asked) {
        Card attacker = asked.attacker();
        if (attacker == null || asked.noCandidates()) {
            return;
        }
        Combat combat = live.game().getPhaseHandler().getCombat();
        LegalSet blockers = LegalSet.of(asked.candidates(), asked.legal(),
                card -> BlockLegality.Responsible.of(card,
                        BlockLegality.blockingStatic(attacker, card),
                        BlockLegality.ownBlockingKeyword(card)),
                card -> BlockLegality.requiredElsewhere(card, attacker, combat));
        List<String> mustBlock = new ArrayList<>();
        for (Card card : asked.candidates()) {
            if (MustBlock.required(card, attacker, combat)) {
                mustBlock.add(Refs.of(card));
            }
        }
        String payload = Json.object()
                .value("anchor_attacker", Refs.of(attacker))
                .raw("legal_blockers", blockers.legal())
                .raw("forbidden", blockers.forbidden())
                .raw("must_block", Json.stringArray(mustBlock))
                .value("min_blockers", asked.minBlockers())
                .render();
        emitLegality(context.newPlayability(PlayabilitySubkind.BLOCKERS).whatIf(false),
                asked.decider(), live.snapshots().ofBoard().state(), payload);
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
                Function<Card, BlockLegality.Responsible> responsible,
                java.util.function.Predicate<Card> requiredElsewhere) {
            List<String> forbidden = new ArrayList<>();
            for (Card card : candidates) {
                if (legal.contains(card)) {
                    continue;
                }
                forbidden.add(BlockLegality.forbiddenJson(
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
        // Keyed on the snapshot as well as the payload (024 FR-028): the same
        // answer on a board that moved is a second observation, which a
        // payload-only key would drop as a repeat of the first.
        context.emitDistinct(record
                .actor(live.playerIdOrActive(decider))
                .state(state)
                .payload(RecordPayload.withoutEvents(payload)), dedup);
    }

    /** Reflective, because the engine exposes these only on a patched checkout. */
    private static StaticAbility cantAttackStatic(Card attacker, GameEntity defender) {
        return (StaticAbility) PatchHooks.CANT_ATTACK_STATIC.invoke(
                new Class<?>[]{Card.class, GameEntity.class}, attacker, defender);
    }

    @SuppressWarnings("unchecked")
    private static List<Card> asCards(Object value) {
        return value instanceof List<?> list ? (List<Card>) list : List.of();
    }
}
