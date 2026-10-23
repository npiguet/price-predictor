package com.pricepredictor.connector.effects;

import forge.game.GameEntity;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.staticability.StaticAbility;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * {@code playability} records: rules-level verdicts only.
 *
 * <p>The AI's policy judgments never reach a record: they are its opinion
 * rather than the game's rules, and training on them would teach the model
 * Forge's play style instead of Magic. So the decision hook hands over only
 * the candidate, and the verdict is {@link PlayabilityVerdict}'s, computed
 * here after the sampler draw. A patch that still passes three verdict
 * booleans after the candidate has them ignored: they were computed by the
 * AI's own cost check, which is policy.
 *
 * <p>The {@code decision} subkind is sampled at {@code --playability-rate}
 * because the AI evaluates every candidate at every priority, which would
 * otherwise swamp the corpus. The {@code attackers} and {@code blockers}
 * subkinds are always logged: they fire once per combat, not once per
 * candidate.
 */
public final class PlayabilityChannel {

    private final LiveGame live;
    private final CollectorContext context;
    private final RecordDedup dedup;
    /** The game's shared sampling stream; see {@link PatchedCollectors}. */
    private final Random sampler;
    private final double playabilityRate;

    PlayabilityChannel(
            LiveGame live, CollectorContext context, RecordDedup dedup, Random sampler,
            double playabilityRate) {
        this.live = live;
        this.context = context;
        this.dedup = dedup;
        this.sampler = sampler;
        this.playabilityRate = playabilityRate;
    }

    // ── the decision subkind ────────────────────────────────────────────

    /**
     * The decision hook's listener.
     *
     * <p>Package-private: the argument order has no compiler behind it, so
     * this hook's own game-identity test calls it with a synthesized argument
     * array.
     */
    PatchHooks.Listener decisionHandler() {
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
            if (live.foreign(candidate)) {
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
     * <p>The sampler draw comes first, so a candidate the sampler drops costs
     * no rules check, and the candidate is read the moment it is sampled in.
     *
     * @param decider the candidate's controller, written as {@code actor_player}
     */
    public void recordCandidate(
            Player decider, SpellAbility candidate, Supplier<PlayabilityVerdict> verdictOf) {
        if (sampler.nextDouble() > playabilityRate) {
            return;
        }
        Optional<DecisionCandidate> read = DecisionCandidate.read(candidate, verdictOf);
        if (read.isEmpty()) {
            return;
        }
        // The candidate goes into the snapshot: without it refs.source is
        // null on every playability record, and a decision record would not
        // say which card the candidate is even on except through its key.
        String state = live.snapshots().ofAbility(
                candidate,
                candidate == null || candidate.getHostCard() == null
                        ? List.of() : List.of(candidate.getHostCard())).render();
        // Deduplicated because the AI re-asks the same question about the
        // same board, and the answer repeats verbatim. The record names no
        // acting line of its own; the candidate's key is in the payload, so
        // two candidates on one host stay two records.
        context.emitDistinct(context.newPlayability(PlayabilitySubkind.DECISION)
                .actor(live.playerIdOrActive(decider))
                .state(state)
                .payload(read.get().payloadJson()), dedup);
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
     * <p>Package-private for the reason {@link #decisionHandler()} is.
     */
    PatchHooks.Listener legalityHandler() {
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
     * @param decider the attacking player the engine names, written as
     *                {@code actor_player}
     */
    private void recordAttackers(
            Player decider, GameEntity defender, List<Card> candidates,
            List<Card> legal) {
        if (candidates == null || candidates.isEmpty()) {
            return;
        }
        LegalSet attackers = LegalSet.of(candidates, legal,
                card -> defender == null ? null : cantAttackStatic(card, defender));
        String payload = Json.object()
                .raw("legal_attackers", attackers.legal())
                .raw("forbidden", attackers.forbidden())
                .render();
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
    private void recordBlockers(
            Player decider, Card attacker, List<Card> candidates, List<Card> legal,
            int minBlockers) {
        if (attacker == null || candidates == null || candidates.isEmpty()) {
            return;
        }
        LegalSet blockers = LegalSet.of(candidates, legal,
                card -> cantBlockByStatic(attacker, card));
        String payload = Json.object()
                .value("anchor_attacker", SnapshotBuilder.entityId(attacker))
                .raw("legal_blockers", blockers.legal())
                .raw("forbidden", blockers.forbidden())
                .value("min_blockers", minBlockers)
                .render();
        emitLegality(PlayabilitySubkind.BLOCKERS, decider, payload);
    }

    /**
     * One declaration's legal set and the entities kept out of it, as the two
     * JSON arrays a legality payload carries.
     */
    private record LegalSet(String legal, String forbidden) {

        /**
         * @param responsible the static keeping a forbidden candidate out, or
         *                    null where the rules themselves do
         */
        static LegalSet of(
                List<Card> candidates, List<Card> legal,
                Function<Card, StaticAbility> responsible) {
            List<String> forbidden = new ArrayList<>();
            for (Card card : candidates) {
                if (legal.contains(card)) {
                    continue;
                }
                forbidden.add(forbiddenJson(card, responsible.apply(card)));
            }
            return new LegalSet(
                    Json.array(legal, card -> Json.string(SnapshotBuilder.entityId(card))),
                    Json.array(forbidden, Function.identity()));
        }
    }

    private void emitLegality(PlayabilitySubkind subkind, Player decider, String payload) {
        String state = live.snapshots().ofBoard().render();
        // Keyed on the snapshot as well as the payload (FR-028): the same
        // answer on a board that moved is a second observation, which a
        // payload-only key would drop as a repeat of the first.
        context.emitDistinct(context.newPlayability(subkind)
                .actor(live.playerIdOrActive(decider))
                .whatIf(false)
                .state(state)
                .payload(payload), dedup);
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
        ProvenanceKey key = responsible == null ? null : ProvenanceResolver.of(responsible);
        return Json.object()
                .value("entity", SnapshotBuilder.entityId(card))
                .raw("responsible_static",
                        ProvenanceKey.toJsonArray(key == null ? List.of() : List.of(key)))
                .render();
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
