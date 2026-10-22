package com.pricepredictor.connector.effects;

import forge.game.GameEntity;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.Supplier;

/**
 * What a {@code decision} record says about its candidate, read into values
 * the moment the candidate is sampled in.
 *
 * <p>Values rather than the ability: a verdict can be abandoned
 * mid-evaluation, and the legality check mutates the checked ability's
 * targets, so nothing may keep a reference to the ability object. What the
 * ability says at this instant is safe to keep, and without it the record
 * names no ability at all.
 *
 * @param keys         the candidate's printed line, as a list of at most one
 * @param legalTargets refs of what the rules allow it to target
 * @param manaCost     its total mana cost, or null where it has no costs
 */
record DecisionCandidate(
        List<ProvenanceKey> keys,
        List<String> legalTargets,
        String manaCost,
        PlayabilityVerdict verdict) {

    /** Legal-target refs one decision may carry before the payload is capped. */
    private static final int MAX_LEGAL_TARGETS = 32;

    /**
     * Read {@code candidate}, asking for its verdict first.
     *
     * @return empty where the verdict's rules check throws: the engine cannot
     *         evaluate that candidate on this board, the way the random seat
     *         treats it, and it is not recorded
     */
    static Optional<DecisionCandidate> read(
            SpellAbility candidate, Supplier<PlayabilityVerdict> verdictOf) {
        PlayabilityVerdict verdict;
        try {
            verdict = verdictOf.get();
        } catch (RuntimeException e) {
            return Optional.empty();
        }
        if (candidate == null) {
            return Optional.of(new DecisionCandidate(List.of(), List.of(), null, verdict));
        }
        return Optional.of(new DecisionCandidate(
                ProvenanceKey.keysOf(candidate),
                legalTargetsOf(candidate),
                candidate.getPayCosts() == null
                        ? null : String.valueOf(candidate.getPayCosts().getTotalMana()),
                verdict));
    }

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

    /** The record's payload: this one candidate. */
    String payloadJson() {
        return "{\"candidates\":[{"
                + "\"ability\":" + ProvenanceKey.toJsonArray(keys)
                + ",\"verdict\":{\"can_play\":" + verdict.canPlay()
                + ",\"affordable\":" + verdict.affordable()
                + ",\"has_legal_target\":" + verdict.hasLegalTarget() + "}"
                + ",\"legal_targets\":" + Json.stringArray(legalTargets)
                + ",\"cost_after_adjustment\":"
                + (manaCost == null
                        ? "{}"
                        : "{\"mana\":" + Json.string(manaCost) + "}")
                // responsible_static needs a cantBeCastStatic hook that does
                // not exist; the attacker and blocker subkinds carry theirs,
                // this one does not.
                + ",\"responsible_static\":[]}]}";
    }
}
