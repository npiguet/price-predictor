package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.json.Json;
import forge.game.GameEntity;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;

/**
 * What a {@code decision} record says about its candidate, read into values
 * the moment the candidate is evaluated.
 *
 * <p>Values rather than the ability: the legality check sets the checked
 * ability's activating player, and the AI may later choose targets on the
 * same object, so nothing may keep a reference to it. What the ability says
 * at this instant is safe to keep, and without it the record names no
 * ability at all.
 *
 * @param keys         the candidate's printed line, as a list of at most one
 * @param legalTargets refs of what the rules allow any of its clauses to target
 * @param manaCost     its total mana cost, or null where it has no costs or
 *                     is a land play
 * @param verdict      what the rules say of it on this board
 * @param landPlay     whether it plays a land (FR-033a), written as
 *                     {@code land_play}
 * @param sample       its stage-2 sampling bucket (FR-033b); never written
 */
public record DecisionCandidate(
        List<ProvenanceKey> keys,
        List<String> legalTargets,
        String manaCost,
        PlayabilityVerdict verdict,
        boolean landPlay,
        DecisionSample sample) {

    /**
     * Read {@code candidate} as {@code activator} would play it.
     *
     * @return empty where the rules check throws: the engine cannot evaluate
     *         that candidate on this board, and it is neither recorded nor
     *         played
     */
    static Optional<DecisionCandidate> read(SpellAbility candidate, Player activator) {
        try {
            return Optional.of(PlayabilityVerdict.asActivator(candidate, activator, () -> {
                boolean landPlay = candidate.isLandAbility();
                PlayabilityVerdict verdict = PlayabilityVerdict.of(candidate, activator);
                return new DecisionCandidate(
                        ProvenanceResolver.keysOf(candidate),
                        // A land play targets nothing and costs nothing (FR-033a).
                        landPlay ? List.of() : legalTargetsOf(candidate),
                        landPlay || candidate.getPayCosts() == null
                                ? null : String.valueOf(candidate.getPayCosts().getTotalMana()),
                        verdict,
                        landPlay,
                        DecisionSample.of(candidate, activator, verdict));
            }));
        } catch (RuntimeException e) {
            return Optional.empty();
        }
    }

    /**
     * What the rules allowed this candidate to target, not what the AI picked:
     * the union over every clause it offers, uncapped (FR-031).
     *
     * <p>Not {@code getTargets()}, which is the chosen set: the per-entity
     * target-legality head is specified to train on the legal set, and an AI's
     * choice is a policy judgment the spec keeps out of the corpus.
     */
    private static List<String> legalTargetsOf(SpellAbility candidate) {
        Set<String> ids = new LinkedHashSet<>();
        for (GameEntity entity : TargetingClauses.legalTargets(candidate)) {
            Refs.ofAny(entity).ifPresent(ids::add);
        }
        return new ArrayList<>(ids);
    }

    /** The record's payload: this one candidate. */
    String payloadJson() {
        Json.ObjectWriter cost = Json.object();
        if (manaCost != null) {
            cost.value("mana", manaCost);
        }
        ProvenanceKey prohibiting = verdict.prohibitingStatic();
        String candidate = Json.object()
                .raw("ability", ProvenanceKey.toJsonArray(keys))
                .raw("verdict", Json.object()
                        .value("can_play", verdict.canPlay())
                        .value("affordable", verdict.affordable())
                        .value("has_legal_target", verdict.hasLegalTarget())
                        .value("timing", verdict.timing())
                        .render())
                .raw("legal_targets", Json.stringArray(legalTargets))
                .raw("cost_after_adjustment", cost.render())
                .raw("responsible_static", keyArray(prohibiting))
                .raw("failed_checks", Json.array(verdict.failedChecks(), check -> Json.object()
                        .value("check", check.check())
                        .raw("responsible_static", keyArray(check.responsibleStatic()))
                        .render()))
                .value("land_play", landPlay)
                .render();
        return Json.object()
                .raw("candidates", Json.array(List.of(candidate), json -> json))
                .render();
    }

    private static String keyArray(ProvenanceKey key) {
        return ProvenanceKey.toJsonArray(key == null ? List.of() : List.of(key));
    }
}
