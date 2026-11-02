package com.pricepredictor.connector.effects;

import forge.ai.ComputerUtilMana;
import forge.game.card.Card;
import forge.game.cost.Cost;
import forge.game.cost.CostPayment;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.staticability.StaticAbility;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Supplier;

/**
 * The rules' answers about a candidate ability, and the one place they are
 * computed for both seats (FR-030–032).
 *
 * <p>Every answer is the game's, never the AI's. Two engine calls look like
 * rules checks and are not: {@code ComputerUtilCost.canPayCost} refuses costs
 * the AI dislikes (a planeswalker's last loyalty, by a draw from the shared
 * random stream; a Ward cost; a Casualty sacrifice) and adds the mana its
 * Nether Void heuristics want, and a candidate's chosen targets are whatever
 * the AI's evaluation left on the ability object, which is nothing when it
 * stopped early and an earlier evaluation's when it did not run at all. So
 * affordability is the mana and additional-cost checks those two delegate to,
 * and target legality counts candidates rather than reading choices.
 *
 * <p>{@code canPlay} is the record's {@code can_play}: Forge's
 * {@code canPlay}, its {@code canCastTiming} and no static forbidding the
 * play, together. {@code failedChecks} says which checks failed, in
 * {@link PlayabilityCheck}'s order.
 *
 * <p>A land play is judged by Forge's {@code canPlayLand} instead
 * (FR-033a): it costs nothing and targets nothing, so {@code affordable} and
 * {@code hasLegalTarget} are true, and its failed checks are the parts of
 * {@code canPlayLand} ({@link PlayabilityChecks#failedLand}).
 *
 * <p>Never writes to the game: the activating player and the host's cast
 * ability, the two things a check sets, are put back after it
 * ({@link #asActivator}).
 *
 * @param timing       Forge's {@code canCastTiming} for the activator
 * @param failedChecks every check the candidate failed, once each
 */
public record PlayabilityVerdict(
        boolean canPlay, boolean affordable, boolean hasLegalTarget, boolean timing,
        List<FailedCheck> failedChecks) {

    /**
     * One check a candidate failed.
     *
     * @param responsibleStatic the static that decided the check, or null
     *                          where the rules themselves did
     */
    public record FailedCheck(PlayabilityCheck check, ProvenanceKey responsibleStatic) {
    }

    public PlayabilityVerdict {
        failedChecks = List.copyOf(failedChecks);
    }

    /**
     * The verdict for {@code candidate} played by {@code activator} as the
     * board stands now, whoever holds priority.
     */
    public static PlayabilityVerdict of(SpellAbility candidate, Player activator) {
        Objects.requireNonNull(activator, "a verdict is for a player");
        return asActivator(candidate, activator, () -> evaluate(candidate, activator));
    }

    /**
     * Run {@code body} with {@code activator} as the candidate's activating
     * player, then put back every activating player and the host's cast
     * ability as they were: the checks set the candidate's activating player,
     * the mana check those of the mana abilities it counts on, and Forge's
     * cast-prohibition check the host's cast ability.
     */
    public static <T> T asActivator(SpellAbility candidate, Player activator, Supplier<T> body) {
        Map<SpellAbility, Player> activators = new IdentityHashMap<>();
        activators.put(candidate, candidate.getActivatingPlayer());
        for (Card card : activator.getGame().getCardsInGame()) {
            for (SpellAbility ability : card.getSpellAbilities()) {
                activators.putIfAbsent(ability, ability.getActivatingPlayer());
            }
        }
        Card host = candidate.getHostCard();
        SpellAbility castBefore = host == null ? null : host.getCastSA();
        candidate.setActivatingPlayer(activator);
        try {
            return body.get();
        } finally {
            activators.forEach((ability, before) -> {
                if (ability.getActivatingPlayer() != before) {
                    ability.setActivatingPlayer(before);
                }
            });
            // The candidate last: an ability of its host may have trickled
            // another player down to it.
            candidate.setActivatingPlayer(activators.get(candidate));
            if (host != null && host.getCastSA() != castBefore) {
                host.setCastSA(castBefore);
            }
        }
    }

    private static PlayabilityVerdict evaluate(SpellAbility candidate, Player activator) {
        if (candidate.isLandAbility()) {
            return evaluateLand(candidate, activator);
        }
        boolean forgeCanPlay = candidate.canPlay();
        boolean timing = candidate.canCastTiming(activator);
        boolean affordable = affordable(candidate, activator);
        StaticAbility prohibiting = PlayabilityChecks.prohibitingStatic(candidate, activator);
        List<FailedCheck> failed = PlayabilityChecks.failed(
                candidate, activator, forgeCanPlay, timing, affordable, prohibiting);
        return new PlayabilityVerdict(
                forgeCanPlay && timing && prohibiting == null, affordable,
                TargetingClauses.reachMinimum(candidate), timing, failed);
    }

    /**
     * A land play: {@code canPlay} is {@code LandAbility.canPlay}, which is
     * {@code canPlayLand}; {@code timing} its player's turn with sorcery
     * timing or the play's flash.
     */
    private static PlayabilityVerdict evaluateLand(SpellAbility play, Player activator) {
        boolean canPlay = play.canPlay();
        boolean timing = PlayabilityChecks.landTiming(play, activator);
        StaticAbility prohibiting = PlayabilityChecks.prohibitingLandStatic(play, activator);
        return new PlayabilityVerdict(canPlay, true, true, timing,
                PlayabilityChecks.failedLand(play, activator, canPlay, timing, prohibiting));
    }

    /** Whether the candidate failed {@code check}. */
    public boolean failed(PlayabilityCheck check) {
        return failedChecks.stream().anyMatch(f -> f.check() == check);
    }

    /**
     * The static forbidding the play, the candidate's own
     * {@code responsible_static} (its gen-2 meaning), or null.
     */
    public ProvenanceKey prohibitingStatic() {
        return failedChecks.stream()
                .filter(f -> f.check() == PlayabilityCheck.STATIC_PROHIBITION)
                .map(FailedCheck::responsibleStatic)
                .filter(Objects::nonNull)
                .findFirst().orElse(null);
    }

    /** Whether the rules let the activator play it now: what the random seat draws from. */
    public boolean playable() {
        return canPlay && affordable && hasLegalTarget;
    }

    /** Whether the player has the mana and the additional costs, ignoring every AI preference. */
    static boolean affordable(SpellAbility candidate, Player payer) {
        Cost cost = candidate.getPayCosts();
        if (cost == null) {
            return true;
        }
        return ComputerUtilMana.canPayManaCost(cost, candidate, payer, 0, false)
                && CostPayment.canPayAdditionalCosts(cost, candidate, false, payer);
    }
}
