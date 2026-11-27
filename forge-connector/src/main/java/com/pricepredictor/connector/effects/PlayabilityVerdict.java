package com.pricepredictor.connector.effects;

import forge.game.cost.Cost;
import forge.game.cost.CostPayment;
import forge.game.player.Player;
import forge.game.spellability.AbilityActivated;
import forge.game.spellability.SpellAbility;
import forge.game.staticability.StaticAbility;

import java.util.List;
import java.util.Objects;
import java.util.function.BiFunction;
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
 * stopped early and an earlier evaluation's when it did not run at all. The
 * AI's mana check, {@code ComputerUtilMana.canPayManaCost}, is no rules check
 * either: it honours the AI's mana reservations and draws from the random
 * stream. Nor is Forge's cost adjustment in its test mode, which asks the
 * payer's controller what to convoke, improvise, waterbend, delve and
 * sacrifice to emerge, and marks the sacrifice on the live card. So
 * affordability is a rules-only mana check ({@link ManaAffordability})
 * against the cost the rules charge, with those reductions as the rules
 * allow them ({@link RulesCost}), and Forge's additional-cost check, and target
 * legality counts candidates rather than reading choices, with the choices
 * left on the ability set aside while it counts: Forge judges a cross-target
 * rule against them ({@link TargetingClauses#withNoneChosen}).
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
 * ability, the two things a check sets, and the targets set aside are put
 * back after it ({@link #asActivator}).
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
        return asActivator(candidate, activator,
                () -> evaluate(candidate, activator, new ChargedCost(candidate, activator)));
    }

    /**
     * The verdict for {@code candidate} played by {@code activator}, as
     * {@link #of} gives it, then {@code reader} over it and the cost it
     * charged, both in one {@link #asActivator} scope: the verdict runs in a
     * scope of its own inside it ({@link ActivatorScope#inner}), which puts
     * back what the checks set before the reader runs, with no second scan
     * of the game's abilities, and the cost is computed once for both.
     */
    static <T> T reading(SpellAbility candidate, Player activator,
                         BiFunction<PlayabilityVerdict, ChargedCost, T> reader) {
        ChargedCost charged = new ChargedCost(candidate, activator);
        ActivatorScope scope = ActivatorScope.over(candidate, activator);
        return scope.run(() -> {
            PlayabilityVerdict verdict = scope.inner().run(() -> evaluate(candidate, activator, charged));
            return reader.apply(verdict, charged);
        });
    }

    /**
     * Run {@code body} with {@code activator} as the candidate's activating
     * player, then put back every activating player and the host's cast
     * ability as they were: the checks set the candidate's activating player,
     * the mana check those of the mana abilities it counts on, and Forge's
     * cast-prohibition check the host's cast ability ({@link ActivatorScope}).
     * The body runs with no target chosen on any clause the candidate offers,
     * and the choices come back after it, also when it throws
     * ({@link TargetingClauses#withNoneChosen}), so every reader inside it,
     * the verdict, {@code legal_targets} and the sample, counts what the
     * rules offer before a choice.
     */
    public static <T> T asActivator(SpellAbility candidate, Player activator, Supplier<T> body) {
        return ActivatorScope.over(candidate, activator).run(body);
    }

    private static PlayabilityVerdict evaluate(SpellAbility candidate, Player activator, ChargedCost charged) {
        if (candidate.isLandAbility()) {
            return evaluateLand(candidate, activator);
        }
        boolean forgeCanPlay = candidate.canPlay();
        boolean timing = candidate.canCastTiming(activator);
        boolean affordable = affordable(candidate, activator, charged);
        StaticAbility prohibiting = PlayabilityChecks.prohibitingStatic(candidate, activator);
        List<FailedCheck> failed = PlayabilityChecks.failed(candidate, activator,
                new PlayabilityChecks.ForgeAnswers(forgeCanPlay, timing, prohibiting), affordable);
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
                PlayabilityChecks.failedLand(play, activator,
                        new PlayabilityChecks.ForgeAnswers(canPlay, timing, prohibiting)));
    }

    /** Whether the candidate failed {@code check}. */
    public boolean failed(PlayabilityCheck check) {
        return failedChecks.stream().anyMatch(f -> f.check() == check);
    }

    /**
     * The static forbidding the play, the candidate's own
     * {@code responsible_static}, or null.
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

    /**
     * Whether the cost cannot be paid at all (118.6): a spell whose mana cost
     * is "no cost" (Ancestral Vision), cast other than through an effect that
     * lets it be cast without paying it, as Forge's own {@code Spell.canPlay}
     * reads it; or an activated ability whose mana part resolves to no cost,
     * as {@code AbilityActivated.canPlay} reads it.
     */
    static boolean unpayable(SpellAbility candidate, Cost cost) {
        if (!cost.hasManaCost()) {
            return false;
        }
        if (candidate.isSpell()) {
            return !candidate.isCastFromPlayEffect() && cost.getCostMana().getMana().isNoCost();
        }
        return candidate instanceof AbilityActivated
                && cost.getCostMana().getManaCostFor(candidate).isNoCost();
    }

    /**
     * Whether the player has the mana, colours included, and the additional
     * costs, by the rules alone: no AI reservation, preference or memory, no
     * draw from Forge's random stream, and the cost's convoke, improvise,
     * waterbend, delve and emerge paid as the rules allow rather than as the
     * AI would choose ({@link RulesCost}, {@link ManaAffordability}). The
     * candidate's activating player is {@code payer}; X is zero.
     */
    public static boolean affordable(SpellAbility candidate, Player payer) {
        return affordable(candidate, payer, new ChargedCost(candidate, payer));
    }

    private static boolean affordable(SpellAbility candidate, Player payer, ChargedCost charged) {
        Cost cost = candidate.getPayCosts();
        if (cost == null) {
            return true;
        }
        if (unpayable(candidate, cost)) {
            return false;
        }
        RulesCost rules;
        try {
            rules = charged.get();
        } catch (RuntimeException e) {
            // The cost cannot be read on this board: the printed cost.
            rules = RulesCost.printed(cost);
        }
        return TapCreatureCost.manaPayable(candidate, payer, rules)
                && CostPayment.canPayAdditionalCosts(cost, candidate, false, payer);
    }
}
