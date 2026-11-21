package com.pricepredictor.connector.effects;

import forge.game.GameObject;
import forge.game.GameObjectPredicates;
import forge.game.ability.AbilityUtils;
import forge.game.card.Card;
import forge.game.card.CardCopyService;
import forge.game.phase.PhaseHandler;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.Spell;
import forge.game.spellability.SpellAbility;
import forge.game.spellability.SpellAbilityRestriction;
import forge.game.staticability.StaticAbilityAdditionalActivations;
import forge.game.staticability.StaticAbilityNumLoyaltyAct;
import forge.util.Expressions;
import forge.util.collect.FCollection;

import java.util.EnumSet;
import java.util.Set;
import java.util.function.BooleanSupplier;
import java.util.function.Predicate;

/**
 * Which of a candidate's {@code SpellAbilityRestriction} checks fail, for
 * {@link PlayabilityChecks}: the activator check ({@code Activator$},
 * may-play permissions), the timing check, the zone check, the other
 * restrictions and the activation limits, each asked as Forge's
 * {@code canPlay} asks it.
 *
 * <p>Two of Forge's checks evaluate several parts at once and say nothing of
 * which failed — {@code checkTimingRestrictions} (whose turn, phase, combat)
 * and {@code checkOtherRestrictions} (a loyalty or boast limit, the script's
 * {@code Activation$} conditions, and a dozen keyword-specific gates) — so
 * when one fails, the parts with an identifier of their own are re-asked
 * here, line for line as Forge asks them, to name the one that failed; a
 * failure none of them explains is {@code script_restriction}.
 */
final class RestrictionFailures {

    private final SpellAbilityRestriction restriction;
    /** The card the checks read ({@link #restrictionHost}). */
    private final Card card;
    private final SpellAbility sa;
    private final Player activator;

    private RestrictionFailures(
            SpellAbilityRestriction restriction, Card card, SpellAbility sa, Player activator) {
        this.restriction = restriction;
        this.card = card;
        this.sa = sa;
        this.activator = activator;
    }

    /** The checks {@code sa}'s restrictions fail for {@code activator}; none when it has none. */
    static Set<PlayabilityCheck> of(SpellAbility sa, Player activator) {
        SpellAbilityRestriction restriction = sa.getRestrictions();
        if (restriction == null) {
            return EnumSet.noneOf(PlayabilityCheck.class);
        }
        return new RestrictionFailures(restriction, restrictionHost(sa, activator), sa, activator).failed();
    }

    private Set<PlayabilityCheck> failed() {
        Set<PlayabilityCheck> failed = EnumSet.noneOf(PlayabilityCheck.class);
        if (fails(() -> restriction.checkActivatorRestrictions(card, sa))) {
            failed.add(PlayabilityCheck.SCRIPT_RESTRICTION);
        }
        if (fails(() -> restriction.checkTimingRestrictions(card, sa))) {
            failed.addAll(timingParts());
        }
        if (fails(() -> restriction.checkZoneRestrictions(card, sa))) {
            failed.add(PlayabilityCheck.ZONE);
        }
        if (fails(() -> restriction.checkOtherRestrictions(card, sa, activator))) {
            failed.addAll(otherParts());
        }
        if (holds(this::limitReached)) {
            failed.add(PlayabilityCheck.ACTIVATION_LIMIT);
        }
        return failed;
    }

    /**
     * The card Forge's restriction checks read: a spell's host as
     * {@code Spell.canPlayFromHost} prepares it (a copy under the activator's
     * control when another player controls it, then the alternate face),
     * an ability's own host.
     */
    private static Card restrictionHost(SpellAbility sa, Player activator) {
        Card card = sa.getHostCard();
        if (!(sa instanceof Spell)) {
            return card;
        }
        if (card.getController() != activator) {
            card = CardCopyService.getLKICopy(card);
            card.setController(activator, 0);
        }
        Card alternate = sa.getAlternateHost(card);
        return alternate == null ? card : alternate;
    }

    // ── checkTimingRestrictions ──────────────────────────────────────

    /** The parts of {@code checkTimingRestrictions}, as it asks them. */
    private Set<PlayabilityCheck> timingParts() {
        PhaseHandler phase = activator.getGame().getPhaseHandler();
        Set<PlayabilityCheck> named = EnumSet.noneOf(PlayabilityCheck.class);
        if (wrongTurn(phase)) {
            named.add(PlayabilityCheck.WHOSE_TURN);
        }
        if (wrongPhase(phase)) {
            named.add(PlayabilityCheck.PHASE);
        }
        if (wrongCombatStep(phase)) {
            named.add(PlayabilityCheck.COMBAT);
        }
        return orRestriction(named);
    }

    private boolean wrongTurn(PhaseHandler phase) {
        return notPlayersTurn(phase) || notOpponentsTurn(phase);
    }

    private boolean notPlayersTurn(PhaseHandler phase) {
        return restriction.isPlayerTurn() && !phase.isPlayerTurn(activator);
    }

    private boolean notOpponentsTurn(PhaseHandler phase) {
        return restriction.isOpponentTurn() && !phase.getPlayerTurn().isOpponentOf(activator);
    }

    private boolean wrongPhase(PhaseHandler phase) {
        return !restriction.getPhases().isEmpty()
                && !restriction.getPhases().contains(phase.getPhase());
    }

    private boolean wrongCombatStep(PhaseHandler phase) {
        return pastFirstCombat(phase) || blockersSkipped(phase) || sneakOutsideBlockers(phase);
    }

    private boolean pastFirstCombat(PhaseHandler phase) {
        return restriction.getFirstCombatOnly()
                && phase.getNumCombat() > (phase.inCombat() ? 1 : 0);
    }

    private boolean blockersSkipped(PhaseHandler phase) {
        return restriction.getAfterBlockersOnly() && phase.skippedDeclareBlockers();
    }

    private boolean sneakOutsideBlockers(PhaseHandler phase) {
        return sa.isSneak() && !phase.is(PhaseType.COMBAT_DECLARE_BLOCKERS);
    }

    // ── checkOtherRestrictions ───────────────────────────────────────

    /** The parts of {@code checkOtherRestrictions} with an identifier of their own. */
    private Set<PlayabilityCheck> otherParts() {
        Set<PlayabilityCheck> named = EnumSet.noneOf(PlayabilityCheck.class);
        if (holds(this::repeatLimitReached)) {
            named.add(PlayabilityCheck.ACTIVATION_LIMIT);
        }
        if (holds(this::conditionUnmet)) {
            named.add(PlayabilityCheck.SCRIPT_CONDITION);
        }
        return orRestriction(named);
    }

    private static Set<PlayabilityCheck> orRestriction(Set<PlayabilityCheck> named) {
        if (named.isEmpty()) {
            named.add(PlayabilityCheck.SCRIPT_RESTRICTION);
        }
        return named;
    }

    /** A loyalty ability's per-turn limit, a boast's, an exhaust or power-up's per-game one. */
    private boolean repeatLimitReached() {
        if (sa.isPwAbility()) {
            return loyaltyLimitReached();
        }
        if (sa.isBoast()) {
            return sa.getActivationsThisTurn() >= additionalActivationsLimit();
        }
        if (sa.isExhaust() || sa.isPowerUp()) {
            return sa.getActivationsThisGame() >= additionalActivationsLimit();
        }
        return false;
    }

    private boolean loyaltyLimitReached() {
        int activated = card.getPlaneswalkerAbilityActivated();
        int limit = StaticAbilityNumLoyaltyAct.limitIncrease(card) ? 2 : 1;
        if (activated < limit) {
            return false;
        }
        limit += StaticAbilityNumLoyaltyAct.additionalActivations(card, sa)
                - (limit == 1 || card.planeswalkerActivationLimitUsed() ? 0 : 1);
        return activated >= limit;
    }

    private int additionalActivationsLimit() {
        return StaticAbilityAdditionalActivations.getLimit(card, sa, activator);
    }

    /**
     * The script's own conditions on the board: {@code Activation$}
     * (threshold, metalcraft, delirium, hellbent, blessing, solved),
     * {@code IsPresent$}, {@code ActivationLifeTotal$}, {@code CheckSVar$},
     * {@code ClassLevel$}.
     */
    private boolean conditionUnmet() {
        return activationUnmet() || presentUnmet() || lifeTotalUnmet() || sVarUnmet()
                || classLevelUnmet();
    }

    /** {@code Activation$}: hellbent, threshold, metalcraft, delirium, blessing, solved. */
    private boolean activationUnmet() {
        return unmet(restriction.isHellbent(), () -> activator.hasHellbent())
                || unmet(restriction.isThreshold(), () -> activator.hasThreshold())
                || unmet(restriction.isMetalcraft(), () -> activator.hasMetalcraft())
                || unmet(restriction.isDelirium(), () -> activator.hasDelirium())
                || unmet(restriction.isBlessing(), () -> activator.hasBlessing())
                || unmet(restriction.isSolved(), () -> card.isSolved());
    }

    /** A condition the script requires, and the board does not meet; asked only when required. */
    private static boolean unmet(boolean required, BooleanSupplier met) {
        return required && !met.getAsBoolean();
    }

    /** {@code IsPresent$}: how many of the objects it defines are valid, against its comparison. */
    private boolean presentUnmet() {
        if (restriction.getIsPresent() == null) {
            return false;
        }
        FCollection<GameObject> present = presentObjects();
        Predicate<GameObject> valid = GameObjectPredicates.restriction(
                restriction.getIsPresent().split(","), activator, card, sa);
        int left = (int) present.stream().filter(valid).count();
        String compare = restriction.getPresentCompare();
        return !Expressions.compare(left, compare,
                AbilityUtils.calculateAmount(card, compare.substring(2), sa));
    }

    private FCollection<GameObject> presentObjects() {
        if (restriction.getPresentDefined() != null) {
            return AbilityUtils.getDefinedObjects(sa.getHostCard(), restriction.getPresentDefined(), sa);
        }
        return new FCollection<>(activator.getGame().getCardsIn(restriction.getPresentZones()));
    }

    /** {@code ActivationLifeTotal$}: the activator's life, against its comparison. */
    private boolean lifeTotalUnmet() {
        if (restriction.getLifeTotal() == null) {
            return false;
        }
        int life = restriction.getLifeTotal().equals("You") ? activator.getLife() : 1;
        String amount = restriction.getLifeAmount();
        return !Expressions.compare(life, amount,
                AbilityUtils.calculateAmount(sa.getHostCard(), amount.substring(2), sa));
    }

    /** {@code CheckSVar$}. */
    private boolean sVarUnmet() {
        return restriction.getsVarToCheck() != null && !Expressions.compare(
                AbilityUtils.calculateAmount(card, restriction.getsVarToCheck(), sa),
                restriction.getsVarOperator(),
                AbilityUtils.calculateAmount(card, restriction.getsVarOperand(), sa));
    }

    /** {@code ClassLevel$}. */
    private boolean classLevelUnmet() {
        return restriction.getClassLevel() != null && !Expressions.compare(
                card.getClassLevel(), restriction.getClassLevelOperator(),
                AbilityUtils.calculateAmount(card, restriction.getClassLevel(), sa));
    }

    // ── activation limits ────────────────────────────────────────────

    /** {@code ActivationLimit$} and {@code GameActivationLimit$}, as {@code canPlay} asks them. */
    private boolean limitReached() {
        return turnLimitReached() || gameLimitReached();
    }

    private boolean turnLimitReached() {
        return restriction.getLimitToCheck() != null && sa.getActivationsThisTurn()
                >= AbilityUtils.calculateAmount(card, restriction.getLimitToCheck(), sa);
    }

    private boolean gameLimitReached() {
        return restriction.getGameLimitToCheck() != null && sa.getActivationsThisGame()
                >= AbilityUtils.calculateAmount(card, restriction.getGameLimitToCheck(), sa);
    }

    // ── asking Forge ─────────────────────────────────────────────────

    /**
     * Whether a check returned false. A check that throws is not reported:
     * it names a failure and never decides one, and the verdict's own
     * {@code canPlay} has already answered for it.
     */
    static boolean fails(BooleanSupplier check) {
        try {
            return !check.getAsBoolean();
        } catch (RuntimeException e) {
            return false;
        }
    }

    /** Whether a question answered true; a question that throws answers false. */
    static boolean holds(BooleanSupplier question) {
        try {
            return question.getAsBoolean();
        } catch (RuntimeException e) {
            return false;
        }
    }
}
