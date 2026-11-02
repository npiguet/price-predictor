package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.GameObject;
import forge.game.GameObjectPredicates;
import forge.game.ability.AbilityUtils;
import forge.game.card.Card;
import forge.game.card.CardCollection;
import forge.game.card.CardCopyService;
import forge.game.cost.CostPayment;
import forge.game.phase.PhaseHandler;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.AbilityActivated;
import forge.game.spellability.AbilityStatic;
import forge.game.spellability.AlternativeCost;
import forge.game.spellability.Spell;
import forge.game.spellability.SpellAbility;
import forge.game.spellability.SpellAbilityRestriction;
import forge.game.staticability.StaticAbility;
import forge.game.staticability.StaticAbilityAdditionalActivations;
import forge.game.staticability.StaticAbilityCantBeCast;
import forge.game.staticability.StaticAbilityMode;
import forge.game.staticability.StaticAbilityNumLoyaltyAct;
import forge.game.zone.Zone;
import forge.game.zone.ZoneType;
import forge.util.Expressions;
import forge.util.collect.FCollection;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.function.BooleanSupplier;
import java.util.function.Predicate;

/**
 * Which of Forge's playability checks a candidate fails, one
 * {@link PlayabilityCheck} each (FR-032).
 *
 * <p>Forge's per-check methods are the authority: a check is reported only
 * where the method that makes it returned false. Two of them evaluate
 * several checks at once and say nothing of which failed —
 * {@code checkTimingRestrictions} (whose turn, phase, combat) and
 * {@code checkOtherRestrictions} (a loyalty or boast limit, the script's
 * {@code Activation$} conditions, and a dozen keyword-specific gates) — so
 * when one fails, the parts with an identifier of their own are re-asked
 * here, line for line as Forge asks them, to name the one that failed. A
 * failure none of them explains is {@code script_restriction}, as is the
 * activator check ({@code Activator$}, may-play permissions), which no
 * other identifier names.
 *
 * <p>The re-asked parts are a reimplementation, for a documented reason:
 * Forge has no per-part method to call. They name a failure and never
 * decide one, so a part that drifts from Forge's can misname a check but
 * not change {@code can_play}.
 */
final class PlayabilityChecks {

    /**
     * The checks that explain Forge's {@code canPlay} being false; a
     * candidate whose {@code canPlay} failed and that failed none of them
     * failed something Forge does not name (a suppressed or detained source,
     * a phased-out one), reported as {@code script_restriction}.
     */
    private static final Set<PlayabilityCheck> IN_CAN_PLAY = EnumSet.of(
            PlayabilityCheck.SPLIT_SECOND, PlayabilityCheck.CASTING_SPEED,
            PlayabilityCheck.WHOSE_TURN, PlayabilityCheck.PHASE, PlayabilityCheck.COMBAT,
            PlayabilityCheck.ACTIVATION_LIMIT, PlayabilityCheck.SCRIPT_CONDITION,
            PlayabilityCheck.ZONE, PlayabilityCheck.SCRIPT_RESTRICTION);

    private PlayabilityChecks() {
    }

    /**
     * Every check {@code sa} fails for {@code activator}, in
     * {@link PlayabilityCheck}'s order.
     *
     * @param canPlay     Forge's {@code canPlay}
     * @param timing      Forge's {@code canCastTiming}
     * @param affordable  the verdict's affordability
     * @param prohibiting the static forbidding the play, or null
     */
    static List<PlayabilityVerdict.FailedCheck> failed(
            SpellAbility sa, Player activator, boolean canPlay, boolean timing,
            boolean affordable, StaticAbility prohibiting) {
        Set<PlayabilityCheck> failed = EnumSet.noneOf(PlayabilityCheck.class);
        Game game = activator.getGame();
        // Special actions (turning a face-down creature up) ignore split second (CR 702.61b).
        if (!(sa instanceof AbilityStatic) && game.getStack().isSplitSecondOnStack()) {
            failed.add(PlayabilityCheck.SPLIT_SECOND);
        }
        if (!timing) {
            failed.add(PlayabilityCheck.CASTING_SPEED);
        }
        SpellAbilityRestriction restriction = sa.getRestrictions();
        if (restriction != null) {
            Card card = restrictionHost(sa, activator);
            if (fails(() -> restriction.checkActivatorRestrictions(card, sa))) {
                failed.add(PlayabilityCheck.SCRIPT_RESTRICTION);
            }
            if (fails(() -> restriction.checkTimingRestrictions(card, sa))) {
                failed.addAll(timingParts(restriction, sa, activator));
            }
            if (fails(() -> restriction.checkZoneRestrictions(card, sa))) {
                failed.add(PlayabilityCheck.ZONE);
            }
            if (fails(() -> restriction.checkOtherRestrictions(card, sa, activator))) {
                failed.addAll(otherParts(restriction, card, sa, activator));
            }
            if (holds(() -> limitReached(restriction, card, sa))) {
                failed.add(PlayabilityCheck.ACTIVATION_LIMIT);
            }
        }
        if (prohibiting != null) {
            failed.add(PlayabilityCheck.STATIC_PROHIBITION);
        }
        if (!affordable) {
            failed.add(PlayabilityCheck.COST);
        }
        if (!canPlay && failed.stream().noneMatch(IN_CAN_PLAY::contains)
                && holds(() -> CostPayment.canPayAdditionalCosts(sa.getPayCosts(), sa, false))) {
            failed.add(PlayabilityCheck.SCRIPT_RESTRICTION);
        }
        return named(failed, prohibiting);
    }

    /** The failed checks in {@link PlayabilityCheck}'s order, the prohibition naming its static. */
    private static List<PlayabilityVerdict.FailedCheck> named(
            Set<PlayabilityCheck> failed, StaticAbility prohibiting) {
        ProvenanceKey prohibitor = prohibiting == null ? null : ProvenanceResolver.of(prohibiting);
        List<PlayabilityVerdict.FailedCheck> checks = new ArrayList<>();
        for (PlayabilityCheck check : failed) {
            checks.add(new PlayabilityVerdict.FailedCheck(
                    check, check == PlayabilityCheck.STATIC_PROHIBITION ? prohibitor : null));
        }
        return checks;
    }

    /**
     * Every check a land play fails for {@code activator}: the parts of
     * {@code Player.canPlayLand}, asked as it asks them (FR-033a).
     *
     * <p>{@code whose_turn} outside the player's turn (rule 305.3);
     * {@code casting_speed} whenever {@code timing} is false; {@code zone}
     * for a land on the battlefield, another owner's without a may-play
     * permission, or one outside the hand without one; the prohibiting
     * static; {@code land_play_limit} once the turn's land plays have reached
     * the player's limit, never when the player may play any number. A
     * {@code canPlay} false none of them explains is
     * {@code script_restriction}.
     *
     * @param canPlay     {@code LandAbility.canPlay}
     * @param timing      {@link #landTiming}
     * @param prohibiting the static forbidding the play, or null
     */
    static List<PlayabilityVerdict.FailedCheck> failedLand(
            SpellAbility play, Player activator, boolean canPlay, boolean timing,
            StaticAbility prohibiting) {
        Set<PlayabilityCheck> failed = EnumSet.noneOf(PlayabilityCheck.class);
        if (!activator.getGame().getPhaseHandler().isPlayerTurn(activator)) {
            failed.add(PlayabilityCheck.WHOSE_TURN);
        }
        if (!timing) {
            failed.add(PlayabilityCheck.CASTING_SPEED);
        }
        if (holds(() -> landZoneFails(play, activator))) {
            failed.add(PlayabilityCheck.ZONE);
        }
        if (prohibiting != null) {
            failed.add(PlayabilityCheck.STATIC_PROHIBITION);
        }
        if (!activator.getMaxLandPlaysInfinite()
                && activator.getLandsPlayedThisTurn() >= activator.getMaxLandPlays()) {
            failed.add(PlayabilityCheck.LAND_PLAY_LIMIT);
        }
        if (!canPlay && failed.isEmpty()) {
            failed.add(PlayabilityCheck.SCRIPT_RESTRICTION);
        }
        return named(failed, prohibiting);
    }

    /** A land play's timing: its player's turn, at sorcery timing or with the play's flash. */
    static boolean landTiming(SpellAbility play, Player activator) {
        return activator.getGame().getPhaseHandler().isPlayerTurn(activator)
                && (activator.canCastSorcery()
                        || holds(() -> play.withFlash(play.getHostCard(), activator)));
    }

    /** {@code canPlayLand}'s zone and ownership checks. */
    private static boolean landZoneFails(SpellAbility play, Player activator) {
        Card land = play.getHostCard();
        if (land.isInZone(ZoneType.Battlefield)) {
            return true;
        }
        boolean mayPlay = play.getMayPlay() != null;
        if (land.getOwner() != activator && !mayPlay) {
            return true;
        }
        Zone zone = activator.getGame().getZoneOf(land);
        return zone != null && !zone.is(ZoneType.Hand) && !mayPlay
                && !play.isAlternativeCost(AlternativeCost.Mayhem);
    }

    /**
     * The static that forbids playing {@code play}'s land, found the way
     * {@code StaticAbilityCantBeCast.cantPlayLandAbility} finds one, which
     * answers only whether one exists.
     */
    static StaticAbility prohibitingLandStatic(SpellAbility play, Player activator) {
        Card land = play.getHostCard();
        CardCollection sources = new CardCollection(
                activator.getGame().getCardsIn(ZoneType.STATIC_ABILITIES_SOURCE_ZONES));
        return firstApplying(sources, StaticAbilityMode.CantPlayLand,
                stAb -> StaticAbilityCantBeCast.applyCantPlayLandAbility(stAb, land, activator));
    }

    /**
     * The static that forbids casting or activating {@code sa}, found the way
     * {@code StaticAbilityCantBeCast} finds one, which answers only whether
     * one exists.
     *
     * <p>A spell is checked on a copy of its host as the AI checks it, last
     * known on the stack and cast from where it is: a prohibition can read
     * the cast's origin, and the copy takes the cast ability Forge's check
     * sets on its card.
     */
    static StaticAbility prohibitingStatic(SpellAbility sa, Player activator) {
        Game game = activator.getGame();
        CardCollection sources = new CardCollection(
                game.getCardsIn(ZoneType.STATIC_ABILITIES_SOURCE_ZONES));
        if (sa instanceof Spell) {
            Card card = castHost(sa, game);
            card.setCastSA(sa);
            sources.add(card);
            return firstApplying(sources, StaticAbilityMode.CantBeCast,
                    stAb -> StaticAbilityCantBeCast.applyCantBeCastAbility(stAb, sa, card, activator));
        }
        if (sa instanceof AbilityActivated && !sa.isTrigger()) {
            Card card = sa.getHostCard();
            return firstApplying(sources, StaticAbilityMode.CantBeActivated,
                    stAb -> StaticAbilityCantBeCast.applyCantBeActivatedAbility(
                            stAb, sa, card, activator));
        }
        return null;
    }

    private static StaticAbility firstApplying(
            CardCollection sources, StaticAbilityMode mode, Predicate<StaticAbility> applies) {
        for (Card source : sources) {
            for (StaticAbility stAb : source.getStaticAbilities()) {
                if (stAb.checkConditions(mode) && applies.test(stAb)) {
                    return stAb;
                }
            }
        }
        return null;
    }

    private static Card castHost(SpellAbility sa, Game game) {
        Card host = sa.getHostCard();
        Card copy = CardCopyService.getLKICopy(host);
        copy.setLKICMC(-1);
        copy.setLastKnownZone(game.getStackZone());
        copy.setCastFrom(host.getZone());
        return copy;
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

    /** The parts of {@code checkTimingRestrictions}, as it asks them. */
    private static Set<PlayabilityCheck> timingParts(
            SpellAbilityRestriction restriction, SpellAbility sa, Player activator) {
        PhaseHandler phase = activator.getGame().getPhaseHandler();
        Set<PlayabilityCheck> named = EnumSet.noneOf(PlayabilityCheck.class);
        if ((restriction.isPlayerTurn() && !phase.isPlayerTurn(activator))
                || (restriction.isOpponentTurn()
                        && !phase.getPlayerTurn().isOpponentOf(activator))) {
            named.add(PlayabilityCheck.WHOSE_TURN);
        }
        if (!restriction.getPhases().isEmpty()
                && !restriction.getPhases().contains(phase.getPhase())) {
            named.add(PlayabilityCheck.PHASE);
        }
        if ((restriction.getFirstCombatOnly()
                        && phase.getNumCombat() > (phase.inCombat() ? 1 : 0))
                || (restriction.getAfterBlockersOnly() && phase.skippedDeclareBlockers())
                || (sa.isSneak() && !phase.is(PhaseType.COMBAT_DECLARE_BLOCKERS))) {
            named.add(PlayabilityCheck.COMBAT);
        }
        return orRestriction(named);
    }

    /** The parts of {@code checkOtherRestrictions} with an identifier of their own. */
    private static Set<PlayabilityCheck> otherParts(
            SpellAbilityRestriction restriction, Card card, SpellAbility sa, Player activator) {
        Set<PlayabilityCheck> named = EnumSet.noneOf(PlayabilityCheck.class);
        if (holds(() -> repeatLimitReached(card, sa, activator))) {
            named.add(PlayabilityCheck.ACTIVATION_LIMIT);
        }
        if (holds(() -> conditionUnmet(restriction, card, sa, activator))) {
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

    /** {@code ActivationLimit$} and {@code GameActivationLimit$}, as {@code canPlay} asks them. */
    private static boolean limitReached(
            SpellAbilityRestriction restriction, Card card, SpellAbility sa) {
        if (restriction.getLimitToCheck() != null && sa.getActivationsThisTurn()
                >= AbilityUtils.calculateAmount(card, restriction.getLimitToCheck(), sa)) {
            return true;
        }
        return restriction.getGameLimitToCheck() != null && sa.getActivationsThisGame()
                >= AbilityUtils.calculateAmount(card, restriction.getGameLimitToCheck(), sa);
    }

    /** A loyalty ability's per-turn limit, a boast's, an exhaust or power-up's per-game one. */
    private static boolean repeatLimitReached(Card card, SpellAbility sa, Player activator) {
        if (sa.isPwAbility()) {
            int activated = card.getPlaneswalkerAbilityActivated();
            int limit = StaticAbilityNumLoyaltyAct.limitIncrease(card) ? 2 : 1;
            if (activated < limit) {
                return false;
            }
            limit += StaticAbilityNumLoyaltyAct.additionalActivations(card, sa)
                    - (limit == 1 || card.planeswalkerActivationLimitUsed() ? 0 : 1);
            return activated >= limit;
        }
        if (sa.isBoast()) {
            return sa.getActivationsThisTurn()
                    >= StaticAbilityAdditionalActivations.getLimit(card, sa, activator);
        }
        if (sa.isExhaust() || sa.isPowerUp()) {
            return sa.getActivationsThisGame()
                    >= StaticAbilityAdditionalActivations.getLimit(card, sa, activator);
        }
        return false;
    }

    /**
     * The script's own conditions on the board: {@code Activation$}
     * (threshold, metalcraft, delirium, hellbent, blessing, solved),
     * {@code IsPresent$}, {@code ActivationLifeTotal$}, {@code CheckSVar$},
     * {@code ClassLevel$}.
     */
    private static boolean conditionUnmet(
            SpellAbilityRestriction restriction, Card card, SpellAbility sa, Player activator) {
        if ((restriction.isHellbent() && !activator.hasHellbent())
                || (restriction.isThreshold() && !activator.hasThreshold())
                || (restriction.isMetalcraft() && !activator.hasMetalcraft())
                || (restriction.isDelirium() && !activator.hasDelirium())
                || (restriction.isBlessing() && !activator.hasBlessing())
                || (restriction.isSolved() && !card.isSolved())) {
            return true;
        }
        if (restriction.getIsPresent() != null) {
            FCollection<GameObject> present = restriction.getPresentDefined() != null
                    ? AbilityUtils.getDefinedObjects(
                            sa.getHostCard(), restriction.getPresentDefined(), sa)
                    : new FCollection<>(activator.getGame().getCardsIn(
                            restriction.getPresentZones()));
            Predicate<GameObject> valid = GameObjectPredicates.restriction(
                    restriction.getIsPresent().split(","), activator, card, sa);
            int left = (int) present.stream().filter(valid).count();
            String compare = restriction.getPresentCompare();
            if (!Expressions.compare(left, compare,
                    AbilityUtils.calculateAmount(card, compare.substring(2), sa))) {
                return true;
            }
        }
        if (restriction.getLifeTotal() != null) {
            int life = restriction.getLifeTotal().equals("You") ? activator.getLife() : 1;
            String amount = restriction.getLifeAmount();
            if (!Expressions.compare(life, amount,
                    AbilityUtils.calculateAmount(sa.getHostCard(), amount.substring(2), sa))) {
                return true;
            }
        }
        if (restriction.getsVarToCheck() != null && !Expressions.compare(
                AbilityUtils.calculateAmount(card, restriction.getsVarToCheck(), sa),
                restriction.getsVarOperator(),
                AbilityUtils.calculateAmount(card, restriction.getsVarOperand(), sa))) {
            return true;
        }
        return restriction.getClassLevel() != null && !Expressions.compare(
                card.getClassLevel(), restriction.getClassLevelOperator(),
                AbilityUtils.calculateAmount(card, restriction.getClassLevel(), sa));
    }

    /**
     * Whether a check returned false. A check that throws is not reported:
     * it names a failure and never decides one, and the verdict's own
     * {@code canPlay} has already answered for it.
     */
    private static boolean fails(BooleanSupplier check) {
        try {
            return !check.getAsBoolean();
        } catch (RuntimeException e) {
            return false;
        }
    }

    /** Whether a question answered true; a question that throws answers false. */
    private static boolean holds(BooleanSupplier question) {
        try {
            return question.getAsBoolean();
        } catch (RuntimeException e) {
            return false;
        }
    }
}
