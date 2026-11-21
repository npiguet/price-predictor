package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CardCollection;
import forge.game.card.CardCopyService;
import forge.game.cost.Cost;
import forge.game.cost.CostPayment;
import forge.game.player.Player;
import forge.game.spellability.AbilityActivated;
import forge.game.spellability.AbilityStatic;
import forge.game.spellability.AlternativeCost;
import forge.game.spellability.Spell;
import forge.game.spellability.SpellAbility;
import forge.game.staticability.StaticAbility;
import forge.game.staticability.StaticAbilityCantBeCast;
import forge.game.staticability.StaticAbilityMode;
import forge.game.zone.Zone;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.function.Predicate;

/**
 * Which of Forge's playability checks a candidate fails, one
 * {@link PlayabilityCheck} each (FR-032).
 *
 * <p>Forge's per-check methods are the authority: a check is reported only
 * where the method that makes it returned false. Split second, casting
 * speed, a prohibiting static and the cost are asked here; the checks of
 * the candidate's {@code SpellAbilityRestriction} are
 * {@link RestrictionFailures}', which re-asks the parts of the two that
 * evaluate several at once. A failure none of them explains is {@code script_restriction}, as is the
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
     * What Forge answered of a candidate, which the failed checks explain.
     *
     * @param canPlay     Forge's {@code canPlay} ({@code LandAbility.canPlay} for a land play)
     * @param timing      Forge's {@code canCastTiming} ({@link #landTiming} for a land play)
     * @param prohibiting the static forbidding the play, or null
     */
    record ForgeAnswers(boolean canPlay, boolean timing, StaticAbility prohibiting) {

        /** {@code casting_speed} when the timing failed, {@code static_prohibition} when a static forbids the play. */
        void addFailed(Set<PlayabilityCheck> failed) {
            if (!timing) {
                failed.add(PlayabilityCheck.CASTING_SPEED);
            }
            if (prohibiting != null) {
                failed.add(PlayabilityCheck.STATIC_PROHIBITION);
            }
        }
    }

    /**
     * Every check {@code sa} fails for {@code activator}, in
     * {@link PlayabilityCheck}'s order.
     *
     * @param forge      Forge's {@code canPlay}, {@code canCastTiming} and the
     *                   static forbidding the play
     * @param affordable the verdict's affordability
     */
    static List<PlayabilityVerdict.FailedCheck> failed(
            SpellAbility sa, Player activator, ForgeAnswers forge, boolean affordable) {
        Set<PlayabilityCheck> failed = EnumSet.noneOf(PlayabilityCheck.class);
        if (splitSecondForbids(sa, activator)) {
            failed.add(PlayabilityCheck.SPLIT_SECOND);
        }
        forge.addFailed(failed);
        failed.addAll(RestrictionFailures.of(sa, activator));
        if (!affordable) {
            failed.add(PlayabilityCheck.COST);
        }
        if (!forge.canPlay() && unexplained(sa, failed)) {
            failed.add(PlayabilityCheck.SCRIPT_RESTRICTION);
        }
        return named(failed, forge.prohibiting());
    }

    private static boolean splitSecondForbids(SpellAbility sa, Player activator) {
        return !ignoresSplitSecond(sa) && activator.getGame().getStack().isSplitSecondOnStack();
    }

    /**
     * Whether a false {@code canPlay} is left unexplained by {@code failed}:
     * none of {@link #IN_CAN_PLAY} failed, the additional costs can be paid,
     * and the cost is not one no one can pay (118.6), which {@code cost}
     * already names.
     */
    private static boolean unexplained(SpellAbility sa, Set<PlayabilityCheck> failed) {
        Cost cost = sa.getPayCosts();
        return failed.stream().noneMatch(IN_CAN_PLAY::contains)
                && !(cost != null && PlayabilityVerdict.unpayable(sa, cost))
                && RestrictionFailures.holds(() -> CostPayment.canPayAdditionalCosts(cost, sa, false));
    }

    /**
     * Whether split second leaves {@code sa} legal (CR 702.61b): a mana
     * ability, or a special action (CR 116.2) — a land play, turning a
     * face-down permanent face up, foretelling, plotting, suspending a card,
     * unlocking a door, and Forge's other special actions, which it builds as
     * {@code AbilityStatic} (an action ending an effect, Circling Vultures'
     * discard, a companion put into hand).
     */
    static boolean ignoresSplitSecond(SpellAbility sa) {
        return sa.isManaAbility() || sa.isLandAbility() || isSpecialAction(sa);
    }

    private static boolean isSpecialAction(SpellAbility sa) {
        return sa instanceof AbilityStatic
                || sa.isTurnFaceUp() || sa.isForetelling() || sa.isPlotting() || sa.isUnlock()
                || !sa.isSpell() && sa.isKeyword(forge.game.keyword.Keyword.SUSPEND);
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
     * @param forge {@code LandAbility.canPlay}, {@link #landTiming} and the
     *              static forbidding the play
     */
    static List<PlayabilityVerdict.FailedCheck> failedLand(
            SpellAbility play, Player activator, ForgeAnswers forge) {
        Set<PlayabilityCheck> failed = EnumSet.noneOf(PlayabilityCheck.class);
        if (!activator.getGame().getPhaseHandler().isPlayerTurn(activator)) {
            failed.add(PlayabilityCheck.WHOSE_TURN);
        }
        forge.addFailed(failed);
        if (RestrictionFailures.holds(() -> landZoneFails(play, activator))) {
            failed.add(PlayabilityCheck.ZONE);
        }
        if (landPlaysUsed(activator)) {
            failed.add(PlayabilityCheck.LAND_PLAY_LIMIT);
        }
        if (!forge.canPlay() && failed.isEmpty()) {
            failed.add(PlayabilityCheck.SCRIPT_RESTRICTION);
        }
        return named(failed, forge.prohibiting());
    }

    /** The turn's land plays have reached the player's limit, which is not "any number". */
    private static boolean landPlaysUsed(Player activator) {
        return !activator.getMaxLandPlaysInfinite()
                && activator.getLandsPlayedThisTurn() >= activator.getMaxLandPlays();
    }

    /** A land play's timing: its player's turn, at sorcery timing or with the play's flash. */
    static boolean landTiming(SpellAbility play, Player activator) {
        return activator.getGame().getPhaseHandler().isPlayerTurn(activator)
                && (activator.canCastSorcery()
                        || RestrictionFailures.holds(() -> play.withFlash(play.getHostCard(), activator)));
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
}
