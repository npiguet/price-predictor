package com.pricepredictor.connector.effects;

import forge.game.card.Card;
import forge.game.keyword.Keyword;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.staticability.StaticAbilityCastWithFlash;
import forge.game.zone.Zone;
import forge.game.zone.ZoneType;

import java.util.EnumSet;
import java.util.Set;

/**
 * The bucket a decision candidate is sampled in once it has been evaluated:
 * stage 2 of decision sampling (FR-033b).
 *
 * <p>Stage 1 keeps a candidate at {@code --playability-rate} before anything
 * is evaluated. Stage 2 then keeps a special candidate always and the two
 * bulk buckets at their own rates, each a share of every candidate enumerated,
 * so a bulk bucket's stage-2 chance is its rate over the stage-1 rate
 * ({@link PlayabilityChannel#keeps}). The bucket is never written: it is how
 * the record was sampled, not what the rules said.
 */
public enum DecisionSample {
    /**
     * A candidate the rules treat unusually: a static prohibition; a failed
     * check other than {@code cost} and {@code casting_speed} on a card in a
     * zone it may be played from; or, its zone allowed, played from outside
     * its usual zone, through an alternative cost or a may-play static, or
     * with flash only through a static. Written whenever stage 1 keeps it.
     */
    SPECIAL,
    /**
     * Not special: the card cannot be played from where it is (fails
     * {@code zone}), or is another player's and fails the activator check.
     */
    ZONE,
    /** Not special, and passes {@code zone}: the card's usual way of being played. */
    PLAIN;

    /**
     * The checks that leave a candidate in a bulk bucket: the ones the
     * ordinary hand-and-battlefield case fails at almost every stop.
     */
    private static final Set<PlayabilityCheck> BULK = EnumSet.of(
            PlayabilityCheck.ZONE, PlayabilityCheck.COST, PlayabilityCheck.CASTING_SPEED);

    /**
     * {@code candidate}'s bucket for {@code activator}, given its verdict.
     *
     * <p>Asks the game only for a candidate whose zone is allowed and whose
     * failed checks are bulk, and then only read-only questions; run it
     * inside {@link PlayabilityVerdict#asActivator}, as the verdict is.
     */
    static DecisionSample of(SpellAbility candidate, Player activator, PlayabilityVerdict verdict) {
        boolean zone = verdict.failed(PlayabilityCheck.ZONE);
        Card host = candidate.getHostCard();
        boolean notItsPlayers = host != null && host.getController() != activator;
        for (PlayabilityVerdict.FailedCheck failed : verdict.failedChecks()) {
            if (!bulk(failed.check(), candidate, zone, notItsPlayers)) {
                return SPECIAL;
            }
        }
        if (zone || (notItsPlayers && verdict.failed(PlayabilityCheck.SCRIPT_RESTRICTION))) {
            return ZONE;
        }
        return playedUnusually(candidate, activator) ? SPECIAL : PLAIN;
    }

    /**
     * Whether a failed check leaves the candidate in bulk.
     *
     * <ul>
     *   <li>{@code zone}, {@code cost} and {@code casting_speed} always do.</li>
     *   <li>A land play's {@code whose_turn} is its timing (rule 305.3), where
     *       a spell's or an ability's is a restriction its own text adds.</li>
     *   <li>Beside a failed {@code zone}, every check but a static prohibition
     *       does: a card that cannot be played from where it is says nothing
     *       of the others, such as the land-play limit of a land on the
     *       battlefield.</li>
     *   <li>On a card another player controls, {@code script_restriction} does:
     *       it is the activator check, which no ability of another player's
     *       permanent passes unless its text lets any player activate it.</li>
     * </ul>
     */
    private static boolean bulk(
            PlayabilityCheck check, SpellAbility candidate, boolean zone, boolean notItsPlayers) {
        return BULK.contains(check)
                || (check == PlayabilityCheck.WHOSE_TURN && candidate.isLandAbility())
                || (zone && check != PlayabilityCheck.STATIC_PROHIBITION)
                || (notItsPlayers && check == PlayabilityCheck.SCRIPT_RESTRICTION);
    }

    /**
     * Whether a candidate the rules let be played from its zone is played
     * otherwise than the usual way: from outside its usual zone, through an
     * alternative cost or a may-play static, or with flash a static alone
     * grants while its player could not act at sorcery speed.
     */
    private static boolean playedUnusually(SpellAbility candidate, Player activator) {
        if (candidate.getAlternativeCost() != null || candidate.getMayPlay() != null) {
            return true;
        }
        Card host = candidate.getHostCard();
        if (host == null) {
            return false;
        }
        boolean castOrLand = candidate.isSpell() || candidate.isLandAbility();
        ZoneType usual = castOrLand ? ZoneType.Hand : ZoneType.Battlefield;
        Zone zone = host.getZone();
        if (zone == null || !zone.is(usual)) {
            return true;
        }
        return castOrLand && !activator.canCastSorcery() && !nativeFlash(candidate, host)
                && flashFromAStatic(candidate, host, activator);
    }

    /**
     * Whether a static grants the play flash. A static whose check throws
     * grants none here: the bucket only decides how often a record is written,
     * and must never cost the random seat a candidate it could play.
     */
    private static boolean flashFromAStatic(SpellAbility candidate, Card host, Player activator) {
        try {
            return StaticAbilityCastWithFlash.anyWithFlash(candidate, host, activator);
        } catch (RuntimeException e) {
            return false;
        }
    }

    /** Flash of its own, the first two of {@code SpellAbility.withFlash}'s answers. */
    private static boolean nativeFlash(SpellAbility candidate, Card host) {
        return candidate.getRestrictions().isInstantSpeed()
                || candidate.isCastFromPlayEffect()
                || host.isInstant()
                || host.hasKeyword(Keyword.FLASH);
    }
}
