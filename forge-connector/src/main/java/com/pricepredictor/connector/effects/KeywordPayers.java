package com.pricepredictor.connector.effects;

import forge.card.mana.ManaCostShard;
import forge.game.ability.AbilityUtils;
import forge.game.card.Card;
import forge.game.cost.Cost;
import forge.game.keyword.Keyword;
import forge.game.mana.ManaCostBeingPaid;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * What a candidate's keywords let pay part of its mana besides mana itself,
 * by the rules and whatever its payer would choose (root spec § 8.8): each
 * untapped creature its payer controls under convoke (702.51a: {1} or one
 * mana of the creature's colour; also an ability's {@code TapCreaturesForMana}),
 * each untapped artifact under improvise (702.126a: {1}), each untapped
 * artifact or creature under waterbend ({1}, no more of them than the
 * waterbend's amount), and each other card in its graveyard under delve
 * (702.66a: {1}).
 *
 * <p>Read as Forge's payment reads them ({@code CostAdjustment}): convoke,
 * improvise and delve on a spell only, and none on a face-down cast, whose
 * card has no keywords; waterbend on any ability whose raised cost carries
 * one. A permanent pays once, as one unit: a creature that both convokes and
 * waterbends convokes. The host of an ability that taps it does not pay.
 *
 * @param tappers each permanent that may tap to pay, in battlefield order,
 *                and the unit it pays as
 * @param limit   how many of the {@link ManaUnit#limited} tappers may pay:
 *                the waterbend's amount
 * @param delve   how many cards delve may exile
 */
record KeywordPayers(Map<Card, ManaUnit> tappers, int limit, int delve) {

    static final KeywordPayers NONE = new KeywordPayers(Map.of(), 0, 0);

    /** Which keywords of the candidate pay, and how much the waterbend allows. */
    private record Payment(boolean convoke, boolean improvise, int waterbend) {

        static Payment of(SpellAbility candidate, Cost raised) {
            Card host = candidate.getHostCard();
            boolean spell = isSpell(candidate);
            return new Payment(
                    spell && host.hasKeyword(Keyword.CONVOKE) || candidate.hasParam("TapCreaturesForMana"),
                    spell && host.hasKeyword(Keyword.IMPROVISE),
                    waterbendAmount(candidate, raised));
        }

        ManaUnit unitOf(Card permanent) {
            if (convoke && permanent.isCreature()) {
                return new ManaUnit(permanent.getColor().getColor(), false, false);
            }
            if (improvise && permanent.isArtifact()) {
                return ManaUnit.generic();
            }
            return waterbend > 0 && bends(permanent) ? new ManaUnit((byte) 0, false, true) : null;
        }

        /** Whether no permanent pays: the scan of the battlefield is skipped. */
        boolean tapsNothing() {
            return !convoke && !improvise && waterbend <= 0;
        }

        private static boolean bends(Card permanent) {
            return permanent.isArtifact() || permanent.isCreature();
        }
    }

    /**
     * What may pay for {@code candidate} played by {@code payer}.
     *
     * @param raised its cost as raised ({@code CostAdjustment.raise}), which
     *               carries a waterbend a static added
     */
    static KeywordPayers of(SpellAbility candidate, Player payer, Cost raised) {
        if (candidate.isTrigger() || candidate.isReplacementAbility()) {
            return NONE;
        }
        Payment payment = Payment.of(candidate, raised);
        Card tapsItself = raised != null && raised.hasTapCost() ? candidate.getHostCard() : null;
        return new KeywordPayers(tappers(payer, payment, tapsItself), payment.waterbend(), delve(candidate, payer));
    }

    /** The payer's untapped permanents that pay, but the host its own tap cost taps. */
    private static Map<Card, ManaUnit> tappers(Player payer, Payment payment, Card tapsItself) {
        if (payment.tapsNothing()) {
            return Map.of();
        }
        Map<Card, ManaUnit> tappers = new LinkedHashMap<>();
        for (Card permanent : payer.getCardsIn(ZoneType.Battlefield)) {
            ManaUnit unit = permanent == tapsItself || !permanent.canTap() ? null : payment.unitOf(permanent);
            if (unit != null) {
                tappers.put(permanent, unit);
            }
        }
        return tappers;
    }

    /** A spell cast face up: a face-down card has no keywords. */
    private static boolean isSpell(SpellAbility candidate) {
        return candidate.isSpell() && !candidate.isCastFaceDown();
    }

    /** How many cards delve may exile: every other card in the payer's graveyard. */
    private static int delve(SpellAbility candidate, Player payer) {
        Card host = candidate.getHostCard();
        if (!isSpell(candidate) || !host.hasKeyword(Keyword.DELVE)) {
            return 0;
        }
        return (int) payer.getCardsIn(ZoneType.Graveyard).stream().filter(card -> card != host).count();
    }

    /** The waterbend's amount, or 0 where the raised cost carries none. */
    private static int waterbendAmount(SpellAbility candidate, Cost raised) {
        String amount = raised == null ? null : raised.getMaxWaterbend();
        return amount == null ? 0 : AbilityUtils.calculateAmount(candidate.getHostCard(), amount, candidate);
    }

    /** Generic units for the cards delve could exile, no more than {@code generic} of them. */
    List<ManaUnit> delved(int generic) {
        List<ManaUnit> units = new ArrayList<>();
        for (int i = Math.min(delve, generic); i > 0; i--) {
            units.add(ManaUnit.generic());
        }
        return units;
    }

    /** How many of {@code units} cannot pay because the waterbend's amount is reached. */
    int beyondLimit(List<ManaUnit> units) {
        long limited = units.stream().filter(ManaUnit::limited).count();
        return (int) Math.max(0, limited - limit);
    }

    /**
     * {@code total} less the most these payers can pay of it, as
     * {@code cost_after_adjustment} reads it: every tapper pays, the
     * coloured shards first, as many of them as one tapper per shard allows
     * (a maximum matching, shards in the cost's order and tappers in
     * battlefield order), then the generic mana with the other tappers and
     * the delved cards. Whether the tappers would rather make mana is
     * {@code affordable}'s question, not this one.
     */
    ManaCostBeingPaid mostReduced(ManaCostBeingPaid total) {
        if (tappers.isEmpty() && delve == 0) {
            return total;
        }
        ManaCostBeingPaid reduced = new ManaCostBeingPaid(total);
        List<ManaUnit> units = new ArrayList<>(tappers.values());
        List<ManaCostShard> shards = ShardDemand.of(total).nonGeneric();
        int[] unitOf = ShardMatching.match(units, shards);
        List<ManaUnit> left = new ArrayList<>(units);
        for (int shard = 0; shard < shards.size(); shard++) {
            if (unitOf[shard] >= 0) {
                reduced.decreaseShard(shards.get(shard), 1);
                left.set(unitOf[shard], null);
            }
        }
        left.removeIf(unit -> unit == null);
        int payers = left.size() - beyondLimit(left) + delve;
        reduced.decreaseGenericMana(Math.min(payers, reduced.getUnpaidShards(ManaCostShard.GENERIC)));
        return reduced;
    }
}
