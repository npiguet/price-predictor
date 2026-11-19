package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.json.Json;
import forge.game.mana.Mana;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

/**
 * What an activation actually paid: the {@code costs} object of a resolution
 * cost half.
 *
 * <p><b>Read at the cast event and nowhere later.</b> Everything here lives on
 * the {@link SpellAbility} object and is cleared by its next activation
 * ({@code PlaySpellAbility} resets the paid hash, {@code CostPartMana} clears
 * the paying mana), so a deferred record holds this value rather than the
 * ability. The cast's outcome is only readable later, which is why the two
 * meet in {@link RecordPayloads#costPayload} rather than here.
 *
 * <p>Deliberately not read off {@code getPayCosts()}, where {@code tapped},
 * {@code sacrificed}, {@code discarded} and {@code exiled} always read empty,
 * for two independent reasons, both in the engine: {@code CostAdjustment.adjust}
 * pays a {@code cost.copy()}, so the parts hanging off the ability are never
 * the ones that were paid; and {@code CostPayment} calls {@code resetLists()}
 * on every list part the moment payment completes, which is before the cast
 * event fires. The paid hash and the paying-mana list are where the payment
 * survives, and are what the rest of Forge reads ({@link CostChannel}).
 *
 * <p>Counters and life survive nowhere on the ability, so they are what the
 * bus published while the payment ran ({@link PaymentLedger}): counters as a
 * signed net count per type (a loyalty cost {@code -2} is {@code LOYALTY: -2},
 * a {@code +1} is {@code LOYALTY: 1}), life as the sum of every payment.
 *
 * <p>Mana is counted one entry per {@link Mana} actually spent, so a hybrid
 * paid as blue counts under {@code U} and under nothing else — where the
 * printed cost would count it under both halves, and ignore reductions and X
 * entirely.
 *
 * @param mana     the mana spent, per symbol
 * @param moved    what the payment did to counters and life
 * @param entities the entity ids each card channel took
 */
record PaidCosts(ManaCounts mana, PaymentTally moved, Map<CostChannel, List<String>> entities) {

    /** What a cost payload says when there is no ability to read: every channel, empty. */
    static final PaidCosts EMPTY = new PaidCosts(ManaCounts.NONE, PaymentTally.NONE, Map.of());

    /** The channels whose entities left where they were: what the game keeps of a payment. */
    private static final List<CostChannel> CARD_MOVING = List.of(
            CostChannel.SACRIFICED, CostChannel.DISCARDED, CostChannel.EXILED,
            CostChannel.RETURNED, CostChannel.DELVED);

    PaidCosts {
        EnumMap<CostChannel, List<String>> all = new EnumMap<>(CostChannel.class);
        for (CostChannel channel : CostChannel.values()) {
            all.put(channel, List.copyOf(entities.getOrDefault(channel, List.of())));
        }
        entities = all;
    }

    /** The entity ids everything the activation tapped, the host included. */
    List<String> tapped() {
        return entities.get(CostChannel.TAPPED);
    }

    /** The life paid. */
    int life() {
        return moved.life();
    }

    /**
     * Whether the payment paid nothing the game keeps, so that paying it again
     * costs nothing (research C9): no mana, no card moved (sacrificed,
     * discarded, exiled, returned to hand, delved), no life and no counter
     * moved. A tap, an untap symbol, a reveal, an exert and a convoke tap
     * leave nothing a seat that repeats the activation at will loses.
     */
    boolean paidNothingKept() {
        return mana.isZero() && life() == 0 && !movedACounter() && !movedACard();
    }

    private boolean movedACounter() {
        return moved.counters().values().stream().anyMatch(net -> net != 0);
    }

    private boolean movedACard() {
        return CARD_MOVING.stream().anyMatch(channel -> !entities.get(channel).isEmpty());
    }

    /** What {@code ability} paid, read now, with no counters or life; {@link #EMPTY} for no ability. */
    static PaidCosts of(SpellAbility ability) {
        return of(ability, PaymentTally.NONE);
    }

    /** What {@code ability} paid, read now, and what the bus told of its counters and life. */
    static PaidCosts of(SpellAbility ability, PaymentTally moved) {
        if (ability == null) {
            return EMPTY;
        }
        List<Byte> spent = new ArrayList<>();
        if (ability.getPayingMana() != null) {
            for (Mana mana : ability.getPayingMana()) {
                if (mana != null) {
                    spent.add(mana.getColor());
                }
            }
        }
        Map<CostChannel, List<String>> read = new EnumMap<>(CostChannel.class);
        for (CostChannel channel : CostChannel.values()) {
            read.put(channel, channel.idsOf(ability));
        }
        return new PaidCosts(ManaCounts.tally(spent), moved, read);
    }

    /**
     * The {@code costs} object. Mana lists only the symbols some was paid in,
     * and in {@link ManaCounts}' fixed order rather than the order it was
     * spent in.
     */
    String toJson() {
        Json.ObjectWriter out = Json.object()
                .raw("mana_by_color", mana.toJsonOmittingZeros())
                .raw("tapped", Json.stringArray(tapped()))
                .value("life", life());
        for (CostChannel channel : CostChannel.values()) {
            if (channel != CostChannel.TAPPED) {
                out.raw(channel.member(), Json.stringArray(entities.get(channel)));
            }
        }
        Json.ObjectWriter counters = Json.object();
        moved.counters().forEach(counters::value);
        return out.raw("counters", counters.render()).render();
    }
}
