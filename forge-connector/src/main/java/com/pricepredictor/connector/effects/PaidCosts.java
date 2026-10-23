package com.pricepredictor.connector.effects;

import forge.game.card.Card;
import forge.game.cost.CostTap;
import forge.game.mana.Mana;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * What an activation actually paid: the {@code costs} object of a resolution
 * cost half.
 *
 * <p><b>Read at the cast event and nowhere later.</b> Everything here lives on
 * the {@link SpellAbility} object and is cleared by its next activation
 * ({@code PlaySpellAbility} resets the paid hash, {@code CostPartMana} clears
 * the paying mana), so a deferred record holds this value rather than the
 * ability. The cast's outcome is only readable later, which is why the two
 * meet in {@link EffectRecord#costPayload} rather than here.
 *
 * <p>Deliberately not read off {@code getPayCosts()}, where {@code tapped},
 * {@code sacrificed}, {@code discarded} and {@code exiled} always read empty,
 * for two independent reasons, both in the engine: {@code CostAdjustment.adjust}
 * pays a {@code cost.copy()}, so the parts hanging off the ability are never
 * the ones that were paid; and {@code CostPayment} calls {@code resetLists()}
 * on every list part the moment payment completes, which is before the cast
 * event fires. The paid hash and the paying-mana list are where the payment
 * survives, and are what the rest of Forge reads.
 *
 * <p>Mana is counted one entry per {@link Mana} actually spent, so a hybrid
 * paid as blue counts under {@code U} and under nothing else — where the
 * printed cost would count it under both halves, and ignore reductions and X
 * entirely.
 *
 * @param mana       the mana spent, per symbol
 * @param tapped     the entity ids of everything the activation tapped
 * @param life       the life paid
 * @param sacrificed the entity ids sacrificed
 * @param discarded  the entity ids discarded
 * @param exiled     the entity ids exiled
 */
record PaidCosts(
        ManaCounts mana, List<String> tapped, int life,
        List<String> sacrificed, List<String> discarded, List<String> exiled) {

    /** What a cost payload says when there is no ability to read: every channel, empty. */
    static final PaidCosts EMPTY =
            new PaidCosts(ManaCounts.NONE, List.of(), 0, List.of(), List.of(), List.of());

    PaidCosts {
        tapped = List.copyOf(tapped);
        sacrificed = List.copyOf(sacrificed);
        discarded = List.copyOf(discarded);
        exiled = List.copyOf(exiled);
    }

    /** What {@code ability} paid, read now; {@link #EMPTY} for no ability. */
    static PaidCosts of(SpellAbility ability) {
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
        return new PaidCosts(
                ManaCounts.tally(spent), tappedFor(ability), ability.getAmountLifePaid(),
                paid(ability, "Sacrificed"), paid(ability, "Discarded"), paid(ability, "Exiled"));
    }

    /**
     * The {@code costs} object. Mana lists only the symbols some was paid in,
     * and in {@link ManaCounts}' fixed order rather than the order it was
     * spent in.
     */
    String toJson() {
        return Json.object()
                .raw("mana_by_color", mana.toJsonOmittingZeros())
                .raw("tapped", Json.stringArray(tapped))
                .value("life", life)
                .raw("sacrificed", Json.stringArray(sacrificed))
                .raw("discarded", Json.stringArray(discarded))
                .raw("exiled", Json.stringArray(exiled))
                .render();
    }

    /**
     * The cards one cost channel took, by the engine's own hash key.
     *
     * <p>{@code CostSacrifice}, {@code CostDiscard}, {@code CostExile} and
     * {@code CostTapType} each report what they took under the name this looks
     * up ({@code getHashForLKIList}), and the entries are last-known-information
     * copies whose id is preserved — so the entity id still names the object the
     * snapshot describes.
     */
    private static List<String> paid(SpellAbility ability, String hashKey) {
        Set<String> ids = new LinkedHashSet<>();
        Iterable<Card> cards = ability.getPaidList(hashKey);
        if (cards != null) {
            for (Card card : cards) {
                if (card != null) {
                    ids.add(SnapshotBuilder.entityId(card));
                }
            }
        }
        return new ArrayList<>(ids);
    }

    /**
     * Everything this activation tapped, the host included.
     *
     * <p>{@code CostTap} — the bare {@code T} symbol, and much the commonest tap
     * cost there is — taps the host directly and is a plain {@code CostPart}
     * with no card list, so it could never have reached this field however the
     * paid lists were read. It is asked for by type rather than through
     * {@code hasTapCost()}, whose flag is only refreshed when a {@code Cost} is
     * copied.
     */
    private static List<String> tappedFor(SpellAbility ability) {
        Set<String> ids = new LinkedHashSet<>(paid(ability, "Tapped"));
        if (ability.getPayCosts() != null
                && ability.getPayCosts().hasSpecificCostType(CostTap.class)
                && ability.getHostCard() != null) {
            ids.add(SnapshotBuilder.entityId(ability.getHostCard()));
        }
        return new ArrayList<>(ids);
    }
}
