package com.pricepredictor.connector.effects;

import forge.game.card.Card;
import forge.game.cost.Cost;
import forge.game.cost.CostPart;
import forge.game.cost.CostTap;
import forge.game.cost.CostUntap;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * One way a cost takes permanents or cards, and where its payment survives.
 *
 * <p>Most channels are one of the engine's paid-list keys (the key
 * {@code getHashForLKIList} of the cost part names), whose entries are
 * last-known-information copies with their ids preserved. The bare {@code T}
 * and {@code Q} symbols have no list and act on the host
 * ({@link #hostPart}); convoke and delve are kept on the ability and the
 * cast card instead.
 */
enum CostChannel {
    TAPPED("tapped", "Tapped", CostTap.class),
    SACRIFICED("sacrificed", "Sacrificed", null),
    DISCARDED("discarded", "Discarded", null),
    EXILED("exiled", "Exiled", null),
    RETURNED("returned", "Returned", null),
    REVEALED("revealed", "Revealed", null),
    EXERTED("exerted", "Exerted", null),
    UNTAPPED("untapped", "Untapped", CostUntap.class),
    CONVOKED("convoked", null, null),
    DELVED("delved", null, null);

    private final String member;
    private final String hashKey;
    private final Class<? extends CostPart> hostPart;

    CostChannel(String member, String hashKey, Class<? extends CostPart> hostPart) {
        this.member = member;
        this.hashKey = hashKey;
        this.hostPart = hostPart;
    }

    /** The {@code costs} member this channel is written under. */
    String member() {
        return member;
    }

    /** The entity ids this channel took in {@code ability}'s payment, each once. */
    List<String> idsOf(SpellAbility ability) {
        Set<String> ids = new LinkedHashSet<>();
        for (Card card : cards(ability)) {
            if (card != null) {
                ids.add(Refs.of(card));
            }
        }
        if (hostPart != null && paidByHost(ability)) {
            ids.add(Refs.of(ability.getHostCard()));
        }
        return new ArrayList<>(ids);
    }

    /**
     * Asked for by type rather than through {@code hasTapCost()}, whose flag
     * is only refreshed when a {@code Cost} is copied.
     */
    private boolean paidByHost(SpellAbility ability) {
        Cost cost = ability.getPayCosts();
        return cost != null && cost.hasSpecificCostType(hostPart) && ability.getHostCard() != null;
    }

    private Iterable<Card> cards(SpellAbility ability) {
        Iterable<Card> cards = switch (this) {
            case CONVOKED -> ability.getTappedForConvoke();
            case DELVED -> ability.getHostCard() == null ? null : ability.getHostCard().getDelved();
            default -> ability.getPaidList(hashKey);
        };
        return cards == null ? List.of() : cards;
    }
}
