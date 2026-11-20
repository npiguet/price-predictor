package com.pricepredictor.connector.effects.directed.operations;

import com.google.common.collect.HashMultiset;
import forge.game.card.Card;
import forge.game.player.Player;

import java.util.Map;

/**
 * What {@link ReplaceCardOperation} carries from the old card to the card
 * that replaced it: the board state and designations that stay with the
 * permanent whatever card it is. Read off {@code old} when {@link #carry}
 * runs, after the replacement is on the battlefield.
 */
record CarriedState(Card old, Card replacement) {

    /** The old card's board state, onto the card that replaced it. */
    void carry() {
        replacement.setGameTimestamp(old.getGameTimestamp());
        replacement.setLayerTimestamp(old.getLayerTimestamp());
        replacement.setTurnInZone(old.getTurnInZone());
        replacement.setTurnInController(old.getTurnInController());
        replacement.copyControllersFrom(old, player -> player);
        replacement.setTapped(old.isTapped());
        replacement.setSickness(old.isFirstTurnControlled());
        replacement.setCounters(HashMultiset.create(old.getCounters()));
        replacement.setDamage(old.getDamage());
        replacement.setHasBeenDealtDeathtouchDamage(old.hasBeenDealtDeathtouchDamage());
        replacement.setDamageHistory(old.getDamageHistory());
        // Every layer row, as GameCopier carries a card's: a resolved
        // effect's (static id 0) stays; a static's is taken off and put
        // back by the static check the edit ends with (swapInStaticEffects).
        replacement.setPTTable(old.getSetPTTable());
        replacement.setPTBoost(old.getPTBoostTable());
        replacement.copyFrom(old);
        replacement.copyResolvedRowsFrom(old);
        // The permanent's designations, which stay with it whatever card it is.
        for (Map.Entry<Long, Player> goad : old.getGoadMap().entrySet()) {
            replacement.addGoad(goad.getKey(), goad.getValue());
        }
        replacement.setMonstrous(old.isMonstrous());
        replacement.setRenowned(old.isRenowned());
        carryDesignations();
        replacement.updateKeywordsCache();
    }

    /** Suspected as {@code old} was, its menace static stamped with the old one's timestamp. */
    private void carrySuspected() {
        if (!old.isSuspected()) {
            return;
        }
        replacement.setSuspected(true);
        if (replacement.isSuspected()) {
            replacement.getSuspectedStatic().putParam("Timestamp", old.getSuspectedStatic().getParam("Timestamp"));
        }
    }

    /**
     * The designations that are plain flags: the Ring-bearer (701.54, the
     * player's pointer repointed as {@code Card.copyStateFrom} leaves it to
     * the copier), suspected (701.60, its menace static stamped as it was) and
     * saddled (702.171b, with how many times and by whom this turn). Detained
     * and encoded are refused instead ({@link #REFUSALS}).
     */
    private void carryDesignations() {
        for (Player player : old.getGame().getPlayers()) {
            if (player.getRingBearer() == old) {
                player.setRingBearer(replacement);
            }
        }
        carrySuspected();
        for (int i = 0; i < old.getTimesSaddledThisTurn(); i++) {
            replacement.setSaddled(true);
        }
        if (!old.isSaddled()) {
            replacement.setSaddled(false);
        }
        replacement.setSaddledByThisTurn(old.getSaddledByThisTurn());
    }
}
