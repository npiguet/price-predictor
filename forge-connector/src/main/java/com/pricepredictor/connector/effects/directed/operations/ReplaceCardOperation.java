package com.pricepredictor.connector.effects.directed.operations;

import com.google.common.collect.HashMultiset;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import forge.game.Game;
import forge.game.GameEntity;
import forge.game.GameObject;
import forge.game.IEntityMap;
import forge.game.StaticEffect;
import forge.game.card.Card;
import forge.game.card.CardCollection;
import forge.game.combat.Combat;
import forge.game.player.Player;
import forge.game.zone.Zone;

import java.util.List;
import java.util.Objects;

/**
 * {@code replace_card}: put another card from the card database in a
 * permanent's place, as the same permanent in every respect but the card.
 *
 * <p>{@link RemovePermanentOperation}'s removing step, then
 * {@link AddPermanentOperation}'s adding step at the old card's place in the
 * battlefield list, with the old card's state carried across: controller,
 * tapped state, summoning sickness, damage (and whether any was deathtouch),
 * counters, timestamps (so its place in the timestamp order and "entered this
 * turn" are the old card's), the effects applied to it (a resolved pump's
 * power and toughness, keywords, types, colours and abilities granted or
 * removed until end of turn), who controls it and since when (a gain-control
 * effect's controller included), its soulbond pair, its damage history (what
 * it attacked, blocked and dealt damage to this turn), what is attached to it
 * and what it is attached to, and its place in combat. The static abilities
 * that applied to the old card are handed the new one in its place, so the
 * static check the edit ends with takes theirs off it and applies them anew
 * where they still apply. Attachments are relinked directly rather than
 * unattached and attached again, which would fire the attach and unattach
 * triggers; the combat is rebuilt with the new card wherever the old one
 * stood, by the same copy {@code GameCopier} builds a fork's combat with, so
 * its bands, blocks and damage-assignment orders stay as they were. Nothing
 * fires: the old card goes to no zone and the new one arrives with no enters
 * trigger or replacement.
 *
 * @param permanent the fork's permanent to replace
 * @param cardName  the replacing card's name in the card database
 */
public record ReplaceCardOperation(Card permanent, String cardName) implements BoardEdit {

    public ReplaceCardOperation {
        Objects.requireNonNull(cardName, "cardName");
    }

    @Override
    public ForkOperation operation() {
        return ForkOperation.REPLACE_CARD;
    }

    @Override
    public void apply(Game fork) {
        Card old = BoardEdit.onBattlefield(permanent);
        Player controller = old.getController();
        Zone battlefield = old.getZone();
        int index = battlefield.getCards(false).indexOf(old);
        Card replacement = NewCards.create(fork, cardName, old.getOwner());

        relinkAttachments(old, replacement);
        relinkPair(old, replacement);
        swapInCombat(fork, old, replacement);
        swapInStaticEffects(fork, old, replacement);
        RemovePermanentOperation.takeOff(fork, old);
        AddPermanentOperation.put(fork, replacement, controller, index);
        carryState(old, replacement);
        BoardEdit.settle(fork);
    }

    /** The old card's board state, onto the card that replaced it. */
    private static void carryState(Card old, Card replacement) {
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
        replacement.updateKeywordsCache();
    }

    /** {@code old}'s soulbond partner, paired with {@code replacement} instead. */
    private static void relinkPair(Card old, Card replacement) {
        Card partner = old.getPairedWith();
        if (partner == null) {
            return;
        }
        old.setPairedWith(null);
        replacement.setPairedWith(partner);
        partner.setPairedWith(replacement);
    }

    /**
     * {@code replacement} in {@code old}'s place among the cards each static
     * ability of the fork applies to, so the next static check takes the
     * static's rows off the card that now carries them.
     */
    private static void swapInStaticEffects(Game fork, Card old, Card replacement) {
        for (StaticEffect effect : fork.getStaticEffects().getEffects()) {
            if (!effect.getAffectedCards().contains(old)) {
                continue;
            }
            CardCollection affected = new CardCollection();
            for (Card card : effect.getAffectedCards()) {
                affected.add(old.equals(card) ? replacement : card);
            }
            effect.setAffectedCards(affected);
        }
    }

    /**
     * Move what is attached to {@code old}, and what {@code old} is attached
     * to, onto {@code replacement}, through the attachment links themselves:
     * no attach or unattach trigger, no new timestamp for the attachments.
     */
    private static void relinkAttachments(Card old, Card replacement) {
        for (Card attached : List.copyOf(old.getAttachedCards())) {
            old.removeAttachedCard(attached);
            attached.setEntityAttachedTo(replacement);
            replacement.addAttachedCard(attached);
        }
        GameEntity host = old.getEntityAttachedTo();
        if (host != null) {
            host.removeAttachedCard(old);
            old.setEntityAttachedTo(null);
            replacement.setEntityAttachedTo(host);
            host.addAttachedCard(replacement);
        }
    }

    /** The fork's combat with {@code replacement} wherever {@code old} stood. */
    private static void swapInCombat(Game fork, Card old, Card replacement) {
        Combat combat = fork.getCombat();
        if (combat == null || !(combat.isAttacking(old) || combat.isBlocking(old)
                || combat.getDefenders().contains(old))) {
            // Left alone otherwise: the copy does not carry which creatures
            // already dealt first-strike damage.
            return;
        }
        fork.getPhaseHandler().setCombat(new Combat(combat, new IEntityMap() {
            @Override
            public Game getGame() {
                return fork;
            }

            @Override
            public GameObject map(GameObject o) {
                return old.equals(o) ? replacement : o;
            }
        }));
    }
}
