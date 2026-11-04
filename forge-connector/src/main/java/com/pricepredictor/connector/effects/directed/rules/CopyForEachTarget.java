package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.ReplaceCardOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.keyword.Keyword;
import forge.game.player.Player;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Optional;

/**
 * 707.10d, an effect copying a spell for each object or player it could
 * target makes one copy per legal target, on scenario {@code 707.10d-1}: A's
 * Radiate copies Shock for each other permanent and player, and B's Abbey
 * Gargoyles, with protection from red, gets no copy.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it holds a card that copies a spell for each target that spell
 * could have, which it has the lands for, beside another instant or sorcery,
 * and a creature on the battlefield has protection. It binds that creature.
 *
 * <p>Variant {@code hill_giant} ({@code replace_card}): Hill Giant, with no
 * protection, is in that creature's place, so a copy targets it too.
 */
public final class CopyForEachTarget {

    public static final String RULE = "707.10d";
    static final String SCENARIO = "707.10d-1";
    static final String HILL_GIANT = "hill_giant";
    static final String GIANT = "Hill Giant";

    private CopyForEachTarget() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, CopyForEachTarget::bind, HILL_GIANT,
                new Application(ForkOperation.REPLACE_CARD,
                        fork -> new ReplaceCardOperation(fork.card("creature"), GIANT)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        List<Card> hand = List.copyOf(active.getCardsIn(ZoneType.Hand));
        Optional<Card> copier = hand.stream()
                .filter(card -> Boards.affordable(active, card) && card.getSpells().stream()
                        .anyMatch(spell -> spell.hasParam("CopyForEachCanTarget")))
                .findFirst();
        if (copier.isEmpty()
                || hand.stream().noneMatch(card -> card != copier.get() && (card.isInstant() || card.isSorcery()))) {
            return Optional.empty();
        }
        return live.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.isCreature() && card.hasKeyword(Keyword.PROTECTION))
                .findFirst()
                .map(creature -> Binding.empty().with("creature", creature));
    }
}
