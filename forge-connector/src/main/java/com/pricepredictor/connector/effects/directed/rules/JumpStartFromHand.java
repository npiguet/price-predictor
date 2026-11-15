package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForcedPlay;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.MoveCardOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.keyword.Keyword;
import forge.game.player.Player;
import forge.game.zone.ZoneType;

import java.util.Optional;

/**
 * 702.133a, a spell cast with jump-start is exiled instead of going anywhere
 * else as it leaves the stack, on scenario {@code 702.133a-2}: A casts
 * Radical Idea from its graveyard, discarding Forest, and it is exiled as it
 * resolves.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, and its graveyard holds a card with jump-start whose spell needs no
 * target and that it has the lands for, which the jump-start cast casts from
 * there. It binds the active player and the card.
 *
 * <p>Variant {@code cast_from_hand} ({@code move_card}): the card is in the
 * player's hand instead. The fork forces its cast from the hand (FR-010b),
 * which jump-start does not touch: the spell goes to its owner's graveyard as
 * it resolves, the rule's near miss, which the counts lack.
 */
public final class JumpStartFromHand {

    public static final String RULE = "702.133a";
    static final String SCENARIO = "702.133a-2";
    static final String CAST_FROM_HAND = "cast_from_hand";

    private JumpStartFromHand() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, JumpStartFromHand::bind, CAST_FROM_HAND,
                new Application(ForkOperation.MOVE_CARD,
                        fork -> MoveCardOperation.move(fork.card("card"), ZoneType.Hand, null))
                        .forcing(ForcedPlay.cast("caster", "card")));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        return active.getCardsIn(ZoneType.Graveyard).stream()
                .filter(card -> card.hasKeyword(Keyword.JUMP_START) && Boards.affordable(active, card))
                .filter(card -> card.getFirstSpellAbility() != null && !card.getFirstSpellAbility().usesTargeting())
                .findFirst()
                .map(card -> Binding.empty().with("caster", active).with("card", card));
    }
}
