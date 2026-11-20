package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForcedPlay;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.Role;
import com.pricepredictor.connector.effects.directed.operations.ReplaceCardOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.keyword.Keyword;
import forge.game.player.Player;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Optional;

/**
 * 702.103e, a bestowed Aura spell whose target became illegal resolves as a
 * creature spell, on scenario {@code 702.103e-2}: Hopeful Eidolon is cast
 * bestowed on Skulking Ghost, which is sacrificed as it becomes the target,
 * and the Eidolon enters as a creature.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it holds a card with bestow whose bestow cost it has the lands for,
 * and exactly one creature is on the battlefield, under either player's
 * control, so the bestowed spell can target that creature alone; a Skulking
 * Ghost put in its place survives the fork's first state-based check
 * ({@link Boards#replacementSurvives}). It binds the active player, the card
 * and that creature. A directed fork is taken at a phase boundary, where the
 * stack is empty, so the base is the board before the bestow cast; the
 * scenario's 702.103e-1, a Shock in response, is no fork's play.
 *
 * <p>Variant {@code skulking_ghost} ({@code replace_card}): Skulking Ghost
 * (when it becomes the target of a spell or ability, sacrifice it) is in that
 * creature's place. The fork forces the card's bestowed cast (FR-010b), its
 * target drawn, which is the Ghost, the one creature: the Ghost's trigger
 * resolves first and sacrifices it, so the Aura spell's target is illegal and
 * it resolves as a creature spell, the side of the rule the counts lack. The
 * card in the creature's place has a new id, so no role names it.
 */
public final class BestowIllegalTarget {

    public static final String RULE = "702.103e";
    static final String SCENARIO = "702.103e-2";
    static final String SKULKING_GHOST = "skulking_ghost";
    static final String GHOST = "Skulking Ghost";

    private static final Role<Player> PLAYER = Role.player("player");
    private static final Role<Card> CARD = Role.card("card");
    private static final Role<Card> CREATURE = Role.card("creature");

    private BestowIllegalTarget() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, BestowIllegalTarget::bind, SKULKING_GHOST,
                new Application(ForkOperation.REPLACE_CARD,
                        fork -> new ReplaceCardOperation(fork.card(CREATURE), GHOST))
                        .forcing(ForcedPlay.cast(PLAYER, CARD)
                                .playing(new ForcedPlay.Line("bestow", ability -> ability.isSpell() && ability.isBestow()))));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        List<Card> creatures = live.getCardsIn(ZoneType.Battlefield).stream().filter(Card::isCreature).toList();
        if (creatures.size() != 1 || !Boards.replacementSurvives(creatures.get(0), GHOST)) {
            return Optional.empty();
        }
        return active.getCardsIn(ZoneType.Hand).stream()
                .filter(card -> Mana.canPayKeyword(active, card, Keyword.BESTOW))
                .findFirst()
                .map(card -> Binding.empty().with(PLAYER, active).with(CARD, card)
                        .with(CREATURE, creatures.get(0)));
    }
}
