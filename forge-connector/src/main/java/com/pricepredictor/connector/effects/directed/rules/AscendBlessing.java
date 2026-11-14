package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.RemovePermanentOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CardCollectionView;
import forge.game.keyword.Keyword;
import forge.game.player.Player;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Optional;

/**
 * 702.131a, a resolving instant or sorcery with ascend gives its controller
 * the city's blessing when it controls ten permanents, on scenario
 * {@code 702.131a-1}: A controls ten Islands and casts Secrets of the Golden
 * City, gets the city's blessing and draws three cards.
 *
 * <p>The base board, read at a boundary: a main phase of the active
 * player's turn begins with the stack empty; the active player controls
 * exactly ten permanents and has no city's blessing, holds an instant or
 * sorcery with ascend it has the untapped lands for, and has no land left to
 * play ({@link Boards#noLandToPlay}) nor any other permanent card in hand it
 * has the lands for, so no permanent of its own can bring a fork at nine
 * back to ten before the spell. It binds a permanent to take away: a tapped
 * land, else a permanent that makes no mana, else an untapped land beyond
 * the spell's mana value, so the spell stays castable.
 *
 * <p>Variant {@code nine_permanents} ({@code remove_permanent}): the player
 * controls nine permanents, so the spell resolves without the blessing.
 */
public final class AscendBlessing {

    public static final String RULE = "702.131a";
    static final String SCENARIO = "702.131a-1";
    static final String NINE_PERMANENTS = "nine_permanents";
    static final int TEN = 10;

    private AscendBlessing() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, AscendBlessing::bind, NINE_PERMANENTS,
                new Application(ForkOperation.REMOVE_PERMANENT,
                        fork -> new RemovePermanentOperation(fork.card("permanent"))));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live) || !live.getStack().isEmpty()) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        List<Card> permanents = active.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> !card.isPhasedOut()).toList();
        if (permanents.size() != TEN || active.hasBlessing() || !Boards.noLandToPlay(active)) {
            return Optional.empty();
        }
        CardCollectionView hand = active.getCardsIn(ZoneType.Hand);
        Optional<Card> spell = hand.stream()
                .filter(card -> (card.isInstant() || card.isSorcery()) && card.hasKeyword(Keyword.ASCEND))
                .filter(card -> Boards.affordable(active, card))
                .findFirst();
        boolean permanentToCast = hand.stream()
                .anyMatch(card -> card.isPermanent() && !card.isLand() && Boards.affordable(active, card));
        if (spell.isEmpty() || permanentToCast) {
            return Optional.empty();
        }
        return spare(permanents, Boards.untappedLands(active) - spell.get().getCMC())
                .map(permanent -> Binding.empty().with("permanent", permanent).with("spell", spell.get()));
    }

    /** The permanent to take away; {@code spareLands} is how many untapped lands the spell does not need. */
    private static Optional<Card> spare(List<Card> permanents, int spareLands) {
        Optional<Card> tappedLand = permanents.stream().filter(card -> card.isLand() && card.isTapped()).findFirst();
        if (tappedLand.isPresent()) {
            return tappedLand;
        }
        Optional<Card> noMana = permanents.stream()
                .filter(card -> !card.isLand() && card.getManaAbilities().isEmpty()).findFirst();
        if (noMana.isPresent()) {
            return noMana;
        }
        return spareLands > 0 ? permanents.stream().filter(Card::isLand).findFirst() : Optional.empty();
    }
}
