package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.Role;
import com.pricepredictor.connector.effects.directed.operations.TapOperation;
import forge.card.CardStateName;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Optional;

/**
 * 405.6d, a special action happens at once without using the stack, on
 * scenario {@code 405.6d-1}: A turns its face-down Exalted Angel face up by
 * paying its morph cost {2}{W}{W} with four Plains.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it controls a face-down permanent whose morph cost is mana alone and
 * that it can pay now, colours included, and the mana its sources make now
 * ({@link Mana#manaSources}, each one activation of its most productive
 * mana ability: a bounce land counts two) is exactly that cost's mana value,
 * so the cost takes every mana it can make. It binds the face-down permanent
 * and one of its untapped lands.
 *
 * <p>The active player cannot add a land to its mana this turn
 * ({@link Mana#noLandToPlay}): it holds no land, or its land drop is
 * used, so the turn's land drop cannot undo the mana the variant takes.
 *
 * <p>Variant {@code cannot_pay} ({@code tap}): the land is already tapped, so
 * the morph cost cannot be paid and turning the permanent face up is not
 * legal.
 *
 * <p>The fork keeps the face-down permanent's decision candidates whatever the decision
 * sample draws (FR-010c): the collector reads their verdicts at each of the
 * fork's priority stops itself, whether or not a seat plays them.
 */
public final class MorphFaceUp {

    public static final String RULE = "405.6d";
    static final String SCENARIO = "405.6d-1";
    static final String CANNOT_PAY = "cannot_pay";

    private static final Role<Card> PERMANENT = Role.card("permanent");
    private static final Role<Card> LAND = Role.card("land");

    private MorphFaceUp() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, MorphFaceUp::bind, CANNOT_PAY,
                new Application(ForkOperation.TAP, fork -> new TapOperation(fork.card(LAND), true))
                        .keepingCandidatesOf(PERMANENT));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        if (!Mana.noLandToPlay(active)) {
            return Optional.empty();
        }
        List<Card> sources = Mana.manaSources(active);
        Optional<Card> land = sources.stream().filter(Card::isLand).findFirst();
        if (land.isEmpty()) {
            return Optional.empty();
        }
        int mana = sources.stream().mapToInt(Mana::manaUnits).sum();
        for (Card permanent : active.getCardsIn(ZoneType.Battlefield)) {
            Optional<SpellAbility> faceUp = morphUp(permanent)
                    .filter(ability -> Mana.manaValue(ability.getPayCosts()) == mana)
                    .filter(ability -> Mana.affordable(active, ability));
            if (faceUp.isPresent()) {
                return Optional.of(Binding.empty().with(PERMANENT, permanent).with(LAND, land.get()));
            }
        }
        return Optional.empty();
    }

    /** A face-down permanent's turning face up for its morph cost, when that cost is mana alone and not zero. */
    private static Optional<SpellAbility> morphUp(Card permanent) {
        if (!permanent.isFaceDown()) {
            return Optional.empty();
        }
        return permanent.getState(CardStateName.Original).getNonManaAbilities().stream()
                .filter(MorphFaceUp::isMorphUp)
                .filter(ability -> Mana.manaValue(ability.getPayCosts()) > 0)
                .findFirst();
    }

    private static boolean isMorphUp(SpellAbility ability) {
        return ability.isTurnFaceUp() && ability.hasParam("MorphUp")
                && "Morph".equals(ability.getParam("PrecostDesc"))
                && ability.getPayCosts() != null && ability.getPayCosts().isOnlyManaCost();
    }
}
