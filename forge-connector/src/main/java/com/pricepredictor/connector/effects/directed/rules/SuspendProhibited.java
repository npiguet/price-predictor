package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.RemovePermanentOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.keyword.Keyword;
import forge.game.keyword.KeywordInterface;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.staticability.StaticAbility;
import forge.game.staticability.StaticAbilityCantBeCast;
import forge.game.staticability.StaticAbilityMode;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Optional;

/**
 * 702.62c, an effect that stops a player casting a card also stops it
 * suspending that card, on scenario {@code 702.62c-1}: under A's Steel Golem
 * ("you can't cast creature spells") A cannot suspend Durkwood Baloth.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it holds a card with suspend whose suspend cost it has the lands for
 * and whose spell its timing allows now, and exactly one permanent on the
 * battlefield has a can't-be-cast static ability that stops it casting that
 * card. It binds that permanent.
 *
 * <p>Variant {@code no_prohibition} ({@code remove_permanent}): the permanent
 * is not on the battlefield, so suspending the card is legal.
 */
public final class SuspendProhibited {

    public static final String RULE = "702.62c";
    static final String SCENARIO = "702.62c-1";
    static final String NO_PROHIBITION = "no_prohibition";

    private SuspendProhibited() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, SuspendProhibited::bind, NO_PROHIBITION,
                new Application(ForkOperation.REMOVE_PERMANENT,
                        fork -> new RemovePermanentOperation(fork.card("permanent"))));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        for (Card card : active.getCardsIn(ZoneType.Hand)) {
            SpellAbility first = card.getFirstSpellAbility();
            if (!Boards.canPayKeyword(active, card, Keyword.SUSPEND)
                    || first == null || !first.canCastTiming(card, active)) {
                continue;
            }
            for (SpellAbility suspend : suspendAbilities(card)) {
                List<Card> prohibiting = live.getCardsIn(ZoneType.Battlefield).stream()
                        .filter(permanent -> prohibits(permanent, suspend, card, active))
                        .toList();
                if (prohibiting.size() == 1) {
                    return Optional.of(Binding.empty().with("permanent", prohibiting.get(0)));
                }
            }
        }
        return Optional.empty();
    }

    /** The special actions the card's suspend instances give it. */
    private static List<SpellAbility> suspendAbilities(Card card) {
        return card.getKeywords(Keyword.SUSPEND).stream()
                .map(KeywordInterface::getAbilities)
                .flatMap(abilities -> abilities.stream())
                .toList();
    }

    /** Whether one of the permanent's static abilities stops {@code player} casting {@code card} now. */
    static boolean prohibits(Card permanent, SpellAbility suspend, Card card, Player player) {
        for (StaticAbility ability : permanent.getStaticAbilities()) {
            if (ability.checkConditions(StaticAbilityMode.CantBeCast)
                    && StaticAbilityCantBeCast.applyCantBeCastAbility(ability, suspend, card, player)) {
                return true;
            }
        }
        return false;
    }
}
