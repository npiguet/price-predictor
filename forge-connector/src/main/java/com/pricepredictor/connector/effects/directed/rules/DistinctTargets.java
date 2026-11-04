package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.AddPermanentOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CardLists;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.spellability.TargetRestrictions;
import forge.game.zone.ZoneType;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;

/**
 * 601.2c, one instance of the word "target" cannot choose the same object
 * twice, on scenario {@code 601.2c-1}: A holds Undo ("return two target
 * creatures") and Grizzly Bears is the only creature, so Undo has no legal
 * choice of targets.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins and it holds a spell it has the mana for whose targets are two or
 * more of any creatures, while exactly one creature on the battlefield is a
 * target the spell could choose. It binds the spell and the opponent.
 *
 * <p>Variant {@code two-creatures} ({@code add_permanent}): the opponent also
 * controls Craw Wurm, a vanilla creature, so the spell has two different
 * creatures to target and can be cast.
 */
public final class DistinctTargets {

    public static final String RULE = "601.2c";
    static final String SCENARIO = "601.2c-1";
    static final String TWO_CREATURES = "two-creatures";
    static final String WURM = "Craw Wurm";

    private DistinctTargets() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, DistinctTargets::bind, TWO_CREATURES,
                new Application(ForkOperation.ADD_PERMANENT,
                        fork -> AddPermanentOperation.of(WURM, fork.player("opponent"), false)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        Optional<Player> opponent = active.getOpponents().stream().findFirst();
        if (opponent.isEmpty()) {
            return Optional.empty();
        }
        return Boards.castableInHand(active, spell -> soleTarget(spell, active)).stream()
                .findFirst()
                .map(card -> Binding.empty().with("spell", card).with("opponent", opponent.get()));
    }

    /**
     * Whether {@code spell} targets two or more of any creatures and exactly
     * one creature on the battlefield is a target it could choose.
     */
    static boolean soleTarget(SpellAbility spell, Player caster) {
        TargetRestrictions targets = spell.getTargetRestrictions();
        if (!spell.usesTargeting() || targets == null
                || !Arrays.asList(targets.getValidTgts()).equals(List.of("Creature"))) {
            return false;
        }
        Card host = spell.getHostCard();
        // Asked as the caster would cast it, on a copy: the live spell stays as it is.
        SpellAbility cast = spell.copy(host, caster, true);
        if (targets.getMinTargets(host, cast) < 2) {
            return false;
        }
        List<Card> candidates = CardLists.getValidCards(caster.getGame().getCardsIn(ZoneType.Battlefield),
                targets.getValidTgts(), caster, host, cast);
        return candidates.stream().filter(card -> card.canBeTargetedBy(cast)).count() == 1;
    }
}
