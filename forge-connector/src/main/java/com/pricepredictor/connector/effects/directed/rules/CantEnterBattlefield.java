package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.AddPermanentOperation;
import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.replacement.ReplacementEffect;
import forge.game.replacement.ReplacementType;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.Optional;

/**
 * 614.17d, an effect saying a permanent can't enter the battlefield stops it
 * entering, on scenario {@code 614.17d-1}: A casts Zombify on Grizzly Bears
 * in its graveyard with no Grafdigger's Cage on the battlefield, and the Bears
 * enters; with the Cage it would stay in the graveyard.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, a creature card is in the active player's graveyard, the active
 * player holds a spell it has the mana for that returns a card from a
 * graveyard to the battlefield, and no permanent on the battlefield has a
 * replacement effect that stops a card entering the battlefield from a
 * graveyard. It binds the active player. The live game, where the creature
 * would enter, is the common side; the fork adds the rare fact.
 *
 * <p>Variant {@code cage} ({@code add_permanent}): Grafdigger's Cage is on
 * the battlefield under the active player's control, so a creature card the
 * spell returns from the graveyard stays there: the rewrite record of the
 * prevented move is what 614.17d-1's declaration reads. Whether the spell is
 * cast is its seat's own decision.
 */
public final class CantEnterBattlefield {

    public static final String RULE = "614.17d";
    static final String SCENARIO = "614.17d-1";
    static final String CAGE = "cage";
    static final String GRAFDIGGERS_CAGE = "Grafdigger's Cage";

    private CantEnterBattlefield() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, CantEnterBattlefield::bind, CAGE,
                new Application(ForkOperation.ADD_PERMANENT,
                        fork -> AddPermanentOperation.of(GRAFDIGGERS_CAGE, fork.player("player"), false)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        if (active.getCardsIn(ZoneType.Graveyard).stream().noneMatch(Card::isCreature)
                || Boards.castableInHand(active, CantEnterBattlefield::reanimates).isEmpty()
                || live.getCardsIn(ZoneType.Battlefield).stream().anyMatch(CantEnterBattlefield::stopsEntering)) {
            return Optional.empty();
        }
        return Optional.of(Binding.empty().with("player", active));
    }

    /** Whether one of the permanent's replacement effects stops a card entering the battlefield from a graveyard. */
    static boolean stopsEntering(Card permanent) {
        for (ReplacementEffect replacement : permanent.getReplacementEffects()) {
            if (replacement.getMode() == ReplacementType.Moved
                    && Boards.lists(replacement.getParam("Destination"), "Battlefield")
                    && Boards.lists(replacement.getParam("Origin"), "Graveyard")
                    && "True".equals(replacement.getParam("Prevent"))) {
                return true;
            }
        }
        return false;
    }

    /** Whether the spell puts a card from a graveyard onto the battlefield. */
    static boolean reanimates(SpellAbility spell) {
        return spell.getApi() == ApiType.ChangeZone
                && Boards.lists(spell.getParam("Origin"), "Graveyard")
                && Boards.lists(spell.getParam("Destination"), "Battlefield");
    }
}
