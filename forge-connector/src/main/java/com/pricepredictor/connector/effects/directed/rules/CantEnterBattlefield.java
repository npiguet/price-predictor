package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.RemovePermanentOperation;
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
 * in its graveyard while Grafdigger's Cage is on the battlefield, and the
 * Bears stays in the graveyard.
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, a permanent on the battlefield has a replacement effect that stops a
 * card entering the battlefield from a graveyard (Grafdigger's Cage), a
 * creature card is in the active player's graveyard, and the active player
 * holds a spell it has the mana for that returns a card from a graveyard to
 * the battlefield. It binds the stopping permanent.
 *
 * <p>Variant {@code no_cage} ({@code remove_permanent}): the permanent is not
 * on the battlefield, so the creature card enters.
 */
public final class CantEnterBattlefield {

    public static final String RULE = "614.17d";
    static final String SCENARIO = "614.17d-1";
    static final String NO_CAGE = "no_cage";

    private CantEnterBattlefield() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, CantEnterBattlefield::bind, NO_CAGE,
                new Application(ForkOperation.REMOVE_PERMANENT,
                        fork -> new RemovePermanentOperation(fork.card("permanent"))));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        Player active = Boards.active(live);
        if (active.getCardsIn(ZoneType.Graveyard).stream().noneMatch(Card::isCreature)
                || Boards.castableInHand(active, CantEnterBattlefield::reanimates).isEmpty()) {
            return Optional.empty();
        }
        return live.getCardsIn(ZoneType.Battlefield).stream()
                .filter(CantEnterBattlefield::stopsEntering)
                .findFirst()
                .map(card -> Binding.empty().with("permanent", card));
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
