package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.ReplaceCardOperation;
import forge.game.Game;

import java.util.Optional;

/**
 * 701.42a, melding exiles a meld pair and puts them onto the battlefield as
 * one permanent, on scenario {@code 701.42a-1}: Graf Rats beside Hill Giant
 * has nothing to meld with at the beginning of combat; beside Midnight
 * Scavengers its trigger would meld the two into Chittering Host.
 *
 * <p>The base board, read at a boundary: the active player's first main phase
 * begins, and it owns and controls a permanent whose beginning-of-combat
 * trigger melds it with a partner it does not own and control, beside a
 * creature (not a token) it owns and controls that the partner can take the
 * place of ({@link Melds#unpairedOfActivePlayer}). It binds the host and that
 * creature. The live game, unpaired, is the common side; the fork adds the
 * rare fact.
 *
 * <p>Variant {@code meld_pair} ({@code replace_card}): the partner is in the
 * creature's place, so at the beginning of combat the trigger fires and the
 * pair melds, the meld resolution 701.42a-1's declaration reads. No seat
 * decides it.
 */
public final class Meld {

    public static final String RULE = "701.42a";
    static final String SCENARIO = "701.42a-1";
    static final String MELD_PAIR = "meld_pair";

    private Meld() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, Meld::bind, MELD_PAIR, pairing());
    }

    /** The edit the three meld rules share: the partner the host names in the stand-in's place. */
    static Application pairing() {
        return new Application(ForkOperation.REPLACE_CARD, fork -> new ReplaceCardOperation(fork.card("stand-in"),
                Melds.partnerName(fork.card("host")).orElseThrow(
                        () -> new IllegalStateException("the fork's host names no partner"))));
    }

    static Optional<Binding> bind(Game live) {
        return Melds.unpairedOfActivePlayer(live).stream()
                .findFirst()
                .map(unpaired -> Binding.empty().with("host", unpaired.host()).with("stand-in", unpaired.standIn()));
    }
}
