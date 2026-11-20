package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.Role;
import com.pricepredictor.connector.effects.directed.operations.ReplaceCardOperation;
import forge.game.Game;
import forge.game.card.Card;

import java.util.Optional;

/**
 * The three meld rules, one registration: they share this board, its
 * binding and its edit, so the board is forked once and its records name all
 * three.
 * <ul>
 * <li>701.42a, melding exiles a meld pair and puts them onto the battlefield
 * as one permanent, on scenario {@code 701.42a-1}: Graf Rats beside Hill
 * Giant has nothing to meld with at the beginning of combat; beside Midnight
 * Scavengers its trigger would meld the two into Chittering Host.
 * <li>712.4a, a meld card's front face and its counterpart meld into one
 * permanent, and 712.14c, a meld card being melded with its counterpart
 * enters the battlefield as a single permanent with its back face up, both
 * on scenario {@code 712.4a-2}, 712.4a-1 inverted: beside Midnight
 * Scavengers both would be exiled and return as Chittering Host.
 * </ul>
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
 * pair is exiled and melds, entering combined: the meld resolution
 * 701.42a-1's declaration reads. No seat decides it.
 */
public final class Meld {

    public static final String RULE = "701.42a";
    public static final String PAIR_RULE = "712.4a";
    public static final String ENTRY_RULE = "712.14c";
    static final String SCENARIO = "701.42a-1";
    static final String PAIR_SCENARIO = "712.4a-2";
    static final String MELD_PAIR = "meld_pair";

    private static final Role<Card> HOST = Role.card("host");
    private static final Role<Card> STAND_IN = Role.card("stand-in");

    private Meld() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, Meld::bind, MELD_PAIR, pairing())
                .alsoFor(PAIR_RULE, PAIR_SCENARIO)
                .alsoFor(ENTRY_RULE, PAIR_SCENARIO);
    }

    /** The partner the host names, in the stand-in's place. */
    private static Application pairing() {
        return new Application(ForkOperation.REPLACE_CARD, fork -> new ReplaceCardOperation(fork.card(STAND_IN),
                Melds.partnerName(fork.card(HOST)).orElseThrow(
                        () -> new IllegalStateException("the fork's host names no partner"))));
    }

    static Optional<Binding> bind(Game live) {
        return Melds.unpairedOfActivePlayer(live).stream()
                .findFirst()
                .map(unpaired -> Binding.empty().with(HOST, unpaired.host()).with(STAND_IN, unpaired.standIn()));
    }
}
