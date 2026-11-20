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

import java.util.List;
import java.util.Optional;
import java.util.function.Function;
import java.util.function.Predicate;

/**
 * The two keywords that search the library for a card of their card's mana
 * value, one class parameterised for each: two rules, two registrations, two
 * forks.
 * <ul>
 * <li>702.53a, transmute discards the card and puts a card of the same mana
 * value from the library into the hand, on scenario {@code 702.53a-1}: A
 * transmutes Dimir Infiltrator (mana value 2) and finds Grizzly Bears.
 * <li>702.71a, transfigure sacrifices the creature and puts a creature card of
 * the same mana value from the library onto the battlefield, on scenario
 * {@code 702.71a-1}: A transfigures Fleshwrither (mana value 4) and Hill Giant
 * enters.
 * </ul>
 *
 * <p>The base board, read at a boundary: a main phase of the active player
 * begins, it holds a card with the keyword where the keyword is activated
 * from (transmute's in its hand, transfigure's a creature it controls) whose
 * keyword cost it has the lands for, and its library holds exactly one card
 * the search can find of that card's mana value (any card for transmute, a
 * creature card for transfigure). It binds the active player, the searching
 * card as {@code source} and that library card.
 *
 * <p>Transmute's variant {@code no_match} and transfigure's
 * {@code no_matching_value} ({@code move_card}): a card of another mana value
 * is in the library card's place, so the search finds nothing (a Centaur
 * Courser, mana value 3, for transmute, a Grizzly Bears when the transmuting
 * card's is 3; a Craw Wurm, mana value 6, for transfigure, a Hill Giant when
 * the creature's is 6). The fork forces the activation (FR-010b), its cost
 * discarding or sacrificing the card, which Forge's AI does not make with
 * nothing to find.
 */
public final class ManaValueSearch {

    public static final String TRANSMUTE_RULE = "702.53a";
    public static final String TRANSFIGURE_RULE = "702.71a";
    static final String COURSER = Cards.COURSER;
    static final String BEARS = Cards.BEARS;
    static final String WURM = Cards.WURM;
    static final String GIANT = Cards.GIANT;

    /**
     * The card the fork puts in the found card's place: {@code name}, or
     * {@code otherwise} when the searching card's mana value is {@code name}'s
     * {@code value}, so the search still finds nothing.
     */
    private record Replacement(String name, int value, String otherwise) {
        String instead(Card source) {
            return source.getCMC() == value ? otherwise : name;
        }
    }

    /**
     * How one keyword searches, named by strings and functions read only
     * when a board is: the registry is built where Forge is not initialised.
     *
     * @param keyword     the keyword's {@link Keyword} name, as the forced line names it
     * @param sources     where the active player's cards with the keyword are
     * @param finds       the library cards the search can find, whatever their mana value
     * @param replacement the card the fork puts in the found one's place
     */
    private record Search(String keyword, Function<Player, Iterable<Card>> sources, Predicate<Card> finds,
                          Replacement replacement) {

        DirectedRule rule(String rule, String scenario, String variant) {
            return DirectedRule.of(rule, scenario, this::bind, variant,
                    new Application(ForkOperation.MOVE_CARD, fork -> MoveCardOperation.replace(fork.card("card"),
                            replacement.instead(fork.card("source"))))
                            .forcing(ForcedPlay.activate("player", "source", ForcedPlay.Line.keyword(keyword))));
        }

        Optional<Binding> bind(Game live) {
            if (!Boards.atMainPhase(live)) {
                return Optional.empty();
            }
            Player active = Boards.active(live);
            for (Card source : sources.apply(active)) {
                if (!Mana.canPayKeyword(active, source, Keyword.valueOf(keyword))) {
                    continue;
                }
                List<Card> matching = active.getCardsIn(ZoneType.Library).stream()
                        .filter(card -> finds.test(card) && card.getCMC() == source.getCMC())
                        .toList();
                if (matching.size() == 1) {
                    return Optional.of(Binding.empty().with("player", active)
                            .with("source", source).with("card", matching.get(0)));
                }
            }
            return Optional.empty();
        }
    }

    private static final Search TRANSMUTE = new Search("TRANSMUTE",
            player -> player.getCardsIn(ZoneType.Hand), card -> true, new Replacement(COURSER, 3, BEARS));
    private static final Search TRANSFIGURE = new Search("TRANSFIGURE",
            Player::getCreaturesInPlay, Card::isCreature, new Replacement(WURM, 6, GIANT));

    private ManaValueSearch() {
    }

    /** 702.53a's registration. */
    public static DirectedRule transmute() {
        return TRANSMUTE.rule(TRANSMUTE_RULE, "702.53a-1", "no_match");
    }

    /** 702.71a's registration. */
    public static DirectedRule transfigure() {
        return TRANSFIGURE.rule(TRANSFIGURE_RULE, "702.71a-1", "no_matching_value");
    }
}
