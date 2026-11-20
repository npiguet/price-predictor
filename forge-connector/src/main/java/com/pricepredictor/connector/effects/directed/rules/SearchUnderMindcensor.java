package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.directed.Application;
import com.pricepredictor.connector.effects.directed.Binding;
import com.pricepredictor.connector.effects.directed.DirectedRule;
import com.pricepredictor.connector.effects.directed.ForcedPlay;
import com.pricepredictor.connector.effects.directed.ForkOperation;
import com.pricepredictor.connector.effects.directed.operations.AddPermanentOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;
import forge.game.trigger.TriggerType;
import forge.game.zone.ZoneType;

import java.util.Optional;

/**
 * 701.23f, searching the top cards of a library instead still counts as
 * searching it, inverted (FR-014a) on scenario {@code 701.23f-2}: A controls
 * Archivist of Oghma ("whenever an opponent searches their library"), and
 * B's search is limited to the top four cards by A's Aven Mindcensor:
 * Archivist's trigger fires on it all the same.
 *
 * <p>The base board, read at a boundary: a main phase begins, a player
 * controls a permanent whose trigger watches an opponent search its library
 * (Archivist of Oghma, Ob Nixilis, Unshackled), and that opponent controls an
 * untapped permanent with an activated ability searching its library it has
 * the mana for (Evolving Wilds), which it may activate at any priority. It
 * binds the watching player, the searcher and the permanent.
 *
 * <p>Variant {@code mindcensor_added} ({@code add_permanent}): the watching
 * player also controls Aven Mindcensor, so the searcher searches the top four
 * cards instead. The fork forces the searcher's activation (FR-010b): the
 * limited search and the trigger it fires are the rule's {@code fires} side,
 * which the counts lack (2 records).
 */
public final class SearchUnderMindcensor {

    public static final String RULE = "701.23f";
    static final String SCENARIO = "701.23f-2";
    static final String MINDCENSOR_ADDED = "mindcensor_added";
    static final String MINDCENSOR = "Aven Mindcensor";

    /** An activated ability that searches its controller's library. */
    static final ForcedPlay.Line LIBRARY_SEARCH = new ForcedPlay.Line("library search",
            ability -> ability.isActivatedAbility() && !ability.isManaAbility() && Scan.searches(ability));

    private SearchUnderMindcensor() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, SearchUnderMindcensor::bind, MINDCENSOR_ADDED,
                new Application(ForkOperation.ADD_PERMANENT,
                        fork -> AddPermanentOperation.of(MINDCENSOR, fork.player("watcher"), false))
                        .forcing(ForcedPlay.activate("searcher", "permanent", LIBRARY_SEARCH)));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        for (Player watcher : live.getPlayers()) {
            if (!watchesAnOpponentSearch(watcher)) {
                continue;
            }
            for (Player searcher : watcher.getOpponents()) {
                Optional<Card> permanent = searcher.getCardsIn(ZoneType.Battlefield).stream()
                        .filter(card -> !card.isTapped() && searchesAtAnyPriority(searcher, card))
                        .findFirst();
                if (permanent.isPresent()) {
                    return Optional.of(Binding.empty().with("watcher", watcher).with("searcher", searcher)
                            .with("permanent", permanent.get()));
                }
            }
        }
        return Optional.empty();
    }

    /** Whether a permanent {@code player} controls has a trigger on an opponent searching its library. */
    static boolean watchesAnOpponentSearch(Player player) {
        for (Card card : player.getCardsIn(ZoneType.Battlefield)) {
            for (Trigger trigger : card.getTriggers()) {
                if (trigger.getMode() == TriggerType.SearchedLibrary
                        && "Player.Opponent".equals(trigger.getParam("ValidPlayer"))) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Whether {@code card} has an activated ability searching its
     * controller's library that it may activate at any priority (no sorcery
     * timing) and that {@code player} can pay ({@link Mana#affordable}). The
     * card taps for its own cost when it is an untapped land, so its mana is
     * not counted toward the mana cost: that cost is covered by the other
     * sources, counted per source ({@link Mana#manaUnits}).
     */
    static boolean searchesAtAnyPriority(Player player, Card card) {
        int others = Mana.manaSources(player).stream()
                .filter(source -> source != card || !card.isLand())
                .mapToInt(Mana::manaUnits).sum();
        for (SpellAbility ability : card.getSpellAbilities()) {
            if (LIBRARY_SEARCH.matches(ability) && !ability.getRestrictions().isSorcerySpeed()
                    && Mana.manaValue(ability.getPayCosts()) <= others
                    && Mana.affordable(player, ability)) {
                return true;
            }
        }
        return false;
    }
}
