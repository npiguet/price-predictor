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
import forge.game.spellability.SpellAbility;
import forge.game.staticability.StaticAbility;
import forge.game.trigger.TriggerType;
import forge.game.zone.ZoneType;

import java.util.Optional;

/**
 * 701.23f, searching only part of a library still counts as searching it, on
 * scenario {@code 701.23f-1}: under A's Aven Mindcensor, B's Vampiric Tutor
 * searches only the top four cards, and A's Archivist of Oghma still triggers
 * on the search.
 *
 * <p>The base board, read at a boundary: a main phase begins, a player
 * controls a permanent whose static ability limits its opponents' searches
 * (Aven Mindcensor) and a permanent that triggers when a player searches a
 * library (Archivist of Oghma), and an opponent of that player holds a card it
 * has the mana for whose spell searches its library. It binds the limiting
 * permanent.
 *
 * <p>Variant {@code no-mindcensor} ({@code remove_permanent}): the limiting
 * permanent is not on the battlefield, so the search covers the whole library;
 * the trigger fires as before.
 */
public final class LimitedSearch {

    public static final String RULE = "701.23f";
    static final String SCENARIO = "701.23f-1";
    static final String NO_MINDCENSOR = "no-mindcensor";
    static final String LIMIT = "LimitSearchLibrary";

    private LimitedSearch() {
    }

    public static DirectedRule rule() {
        return DirectedRule.of(RULE, SCENARIO, LimitedSearch::bind, NO_MINDCENSOR,
                new Application(ForkOperation.REMOVE_PERMANENT,
                        fork -> new RemovePermanentOperation(fork.card("permanent"))));
    }

    static Optional<Binding> bind(Game live) {
        if (!Boards.atMainPhase(live)) {
            return Optional.empty();
        }
        for (Player player : live.getPlayers()) {
            boolean watches = player.getCardsIn(ZoneType.Battlefield).stream()
                    .flatMap(card -> card.getTriggers().stream())
                    .anyMatch(trigger -> trigger.getMode() == TriggerType.SearchedLibrary);
            boolean searching = player.getOpponents().stream()
                    .anyMatch(opponent -> !Boards.castableInHand(opponent, LimitedSearch::searches).isEmpty());
            if (!watches || !searching) {
                continue;
            }
            Optional<Card> limiting = player.getCardsIn(ZoneType.Battlefield).stream()
                    .filter(LimitedSearch::limitsOpponentsSearches)
                    .findFirst();
            if (limiting.isPresent()) {
                return Optional.of(Binding.empty().with("permanent", limiting.get()));
            }
        }
        return Optional.empty();
    }

    /** Whether one of the permanent's static abilities limits its controller's opponents' library searches. */
    static boolean limitsOpponentsSearches(Card permanent) {
        for (StaticAbility ability : permanent.getStaticAbilities()) {
            String affected = ability.getParam("Affected");
            String keywords = ability.getParam("AddKeyword");
            if (affected != null && affected.contains("Opponent") && keywords != null && keywords.contains(LIMIT)) {
                return true;
            }
        }
        return false;
    }

    /** Whether the spell searches a library: it moves a card of a stated type out of one, its contents hidden. */
    static boolean searches(SpellAbility spell) {
        return spell.getApi() == ApiType.ChangeZone && Boards.lists(spell.getParam("Origin"), "Library")
                && spell.hasParam("ChangeType");
    }
}
