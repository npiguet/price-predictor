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

import java.util.List;
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
 * library (Archivist of Oghma), no other permanent limits searches, and an
 * opponent of that player holds a card whose spell searches its library and
 * that it can cast now ({@link #searchersInHand}). It binds the limiting
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
        List<Card> limiting = live.getCardsIn(ZoneType.Battlefield).stream()
                .filter(LimitedSearch::limitsOpponentsSearches)
                .toList();
        if (!Boards.atMainPhase(live) || limiting.size() != 1) {
            return Optional.empty();
        }
        Player player = limiting.get(0).getController();
        boolean watches = player.getCardsIn(ZoneType.Battlefield).stream()
                .flatMap(card -> card.getTriggers().stream())
                .anyMatch(trigger -> trigger.getMode() == TriggerType.SearchedLibrary);
        boolean searching = player.getOpponents().stream()
                .anyMatch(opponent -> !searchersInHand(opponent).isEmpty());
        return watches && searching
                ? Optional.of(Binding.empty().with("permanent", limiting.get(0)))
                : Optional.empty();
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

    /**
     * The cards in {@code player}'s hand it has the mana for whose spell
     * searches its library and that it can cast at this boundary
     * ({@link Boards#castableAtItsSpeed}: a sorcery only as the active
     * player).
     */
    static List<Card> searchersInHand(Player player) {
        return Boards.castableInHand(player, LimitedSearch::searches).stream()
                .filter(card -> Boards.castableAtItsSpeed(player, card))
                .toList();
    }

    /** Whether the spell searches a library: it moves a card of a stated type out of one, its contents hidden. */
    static boolean searches(SpellAbility spell) {
        return spell.getApi() == ApiType.ChangeZone && Boards.lists(spell.getParam("Origin"), "Library")
                && spell.hasParam("ChangeType");
    }
}
