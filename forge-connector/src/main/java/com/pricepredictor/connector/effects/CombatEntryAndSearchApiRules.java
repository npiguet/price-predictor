package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.effects.ApiEvents.Rule;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static com.pricepredictor.connector.effects.ApiEvents.put;

/**
 * The {@link ApiEvents} rules for entering combat, becoming blocked and
 * searching a library.
 */
final class CombatEntryAndSearchApiRules {

    private CombatEntryAndSearchApiRules() {
    }

    /**
     * Who entered combat as a clause resolved, an attacker that became
     * blocked, and a library searched.
     *
     * <p>A creature put onto the battlefield attacking or blocking (a token
     * made attacking, a ninja, a card put in "tapped and attacking") joins
     * combat without being declared, and the bus says only that the combat
     * changed. Read as the combatants after the clause less those before it,
     * so the rule needs no list of the parameters ({@code TokenAttacking$},
     * {@code Attacking$}, {@code Blocking$}, ...) a script asks for it with.
     *
     * <p>{@code ChangeZone} also searches libraries. Forge counts a player's
     * searches of their own library ({@code Player.incLibrarySearched}, in
     * {@code ChangeZoneEffect} alone), so a count that moved is a search,
     * which no bus event announces; written as a look at that library named
     * {@value BusEvents#SEARCHED_LIBRARY}, Forge's trigger mode for it, so a
     * search reads apart from a scry and a surveil. A search of another
     * player's library is not counted by Forge and reaches no event. One
     * event per clause: a clause that both searched and put its card in
     * attacking reports the attack.
     */
    static void combatEntryAndSearchRules(Map<String, Rule<?>> rules) {
        put(rules, Rule.one(EventType.STATE_FLAG_CHANGE, CombatEntryAndSearchApiRules::combatants,
                        (sa, host, before) -> enteredCombat(host, before)),
                "Token", "CopyPermanent", "ChangeZoneAll");
        put(rules, Rule.one(EventType.STATE_FLAG_CHANGE, CombatEntryAndSearchApiRules::beforeChangeZone,
                        (sa, host, before) -> changedZone(host, before)),
                "ChangeZone");
        put(rules, Rule.one(EventType.BECAME_BLOCKED, CombatEntryAndSearchApiRules::unblockedAttackers,
                        (sa, host, before) -> becameBlocked(host, before)),
                "BecomesBlocked", "Block");
    }

    /** The ids of every attacker and blocker, empty outside combat. */
    private static Set<Integer> combatants(SpellAbility sa, Card host) {
        Set<Integer> ids = new HashSet<>();
        Combat combat = host.getGame().getCombat();
        if (combat != null) {
            combat.getAttackers().forEach(card -> ids.add(card.getId()));
            combat.getAllBlockers().forEach(card -> ids.add(card.getId()));
        }
        return ids;
    }

    /**
     * {@code state_flag_change} {@code entered_attacking} on the attackers
     * not in combat before the clause, else {@code entered_blocking} on such
     * blockers; null when nobody joined.
     */
    private static EffectEvent enteredCombat(Card host, Set<Integer> before) {
        Combat combat = host.getGame().getCombat();
        if (before == null || combat == null) {
            return null;
        }
        EffectEvent attacking = joined(combat.getAttackers(), before, StateFlag.ENTERED_ATTACKING);
        return attacking != null ? attacking
                : joined(combat.getAllBlockers(), before, StateFlag.ENTERED_BLOCKING);
    }

    private static EffectEvent joined(Iterable<Card> now, Set<Integer> before, StateFlag flag) {
        EffectEvent event = new EffectEvent(EventType.STATE_FLAG_CHANGE)
                .flag(flag, true);
        for (Card card : now) {
            if (!before.contains(card.getId())) {
                event.subject(Refs.of(card));
            }
        }
        return event.subjects().isEmpty() ? null : event;
    }

    /** What a zone change is read against: the combatants and every player's searches. */
    private record BeforeChangeZone(Set<Integer> combatants, Map<Player, Integer> searches) {
    }

    private static BeforeChangeZone beforeChangeZone(SpellAbility sa, Card host) {
        Map<Player, Integer> searches = new LinkedHashMap<>();
        for (Player player : host.getGame().getPlayers()) {
            searches.put(player, player.getLibrarySearched());
        }
        return new BeforeChangeZone(combatants(sa, host), searches);
    }

    private static EffectEvent changedZone(Card host, BeforeChangeZone before) {
        if (before == null) {
            return null;
        }
        EffectEvent entered = enteredCombat(host, before.combatants());
        if (entered != null) {
            return entered;
        }
        EffectEvent searched = new EffectEvent(EventType.CARD_LOOKED_AT)
                .param(EventParam.FROM_ZONE, CardSpelling.zoneName(ZoneType.Library))
                .param(EventParam.MODE, BusEvents.SEARCHED_LIBRARY);
        before.searches().forEach((player, count) -> {
            if (player.getLibrarySearched() > count) {
                searched.subject(Refs.of(player));
            }
        });
        return searched.subjects().isEmpty() ? null : searched;
    }

    /** The attackers not blocked before the clause. */
    private static List<Card> unblockedAttackers(SpellAbility sa, Card host) {
        Combat combat = host.getGame().getCombat();
        List<Card> unblocked = new ArrayList<>();
        if (combat != null) {
            for (Card attacker : combat.getAttackers()) {
                if (!combat.isBlocked(attacker)) {
                    unblocked.add(attacker);
                }
            }
        }
        return unblocked;
    }

    /**
     * {@code became_blocked} on every attacker the clause blocked, with the
     * blockers now blocking them ({@code BecomesBlocked} leaves none, a
     * {@code Block} clause names its blocker); null when none became blocked.
     */
    private static EffectEvent becameBlocked(Card host, List<Card> before) {
        Combat combat = host.getGame().getCombat();
        if (before == null || combat == null) {
            return null;
        }
        EffectEvent event = new EffectEvent(EventType.BECAME_BLOCKED);
        List<String> blockers = new ArrayList<>();
        for (Card attacker : before) {
            if (combat.isAttacking(attacker) && combat.isBlocked(attacker)) {
                event.subject(Refs.of(attacker));
                addBlockers(combat.getBlockers(attacker), blockers);
            }
        }
        return event.subjects().isEmpty() ? null : event.param(EventParam.BLOCKERS, blockers);
    }

    /** Each of {@code blocking}'s refs not already in {@code blockers}, onto it, in order. */
    private static void addBlockers(Iterable<Card> blocking, List<String> blockers) {
        for (Card blocker : blocking) {
            String ref = Refs.of(blocker);
            if (!blockers.contains(ref)) {
                blockers.add(ref);
            }
        }
    }
}
