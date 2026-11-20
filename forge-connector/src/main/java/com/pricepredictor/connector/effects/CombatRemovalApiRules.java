package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.effects.ApiEvents.Rule;
import forge.game.GameEntity;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static com.pricepredictor.connector.effects.ApiEvents.put;

/**
 * The {@link ApiEvents} rules for combat removals: a creature removed from
 * combat by the clause, and a combat phase ended.
 */
final class CombatRemovalApiRules {

    private CombatRemovalApiRules() {
    }

    /**
     * The APIs whose clause rule tells every combatant that leaves combat
     * while the clause runs ({@link #everyCombatant}), whatever took it out.
     * Read when {@link ApiEvents#RULES} is built, and by
     * {@link #tellsEveryCombatRemoval}.
     */
    private static final Set<String> EVERY_COMBATANT_APIS =
            Set.of("Regeneration", "GainControl", "Phases");

    /**
     * Whether {@code clause}, the clause resolving now, tells every
     * combatant that leaves combat while it runs, so a removal seen
     * elsewhere meanwhile ({@link ControlCorrection}) is already told.
     */
    static boolean tellsEveryCombatRemoval(Object clause) {
        return clause instanceof SpellAbility sa && sa.getApi() != null
                && EVERY_COMBATANT_APIS.contains(sa.getApi().name());
    }

    static void combatRemovalRules(Map<String, Rule<?>> rules) {
        put(rules, Rule.one(EventType.REMOVED_FROM_COMBAT,
                        CombatRemovalApiRules::combatantsBefore,
                        (sa, host, memo) -> noLongerInCombat(host, memo)),
                "RemoveFromCombat");
        put(rules, Rule.one(EventType.REMOVED_FROM_COMBAT,
                        CombatRemovalApiRules::attackerDefendersBefore,
                        (sa, host, memo) -> reselectedAttackers(host, memo)),
                "ChangeCombatants");
        put(rules, Rule.one(EventType.REMOVED_FROM_COMBAT,
                        CombatRemovalApiRules::everyCombatant,
                        (sa, host, memo) -> noLongerInCombat(host, memo)),
                EVERY_COMBATANT_APIS.toArray(new String[0]));
        put(rules, Rule.shaped(EventType.COMBAT_ENDED,
                        CombatRemovalApiRules::combatAttacker, CombatRemovalApiRules::endedCombatOf),
                "EndCombatPhase");
    }

    /**
     * {@code RemoveFromCombat}'s memo: which of the clause's own targets are
     * genuinely attacking or blocking right now, before the clause runs.
     *
     * <p>removed_from_combat names every creature actually removed, not just
     * the first. {@code RemoveFromCombatEffect} gates each target on several
     * conditions (combat null, not in play, a stale LKI reference) this file
     * cannot see from outside; reading whether each memoized combatant is
     * still attacking or blocking afterward ({@link #noLongerInCombat})
     * sidesteps replicating them -- the before/after shape of {@link
     * HealGrantAndRetargetApiRules#damageBefore} and {@link TitleApiRules#newMonarch}.
     */
    private static List<Card> combatantsBefore(SpellAbility sa, Card host) {
        Combat combat = host.getGame().getCombat();
        List<Card> inCombat = new ArrayList<>();
        if (combat != null) {
            for (Card c : ClauseTargets.affectedCards(sa)) {
                if (CombatShape.isCombatant(combat, c)) {
                    inCombat.add(c);
                }
            }
        }
        return inCombat;
    }

    /**
     * Every attacker and blocker, before a clause that removes a creature
     * from combat as a side of what it does: a regeneration (701.15),
     * a change of control (506.4) or a phasing out (702.26), each of which
     * Forge applies by {@code Combat.removeFromCombat} inside the clause
     * ({@code RegenerationEffect}, {@code ControlGainEffect}'s zone
     * correction, {@code Card.phase}). Every combatant rather than the
     * clause's own cards, because each of the three APIs finds its cards its
     * own way; only a combatant the clause removed is no longer one after it.
     *
     * <p>Not {@code ExchangeControl} and the other control APIs: they set the
     * new controller and leave the zone correction, and with it the removal,
     * to the static check after the clause, where {@link ControlCorrection}
     * tells it.
     */
    private static List<Card> everyCombatant(SpellAbility sa, Card host) {
        Combat combat = host.getGame().getCombat();
        List<Card> inCombat = new ArrayList<>();
        if (combat != null) {
            inCombat.addAll(combat.getAttackers());
            inCombat.addAll(combat.getAllBlockers());
        }
        return inCombat;
    }

    /** Every memoized combatant no longer attacking or blocking, or {@code null} if none. */
    private static EffectEvent noLongerInCombat(Card host, List<Card> before) {
        if (before == null || before.isEmpty()) {
            return null;
        }
        Combat combat = host.getGame().getCombat();
        EffectEvent event = new EffectEvent(EventType.REMOVED_FROM_COMBAT);
        for (Card c : before) {
            if (!CombatShape.isCombatant(combat, c)) {
                event.subject(Refs.of(c));
            }
        }
        return event.subjects().isEmpty() ? null : event;
    }

    /**
     * {@code ChangeCombatants}'s memo: each targeted card's current attacker
     * assignment, the same pair ({@code originalDefender}/{@code defender})
     * {@code ChangeCombatantsEffect.resolve()} itself reads around its own
     * {@code addToCombat} call, for the same reselection-retargeting reason.
     * A card absent from combat entirely (not currently attacking) maps to
     * {@code null} rather than being omitted, so {@link #reselectedAttackers}
     * can tell "wasn't attacking" from "was attacking, still is" without a
     * second lookup.
     */
    private static Map<Card, GameEntity> attackerDefendersBefore(SpellAbility sa, Card host) {
        Combat combat = host.getGame().getCombat();
        Map<Card, GameEntity> map = new LinkedHashMap<>();
        if (combat != null) {
            for (Card c : ClauseTargets.affectedCards(sa)) {
                map.put(c, combat.getDefenderByAttacker(c));
            }
        }
        return map;
    }

    /**
     * Every memoized attacker whose defender assignment actually changed --
     * exactly the creatures {@code SpellAbilityEffect.addToCombat}'s
     * {@code combat.removeFromCombat(c)} call ran for, since that call fires
     * only immediately before re-adding the same creature against a new
     * defender. A card that was never attacking
     * ({@code null} before) is skipped: nothing could have removed it.
     *
     * <p>So "removed from combat" for {@code ChangeCombatants} means a
     * targeted attacker's defender assignment actually changed, read the same
     * way {@code ChangeCombatantsEffect} itself reads its own {@code
     * originalDefender}/{@code defender} pair for retargeting triggers, not
     * replicated from the {@code Optional}-decline or
     * already-attacking-that-defender guards.
     */
    private static EffectEvent reselectedAttackers(Card host, Map<Card, GameEntity> before) {
        if (before == null || before.isEmpty()) {
            return null;
        }
        Combat combat = host.getGame().getCombat();
        EffectEvent event = new EffectEvent(EventType.REMOVED_FROM_COMBAT);
        for (Map.Entry<Card, GameEntity> entry : before.entrySet()) {
            Card c = entry.getKey();
            if (entry.getValue() == null) {
                continue;
            }
            GameEntity now = combat == null ? null : combat.getDefenderByAttacker(c);
            if (!entry.getValue().equals(now)) {
                event.subject(Refs.of(c));
            }
        }
        return event.subjects().isEmpty() ? null : event;
    }

    /**
     * {@code EndCombatPhase}'s memo: the attacking player whose combat phase
     * the clause ends, read before the clause clears it, or null outside
     * combat. {@code EndCombatPhaseEffect} does nothing outside a combat
     * phase (CR 723.2g), so no combat emits nothing.
     */
    private static Player combatAttacker(SpellAbility sa, Card host) {
        Combat combat = host.getGame().getCombat();
        return combat == null ? null : combat.getAttackingPlayer();
    }

    /** combat_ended: the attacking player whose combat phase ended, if one did. */
    private static EffectEvent endedCombatOf(
            EffectEvent event, SpellAbility sa, Card host, Player attacker) {
        return attacker == null ? null : event.subject(Refs.of(attacker));
    }
}
