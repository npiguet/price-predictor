package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.combat.Combat;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.RecordLines.events;
import static com.pricepredictor.connector.effects.RecordLines.params;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Rules whose point combat makes, played forward: first-strike damage
 * (510.2), the engine's own question before a declaration of blockers
 * (702.118b's blockers record, a record no other kind of point writes) and
 * combat damage trampling over a planeswalker (702.19c, 702.19f).
 */
@ExtendWith(ForgeExtension.class)
class DirectedForkPlayedForwardCombatTest extends DirectedForkPlayedForwardSupport {

    /**
     * {@code replace_card}, 510.2: A's Grizzly Bears blocked by B's Grizzly
     * Bears as the first-strike damage step begins. The live creatures trade
     * in the regular step; the fork's blocker is Youthful Knight, which kills
     * the attacker in the first-strike step and takes no damage.
     */
    @Test
    void aFirstStrikeBlockerInTheBlockersPlaceKillsFirst() throws Throwable {
        Game game = board(PhaseType.COMBAT_FIRST_STRIKE_DAMAGE);
        Player active = game.getPlayers().get(0);
        Player defending = game.getPlayers().get(1);
        Card attacker = game.getCardsIn(forge.game.zone.ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals("Grizzly Bears")).findFirst().orElseThrow();
        attacker.setSickness(false);
        Card blocker = permanent(game, defending, "Grizzly Bears");
        TestBoard.blockedCombat(game, active, defending, attacker, blocker);

        List<String> fork = forkOf(game, PhaseType.COMBAT_FIRST_STRIKE_DAMAGE, "510.2",
                "knight_in_the_bears_place");

        List<Map<String, Object>> moves = eventsOf(fork, "zone_change", ref(attacker));
        assertTrue(moves.stream().anyMatch(move -> "graveyard".equals(params(move).get("to_zone"))),
                "the attacker dies in the fork: " + moves);
        String knight = null;
        for (String record : fork) {
            for (Map<String, Object> event : events(record)) {
                if ("damage_dealt".equals(event.get("type"))
                        && List.of(ref(attacker)).equals(event.get("subjects"))) {
                    knight = (String) params(event).get("source");
                }
            }
        }
        assertTrue(knight != null && !knight.equals(ref(blocker)), "a new card dealt the attacker's damage");
        assertTrue(eventsOf(fork, "damage_dealt", knight).isEmpty(), "the Knight is dealt no damage");
    }

    /**
     * {@code replace_card}, 702.118b: Furtive Homunculus (2/1, skulk)
     * attacks and B's Centaur Courser could block it but for its power, as
     * the declare-blockers step begins. The fork's Courser is Llanowar
     * Elves, which the blockers record of the engine's question before the
     * declaration lists among the Homunculus's legal blockers, whatever B
     * then blocks with.
     */
    @Test
    void anElfOfLesserPowerInTheCoursersPlaceIsALegalBlocker() throws Throwable {
        Game game = board(PhaseType.COMBAT_DECLARE_BLOCKERS);
        Player active = game.getPlayers().get(0);
        Player defending = game.getPlayers().get(1);
        Card homunculus = permanent(game, active, "Furtive Homunculus");
        Card courser = permanent(game, defending, "Centaur Courser");
        Combat combat = new Combat(active);
        combat.addAttacker(homunculus, defending);
        game.getPhaseHandler().setCombat(combat);

        List<String> fork = forkOf(game, PhaseType.COMBAT_DECLARE_BLOCKERS, "702.118b", "lesser_power");

        List<Object> legal = new ArrayList<>();
        for (String record : fork) {
            if ("blockers".equals(parse(record).get("subkind"))
                    && ref(homunculus).equals(RecordLines.payload(record).get("anchor_attacker"))) {
                legal.add(RecordLines.payload(record).get("legal_blockers"));
            }
        }
        assertEquals(1, legal.size(), "one blockers record for the Homunculus: " + legal);
        assertTrue(legal.get(0) instanceof List<?> blockers && blockers.size() == 1
                && !blockers.contains(ref(courser)), "the Elves, a new card, may block: " + legal);
    }

    /** The Grizzly Bears attacking B beside the attacker of {@link #goneWalkerTurn}. */
    private String bearsRef;

    /**
     * Every record of a live turn whose {@code attacker} attacks B's Jace
     * Beleren unblocked beside a Grizzly Bears attacking B, Jace having left
     * the battlefield before the damage step (removed by whatever removed it),
     * so the attacker attacks Forge's placeholder (506.4c). The live game
     * leaves the declare-blockers step and begins the damage step itself,
     * {@code rule}'s directed fork taken at that boundary.
     */
    private List<String> goneWalkerTurn(String attackerName, String rule) throws Throwable {
        Game game = board(PhaseType.COMBAT_DECLARE_BLOCKERS);
        Player active = game.getPlayers().get(0);
        Player defending = game.getPlayers().get(1);
        Card attacker = permanent(game, active, attackerName);
        Card bears = permanent(game, active, "Grizzly Bears");
        bearsRef = ref(bears);
        Card jace = permanent(game, defending, "Jace Beleren");
        Combat combat = new Combat(active);
        combat.addAttacker(attacker, jace);
        combat.addAttacker(bears, defending);
        combat.setBlocked(attacker, false);
        combat.setBlocked(bears, false);
        game.getPhaseHandler().setCombat(combat);
        game.getAction().moveToGraveyard(jace, null);
        assertTrue(Combat.isRemovedDefender(combat.getDefenderByAttacker(attacker)));
        return played(game, null, rule, Decisions.SAMPLED);
    }

    /**
     * The combat damage {@code records} (of the fork, or of the live game)
     * dealt to player {@code P1} by anything but the Bears beside the attacker.
     */
    private List<Map<String, Object>> combatDamageToB(List<String> records, boolean fork) {
        List<Map<String, Object>> found = new ArrayList<>();
        for (String record : records) {
            if (isCombatOf(parse(record), fork)) {
                found.addAll(combatDamageToBBesideTheBears(record));
            }
        }
        return found;
    }

    /** The combat damage events of {@code record} dealt to player {@code P1} by anything but the Bears. */
    private List<Map<String, Object>> combatDamageToBBesideTheBears(String record) {
        return eventsOf(List.of(record), "damage_dealt", "P1").stream()
                .filter(event -> Boolean.TRUE.equals(params(event).get("combat"))
                        && !bearsRef.equals(params(event).get("source")))
                .toList();
    }

    /** Whether {@code records} hold a combat record (of the fork, or of the live game). */
    private static boolean hasCombat(List<String> records, boolean fork) {
        return records.stream().map(RecordLines::parse).anyMatch(parsed -> isCombatOf(parsed, fork));
    }

    /** Whether a parsed record is a combat record of the fork ({@code fork}) or of the live game. */
    private static boolean isCombatOf(Map<String, Object> parsed, boolean fork) {
        return Boolean.TRUE.equals(parsed.get("fork")) == fork && "combat".equals(parsed.get("kind"));
    }

    /**
     * {@code replace_card}, 702.19c, inverted: Colossal Dreadmaw attacks B's
     * Jace Beleren (3 loyalty) unblocked, and all its damage stays on Jace;
     * the fork's Thrasta, Tempest's Roar, in its place, has trample over
     * planeswalkers, and its damage beyond Jace's loyalty reaches B.
     */
    @Test
    void thrastaInTheDreadmawsPlaceTramplesOverThePlaneswalker() throws Throwable {
        Game game = board(PhaseType.COMBAT_DECLARE_BLOCKERS);
        Player active = game.getPlayers().get(0);
        Player defending = game.getPlayers().get(1);
        Card dreadmaw = permanent(game, active, "Colossal Dreadmaw");
        Card bears = game.getCardsIn(forge.game.zone.ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals("Grizzly Bears")).findFirst().orElseThrow();
        bearsRef = ref(bears);
        Card jace = permanent(game, defending, "Jace Beleren");
        jace.setCounters(CounterEnumType.LOYALTY, 3);
        Combat combat = new Combat(active);
        combat.addAttacker(dreadmaw, jace);
        combat.setBlocked(dreadmaw, false);
        game.getPhaseHandler().setCombat(combat);

        List<String> records = played(game, null, "702.19c", Decisions.SAMPLED);

        assertTrue(hasCombat(records, true) && hasCombat(records, false), "both games recorded the step");
        assertEquals(List.of(), combatDamageToB(records, false), "the Dreadmaw's damage stays on Jace live");
        List<Map<String, Object>> fork = combatDamageToB(records, true);
        assertEquals(1, fork.size(), "Thrasta's excess reaches B in the fork: " + fork);
        assertEquals(4, ((Number) params(fork.get(0)).get("amount")).intValue());
    }

    /**
     * {@code replace_card}, 702.19f: Craw Wurm attacks a planeswalker that
     * has left combat and deals no combat damage; the fork's Thrasta,
     * Tempest's Roar deals its 7 to B. The same fork is 702.19e's
     * {@code fires} side (an attacker with trample over planeswalkers whose
     * planeswalker left combat), which is why 702.19e has no fork of its own.
     */
    @Test
    void thrastaInTheWurmsPlaceHitsBOnceThePlaneswalkerIsGone() throws Throwable {
        List<String> records = goneWalkerTurn("Craw Wurm", "702.19f");

        assertTrue(hasCombat(records, true) && hasCombat(records, false), "both games recorded the step");
        assertEquals(List.of(), combatDamageToB(records, false), "the Wurm's damage reaches nobody live");
        List<Map<String, Object>> fork = combatDamageToB(records, true);
        assertEquals(1, fork.size(), "Thrasta's reaches B in the fork: " + fork);
        assertEquals(7, ((Number) params(fork.get(0)).get("amount")).intValue());
    }
}
