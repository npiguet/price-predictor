package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.GameStage;
import forge.game.card.Card;
import forge.game.card.CardView;
import forge.game.combat.Combat;
import forge.game.event.GameEventCardDamaged;
import forge.game.event.GameEventCardDamaged.DamageType;
import forge.game.phase.PhaseHandler;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.RecordLines.asMap;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A creature's {@code damage_dealt} says how the damage was dealt, as
 * {@code damage_type}: marked, turned into -1/-1 counters by wither or
 * infect, or from a deathtouch source (V1 #13).
 */
@ExtendWith(ForgeExtension.class)
class DamageTypeTest {

    @TempDir
    Path tempDir;

    /**
     * Each Forge damage type has its spelling, and a card damage carries it.
     */
    @Test
    void everyForgeDamageTypeHasItsSpelling() {
        Card bears = TestCards.build("Grizzly Bears");
        Map<DamageType, String> spelled = new HashMap<>();
        for (DamageType type : DamageType.values()) {
            EffectEvent event = BusEvents.cardDamaged(new GameEventCardDamaged(
                    CardView.get(bears), CardView.get(bears), 2, type), false);
            spelled.put(type, (String) event.params().get("damage_type"));
        }
        assertEquals(Map.of(
                DamageType.Normal, "normal",
                DamageType.M1M1Counters, "m1m1_counters",
                DamageType.Deathtouch, "deathtouch",
                DamageType.LoyaltyLoss, "loyalty_loss"), spelled);
    }

    /**
     * Boggart Ram-Gang (wither) and Typhoid Rats (deathtouch) attack, Hill
     * Giant and Grizzly Bears block them: the giant takes -1/-1 counters,
     * the bears deathtouch damage, and each attacker plain damage back.
     */
    @Test
    void aCombatRecordsDamageSaysHowItWasDealt() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.COMBAT_DECLARE_BLOCKERS);
        Player attacking = game.getPlayers().get(0);
        Player defending = game.getPlayers().get(1);
        game.setAge(GameStage.Play);
        Card ramGang = TestBoard.inPlay(game, attacking, "Boggart Ram-Gang");
        Card rats = TestBoard.inPlay(game, attacking, "Typhoid Rats");
        Card giant = TestBoard.inPlay(game, defending, "Hill Giant");
        Card bears = TestBoard.inPlay(game, defending, "Grizzly Bears");
        Combat combat = new Combat(attacking);
        combat.addAttacker(ramGang, defending);
        combat.addAttacker(rats, defending);
        combat.addBlocker(ramGang, giant);
        combat.addBlocker(rats, bears);
        combat.setBlocked(ramGang, true);
        combat.setBlocked(rats, true);
        combat.orderBlockersForDamageAssignment();
        combat.orderAttackersForDamageAssignment();
        game.getPhaseHandler().setCombat(combat);

        PhaseHandler phases = game.getPhaseHandler();
        List<String> records = ShardRun.observed(tempDir, game, collector -> {
            for (int step = 0; step < 20 && !phases.is(PhaseType.COMBAT_END); step++) {
                phases.mainLoopStep();
            }
            assertTrue(phases.is(PhaseType.COMBAT_END), "the game reached the end of combat");
        });

        Map<String, String> typeBySubject = new HashMap<>();
        for (String record : records) {
            if (!"combat".equals(RecordLines.parse(record).get("kind"))) {
                continue;
            }
            for (Map<String, Object> event : RecordLines.events(record)) {
                if ("damage_dealt".equals(event.get("type"))) {
                    typeBySubject.put(((List<?>) event.get("subjects")).get(0).toString(),
                            (String) asMap(event.get("params")).get("damage_type"));
                }
            }
        }
        assertEquals(Map.of(
                Refs.of(giant), "m1m1_counters",
                Refs.of(bears), "deathtouch",
                Refs.of(ramGang), "normal",
                Refs.of(rats), "normal"), typeBySubject, records.toString());
    }
}
