package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.GameStage;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.phase.PhaseHandler;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A clause a replacement runs inside the combat damage step has no resolution
 * bracket open, so its clause-hook events are filed on the step's combat
 * record ({@code BusBracketCollector.route}, the consequence case), not in an
 * effect half: Drudge Skeletons' regeneration shield, used by the lethal
 * damage of a Hill Giant it blocks, writes its {@code regenerated} and
 * {@code removed_from_combat} there.
 */
@ExtendWith(ForgeExtension.class)
class CombatReplacementClauseTest {

    @TempDir
    Path tempDir;

    @Test
    void aRegenerationShieldUsedByCombatDamageFilesItsClauseEventsOnTheCombatRecord()
            throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.COMBAT_DECLARE_BLOCKERS);
        Player attacking = game.getPlayers().get(0);
        Player defending = game.getPlayers().get(1);
        game.setAge(GameStage.Play);
        Card giant = TestBoard.inPlay(game, attacking, "Hill Giant");
        Card skeletons = TestBoard.inPlay(game, defending, "Drudge Skeletons");
        SpellAbility shield = TestBoard.ability(skeletons, ApiType.Regenerate);
        shield.setActivatingPlayer(defending);
        game.getStack().add(shield);
        game.getStack().resolveStack();
        TestBoard.blockedCombat(new TestBoard.CombatSides(game, attacking, defending), giant, skeletons);

        PhaseHandler phases = game.getPhaseHandler();
        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game,
                "for the clause hook", () -> {
                    for (int step = 0; step < 20 && !phases.is(PhaseType.COMBAT_END); step++) {
                        phases.mainLoopStep();
                    }
                });

        assertTrue(skeletons.isInPlay(), "the premise: the shield saved it");
        List<String> combats = records.stream()
                .filter(record -> "combat".equals(RecordLines.parse(record).get("kind"))).toList();
        List<String> shielded = combats.stream()
                .filter(record -> RecordEvents.eventTypes(record).contains("regenerated")).toList();
        assertEquals(1, shielded.size(), combats.toString());
        assertTrue(RecordEvents.eventTypes(shielded.get(0)).contains("removed_from_combat"),
                shielded.get(0));
    }
}
