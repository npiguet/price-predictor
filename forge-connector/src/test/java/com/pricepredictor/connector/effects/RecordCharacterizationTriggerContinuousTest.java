package com.pricepredictor.connector.effects;

import forge.game.ability.AbilityKey;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.trigger.Trigger;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

/**
 * The trigger records, fired and played, and the continuous records of the statics.
 *
 * <p>Each test drives the production collectors over a fresh game and
 * compares the shard's lines with an expected file; only the envelope's
 * wall-clock timestamp is replaced.
 */
@CharacterizationTest
class RecordCharacterizationTriggerContinuousTest extends RecordCharacterizationSupport {

    /**
     * A fired evaluation, the non-fired one it buys, and the Monarch's
     * engine-built trigger, whose line no script holds.
     */
    @Test
    void triggerRecords() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Player you = board.you;
        Card warden = board.battlefield("Soul Warden", you);
        Card bears = board.battlefield("Grizzly Bears", board.opponent);
        board.game.getAction().becomeMonarch(you, "M12");
        Trigger monarch = board.game.getCardsIn(ZoneType.Command).get(0)
                .getTriggers().iterator().next();

        List<String> records = ShardRun.records(tempDir, board.game, CAPS, (bracket, collectors) -> {
            HookShapes.TriggerFire listener =
                    HookShapes.proxy(HookShapes.TriggerFire.class, collectors.triggers().handler());
            Trigger entered = warden.getTriggers().iterator().next();

            Map<AbilityKey, Object> bearsEntered = AbilityKey.newMap();
            bearsEntered.put(AbilityKey.Card, bears);
            bearsEntered.put(AbilityKey.Destination, ZoneType.Battlefield.name());
            listener.onConditionEvaluated(entered, bearsEntered, true);
            listener.onTriggerRun(entered, bearsEntered, null);

            Map<AbilityKey, Object> itselfEntered = AbilityKey.newMap();
            itselfEntered.put(AbilityKey.Card, warden);
            itselfEntered.put(AbilityKey.Destination, ZoneType.Battlefield.name());
            listener.onConditionEvaluated(entered, itselfEntered, false);

            Map<AbilityKey, Object> endStep = AbilityKey.newMap();
            endStep.put(AbilityKey.Phase, PhaseType.END_OF_TURN.nameForScripts);
            endStep.put(AbilityKey.Player, you);
            listener.onConditionEvaluated(monarch, endStep, true);
            listener.onTriggerRun(monarch, endStep, null);
        });

        Characterization.assertMatches(
                "record-trigger.jsonl", Characterization.recordLines(records));
    }

    /**
     * Triggers the engine runs, with every hook installed: Exquisite Blood
     * seeing an opponent lose 3 life (a loss written negative), Bloodcrazed
     * Hoplite seeing a +1/+1 counter put on it, one per counter.
     */
    @Test
    void playedTriggerRecords() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        board.battlefield("Exquisite Blood", board.you);
        Card hoplite = board.battlefield("Bloodcrazed Hoplite", board.you);
        board.game.getTriggerHandler().resetActiveTriggers();

        List<String> records = ShardRun.readShard(ShardRun.recordedWithClauseHook(
                tempDir, new ShardRun.Hooked(board.game, CAPS, "for the trigger hook"), () -> {
                    board.opponent.loseLife(3, false, false, null);
                    forge.game.GameEntityCounterTable table = new forge.game.GameEntityCounterTable();
                    hoplite.addCounter(forge.game.card.CounterType.getType("P1P1"), 1, board.you, table);
                    table.replaceCounterEffect(board.game, null);
                }).path()).stream()
                .filter(record -> "trigger".equals(RecordLines.parse(record).get("kind")))
                .toList();

        Characterization.assertMatches(
                "record-trigger-played.jsonl", Characterization.recordLines(records));
    }

    /** One record per static: a P/T boost, a granted keyword, a type and a colour change. */
    @Test
    void continuousRecords() throws Throwable {
        Assumptions.assumeTrue(ContinuousChannel.CAN_SUPPRESS,
                "../forge cannot remove one static's contributions");
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        board.battlefield("Glorious Anthem", board.you);
        board.battlefield("Levitation", board.you);
        board.battlefield("Grizzly Bears", board.you);
        board.battlefield("Mycosynth Lattice", board.opponent);
        board.battlefield("Serra Angel", board.opponent);
        board.game.getAction().checkStaticAbilities();

        List<String> records = ShardRun.records(tempDir, board.game, CAPS,
                (bracket, collectors) -> collectors.continuous().collect());

        Characterization.assertMatches(
                "record-continuous.jsonl", Characterization.recordLines(records));
    }

    /**
     * Statics that change abilities and hidden keywords: Humility (every
     * keyword and ability of Prodigal Sorcerer taken away), Cryptolith Rite
     * (an ability granted) and Errantry (a hidden keyword granted).
     */
    @Test
    void continuousRecordsOfAbilities() throws Throwable {
        Assumptions.assumeTrue(ContinuousChannel.CAN_SUPPRESS,
                "../forge cannot remove one static's contributions");
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        board.battlefield("Cryptolith Rite", board.you);
        Card bears = board.battlefield("Grizzly Bears", board.you);
        Card errantry = board.battlefield("Errantry", board.you);
        errantry.attachToEntity(bears, null);
        board.battlefield("Humility", board.opponent);
        board.battlefield("Prodigal Sorcerer", board.opponent);
        board.game.getAction().checkStaticAbilities();

        List<String> records = ShardRun.records(tempDir, board.game, CAPS,
                (bracket, collectors) -> collectors.continuous().collect());

        Characterization.assertMatches(
                "record-continuous-abilities.jsonl", Characterization.recordLines(records));
    }
}
