package com.pricepredictor.connector.effects;

import forge.game.ability.AbilityKey;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.replacement.ReplacementEffect;
import forge.game.replacement.ReplacementResult;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

/**
 * The rewrite records (a substitution, a buffered ability, counter types) and the mana record, as the collectors build them.
 *
 * <p>Each test drives the production collectors over a fresh game and
 * compares the shard's lines with an expected file; only the envelope's
 * wall-clock timestamp is replaced.
 */
@CharacterizationTest
class RecordCharacterizationRewriteTest extends RecordCharacterizationSupport {

    /**
     * A substitution that ran another ability, and an in-place edit of the
     * damage amount, through the replacement hook's handler.
     */
    @Test
    void rewriteRecords() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Player you = board.you;
        Card rest = board.battlefield("Rest in Peace", you);
        Card armor = board.battlefield("Urza's Armor", you);
        Card bears = board.battlefield("Grizzly Bears", board.opponent);
        Card angel = board.battlefield("Serra Angel", board.opponent);

        List<String> records = ShardRun.records(tempDir, board.game, CAPS, (bracket, collectors) -> {
            HookShapes.Replacement listener =
                    HookShapes.proxy(HookShapes.Replacement.class, collectors.rewrites().handler());

            ReplacementEffect exile = rest.getReplacementEffects().iterator().next();
            SpellAbility ran = exile.ensureAbility();
            ran.setReplacementEffect(exile);
            Map<AbilityKey, Object> dying = AbilityKey.newMap();
            dying.put(AbilityKey.Affected, bears);
            dying.put(AbilityKey.Origin, ZoneType.Battlefield.name());
            dying.put(AbilityKey.Destination, ZoneType.Graveyard.name());
            listener.onReplacement(exile, dying, AbilityKey.newMap(dying),
                    ReplacementResult.Replaced, ran);

            ReplacementEffect prevent = armor.getReplacementEffects().iterator().next();
            Map<AbilityKey, Object> before = AbilityKey.newMap();
            before.put(AbilityKey.Affected, you);
            before.put(AbilityKey.DamageSource, angel);
            before.put(AbilityKey.DamageAmount, 4);
            before.put(AbilityKey.IsCombat, true);
            Map<AbilityKey, Object> after = AbilityKey.newMap(before);
            after.put(AbilityKey.DamageAmount, 3);
            listener.onReplacement(prevent, before, after, ReplacementResult.Updated, null);

            // Run parameters Forge could not copy: incoming_unknown.
            listener.onReplacement(prevent, null, after, ReplacementResult.Updated, null);
        });

        Characterization.assertMatches(
                "record-rewrite.jsonl", Characterization.recordLines(records));
    }

    /**
     * Shock at Swans of Bryn Argoll, played on the hooked path: the
     * prevention's rewrite record names the buffered draw and takes the
     * events of its run.
     */
    @Test
    void bufferedRewriteRecords() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Player you = board.you;
        for (int i = 0; i < 2; i++) {
            board.game.getAction().moveToLibrary(board.hand("Island", you), null, null);
        }
        Card swans = board.battlefield("Swans of Bryn Argoll", board.opponent);

        List<String> records = ShardRun.recordsWithClauseHook(
                tempDir, board.game, "for the buffered-ability hooks", () -> {
                    board.cast(board.card("Shock", you), you, swans);
                    board.game.getStack().resolveStack();
                    board.game.getAction().checkStateEffects(false);
                });

        Characterization.assertMatches("record-rewrite-buffered.jsonl",
                Characterization.recordLines(records.stream()
                        .filter(line -> line.contains("\"kind\":\"rewrite\"")).toList()));
    }

    /**
     * Captain Marvel's power-up (a +1/+1 and an indestructible counter at
     * once) under Hardened Scales and Doubling Season, played on the hooked
     * path: the scales' record is the +1/+1 counters' alone, and the
     * season's is two, one per type it doubled.
     */
    @Test
    void counterRewriteRecordsOverSeveralTypes() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Player you = board.you;
        board.battlefield("Hardened Scales", you);
        board.battlefield("Doubling Season", you);
        Card marvel = board.battlefield("Captain Marvel, Earth's Protector", you);

        List<String> records = ShardRun.recordsWithClauseHook(
                tempDir, board.game, "for rewrite records over a counter map", () -> {
                    SpellAbility powerUp = board.ability(marvel, ApiType.PutCounter);
                    powerUp.setActivatingPlayer(you);
                    board.castAndResolve(powerUp);
                });

        Characterization.assertMatches("record-rewrite-counter-types.jsonl",
                Characterization.recordLines(records.stream()
                        .filter(line -> line.contains("\"kind\":\"rewrite\"")).toList()));
    }

    /**
     * A pain land tapped for coloured mana, played on the hooked path: its
     * mana record holds the tap, the mana, and the damage its sub-ability
     * dealt its controller.
     */
    @Test
    void manaRecords() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Player you = board.you;
        Card wastes = board.battlefield("Adarkar Wastes", you);
        SpellAbility pain = wastes.getManaAbilities().stream()
                .filter(sa -> sa.getSubAbility() != null).findFirst().orElseThrow();
        pain.setActivatingPlayer(you);

        List<String> records = ShardRun.recordsWithClauseHook(
                tempDir, board.game, "for the payment and clause hooks", () ->
                        forge.ai.ComputerUtil.playNoStack(you, pain, board.game, false));

        Characterization.assertMatches(
                "record-mana.jsonl", Characterization.recordLines(records));
    }
}
