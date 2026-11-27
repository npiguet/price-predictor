package com.pricepredictor.connector.effects;

import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The combat record and its damage-step probe, and the fork records: interventional, protection and directed forks.
 *
 * <p>Each test drives the production collectors over a fresh game and
 * compares the shard's lines with an expected file; only the envelope's
 * wall-clock timestamp is replaced.
 */
@CharacterizationTest
class RecordCharacterizationForkTest extends RecordCharacterizationSupport {

    /**
     * A trampler blocked by a smaller creature, the observed damage step's
     * record, and the probe branch that re-ran the step without trample and
     * mirrors it, wired as a worker wires them.
     */
    @Test
    void combatRecordAndItsDamageStepProbe() throws Throwable {
        Assumptions.assumeTrue(PatchHooks.ASSIGNED_DAMAGE.present(),
                "../forge does not keep the step's damage assignment");
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.COMBAT_DAMAGE);
        Player you = board.you;
        Player opponent = board.opponent;
        Card dreadmaw = board.battlefield("Colossal Dreadmaw", you);
        Card soldier = board.token("w_1_1_soldier", you);
        Card bears = board.battlefield("Grizzly Bears", opponent);
        Combat combat = TestBoard.blockedCombat(
                new TestBoard.CombatSides(board.game, you, opponent), dreadmaw, bears, soldier);
        CollectionCaps caps = TestCaps.caps().everyDecision()
                .interventions(0).probes(1).probeKeywords("trample").build();

        RecordShardWriter writer = ShardRun.writer(tempDir);
        try {
            ProbeScheduler probing = new ProbeScheduler(board.game, ShardRun.GAME_ID, caps,
                    new DamageStepProbe(board.game, TestCollectors.context(writer, ShardRun.GAME_ID), new ForkBudget(caps)));
            BusBracketCollector bracket = new BusBracketCollector(
                    board.game, TestCollectors.context(writer, ShardRun.GAME_ID), probing::writeHeldProbes);
            board.game.subscribeToEvents(bracket);
            probing.collect();
            assertTrue(combat.assignCombatDamage(false), "the step assigned no damage");
            combat.dealAssignedDamage();
            board.game.getAction().checkStateEffects(false);
            TestBoard.enterStep(board.game, PhaseType.COMBAT_END);
        } finally {
            writer.close();
        }
        List<String> records = ShardRun.readShard(writer.path());

        List<String> observed = new ArrayList<>();
        List<String> probes = new ArrayList<>();
        for (String record : records) {
            (record.contains("\"fork\":true") ? probes : observed).add(record);
        }
        assertTrue(!probes.isEmpty(), "no probe branch was written: " + records);
        Characterization.assertMatches(
                "record-combat.jsonl", Characterization.recordLines(observed));
        Characterization.assertMatches(
                "record-combat-probe.jsonl", Characterization.recordLines(probes));
    }

    /** A forced resolution of a spell nobody cast, in a copy of the game. */
    @Test
    void anInterventionalResolution() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Player you = board.you;
        board.battlefield("Mountain", you);
        board.battlefield("Grizzly Bears", board.opponent);
        SpellAbility pyroclasm = board.spell(board.hand("Pyroclasm", you), you);
        CollectionCaps caps = TestCaps.caps().everyDecision()
                .interventions(1).probes(0).build();

        ShardRun.Fork fork = ShardRun.forked(tempDir, new ShardRun.ForkRun(board.game, caps, 1L, pyroclasm));
        assertTrue(fork.wrote(), "the fork observed nothing");

        Characterization.assertMatches(
                "record-intervention.jsonl", Characterization.recordLines(fork.records()));
    }

    /**
     * A forced resolution through the hooks, on a copy that left something
     * out: Raise the Alarm's effect half holds the {@code token_created} the
     * outcome hook reports, and the record carries {@code copy_dropped} for
     * the one remembered object (a token that ceased to exist) the copier
     * could not map.
     */
    @Test
    void anInterventionalResolutionOnALossyCopyThroughTheHooks() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Player you = board.you;
        Card holder = board.battlefield("Hill Giant", you);
        Card token = board.token("w_1_1_soldier", you);
        holder.addRemembered(token);
        token.getZone().remove(token);
        SpellAbility alarm = board.spell(board.hand("Raise the Alarm", you), you);
        CollectionCaps caps = TestCaps.caps().everyDecision()
                .interventions(1).probes(0).build();

        ShardRun.Fork fork = ShardRun.forkedWithClauseHook(
                tempDir, new ShardRun.ForkRun(board.game, caps, 1L, alarm), "to hear the outcome hook");
        assertTrue(fork.wrote(), "the fork observed nothing");
        String record = fork.records().get(0);
        assertTrue(record.contains("\"copy_dropped\":{\"remembered\":1}"), record);
        assertTrue(record.contains("\"token_created\""), record);

        Characterization.assertMatches(
                "record-intervention-hooked.jsonl", Characterization.recordLines(fork.records()));
    }

    /**
     * Two forced resolutions that fizzle: Murder at Skulking Ghost, which
     * sacrifices itself on becoming a target (the spell's half holds its
     * card's move off the stack), and Royal Assassin's ability at the tapped
     * Ghost (the ability's half holds nothing). Both read fizzled.
     */
    @Test
    void forcedResolutionsThatFizzle() throws Throwable {
        List<String> records = new ArrayList<>();
        for (String forced : List.of("Murder", "Royal Assassin")) {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card ghost = board.battlefield("Skulking Ghost", board.opponent);
            ghost.setTapped(true);
            board.game.getTriggerHandler().resetActiveTriggers();
            SpellAbility ability;
            if (forced.equals("Murder")) {
                ability = board.spell(board.hand("Murder", board.you), board.you);
            } else {
                Card assassin = board.battlefield("Royal Assassin", board.you);
                ability = assassin.getSpellAbilities().stream()
                        .filter(sa -> sa.getApi() == ApiType.Destroy).findFirst().orElseThrow();
                ability.setActivatingPlayer(board.you);
            }
            CollectionCaps caps = TestCaps.caps().everyDecision().interventions(1).probes(0).build();
            ShardRun.Fork fork = ShardRun.forkedWithClauseHook(
                    tempDir.resolve(forced.replace(' ', '-')), new ShardRun.ForkRun(board.game, caps, 1L, ability), "to read the resolution hook's fizzle");
            assertTrue(fork.wrote(), forced + ": the fork wrote nothing");
            records.addAll(fork.records());
        }
        assertTrue(records.stream().allMatch(record -> record.contains("\"outcome\":\"fizzled\"")),
                records.toString());

        Characterization.assertMatches(
                "record-intervention-fizzled.jsonl", Characterization.recordLines(records));
    }

    /**
     * A forced "you may" its controller declines: Druidic Ritual's "you may
     * mill three" refused by a seat that refuses every "you may", its half
     * holding the card's move off the stack and reading declined.
     */
    @Test
    void aForcedYouMayDeclined() throws Throwable {
        forge.game.Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1,
                new DecliningSeat("a"), new forge.ai.LobbyPlayerAi("b", null));
        game.setAge(forge.game.GameStage.Play);
        CharacterizationBoard board = CharacterizationBoard.on(game);
        for (int i = 0; i < 5; i++) {
            game.getAction().moveToLibrary(TestBoard.card(game, board.you, "Shock"), null, null);
        }
        SpellAbility ritual = board.spell(board.hand("Druidic Ritual", board.you), board.you);
        CollectionCaps caps = TestCaps.caps().everyDecision().interventions(1).probes(0).build();

        ShardRun.Fork fork = ShardRun.forkedWithClauseHook(
                tempDir, new ShardRun.ForkRun(game, caps, 1L, ritual), "to hear the confirm hook");
        assertTrue(fork.wrote(), "the fork wrote nothing");
        assertTrue(fork.records().get(0).contains("\"outcome\":\"declined\""), fork.records().toString());

        Characterization.assertMatches(
                "record-intervention-declined.jsonl", Characterization.recordLines(fork.records()));
    }

    /**
     * A probe whose combat record holds the damage a protection prevented:
     * Colossal Dreadmaw blocked by Coast Watcher (protection from green),
     * trample probed, through the hooks.
     */
    @Test
    void aProbeHoldingThePreventedDamage() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.COMBAT_DAMAGE);
        Card dreadmaw = board.battlefield("Colossal Dreadmaw", board.you);
        Card watcher = board.battlefield("Coast Watcher", board.opponent);
        Combat combat = TestBoard.blockedCombat(new TestBoard.CombatSides(board.game, board.you, board.opponent), dreadmaw, watcher);
        CollectionCaps caps = TestCaps.caps().everyDecision().manaCap(2000).interventions(0).probes(1)
                .probeKeywords("trample").build();

        RecordShardWriter writer = ShardRun.recordedWithClauseHook(
                tempDir, new ShardRun.Hooked(board.game, caps, "to hear the outcome hook"), () -> {
                    TestBoard.enterStep(board.game, PhaseType.COMBAT_DAMAGE);
                    assertTrue(combat.assignCombatDamage(false), "the step assigned no damage");
                    combat.dealAssignedDamage();
                    board.game.getAction().checkStateEffects(false);
                    TestBoard.enterStep(board.game, PhaseType.COMBAT_END);
                });
        List<String> probes = ShardRun.readShard(writer.path()).stream()
                .filter(record -> record.contains("\"fork\":true")).toList();
        assertTrue(probes.stream().anyMatch(record -> record.contains("\"damage_prevented\"")), probes.toString());

        Characterization.assertMatches(
                "record-combat-probe-prevented.jsonl", Characterization.recordLines(probes));
    }

    /**
     * A protection fork: Grizzly Bears granted protection from Goblins, the
     * copy played to the end of the turn, every record interventional.
     */
    @Test
    void aProtectionFork() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.you);
        board.battlefield("Goblin Piker", board.opponent);
        // The default sampling rates, from the game's seed: the turn's
        // decisions mostly unwritten, so the golden holds the fork's shape.
        CollectionCaps caps = TestCaps.caps().interventions(0).probes(0).build();
        RecordShardWriter writer = ShardRun.writer(tempDir);
        boolean kept;
        try (GameRecording recording = new GameRecording(
                GameRecordingRequest.of(board.game, writer, ShardRun.GAME_ID, caps)
                        .seeded(7L))) {
            recording.open();
            kept = new ProtectionFork(board.game, new ForkBudget(caps), recording.playForward(), 7L)
                    .take(new ProtectionFork.Grant(bears, "Protection:Goblin"));
        } finally {
            writer.close();
        }
        assertTrue(kept, "the fork was not kept");
        List<String> records = ShardRun.readShard(writer.path());

        Characterization.assertMatches(
                "record-protection-fork.jsonl", Characterization.recordLines(records));
    }

    /**
     * A directed fork: 704.5f's variant "tougher" (a pt edit) on Grizzly
     * Bears, the copy played to the end of the turn, every record naming the
     * rule, the variant and the operation.
     */
    @Test
    void aDirectedFork() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.you);
        board.battlefield("Hill Giant", board.opponent);
        CollectionCaps caps = TestCaps.caps().interventions(0).probes(0)
                .directed(1, 1.0, "704.5f").build();
        com.pricepredictor.connector.effects.directed.DirectedRule rule =
                com.pricepredictor.connector.effects.directed.DirectedRule.of("704.5f", "704.5f-1",
                        live -> java.util.Optional.of(
                                com.pricepredictor.connector.effects.directed.Binding.empty()
                                        .with("creature", bears)),
                        new com.pricepredictor.connector.effects.directed.DirectedRule.Variant("tougher", new com.pricepredictor.connector.effects.directed.Application(
                                com.pricepredictor.connector.effects.directed.ForkOperation.PT,
                                binding -> new com.pricepredictor.connector.effects.directed.operations
                                        .PtOperation(binding.card("creature"), null, 5))));
        RecordShardWriter writer = ShardRun.writer(tempDir);
        boolean kept;
        try (GameRecording recording = new GameRecording(
                GameRecordingRequest.of(board.game, writer, ShardRun.GAME_ID, caps)
                        .seeded(7L))) {
            recording.open();
            kept = new DirectedFork(board.game, recording.playForward(), () -> board.you)
                    .take(rule, "tougher", rule.variants().get("tougher"),
                            com.pricepredictor.connector.effects.directed.Binding.empty().with("creature", bears));
        } finally {
            writer.close();
        }
        assertTrue(kept, "the fork was not kept");
        List<String> records = ShardRun.readShard(writer.path());

        Characterization.assertMatches(
                "record-directed-fork.jsonl", Characterization.recordLines(records));
    }
}
