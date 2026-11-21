package com.pricepredictor.connector.effects;

import forge.card.MagicColor;
import forge.game.ability.AbilityFactory;
import forge.game.ability.ApiType;
import forge.game.ability.SpellAbilityEffect;
import forge.game.card.Card;
import forge.game.event.GameEventCardDamaged;
import forge.game.event.GameEventPlayerDamaged;
import forge.game.event.GameEventPlayerLivesChanged;
import forge.game.mana.Mana;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Every resolution record's rendered line: the resolution records of a patched worker, a modal spell's halves, a turn wrap's turn actions, and a modal spell as a worker without the patch records it.
 *
 * <p>Each test drives the production collectors over a fresh game and
 * compares the shard's lines with an expected file; only the envelope's
 * wall-clock timestamp is replaced.
 */
@CharacterizationTest
class RecordCharacterizationResolutionTest extends RecordCharacterizationSupport {

    /**
     * Both moments, through the engine with every hook installed, as a
     * patched worker records them: a two-colour spell that kills its target,
     * a spell that fizzles (its targets as cast), a tap ability, an ability
     * hosted on an engine-built effect card, which keys to the line that made
     * the effect, a "can't block" pump (a hidden keyword), Arc Trail losing
     * one of its two targets (partially fizzled, its targets as cast apart
     * from those left) and Cryptolith Rite entering (an ability granted).
     */
    @Test
    void resolutionRecords() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Player you = board.you;
        Player opponent = board.opponent;
        Card mountain = board.battlefield("Mountain", you);
        Card plains = board.battlefield("Plains", you);
        Card pyromancer = board.battlefield("Prodigal Pyromancer", you);
        pyromancer.setSickness(false);
        Card bears = board.battlefield("Grizzly Bears", opponent);
        Card giant = board.battlefield("Hill Giant", opponent);
        Card elves = board.battlefield("Llanowar Elves", opponent);
        Card ogre = board.battlefield("Gray Ogre", opponent);
        Card wall = board.battlefield("Wall of Wood", you);

        RecordShardWriter writer = ShardRun.recordedWithClauseHook(tempDir, board.game, CAPS,
                "for the resolution hook's targets as cast", () -> {
            SpellAbility helix = board.spell(
                    board.game.getAction().moveToStack(board.card("Lightning Helix", you), null),
                    you);
            helix.resetTargets();
            helix.getTargets().add(bears);
            helix.getPayingMana().add(new Mana(MagicColor.RED, mountain, null, you));
            helix.getPayingMana().add(new Mana(MagicColor.WHITE, plains, null, you));
            board.castAndResolve(helix);

            board.cast(board.card("Shock", you), you, giant);
            board.game.getAction().exile(giant, null, null);
            board.game.getStack().resolveStack();
            board.game.getAction().checkStateEffects(false);

            SpellAbility ping = board.ability(pyromancer, ApiType.DealDamage);
            ping.setActivatingPlayer(you);
            ping.resetTargets();
            ping.getTargets().add(opponent);
            board.castAndResolve(ping);

            Card effect = SpellAbilityEffect.createEffect(
                    ping, you, "Characterization Effect", null);
            board.game.getAction().moveToCommand(effect, null);
            SpellAbility gain = AbilityFactory.getAbility(
                    "AB$ GainLife | Cost$ 0 | Defined$ You | LifeAmount$ 2", effect);
            gain.setActivatingPlayer(you);
            board.castAndResolve(gain);

            board.cast(board.card("Abandon the Post", you), you, elves);
            board.game.getStack().resolveStack();
            board.game.getAction().checkStateEffects(false);

            SpellAbility arcTrail = board.spell(
                    board.game.getAction().moveToStack(board.card("Arc Trail", you), null), you);
            arcTrail.resetTargets();
            arcTrail.getTargets().add(ogre);
            arcTrail.getSubAbility().getTargets().add(opponent);
            board.game.getStack().add(arcTrail);
            board.game.getAction().exile(ogre, null, null);
            board.game.getStack().resolveStack();
            board.game.getAction().checkStateEffects(false);

            SpellAbility rite = board.spell(
                    board.game.getAction().moveToStack(board.card("Cryptolith Rite", you), null), you);
            board.castAndResolve(rite);

            board.nextStep();
        });
        List<String> records = ShardRun.readShard(writer.path()).stream()
                .filter(record -> "resolution".equals(RecordLines.parse(record).get("kind")))
                .toList();
        assertTrue(wall.isInPlay());

        Characterization.assertMatches(
                "record-resolution.jsonl", Characterization.recordLines(records));
    }

    /**
     * What a board holds beside its permanents' front faces, through the
     * engine with every hook installed: Graceful Reprieve leaves a pending
     * delayed trigger on Grizzly Bears, and a transformed Delver of Secrets
     * (Insectile Aberration) is on the battlefield on its back face, when
     * Lightning Bolt is cast at the opponent and resolves. Both moments of
     * both spells.
     */
    @Test
    void aBackFaceAndAPendingDelayedTrigger() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Player you = board.you;
        Player opponent = board.opponent;
        Card bears = board.battlefield("Grizzly Bears", you);
        Card delver = board.battlefield("Delver of Secrets", you);
        delver.setState(forge.card.CardStateName.Backside, true);

        RecordShardWriter writer = ShardRun.recordedWithClauseHook(tempDir, board.game, CAPS,
                "for the resolution hook", () -> {
            board.cast(board.card("Graceful Reprieve", you), you, bears);
            board.game.getStack().resolveStack();
            TestBoard.cast(board.game, board.card("Lightning Bolt", you), you, opponent);
            board.game.getStack().resolveStack();
            board.game.getAction().checkStateEffects(false);
            board.nextStep();
        });
        List<String> records = ShardRun.readShard(writer.path()).stream()
                .filter(record -> "resolution".equals(RecordLines.parse(record).get("kind")))
                .toList();
        assertEquals(4, records.size(), records.toString());
        String bolt = RecordLines.only(records, "activation", "cardsfolder/l/lightning_bolt.txt");
        assertTrue(bolt.contains("\"delayed_triggers\":[{"), "the reprieve pending: " + bolt);
        assertTrue(bolt.contains("\"name\":\"Insectile Aberration\"")
                && bolt.contains("\"face_state\":\"back\""), "the delver's back face: " + bolt);

        Characterization.assertMatches(
                "record-back-face-and-delayed-trigger.jsonl", Characterization.recordLines(records));
    }

    /**
     * Every turn-based action a turn wrap performs, each a {@code turn_action}
     * record of its own, through the engine's real loop with every hook
     * installed: the ending player's cleanup discard to hand size, then the
     * next player's phasing (Breezekeeper), day becoming night, untap (a
     * tapped Forest), draw and a lore counter on History of Benalia.
     */
    @Test
    void theTurnActionsOfATurnWrap() throws Throwable {
        forge.game.Game game = TestBoard.twoPlayerGame(PhaseType.END_OF_TURN);
        CharacterizationBoard board = CharacterizationBoard.on(game);
        Player next = game.getPlayers().get(0);
        Player ending = game.getPlayers().get(1);
        // Turn 1 is the ending player's, so the next one's draw is not skipped.
        game.getPhaseHandler().devModeSet(PhaseType.END_OF_TURN, ending, 1);
        game.setAge(forge.game.GameStage.Play);
        for (int i = 0; i < 8; i++) {
            game.getAction().moveTo(ZoneType.Hand, board.card("Craw Wurm", ending), null, null);
        }
        for (Player player : List.of(ending, next)) {
            for (int i = 0; i < 2; i++) {
                game.getAction().moveTo(ZoneType.Library, board.card("Craw Wurm", player), null, null);
            }
        }
        board.battlefield("Forest", next).setTapped(true);
        board.battlefield("Breezekeeper", next);
        board.battlefield("History of Benalia", next);
        game.getTriggerHandler().clearWaitingTriggers();
        game.setDayTime(false);

        forge.game.phase.PhaseHandler phases = game.getPhaseHandler();
        RecordShardWriter writer = ShardRun.recordedWithClauseHook(tempDir, game, CAPS,
                "for the hooks a worker installs", () -> {
            for (int step = 0; step < 100
                    && !(phases.is(PhaseType.MAIN1) && phases.getPlayerTurn() == next); step++) {
                phases.mainLoopStep();
            }
            assertTrue(phases.is(PhaseType.MAIN1) && phases.getPlayerTurn() == next,
                    "the game reached the next player's main phase");
            game.fireEvent(new forge.game.event.GameEventPlayerPriority(
                    forge.game.player.PlayerView.get(next), PhaseType.MAIN1,
                    forge.game.player.PlayerView.get(next)));
        });
        List<String> records = ShardRun.readShard(writer.path()).stream()
                .filter(record -> "turn_action".equals(RecordLines.parse(record).get("kind")))
                .toList();
        assertEquals(List.of("cleanup_discard", "phasing", "day_night", "untap", "draw", "lore_counter"),
                records.stream().map(record -> RecordLines.parse(record).get("subkind")).toList());

        Characterization.assertMatches(
                "record-turn-actions.jsonl", Characterization.recordLines(records));
    }

    /**
     * A modal spell through the engine with the clause hook installed: one
     * cost half through the root key and one effect half per chosen mode,
     * each through its {@code option} key.
     */
    @Test
    void aModalResolutionRecordsOneHalfPerMode() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Player you = board.you;
        board.battlefield("Llanowar Elves", board.opponent);
        board.battlefield("Hill Giant", board.opponent);
        SpellAbility charm = TestBoard.charmOf(
                board.game.getAction().moveToStack(board.card("Fiery Confluence", you), null),
                you);
        TestBoard.chain(charm, 0, 1, 1);

        Assumptions.assumeTrue(PatchHooks.CLAUSE_LISTENER.present(),
                "../forge has no clause hook to split the modes");
        RecordShardWriter writer = ShardRun.writer(tempDir);
        try (GameRecording recording =
                new GameRecording(
                        GameRecordingRequest.of(board.game, writer, ShardRun.GAME_ID, CAPS)
                                .seeded(1L))) {
            recording.open();
            board.castAndResolve(charm);
            board.nextStep();
        } finally {
            writer.close();
        }
        List<String> records = ShardRun.readShard(writer.path());

        Characterization.assertMatches(
                "record-resolution-modal.jsonl", Characterization.recordLines(records));
    }

    /**
     * A modal spell as a worker without the patch records it: {@code mode}
     * degraded, one cost half, and one effect half through the root line
     * holding every mode's bus events, since no clause hook splits them.
     *
     * <p>The collector is handed the mode a stock checkout's probe answers.
     * The events are published outside any resolution, where the
     * sub-ability pointer is empty as it always is unpatched.
     */
    @Test
    void aDegradedWorkersModalResolution() throws Exception {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Player you = board.you;
        Player opponent = board.opponent;
        Card elves = board.battlefield("Llanowar Elves", opponent);
        Card confluence = board.game.getAction().moveToStack(
                board.card("Fiery Confluence", you), null);
        SpellAbility charm = TestBoard.charmOf(confluence, you);
        TestBoard.chain(charm, 0, 1);

        RecordShardWriter writer = ShardRun.writer(tempDir);
        try {
            BusBracketCollector bracket = new BusBracketCollector(board.game,
                    new CollectorContext(
                            writer, ShardRun.GAME_ID, AttributionMode.DEGRADED, null, null),
                    recordId -> { });
            board.game.subscribeToEvents(bracket);
            bracket.beginBracket(charm);
            // The charm's resolution, as MagicStack.resolveStack marks it:
            // the card is in the stack zone though no stack item was pushed,
            // which with nothing resolving reads as a cast under way.
            board.game.getStack().setResolving(true);
            try {
                board.game.fireEvent(new GameEventCardDamaged(
                        elves.getView(), confluence.getView(), 1,
                        GameEventCardDamaged.DamageType.Normal));
                board.game.fireEvent(new GameEventPlayerDamaged(
                        opponent.getView(), confluence.getView(), 2, false, false));
                board.game.fireEvent(new GameEventPlayerLivesChanged(opponent, 20, 18));
                bracket.endBracket(charm.getId(), false);
            } finally {
                board.game.getStack().setResolving(false);
            }
            bracket.onNextAction();
        } finally {
            writer.close();
        }

        Characterization.assertMatches("record-degraded.jsonl",
                Characterization.recordLines(ShardRun.readShard(writer.path())));
    }
}
