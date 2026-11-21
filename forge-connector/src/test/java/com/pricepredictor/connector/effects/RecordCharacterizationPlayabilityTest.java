package com.pricepredictor.connector.effects;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Multimap;
import forge.game.GameEntity;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.event.GameEventAttackersDeclared;
import forge.game.event.GameEventCardTapped;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;

import java.util.List;

/**
 * The playability records: the decision, attackers and blockers subkinds, and the what-if legality record.
 *
 * <p>Each test drives the production collectors over a fresh game and
 * compares the shard's lines with an expected file; only the envelope's
 * wall-clock timestamp is replaced.
 */
@CharacterizationTest
class RecordCharacterizationPlayabilityTest extends RecordCharacterizationSupport {

    /**
     * The three subkinds: a {@code decision} with no {@code what_if}, the
     * {@code attackers} records anchored one per attacker, with no
     * {@code what_if} either, and the {@code blockers} record, which carries
     * it. The decision is the one candidate written as a priority stop writes
     * each it samples in; {@link PriorityCandidatesTest} covers the stop.
     * The bears are declared and tapped by the attack, so their attackers
     * record carries {@code tapped_by_attack}, and the elves' does not. The
     * defending player is the random seat, so the blockers record is the
     * one stamped {@code random_seat}.
     */
    @Test
    void playabilityRecords() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Player you = board.you;
        Player opponent = board.opponent;
        board.battlefield("Mountain", you);
        Card bears = board.battlefield("Grizzly Bears", you);
        bears.setSickness(false);
        Card elves = board.battlefield("Llanowar Elves", you);
        elves.setSickness(false);
        board.battlefield("Pacifism", you).attachToEntity(elves, null);
        Card angel = board.battlefield("Serra Angel", opponent);
        Card giant = board.battlefield("Hill Giant", opponent);
        giant.setTapped(true);
        board.game.getAction().checkStaticAbilities();
        SpellAbility bolt = board.spell(board.hand("Lightning Bolt", you), you);

        List<String> records = ShardRun.records(
                tempDir, board.game, CAPS, Refs.of(opponent),
                (bracket, collectors) -> {
            collectors.playability().recordDecision(
                    you, bolt, DecisionCandidate.read(bolt, you).orElseThrow());

            HookShapes.CombatLegality listener = HookShapes.proxy(
                    HookShapes.CombatLegality.class, collectors.playability().legalityHandler());
            listener.onAttackersDeclarable(
                    you, opponent, List.of(bears, elves), List.of(bears));
            AttackDeclaration declaration = collectors.playability().declaration();
            declaration.onTapped(new GameEventCardTapped(bears, true));
            Multimap<GameEntity, Card> declared = ArrayListMultimap.create();
            declared.put(opponent, bears);
            declaration.onAttackersDeclared(new GameEventAttackersDeclared(you, declared));
            listener.onBlockersDeclarable(
                    opponent, bears, List.of(angel, giant), List.of(angel), 1);

            // A blocker Pacifism keeps out (responsible_static), one a lure
            // attacking beside the anchor calls away (required_elsewhere),
            // one its own keyword line keeps out (responsible_own), and one
            // an engine-built effect's static keeps out, which no line names
            // (responsible_unresolved).
            Card pacified = board.battlefield("Gray Ogre", opponent);
            board.battlefield("Pacifism", you).attachToEntity(pacified, null);
            Card unicorn = board.battlefield("Prized Unicorn", you);
            Card lured = board.battlefield("Runeclaw Bear", opponent);
            Card grakk = board.battlefield("Grakk the Pacifist", opponent);
            Card stopped = board.battlefield("Walking Corpse", opponent);
            Card effect = new Card(board.game.nextCardId(), board.game);
            effect.setName("Engine Effect");
            effect.setOwner(you);
            effect.setGamePieceType(forge.card.GamePieceType.EFFECT);
            effect.addStaticAbility("Mode$ CantBlock | ValidCard$ Creature.namedWalking Corpse"
                    + " | EffectZone$ Command | Description$ Walking Corpse can't block.");
            board.game.getAction().moveTo(ZoneType.Command, effect, null, null);
            board.game.getAction().checkStaticAbilities();
            Combat lure = new Combat(you);
            lure.addAttacker(bears, opponent);
            lure.addAttacker(unicorn, opponent);
            board.game.getPhaseHandler().setCombat(lure);
            listener.onBlockersDeclarable(
                    opponent, bears, List.of(pacified, lured, grakk, stopped), List.of(), 1);
        });

        Characterization.assertMatches(
                "record-playability.jsonl", Characterization.recordLines(records));
    }

    /**
     * A legality record answering an AI what-if query: {@code what_if} true,
     * right after {@code random_seat}.
     *
     * <p>No collector writes one any more: the engine's declarations are the
     * only legality source, and they are stamped false. The value stays part
     * of the record schema, and gen-2 shards carry it, so its rendering is
     * pinned through the record builder the legality path uses, over the
     * board's own snapshot.
     */
    @Test
    void aWhatIfLegalityRecord() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        board.battlefield("Grizzly Bears", board.you);
        board.battlefield("Serra Angel", board.opponent);

        String record = EffectRecord.playability(
                new RecordHeader(
                        "run.0-l1.0", "run", RecordShardWriter.timestamp(), ShardRun.GAME_ID),
                PlayabilitySubkind.BLOCKERS, AttributionMode.detect())
                .actor(Refs.of(board.opponent))
                .whatIf(true)
                .state(new SnapshotBuilder(board.game).ofBoard().state())
                .toJson();

        Characterization.assertMatches(
                "record-what-if.jsonl", Characterization.recordLines(List.of(record)));
    }
}
