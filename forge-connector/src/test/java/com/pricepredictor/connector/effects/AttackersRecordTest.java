package com.pricepredictor.connector.effects;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Multimap;
import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.GameEntity;
import forge.game.GameStage;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.combat.CombatUtil;
import forge.game.event.GameEventAttackersDeclared;
import forge.game.event.GameEventCardTapped;
import forge.game.phase.PhaseHandler;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * An {@code attackers} record is anchored to one attacker (FR-037, story 2's
 * acceptance 11): its legal defenders, whether it must attack, and, once the
 * declaration is made, whether declaring the attack tapped it.
 *
 * <p>The engine asks once per defender ({@code CombatLegalityObserver}); the
 * hand-driven cases ask as it does, through the hook's own shape, one call per
 * defender of the combat with the creatures {@code CombatUtil.canAttack}
 * lets attack it. The last case plays the real declaration.
 */
@ExtendWith(ForgeExtension.class)
class AttackersRecordTest {

    @TempDir
    Path tempDir;

    /** The attacking side: one creature that must attack, one with vigilance, one tapped. */
    private record Board(Game game, Player attacking, Player defending, Combat combat,
                         Card juggernaut, Card angel, Card tapped, Card jace) {

        static Board of() {
            Game game = TestBoard.twoPlayerGame(PhaseType.COMBAT_DECLARE_ATTACKERS);
            Player attacking = game.getPlayers().get(0);
            Player defending = game.getPlayers().get(1);
            Card juggernaut = TestBoard.inPlay(game, attacking, "Juggernaut");
            Card angel = TestBoard.inPlay(game, attacking, "Serra Angel");
            Card tapped = TestBoard.inPlay(game, attacking, "Runeclaw Bear");
            tapped.setTapped(true);
            Card jace = TestBoard.inPlay(game, defending, "Jace Beleren");
            // A new combat initializes its constraints, as the declare-attackers
            // step does just before the engine asks.
            Combat combat = new Combat(attacking);
            game.getPhaseHandler().setCombat(combat);
            return new Board(game, attacking, defending, combat, juggernaut, angel, tapped, jace);
        }

        List<Card> candidates() {
            return List.of(juggernaut, angel, tapped);
        }

        /** What the engine asks before the declaration: once per defender. */
        void ask(Channels collectors) {
            HookShapes.CombatLegality listener = HookShapes.proxy(
                    HookShapes.CombatLegality.class, collectors.playability().legalityHandler());
            for (GameEntity defender : combat.getDefenders()) {
                List<Card> legal = new ArrayList<>();
                for (Card creature : candidates()) {
                    if (CombatUtil.canAttack(creature, defender)) {
                        legal.add(creature);
                    }
                }
                listener.onAttackersDeclarable(attacking, defender, candidates(), legal);
            }
        }
    }

    private static List<Map<String, Object>> attackersRecords(List<String> lines) {
        return lines.stream().map(RecordLines::parse)
                .filter(r -> "playability".equals(r.get("kind"))
                        && "attackers".equals(r.get("subkind")))
                .toList();
    }

    private static Map<String, Object> anchoredOn(List<Map<String, Object>> records, Card anchor) {
        List<Map<String, Object>> found = records.stream()
                .filter(r -> Refs.of(anchor).equals(payload(r).get("anchor_attacker")))
                .toList();
        assertEquals(1, found.size(), anchor + " anchors " + found.size() + ": " + records);
        return found.get(0);
    }

    /**
     * The members of {@code names} that {@code written} holds, in the order
     * it writes them: {@link RecordLines#parse} does not keep key order.
     */
    static List<String> membersInOrder(String written, String... names) {
        return Arrays.stream(names)
                .filter(name -> written.contains("\"" + name + "\":"))
                .sorted(Comparator.comparingInt(
                        name -> written.indexOf("\"" + name + "\":")))
                .toList();
    }

    private static Map<String, Object> payload(Map<String, Object> record) {
        return RecordLines.asMap(record.get("payload"));
    }

    /**
     * Two creatures that may attack and one tapped give three records, each
     * anchored to one of them: the two list both defenders, the player and
     * the planeswalker, the tapped one none. The one a static makes attack
     * says so. Nothing was declared, so none carries {@code tapped_by_attack},
     * and no attackers record carries {@code what_if}.
     */
    @Test
    void eachAttackerGetsARecordWithItsLegalDefendersAndItsRequirement() throws Throwable {
        Board board = Board.of();
        List<String> lines = ShardRun.records(tempDir, board.game(),
                (bracket, collectors) -> board.ask(collectors));

        List<Map<String, Object>> records = attackersRecords(lines);
        assertEquals(3, records.size(), records.toString());
        List<String> bothDefenders = List.of(Refs.of(board.defending()), Refs.of(board.jace()));

        Map<String, Object> juggernaut = payload(anchoredOn(records, board.juggernaut()));
        assertEquals(bothDefenders, juggernaut.get("legal_defenders"));
        assertEquals(true, juggernaut.get("must_attack"), "Juggernaut attacks each combat if able");

        Map<String, Object> angel = payload(anchoredOn(records, board.angel()));
        assertEquals(bothDefenders, angel.get("legal_defenders"));
        assertEquals(false, angel.get("must_attack"));

        Map<String, Object> tapped = payload(anchoredOn(records, board.tapped()));
        assertEquals(List.of(), tapped.get("legal_defenders"), "a tapped creature attacks nothing");
        assertEquals(false, tapped.get("must_attack"));

        for (Map<String, Object> record : records) {
            Map<String, Object> payload = payload(record);
            assertFalse(record.containsKey("what_if"), "no what-if attackers record: " + record);
            assertFalse(payload.containsKey("tapped_by_attack"), "nothing declared: " + payload);
            assertEquals(Refs.of(board.attacking()), record.get("actor_player"));
            assertEquals(List.of(Refs.of(board.juggernaut()), Refs.of(board.angel())),
                    payload.get("legal_attackers"), "the declaration's, on every record");
            assertEquals(Refs.of(board.tapped()),
                    RecordLines.asMap(((List<?>) payload.get("forbidden")).get(0)).get("entity"));
        }
        String written = RecordLines.payloadOf(lines.stream()
                .filter(l -> l.contains("\"subkind\":\"attackers\"")).findFirst().orElseThrow());
        List<String> order = List.of(
                "anchor_attacker", "legal_defenders", "must_attack", "legal_attackers", "forbidden");
        assertEquals(order, membersInOrder(written, order.toArray(String[]::new)),
                "the member order the contract gives: " + written);
    }

    /**
     * Acceptance 11 by hand: the engine taps the declared attacker without
     * vigilance and leaves the one with vigilance untapped, then publishes
     * the declaration. The records wait for it and are written then, so a tap
     * after it changes nothing: the first carries
     * {@code tapped_by_attack = true}, the second {@code false}, and the
     * creature nobody declared carries none.
     */
    @Test
    void aDeclaredAnchorCarriesWhetherTheAttackTappedIt() throws Throwable {
        Board board = Board.of();
        List<String> lines = ShardRun.records(tempDir, board.game(), (bracket, collectors) -> {
            board.game().subscribeToEvents(collectors.playability().declaration());
            board.ask(collectors);
            // A tap before the declaration is made is the declaration's.
            board.combat().addAttacker(board.juggernaut(), board.defending());
            board.combat().addAttacker(board.angel(), board.jace());
            board.juggernaut().tap(true, true, null, null);
            Multimap<GameEntity, Card> declared = ArrayListMultimap.create();
            declared.put(board.defending(), board.juggernaut());
            declared.put(board.jace(), board.angel());
            board.game().fireEvent(new GameEventAttackersDeclared(board.attacking(), declared));
            // A tap after it is not the declaration's: the records are written.
            board.game().fireEvent(new GameEventCardTapped(board.angel(), true));
        });

        List<Map<String, Object>> records = attackersRecords(lines);
        assertEquals(3, records.size(), records.toString());
        assertEquals(true, payload(anchoredOn(records, board.juggernaut())).get("tapped_by_attack"));
        assertEquals(false, payload(anchoredOn(records, board.angel())).get("tapped_by_attack"));
        assertFalse(payload(anchoredOn(records, board.tapped())).containsKey("tapped_by_attack"));
    }

    /**
     * Acceptance 11 on the engine: the declare-attackers step asks, the AI
     * declares both creatures, the engine taps the one without vigilance and
     * publishes the declaration, and the records say so.
     */
    @Test
    void theEnginesDeclarationTapsTheAnchorWithoutVigilance() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player attacking = game.getPlayers().get(0);
        Player defending = game.getPlayers().get(1);
        game.setAge(GameStage.Play);
        Card juggernaut = TestBoard.inPlay(game, attacking, "Juggernaut");
        Card angel = TestBoard.inPlay(game, attacking, "Serra Angel");

        PhaseHandler phases = game.getPhaseHandler();
        List<String> lines = ShardRun.recordsWithClauseHook(tempDir, game,
                "the combat legality hook", () -> {
            for (int step = 0; step < 40 && !phases.getPhase().isAfter(
                    PhaseType.COMBAT_DECLARE_ATTACKERS) && phases.getPlayerTurn() == attacking;
                    step++) {
                phases.mainLoopStep();
            }
            assertTrue(phases.getPhase().isAfter(PhaseType.COMBAT_DECLARE_ATTACKERS),
                    "the game got past the declaration: " + phases.getPhase());
        });

        List<Map<String, Object>> records = attackersRecords(lines);
        assertEquals(2, records.size(), records.toString());
        Map<String, Object> declaredJuggernaut = payload(anchoredOn(records, juggernaut));
        Map<String, Object> declaredAngel = payload(anchoredOn(records, angel));
        assertEquals(List.of(Refs.of(defending)), declaredJuggernaut.get("legal_defenders"));
        assertEquals(List.of(Refs.of(defending)), declaredAngel.get("legal_defenders"));
        assertEquals(true, declaredJuggernaut.get("must_attack"));
        assertEquals(false, declaredAngel.get("must_attack"));
        assertEquals(true, declaredJuggernaut.get("tapped_by_attack"),
                "declared without vigilance: " + declaredJuggernaut);
        assertEquals(false, declaredAngel.get("tapped_by_attack"),
                "declared with vigilance: " + declaredAngel);
        assertTrue(juggernaut.isTapped());
        assertFalse(angel.isTapped());
    }
}
