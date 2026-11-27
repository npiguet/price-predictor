package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.AbilitySub;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.RecordLines.asMap;
import static com.pricepredictor.connector.effects.RecordPicks.only;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * A stack object's {@code targets} links as every record writes them, from
 * the one reading of its targets ({@code ChainTargets.Target}): each ref
 * once, indexed by its position among them, an ability by its line, on a
 * cost half and an effect half alike.
 */
@ExtendWith(ForgeExtension.class)
class TargetLinksTest {

    private static final String WHY = "for the resolution hook's targets as cast";

    @TempDir
    Path tempDir;

    /** {@code record}'s {@code targets} links from {@code line}, each as {@code [to, index]}. */
    private static List<List<Object>> targetsFrom(String record, String line) {
        Object from = parse("{\"line\":" + line + "}").get("line");
        List<?> links = (List<?>) asMap(parse(record).get("state")).get("links");
        return links.stream().map(RecordLines::asMap)
                .filter(link -> "targets".equals(link.get("type")) && from.equals(link.get("from")))
                .map(link -> List.of(link.get("to"), link.get("index"))).toList();
    }

    /**
     * Stifle at Prodigal Pyromancer's ability on the stack (the stack copy
     * {@code MagicStack.add} made of it): its cost half and its effect half
     * both link the ability at its line, never at its card, at index 0.
     */
    @Test
    void anAbilityTargetedEndsAtItsLineOnTheCostAndTheEffectHalf() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card pyromancer = TestBoard.inPlay(game, opponent, "Prodigal Pyromancer");
        SpellAbility ping = TestBoard.ability(pyromancer, ApiType.DealDamage);
        ping.setActivatingPlayer(opponent);
        ping.resetTargets();
        ping.getTargets().add(caster);
        Card stifle = TestBoard.inHand(game, caster, "Stifle");
        SpellAbility[] cast = {null};

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            game.getStack().add(ping);
            // MagicStack.add puts a copy of an activated ability on the stack
            // and clears the original's targets: Stifle aims at that copy.
            cast[0] = TestBoard.cast(game, stifle, caster, game.getStack().peekAbility());
            game.getStack().resolveStack();
            TestBoard.priority(game);
            TestBoard.nextStep(game);
        });

        String line = SnapshotLinks.line(cast[0].getHostCard(), cast[0]);
        Object pingLine = parse("{\"line\":" + SnapshotLinks.line(pyromancer, ping) + "}").get("line");
        List<List<Object>> expected = List.of(List.of(pingLine, 0L));
        String script = "cardsfolder/s/stifle.txt";
        String cost = only(records, "activation", script);
        assertEquals(expected, targetsFrom(cost, line), cost);
        String effect = only(records, "resolution", script);
        assertEquals(expected, targetsFrom(effect, line), effect);
    }

    /**
     * Kolaghan's Command destroying an Ornithopter and dealing it 2 damage,
     * under a Shock: on the Shock's records the Command's two links to the
     * Ornithopter are one, at index 0, as {@code refs.targets} lists the
     * Ornithopter once.
     */
    @Test
    void aTargetTwoLinksShareIsOneLinkAtItsPositionAmongTheRefs() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card thopter = TestBoard.inPlay(game, opponent, "Ornithopter");
        Card command = TestBoard.inHand(game, caster, "Kolaghan's Command");
        Card shock = TestBoard.inHand(game, caster, "Shock");
        SpellAbility[] charm = {null};

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            charm[0] = TestBoard.charmOf(game.getAction().moveToStack(command, null), caster);
            List<AbilitySub> modes = TestBoard.chain(charm[0], 2, 3);
            modes.get(0).getTargets().add(thopter);
            modes.get(1).getTargets().add(thopter);
            game.getStack().add(charm[0]);
            TestBoard.cast(game, shock, caster, opponent);
            game.getStack().resolveStack();
            TestBoard.priority(game);
            TestBoard.nextStep(game);
        });

        String line = SnapshotLinks.line(charm[0].getHostCard(), charm[0]);
        List<List<Object>> once = List.of(List.of(Refs.of(thopter), 0L));
        String script = "cardsfolder/s/shock.txt";
        for (String record : List.of(only(records, "activation", script),
                only(records, "resolution", script))) {
            assertEquals(once, targetsFrom(record, line), record);
        }
        Map<String, Object> commandCost = asMap(parse(
                only(records, "activation", "cardsfolder/k/kolaghans_command.txt")).get("state"));
        assertEquals(List.of(Refs.of(thopter)), asMap(commandCost.get("refs")).get("targets"));
    }

    /** The links of {@code family} in {@code state} that end at {@code ref}, each once. */
    private static List<Object> linksTo(Map<String, Object> state, String family, String ref) {
        return ((List<?>) state.get("links")).stream().map(RecordLines::asMap)
                .filter(link -> family.equals(link.get("type")) && ref.equals(link.get("to")))
                .map(link -> link.get("from")).toList();
    }

    /**
     * A Lightning Bolt's targets chosen after the board was read: a Grizzly
     * Bears in exile (a card the board lacked) and a Hill Giant in a
     * graveyard (one it held). Both get their {@code targets} link, and each
     * its owner's {@code owns} and {@code controls} link once.
     */
    @Test
    void aTargetAddedAtTheCastEventGetsItsControlsAndOwnsLinks() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player you = game.getPlayers().get(0);
        Player opp = game.getPlayers().get(1);
        Card exiled = game.getAction().moveTo(ZoneType.Exile, TestBoard.card(game, opp, "Grizzly Bears"), null, null);
        Card graveyard = TestBoard.inGraveyard(game, opp, "Hill Giant");
        Card bolt = TestBoard.inHand(game, you, "Lightning Bolt");
        SpellAbility spell = TestBoard.spellOf(bolt, you);
        SnapshotBuilder builder = new SnapshotBuilder(game);
        Snapshot before = builder.ofAbility(spell, List.of(bolt));
        spell.resetTargets();
        spell.getTargets().add(exiled);
        spell.getTargets().add(graveyard);

        String state = builder.withTargetsChosen(before, spell).render();

        Map<String, Object> parsed = asMap(parse(state));
        String owner = (String) linksTo(parsed, "owns", Refs.of(graveyard)).get(0);
        for (Card target : List.of(exiled, graveyard)) {
            String ref = Refs.of(target);
            assertEquals(List.of(owner), linksTo(parsed, "controls", ref), state);
            assertEquals(List.of(owner), linksTo(parsed, "owns", ref), state);
            assertEquals(1, linksTo(parsed, "targets", ref).size(), state);
        }
        List<String> all = new ArrayList<>();
        ((List<?>) parsed.get("links")).forEach(link -> all.add(link.toString()));
        assertEquals(all.size(), new HashSet<>(all).size(), "a link written twice: " + state);
    }
}
