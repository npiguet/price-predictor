package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.ai.LobbyPlayerAi;
import forge.ai.PlayerControllerAi;
import forge.game.Game;
import forge.game.GameStage;
import forge.game.ability.ApiType;
import forge.game.ability.effects.CharmEffect;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.player.PlayerActionConfirmMode;
import forge.game.spellability.AbilitySub;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.function.Consumer;
import java.util.function.Function;

import static com.pricepredictor.connector.effects.RecordLines.parse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A forced resolution's effect half is the live one's (D3, FR-042): one play
 * resolved live with the hooks on, and the same play forced in a fork of the
 * same board, write effect halves that agree member for member but for the
 * envelope's fork members (its id, its write time, its link, the fork flags
 * and what the copy left out). Each play here has one legal set of choices,
 * so the fork's random draw is the live one.
 */
@ExtendWith(ForgeExtension.class)
class ForcedAndLiveHalvesTest {

    private static final String WHY = "to compare a forced half with a live one";

    /** The envelope members a fork's record differs in by construction. */
    private static final Set<String> FORK_MEMBERS =
            Set.of("record_id", "timestamp", "link_id", "interventional", "fork", "copy_dropped");

    @TempDir
    Path tempDir;

    private final java.util.concurrent.atomic.AtomicInteger runs = new java.util.concurrent.atomic.AtomicInteger();

    /** A seat whose controller refuses every "you may", in the live game and in its copies. */
    private static final class DecliningSeat extends LobbyPlayerAi {
        DecliningSeat(String name) {
            super(name, null);
        }

        @Override
        public Player createIngamePlayer(Game game, int id) {
            Player player = new Player(getName(), game, id);
            player.setFirstController(new Declining(game, player, this));
            return player;
        }
    }

    private static final class Declining extends PlayerControllerAi {
        Declining(Game game, Player player, LobbyPlayerAi lobby) {
            super(game, player, lobby);
        }

        @Override
        public boolean confirmAction(SpellAbility sa, PlayerActionConfirmMode mode, String message,
                                     List<String> options, Card cardToShow, Map<String, Object> params) {
            try {
                java.lang.reflect.Method note = PlayerControllerAi.class.getDeclaredMethod(
                        "noteEffectRecordConfirm", SpellAbility.class, boolean.class);
                note.setAccessible(true);
                return (Boolean) note.invoke(null, sa, false);
            } catch (ReflectiveOperationException e) {
                throw new AssertionError(e);
            }
        }
    }

    /** A board, built afresh for each side, and the spell its first player holds. */
    private record Play(Function<Boolean, Game> board, String spell, Consumer<SpellAbility> liveChoices) {
    }

    private static Game game(boolean declining) {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1,
                declining ? new DecliningSeat("a") : new LobbyPlayerAi("a", null),
                new LobbyPlayerAi("b", null));
        game.setAge(GameStage.Play);
        return game;
    }

    private static SpellAbility spellInHand(Game game, String name) {
        Player caster = game.getPlayers().get(0);
        for (Card card : caster.getCardsIn(forge.game.zone.ZoneType.Hand)) {
            if (card.getName().equals(name)) {
                SpellAbility spell = card.getFirstSpellAbility();
                spell.setActivatingPlayer(caster);
                return spell;
            }
        }
        throw new AssertionError("no " + name + " in hand");
    }

    /** The effect half of the play resolved live, hooks on. */
    private String liveHalf(Play play, boolean declining) throws Throwable {
        Game game = play.board().apply(declining);
        SpellAbility spell = spellInHand(game, play.spell());
        play.liveChoices().accept(spell);
        List<String> records = ShardRun.recordsWithClauseHook(tempDir.resolve("live" + runs.incrementAndGet()), game, WHY, () -> {
            game.getStack().addAndUnfreeze(spell);
            game.getStack().resolveStack();
            game.getAction().checkStateEffects(false);
            TestBoard.nextStep(game);
        });
        List<String> halves = records.stream()
                .filter(line -> "resolution".equals(parse(line).get("kind"))
                        && "resolution".equals(parse(line).get("moment")))
                .toList();
        assertEquals(1, halves.size(), records.toString());
        return halves.get(0);
    }

    /** The effect half of the same play forced in a fork of the same board. */
    private String forcedHalf(Play play, boolean declining) throws java.io.IOException {
        Game game = play.board().apply(declining);
        SpellAbility spell = spellInHand(game, play.spell());
        CollectionCaps caps = TestCaps.caps().manaCap(2000).interventions(1).probes(0).build();
        ShardRun.Fork fork = ShardRun.forkedWithClauseHook(tempDir.resolve("fork" + runs.incrementAndGet()), game, caps, 1L, spell, WHY);
        assertTrue(fork.wrote(), "the forced play wrote nothing");
        List<String> records = fork.records();
        assertEquals(1, records.size(), records.toString());
        return records.get(0);
    }

    private static Map<String, Object> withoutForkMembers(String record) {
        Map<String, Object> members = new TreeMap<>(parse(record));
        members.keySet().removeAll(FORK_MEMBERS);
        return members;
    }

    private void assertAlike(Play play, boolean declining) throws Throwable {
        String live = liveHalf(play, declining);
        String forced = forcedHalf(play, declining);
        Map<String, Object> liveMembers = withoutForkMembers(live);
        Map<String, Object> forcedMembers = withoutForkMembers(forced);
        for (String member : liveMembers.keySet()) {
            assertEquals(liveMembers.get(member), forcedMembers.get(member),
                    play.spell() + ": " + member + "\nlive:   " + live + "\nforced: " + forced);
        }
        assertEquals(liveMembers.keySet(), forcedMembers.keySet(), play.spell());
    }

    private static Game withLibrary(Game game, String card, int count) {
        Player caster = game.getPlayers().get(0);
        for (int i = 0; i < count; i++) {
            game.getAction().moveToLibrary(TestBoard.card(game, caster, card), null, null);
        }
        return game;
    }

    /** Flame Slash with one creature out: its one legal target. */
    @Test
    void aTargetedDamageSpell() throws Throwable {
        Play play = new Play(declining -> {
            Game game = game(declining);
            TestBoard.inPlay(game, game.getPlayers().get(1), "Hill Giant");
            TestBoard.inHand(game, game.getPlayers().get(0), "Flame Slash");
            return game;
        }, "Flame Slash", spell -> {
            spell.resetTargets();
            spell.getTargets().add(spell.getActivatingPlayer().getOpponents().get(0)
                    .getCreaturesInPlay().get(0));
        });
        assertAlike(play, false);
    }

    @Test
    void aTokenMaker() throws Throwable {
        Play play = new Play(declining -> {
            Game game = game(declining);
            TestBoard.inHand(game, game.getPlayers().get(0), "Raise the Alarm");
            return game;
        }, "Raise the Alarm", spell -> { });
        assertAlike(play, false);
    }

    /** Druidic Ritual's "you may mill three", accepted by the AI and refused by a declining seat. */
    @Test
    void aYouMaySpellAcceptedAndDeclined() throws Throwable {
        Play play = new Play(declining -> {
            Game game = withLibrary(game(declining), "Shock", 5);
            TestBoard.inHand(game, game.getPlayers().get(0), "Druidic Ritual");
            return game;
        }, "Druidic Ritual", spell -> { });
        assertAlike(play, false);
        assertAlike(play, true);
        assertEquals("declined", RecordLines.payload(liveHalf(play, true)).get("outcome"));
    }

    /** Abzan Charm with no creature out: its one legal mode, draw two and lose 2 life. */
    @Test
    void aCharmMode() throws Throwable {
        Play play = new Play(declining -> {
            Game game = withLibrary(game(declining), "Shock", 5);
            TestBoard.inHand(game, game.getPlayers().get(0), "Abzan Charm");
            return game;
        }, "Abzan Charm", spell -> {
            List<AbilitySub> choices = spell.getAdditionalAbilityList("Choices");
            List<AbilitySub> chosen = new java.util.ArrayList<>(List.of(choices.get(1)));
            spell.setChosenList(chosen);
            CharmEffect.chainAbilities(spell, chosen);
        });
        assertEquals(ApiType.Charm, spellInHand(play.board().apply(false), "Abzan Charm").getApi());
        assertAlike(play, false);
    }

    /** Hurricane with no land out: X is 1, the least a fork announces and all a payer without mana has. */
    @Test
    void aSpellWithX() throws Throwable {
        Play play = new Play(declining -> {
            Game game = game(declining);
            TestBoard.inPlay(game, game.getPlayers().get(1), "Serra Angel");
            TestBoard.inHand(game, game.getPlayers().get(0), "Hurricane");
            return game;
        }, "Hurricane", spell -> spell.setXManaCostPaid(1));
        assertAlike(play, false);
    }
}
