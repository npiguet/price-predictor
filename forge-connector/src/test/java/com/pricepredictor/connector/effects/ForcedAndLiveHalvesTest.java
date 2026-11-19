package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.ai.LobbyPlayerAi;
import forge.game.Game;
import forge.game.GameStage;
import forge.game.ability.ApiType;
import forge.game.ability.effects.CharmEffect;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
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

    /** A board, built afresh for each side, and the spell or ability its first player plays on it. */
    private record Play(Function<Boolean, Game> board, String spell, Function<Game, SpellAbility> pick,
                        Consumer<SpellAbility> liveChoices) {

        /** The spell named {@code spell}, from its first player's hand. */
        Play(Function<Boolean, Game> board, String spell, Consumer<SpellAbility> liveChoices) {
            this(board, spell, game -> spellInHand(game, spell), liveChoices);
        }
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
        SpellAbility spell = play.pick().apply(game);
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
        SpellAbility spell = play.pick().apply(game);
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
        assertAlike(play, declining, live -> { });
    }

    /** The same, with a check of the live half, which the forced one then matches. */
    private void assertAlike(Play play, boolean declining, Consumer<String> checkLive) throws Throwable {
        String live = liveHalf(play, declining);
        checkLive.accept(live);
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
        assertAlike(atTheOpponentsCreature("Flame Slash"), false);
    }

    /**
     * {@code spell}, from the first player's hand, aimed at the one creature
     * out, the opponent's Hill Giant.
     */
    private static Play atTheOpponentsCreature(String spell) {
        return new Play(declining -> {
            Game game = game(declining);
            TestBoard.inPlay(game, game.getPlayers().get(1), "Hill Giant");
            TestBoard.inHand(game, game.getPlayers().get(0), spell);
            return game;
        }, spell, aimed -> {
            aimed.resetTargets();
            aimed.getTargets().add(aimed.getActivatingPlayer().getOpponents().get(0)
                    .getCreaturesInPlay().get(0));
        });
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

    /**
     * Bonesplitter equipped to the one creature out: the half holds the
     * attaching and the +2/+0 the static check after it applies, both the
     * equip's one clause's, live and forced alike (finding 2.1).
     */
    @Test
    void anEquip() throws Throwable {
        Play play = new Play(declining -> {
            Game game = game(declining);
            Player caster = game.getPlayers().get(0);
            TestBoard.inPlay(game, caster, "Grizzly Bears");
            TestBoard.inPlay(game, caster, "Bonesplitter");
            return game;
        }, "Bonesplitter", ForcedAndLiveHalvesTest::equipOfBonesplitter, equip -> {
            equip.resetTargets();
            equip.getTargets().add(equip.getActivatingPlayer().getCreaturesInPlay().get(0));
        });
        assertAlike(play, false, linksAre(
                List.of(List.of("attached", "root"), List.of("pt_change", "root"))));
    }

    /**
     * Brass Squire attaching Bonesplitter to itself, the one creature out:
     * a clause targeting the Equipment, then the Attach sub-ability. The
     * attaching, and the +2/+0 the static check after that last clause
     * applies, are the sub-ability's, {@code 0}, live and forced alike
     * (finding 2.1).
     */
    @Test
    void anEquipWithASubAbility() throws Throwable {
        Play play = new Play(declining -> {
            Game game = game(declining);
            Player caster = game.getPlayers().get(0);
            TestBoard.inPlay(game, caster, "Brass Squire");
            TestBoard.inPlay(game, caster, "Bonesplitter");
            return game;
        }, "Brass Squire", game -> abilityOf(game, ApiType.Pump), attach -> {
            Player caster = attach.getActivatingPlayer();
            attach.resetTargets();
            attach.getTargets().add(caster.getCardsIn(forge.game.zone.ZoneType.Battlefield).stream()
                    .filter(Card::isEquipment).findFirst().orElseThrow());
            attach.getSubAbility().resetTargets();
            attach.getSubAbility().getTargets().add(caster.getCreaturesInPlay().get(0));
        });
        assertAlike(play, false, linksAre(
                List.of(List.of("attached", "0"), List.of("pt_change", "0"))));
    }

    /**
     * Act of Treason on the one creature out, the opponent's: its one clause
     * gains control of it, untaps it and gives it haste. The control change
     * is the clause's, {@code root}, live and forced alike (finding 2.1).
     */
    @Test
    void aControlChange() throws Throwable {
        assertAlike(atTheOpponentsCreature("Act of Treason"), false, linksAre(List.of(
                List.of("keyword_change", "root"), List.of("control_change", "root"),
                List.of("zone_change", "unresolved"))));
    }

    /** A check that a record's events are {@code expected}: each one's type and {@code attributed_to}, in order. */
    private static Consumer<String> linksAre(List<List<Object>> expected) {
        return record -> assertEquals(expected, RecordLines.events(record).stream()
                .map(event -> List.of(event.get("type"), event.get("attributed_to"))).toList(), record);
    }

    private static SpellAbility equipOfBonesplitter(Game game) {
        return abilityOf(game, ApiType.Attach);
    }

    /** The first ability whose API is {@code api} of the first player's permanents. */
    private static SpellAbility abilityOf(Game game, ApiType api) {
        Player caster = game.getPlayers().get(0);
        SpellAbility ability = caster.getCardsIn(forge.game.zone.ZoneType.Battlefield).stream()
                .flatMap(card -> card.getSpellAbilities().stream())
                .filter(line -> line.getApi() == api).findFirst().orElseThrow();
        ability.setActivatingPlayer(caster);
        return ability;
    }
}
