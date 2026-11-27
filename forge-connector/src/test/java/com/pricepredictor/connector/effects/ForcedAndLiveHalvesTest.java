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
            Set.of("record_id", "timestamp", "link_id", "interventional", "fork", "fork_kind", "copy_dropped");

    @TempDir
    Path tempDir;

    /** The most stack items a live play resolves before the next step: the play and what it set off. */
    private static final int MAX_ITEMS = 10;

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

    /**
     * The effect halves of the play resolved live, hooks on: the resolution
     * records under a cost half's link, which a trigger or a copy the play
     * set off has none of. What the play triggered goes on the stack, and
     * the stack is resolved until it is empty, each item followed by the
     * state-based check and the triggers it left, then the next step begins.
     */
    private List<String> liveHalves(Play play, boolean declining) throws Throwable {
        Game game = play.board().apply(declining);
        SpellAbility spell = play.pick().apply(game);
        play.liveChoices().accept(spell);
        List<String> records = ShardRun.recordsWithClauseHook(tempDir.resolve("live" + runs.incrementAndGet()), game, WHY, () -> {
            game.getStack().addAndUnfreeze(spell);
            game.getStack().addAllTriggeredAbilitiesToStack();
            for (int items = 0; items < MAX_ITEMS && !game.getStack().isEmpty(); items++) {
                game.getStack().resolveStack();
                observedStateBasedLoop(game);
            }
            TestBoard.nextStep(game);
        });
        List<String> halves = records.stream()
                .filter(line -> "resolution".equals(parse(line).get("kind"))
                        && "resolution".equals(parse(line).get("moment"))
                        && RecordAbilities.linkOf(line) != null)
                .toList();
        assertTrue(!halves.isEmpty(), records.toString());
        return halves;
    }

    /**
     * The state-based loop a real game runs before a player gets priority:
     * Forge's private {@code PhaseHandler.checkStateBasedEffects}, which
     * signals the loop's begin and end to the hook and repeats the check
     * while triggers go on the stack. Reflection, as no public entry runs it.
     */
    private static void observedStateBasedLoop(Game game) throws Exception {
        java.lang.reflect.Method loop = forge.game.phase.PhaseHandler.class.getDeclaredMethod("checkStateBasedEffects");
        loop.setAccessible(true);
        loop.invoke(game.getPhaseHandler());
    }

    /** The effect halves of the same play forced in a fork of the same board. */
    private List<String> forcedHalves(Play play, boolean declining) throws java.io.IOException {
        Game game = play.board().apply(declining);
        SpellAbility spell = play.pick().apply(game);
        CollectionCaps caps = TestCaps.caps().manaCap(2000).interventions(1).probes(0).build();
        ShardRun.Fork fork = ShardRun.forkedWithClauseHook(tempDir.resolve("fork" + runs.incrementAndGet()), game, caps, 1L, spell, WHY);
        assertTrue(fork.wrote(), "the forced play wrote nothing");
        return fork.records();
    }

    private static Map<String, Object> withoutForkMembers(String record) {
        Map<String, Object> members = new TreeMap<>(parse(record));
        members.keySet().removeAll(FORK_MEMBERS);
        return members;
    }

    private void assertAlike(Play play, boolean declining) throws Throwable {
        assertAlike(play, declining, live -> { });
    }

    /** The same, with a check of the live halves, which the forced ones then match. */
    private void assertAlike(Play play, boolean declining, Consumer<String> checkLive) throws Throwable {
        List<String> live = liveHalves(play, declining);
        live.forEach(checkLive);
        List<String> forced = forcedHalves(play, declining);
        assertEquals(live.size(), forced.size(), play.spell() + "\nlive:   " + live + "\nforced: " + forced);
        for (int index = 0; index < live.size(); index++) {
            // Member for member: the maps differ in a member or in the members they hold.
            assertEquals(withoutForkMembers(live.get(index)), withoutForkMembers(forced.get(index)),
                    play.spell() + "\nlive:   " + live.get(index) + "\nforced: " + forced.get(index));
        }
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
        assertEquals("declined", RecordEvents.payload(liveHalves(play, true).get(0)).get("outcome"));
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
     * equip's one clause's, live and forced alike.
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
     *, and so is the duration: the Attach's, permanent, not
     * the Pump root's end of turn.
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
                List.of(List.of("attached", "0"), List.of("pt_change", "0")))
                .andThen(ptChangeIsPermanent()));
    }

    /**
     * Act of Treason on the one creature out, the opponent's: its one clause
     * gains control of it, untaps it and gives it haste. The control change
     * is the clause's, {@code root}, live and forced alike.
     */
    @Test
    void aControlChange() throws Throwable {
        assertAlike(atTheOpponentsCreature("Act of Treason"), false, linksAre(List.of(
                List.of("keyword_change", "root"), List.of("control_change", "root"),
                List.of("zone_change", "unresolved"))));
    }

    /**
     * Switcheroo exchanging the first player's attacking Grizzly Bears for
     * the defender's Hill Giant, the two creatures out: the static check
     * after its one clause corrects both zones, and the bears leave combat,
     * the clause's, {@code root}, live and forced alike.
     */
    @Test
    void anExchangeOfAnAttacker() throws Throwable {
        Play play = new Play(declining -> {
            Game game = game(declining);
            Player caster = game.getPlayers().get(0);
            Player defender = game.getPlayers().get(1);
            Card bears = TestBoard.inPlay(game, caster, "Grizzly Bears");
            TestBoard.inPlay(game, defender, "Hill Giant");
            TestBoard.inHand(game, caster, "Switcheroo");
            game.getPhaseHandler().devModeSet(PhaseType.COMBAT_DECLARE_BLOCKERS, caster);
            forge.game.combat.Combat combat = new forge.game.combat.Combat(caster);
            combat.addAttacker(bears, defender);
            combat.setBlocked(bears, false);
            game.getPhaseHandler().setCombat(combat);
            return game;
        }, "Switcheroo", exchange -> {
            exchange.resetTargets();
            for (Player each : exchange.getHostCard().getGame().getPlayers()) {
                exchange.getTargets().add(each.getCreaturesInPlay().get(0));
            }
        });
        assertAlike(play, false, linksAre(List.of(
                List.of("removed_from_combat", "root"), List.of("control_change", "root"),
                List.of("control_change", "root"), List.of("zone_change", "unresolved"))));
    }

    /**
     * Flame Slash killing the one creature out, the opponent's Grizzly Bears
     * under the first player's Pacifism: the state-based check that kills
     * the bears takes the Aura off, and both halves hold its {@code
     * unattached}.
     */
    @Test
    void anEnchantedCreatureKilledInTheStateBasedCheck() throws Throwable {
        Play play = new Play(declining -> {
            Game game = game(declining);
            Player caster = game.getPlayers().get(0);
            Card bears = TestBoard.inPlay(game, game.getPlayers().get(1), "Grizzly Bears");
            TestBoard.inPlay(game, caster, "Pacifism").attachToEntity(bears, null);
            TestBoard.inHand(game, caster, "Flame Slash");
            return game;
        }, "Flame Slash", aimed -> {
            aimed.resetTargets();
            aimed.getTargets().add(aimed.getActivatingPlayer().getOpponents().get(0)
                    .getCreaturesInPlay().get(0));
        });
        assertAlike(play, false, live -> assertTrue(
                RecordEvents.eventTypes(live).contains("unattached"), live));
    }

    /**
     * Silumgar's Command with the opponent's Phantasmal Bear the one
     * permanent out: its one legal pair of modes, return it and -3/-3, both
     * at the bear, which its own trigger sacrifices before the Command
     * resolves. The Command fizzles, and each chosen mode writes its own
     * empty half, the card's move off the stack in the last.
     */
    @Test
    void aFizzledCharm() throws Throwable {
        Play play = new Play(declining -> {
            Game game = game(declining);
            TestBoard.inPlay(game, game.getPlayers().get(1), "Phantasmal Bear");
            TestBoard.inHand(game, game.getPlayers().get(0), "Silumgar's Command");
            return game;
        }, "Silumgar's Command", spell -> {
            Card bear = spell.getActivatingPlayer().getOpponents().get(0).getCreaturesInPlay().get(0);
            for (AbilitySub mode : TestBoard.chain(spell, 1, 2)) {
                mode.resetTargets();
                mode.getTargets().add(bear);
            }
        });
        assertEquals(2, liveHalves(play, false).size(), "one half per chosen mode");
        assertAlike(play, false, live -> assertEquals("fizzled", RecordAbilities.outcomeOf(live), live));
    }

    /**
     * Sundial of the Infinite ending the turn of a player holding eight
     * cards: the cleanup step it skips to discards one, which is the step's
     * and not the half's.
     */
    @Test
    void anEndTurn() throws Throwable {
        Play play = new Play(declining -> {
            Game game = game(declining);
            Player caster = game.getPlayers().get(0);
            TestBoard.inPlay(game, caster, "Sundial of the Infinite");
            for (int i = 0; i < 8; i++) {
                TestBoard.inHand(game, caster, "Shock");
            }
            return game;
        }, "Sundial of the Infinite", game -> abilityOf(game, ApiType.EndTurn), sundial -> { });
        assertAlike(play, false, live -> assertTrue(RecordEvents.events(live).stream()
                .noneMatch(event -> "hand".equals(paramsOf(event).get("from_zone"))), live));
    }

    /**
     * Chain Lightning killing the opponent's Grizzly Bears, the one target
     * two Ivory Masks leave: the opponent pays {R}{R} and copies it as it
     * finishes, and the bears' death in the state-based check after is still
     * the spell's.
     */
    @Test
    void aSelfCopyingSpell() throws Throwable {
        Play play = new Play(declining -> {
            Game game = game(declining);
            Player caster = game.getPlayers().get(0);
            Player opponent = game.getPlayers().get(1);
            TestBoard.inPlay(game, caster, "Ivory Mask");
            TestBoard.inPlay(game, opponent, "Ivory Mask");
            TestBoard.inPlay(game, opponent, "Mountain");
            TestBoard.inPlay(game, opponent, "Mountain");
            TestBoard.inPlay(game, opponent, "Grizzly Bears");
            TestBoard.inHand(game, caster, "Chain Lightning");
            game.getAction().checkStaticAbilities();
            return game;
        }, "Chain Lightning", aimed -> {
            aimed.resetTargets();
            aimed.getTargets().add(aimed.getActivatingPlayer().getOpponents().get(0)
                    .getCreaturesInPlay().get(0));
        });
        assertAlike(play, false, live -> assertTrue(RecordEvents.events(live).stream()
                .anyMatch(event -> "battlefield".equals(paramsOf(event).get("from_zone"))
                        && "graveyard".equals(paramsOf(event).get("to_zone"))), live));
    }

    /**
     * Wild Might on the first player's Grizzly Bears, the one creature out:
     * the opponent taps a Forest and pays {2} to deny the +4/+4. The mana
     * abilities it activated are its payment's, in no effect half.
     */
    @Test
    void anUnlessCostPaidWithMana() throws Throwable {
        Play play = new Play(declining -> {
            Game game = game(declining);
            Player caster = game.getPlayers().get(0);
            Player opponent = game.getPlayers().get(1);
            TestBoard.inPlay(game, caster, "Grizzly Bears");
            TestBoard.inPlay(game, opponent, "Forest");
            TestBoard.inPlay(game, opponent, "Forest");
            TestBoard.inHand(game, caster, "Wild Might");
            return game;
        }, "Wild Might", aimed -> {
            aimed.resetTargets();
            aimed.getTargets().add(aimed.getActivatingPlayer().getCreaturesInPlay().get(0));
        });
        assertAlike(play, false, live -> assertTrue(
                !RecordEvents.eventTypes(live).contains("tapped"), live));
    }

    /** An event's parameters, none when it has none. */
    private static Map<String, Object> paramsOf(Map<String, Object> event) {
        Map<String, Object> params = RecordLines.asMap(event.get("params"));
        return params == null ? Map.of() : params;
    }

    /** A check that a record's events are {@code expected}: each one's type and {@code attributed_to}, in order. */
    private static Consumer<String> linksAre(List<List<Object>> expected) {
        return record -> assertEquals(expected, RecordEvents.events(record).stream()
                .map(event -> List.of(event.get("type"), event.get("attributed_to"))).toList(), record);
    }

    /** A check that the record's {@code pt_change} is permanent. */
    private static Consumer<String> ptChangeIsPermanent() {
        return record -> assertEquals("permanent", RecordEvents.events(record).stream()
                .filter(event -> "pt_change".equals(event.get("type")))
                .map(event -> event.get("duration")).findFirst().orElseThrow(), record);
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
