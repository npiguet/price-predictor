package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.card.CardStateName;
import forge.game.Game;
import forge.game.ability.AbilityFactory;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.event.GameEventCardChangeZone;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Outcomes no bus event announces as themselves, read as the collector reads
 * them: where in a library a card went, a player's city's blessing and
 * completed dungeons, and the clause outcomes {@link ApiEvents} reads from
 * the state before and after a clause (an animation's cards, a search, a
 * creature put into combat, an attacker made blocked, a Class level, a door,
 * a soulbond pair, a merge).
 */
@ExtendWith(ForgeExtension.class)
class UnannouncedOutcomesTest {

    /** A reader that keeps every outcome and move it is handed. */
    private static final class Capture extends BusEventReader {

        final List<EffectEvent> events = new ArrayList<>();

        Capture(Game game) {
            super(game);
        }

        @Override
        boolean cardDamageIsCombat() {
            return false;
        }

        @Override
        void fileDamage(EffectEvent damage, String namedCause, boolean combat) {
            events.add(damage);
        }

        @Override
        void fileOutcome(EffectEvent event, String namedCause) {
            events.add(event);
        }

        @Override
        void fileMove(GameEventCardChangeZone event, EffectEvent moved) {
            events.add(moved);
        }

        List<EffectEvent> of(EventType type) {
            return events.stream().filter(e -> e.type() == type).toList();
        }

        /** The last move of {@code card}. */
        EffectEvent moveOf(Card card) {
            List<EffectEvent> moves = events.stream()
                    .filter(e -> e.type() == EventType.ZONE_CHANGE
                            && e.subjects().contains(Refs.of(card)))
                    .toList();
            assertFalse(moves.isEmpty(), "no move of " + card + ": " + events);
            return moves.get(moves.size() - 1);
        }

        /** The {@code state_flag_change} events naming {@code flag}. */
        List<EffectEvent> flagged(String flag) {
            return of(EventType.STATE_FLAG_CHANGE).stream()
                    .filter(e -> flag.equals(e.params().get("flag")))
                    .toList();
        }
    }

    private static Capture listening(Game game) {
        Capture capture = new Capture(game);
        game.subscribeToEvents(capture);
        return capture;
    }

    // ── library positions ───────────────────────────────────────────────

    /**
     * A move into a library names where the card went: 0 the top, -1 the
     * bottom, else its index from the top.
     */
    @Test
    void aMoveIntoALibraryNamesItsPosition() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        for (String name : List.of("Forest", "Island", "Mountain")) {
            game.getAction().moveToLibrary(TestBoard.card(game, a, name), null);
        }
        Card top = TestBoard.inHand(game, a, "Grizzly Bears");
        Card bottom = TestBoard.inHand(game, a, "Hill Giant");
        Card second = TestBoard.inHand(game, a, "Shock");
        Capture capture = listening(game);

        game.getAction().moveToLibrary(top, null);
        game.getAction().moveToLibrary(bottom, -1, null);
        game.getAction().moveToLibrary(second, 1, null);

        assertEquals(0, capture.moveOf(top).params().get("library_position"));
        assertEquals(-1, capture.moveOf(bottom).params().get("library_position"));
        assertEquals(1, capture.moveOf(second).params().get("library_position"));
    }

    /** A move anywhere else names no position. */
    @Test
    void aMoveElsewhereNamesNoPosition() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Card bears = TestBoard.inHand(game, a, "Grizzly Bears");
        Capture capture = listening(game);

        game.getAction().moveTo(ZoneType.Graveyard, bears, null, null);

        assertFalse(capture.moveOf(bears).params().containsKey("library_position"));
    }

    // ── a player's command-zone marks ───────────────────────────────────

    @Test
    void theCitysBlessingIsAStateFlagOnThePlayer() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Capture capture = listening(game);

        a.setBlessing(true, null);
        a.setBlessing(true, null);

        List<EffectEvent> blessed = capture.flagged("city_blessing");
        assertEquals(1, blessed.size(), capture.events.toString());
        assertEquals(List.of(Refs.of(a)), blessed.get(0).subjects());
        assertEquals(true, blessed.get(0).params().get("value"));
    }

    @Test
    void aCompletedDungeonIsAStateFlagOnThePlayer() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Card dungeon = TestBoard.card(game, a, "Grizzly Bears");
        game.getAction().moveToCommand(dungeon, null);
        Capture capture = listening(game);

        game.getAction().completeDungeon(a, dungeon);

        List<EffectEvent> completed = capture.flagged("dungeon_completed");
        assertEquals(1, completed.size(), capture.events.toString());
        assertEquals(List.of(Refs.of(a)), completed.get(0).subjects());
    }

    // ── clause outcomes read before and after ───────────────────────────

    private static ApiEvents.ClauseObservation<?> observe(SpellAbility sa) {
        return ApiEvents.observe(sa, ApiEvents::ruleFor);
    }

    private static SpellAbility scripted(String script, Card host, Player activator) {
        SpellAbility sa = AbilityFactory.getAbility(script, host);
        sa.setActivatingPlayer(activator);
        return sa;
    }

    /**
     * Den of the Bugbear animates itself and grants a trigger: the event names
     * the card and the trigger, which {@code Triggers$} grants rather than
     * {@code Abilities$}.
     */
    @Test
    void anAnimationNamesItsCardAndEveryTraitItGrants() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Card den = TestBoard.inPlay(game, a, "Den of the Bugbear");
        SpellAbility animate = TestBoard.ability(den, ApiType.Animate);
        animate.setActivatingPlayer(a);

        EffectEvent event = observe(animate).finish();

        assertNotNull(event);
        assertEquals(EventType.ABILITY_CHANGE, event.type());
        assertEquals(List.of(Refs.of(den)), event.subjects());
        assertEquals("TrigAttack", event.params().get("abilities"));
        assertEquals(false, event.params().get("removed"));
    }

    /** A search of one's own library is a look at it named for the search. */
    @Test
    void aSearchIsALookAtTheLibraryNamedForTheSearch() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Card host = TestBoard.inHand(game, a, "Grizzly Bears");
        SpellAbility search = scripted(
                "SP$ ChangeZone | Origin$ Library | Destination$ Hand | ChangeType$ Card", host, a);
        ApiEvents.ClauseObservation<?> observed = observe(search);

        a.incLibrarySearched();
        EffectEvent event = observed.finish();

        assertNotNull(event);
        assertEquals(EventType.CARD_LOOKED_AT, event.type());
        assertEquals(List.of(Refs.of(a)), event.subjects());
        assertEquals("SearchedLibrary", event.params().get("mode"));
        assertEquals("library", event.params().get("from_zone"));
    }

    /** A zone change that searched nothing and put nobody into combat says nothing. */
    @Test
    void aPlainZoneChangeSaysNothingHere() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Card host = TestBoard.inHand(game, a, "Grizzly Bears");
        SpellAbility bounce = scripted(
                "SP$ ChangeZone | Origin$ Battlefield | Destination$ Hand | Defined$ Self", host, a);

        assertNull(observe(bounce).finish());
    }

    /** A token put onto the battlefield attacking entered combat; the declared attacker did not. */
    @Test
    void aCreatureMadeAttackingEnteredCombat() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Player b = game.getPlayers().get(1);
        Card declared = TestBoard.inPlay(game, a, "Grizzly Bears");
        Combat combat = new Combat(a);
        combat.addAttacker(declared, b);
        game.getPhaseHandler().setCombat(combat);
        Card made = TestBoard.inPlay(game, a, "Hill Giant");
        SpellAbility token = scripted(
                "SP$ Token | TokenScript$ r_1_1_goblin | TokenAttacking$ True", made, a);
        ApiEvents.ClauseObservation<?> observed = observe(token);

        combat.addAttacker(made, b);
        EffectEvent event = observed.finish();

        assertNotNull(event);
        assertEquals(EventType.STATE_FLAG_CHANGE, event.type());
        assertEquals("entered_attacking", event.params().get("flag"));
        assertEquals(List.of(Refs.of(made)), event.subjects());
    }

    /** An attacker a clause made blocked became blocked, by no blocker. */
    @Test
    void anAttackerMadeBlockedBecameBlocked() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Player b = game.getPlayers().get(1);
        Card attacker = TestBoard.inPlay(game, a, "Grizzly Bears");
        Combat combat = new Combat(a);
        combat.addAttacker(attacker, b);
        combat.setBlocked(attacker, false);
        game.getPhaseHandler().setCombat(combat);
        Card host = TestBoard.inHand(game, b, "Choking Vines");
        SpellAbility vines = scripted(
                "SP$ BecomesBlocked | Defined$ Valid Creature.attacking", host, b);
        ApiEvents.ClauseObservation<?> observed = observe(vines);

        combat.setBlocked(attacker, true);
        EffectEvent event = observed.finish();

        assertNotNull(event);
        assertEquals(EventType.BECAME_BLOCKED, event.type());
        assertEquals(List.of(Refs.of(attacker)), event.subjects());
        assertEquals(List.of(), event.params().get("blockers"));
    }

    @Test
    void aClassLevelGainedNamesTheLevelReached() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Card sorcerer = TestBoard.inPlay(game, a, "Sorcerer Class");
        SpellAbility levelUp = TestBoard.ability(sorcerer, ApiType.ClassLevelUp);
        levelUp.setActivatingPlayer(a);
        ApiEvents.ClauseObservation<?> observed = observe(levelUp);

        sorcerer.setClassLevel(sorcerer.getClassLevel() + 1);
        EffectEvent event = observed.finish();

        assertNotNull(event);
        assertEquals("class_level", event.params().get("flag"));
        assertEquals(sorcerer.getClassLevel(), event.params().get("value"));
        assertEquals(List.of(Refs.of(sorcerer)), event.subjects());
    }

    @Test
    void anUnlockedDoorIsAStateFlagOnTheRoom() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Card room = TestBoard.inPlay(game, a, "Bottomless Pool // Locker Room");
        SpellAbility unlock = scripted("AB$ UnlockDoor | Cost$ 0 | Mode$ ThisDoor", room, a);
        ApiEvents.ClauseObservation<?> observed = observe(unlock);

        room.unlockRoom(a, CardStateName.LeftSplit);
        EffectEvent event = observed.finish();

        assertNotNull(event);
        assertEquals("door_unlocked", event.params().get("flag"));
        assertEquals(List.of(Refs.of(room)), event.subjects());
    }

    @Test
    void aSoulbondPairNamesBothCreatures() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Card paladin = TestBoard.inPlay(game, a, "Silverblade Paladin");
        Card bears = TestBoard.inPlay(game, a, "Grizzly Bears");
        SpellAbility bond = scripted("DB$ Bond | Defined$ Self | ValidCards$ Creature.YouCtrl+Other",
                paladin, a);
        ApiEvents.ClauseObservation<?> observed = observe(bond);

        paladin.setPairedWith(bears);
        bears.setPairedWith(paladin);
        EffectEvent event = observed.finish();

        assertNotNull(event);
        assertEquals("paired", event.params().get("flag"));
        assertEquals(List.of(Refs.of(paladin), Refs.of(bears)).stream().sorted().toList(),
                event.subjects().stream().sorted().toList());
    }

    @Test
    void aMergeNamesThePermanentMergedInto() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Card bears = TestBoard.inPlay(game, a, "Grizzly Bears");
        Card mutating = TestBoard.inHand(game, a, "Archipelagore");
        SpellAbility mutate = scripted("SP$ Mutate | Cost$ 0", mutating, a);
        ApiEvents.ClauseObservation<?> observed = observe(mutate);

        mutating.setMergedToCard(bears);
        EffectEvent event = observed.finish();

        assertNotNull(event);
        assertEquals("merged", event.params().get("flag"));
        assertEquals(List.of(Refs.of(bears)), event.subjects());
    }

    /** Each new rule's API is a real {@code ApiType} member. */
    @Test
    void theNewRulesAreKeyedByRealApis() {
        for (String api : List.of("ChangeZone", "ChangeZoneAll", "Token", "CopyPermanent",
                "BecomesBlocked", "Block", "ClassLevelUp", "UnlockDoor", "Bond", "Mutate")) {
            assertTrue(ApiEvents.apiNames().contains(api), api);
            assertNotNull(ApiEvents.ruleFor(ApiType.smartValueOf(api)), api);
        }
    }
}
