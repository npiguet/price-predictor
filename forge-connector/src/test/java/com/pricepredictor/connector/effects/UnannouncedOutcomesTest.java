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
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

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
                    .filter(e -> flag.equals(e.param(EventParam.FLAG)))
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

        assertEquals(0, capture.moveOf(top).param(EventParam.LIBRARY_POSITION));
        assertEquals(-1, capture.moveOf(bottom).param(EventParam.LIBRARY_POSITION));
        assertEquals(1, capture.moveOf(second).param(EventParam.LIBRARY_POSITION));
    }

    /** A move anywhere else names no position. */
    @Test
    void aMoveElsewhereNamesNoPosition() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Card bears = TestBoard.inHand(game, a, "Grizzly Bears");
        Capture capture = listening(game);

        game.getAction().moveTo(ZoneType.Graveyard, bears, null, null);

        assertFalse(capture.moveOf(bears).hasParam(EventParam.LIBRARY_POSITION));
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
        assertEquals(true, blessed.get(0).param(EventParam.VALUE));
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

    /** The event exists, is of {@code type} and names {@code subjects}. */
    private static void assertEventNaming(EffectEvent event, EventType type, List<?> subjects) {
        assertNotNull(event);
        assertEquals(type, event.type());
        assertEquals(subjects, event.subjects());
    }

    /** The event exists and flags a Class level, with {@code level}. */
    private static void assertClassLevelFlagged(EffectEvent event, int level) {
        assertNotNull(event);
        assertEquals("class_level", event.param(EventParam.FLAG));
        assertEquals(level, event.param(EventParam.VALUE));
    }

    /** The animation's {@code ability_change} names {@code den} and the trigger it grants, granted not removed. */
    private static void assertAnimationGrantingItsAttackTrigger(EffectEvent event, Card den) {
        assertEventNaming(event, EventType.ABILITY_CHANGE, List.of(Refs.of(den)));
        assertEquals("TrigAttack", event.param(EventParam.ABILITIES));
        assertEquals(false, event.param(EventParam.REMOVED));
    }

    /** The event is {@code player}'s look at the library named for the search. */
    private static void assertLibrarySearchedBy(EffectEvent event, Player player) {
        assertEventNaming(event, EventType.CARD_LOOKED_AT, List.of(Refs.of(player)));
        assertEquals("SearchedLibrary", event.param(EventParam.MODE));
        assertEquals("library", event.param(EventParam.FROM_ZONE));
    }

    /** The event flags {@code made} as having entered combat attacking. */
    private static void assertEnteredCombatAttacking(EffectEvent event, Card made) {
        assertEventNaming(event, EventType.STATE_FLAG_CHANGE, List.of(Refs.of(made)));
        assertEquals("entered_attacking", event.param(EventParam.FLAG));
    }

    /** The event says {@code attacker} became blocked, by exactly {@code blockers}. */
    private static void assertBecameBlocked(EffectEvent event, Card attacker, List<?> blockers) {
        assertEventNaming(event, EventType.BECAME_BLOCKED, List.of(Refs.of(attacker)));
        assertEquals(blockers, event.param(EventParam.BLOCKERS));
    }

    /** The event flags {@code sorcerer}'s Class level, with the level it now holds. */
    private static void assertClassLevelReached(EffectEvent event, Card sorcerer) {
        assertClassLevelFlagged(event, sorcerer.getClassLevel());
        assertEquals(List.of(Refs.of(sorcerer)), event.subjects());
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

        EffectEvent event = TestCards.onlyEvent(observe(animate));

        assertAnimationGrantingItsAttackTrigger(event, den);
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
        EffectEvent event = TestCards.onlyEvent(observed);

        assertLibrarySearchedBy(event, a);
    }

    /** A zone change that searched nothing and put nobody into combat says nothing. */
    @Test
    void aPlainZoneChangeSaysNothingHere() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Card host = TestBoard.inHand(game, a, "Grizzly Bears");
        SpellAbility bounce = scripted(
                "SP$ ChangeZone | Origin$ Battlefield | Destination$ Hand | Defined$ Self", host, a);

        assertNull(TestCards.onlyEvent(observe(bounce)));
    }

    /**
     * A creature put onto the battlefield attacking entered combat; the
     * declared attacker did not. One case per API the rule is keyed by,
     * each through its own clause, so a key missing from the rule fails here.
     */
    @ParameterizedTest
    @ValueSource(strings = {
            "SP$ Token | TokenScript$ r_1_1_goblin | TokenAttacking$ True",
            "SP$ CopyPermanent | Defined$ Self | TokenAttacking$ True",
            "SP$ ChangeZoneAll | ChangeType$ Creature.YouOwn | Origin$ Hand"
                    + " | Destination$ Battlefield | Attacking$ True"})
    void aCreatureMadeAttackingEnteredCombat(String script) {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Player b = game.getPlayers().get(1);
        Card declared = TestBoard.inPlay(game, a, "Grizzly Bears");
        Combat combat = new Combat(a);
        combat.addAttacker(declared, b);
        game.getPhaseHandler().setCombat(combat);
        Card made = TestBoard.inPlay(game, a, "Hill Giant");
        SpellAbility token = scripted(script, made, a);
        ApiEvents.ClauseObservation<?> observed = observe(token);

        combat.addAttacker(made, b);
        EffectEvent event = TestCards.onlyEvent(observed);

        assertEnteredCombatAttacking(event, made);
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
        EffectEvent event = TestCards.onlyEvent(observed);

        assertBecameBlocked(event, attacker, List.of());
    }

    /** A clause making a creature block an attacker: the attacker became blocked, by it. */
    @Test
    void anAttackerAClauseBlockedNamesItsBlocker() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Player b = game.getPlayers().get(1);
        Card attacker = TestBoard.inPlay(game, a, "Grizzly Bears");
        Card blocker = TestBoard.inPlay(game, b, "Hill Giant");
        Combat combat = new Combat(a);
        combat.addAttacker(attacker, b);
        combat.setBlocked(attacker, false);
        game.getPhaseHandler().setCombat(combat);
        SpellAbility block = scripted(
                "SP$ Block | DefinedAttacker$ Valid Creature.attacking | DefinedBlocker$ Self",
                blocker, b);
        ApiEvents.ClauseObservation<?> observed = observe(block);

        combat.addBlocker(attacker, blocker);
        combat.setBlocked(attacker, true);
        EffectEvent event = TestCards.onlyEvent(observed);

        assertBecameBlocked(event, attacker, List.of(Refs.of(blocker)));
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
        EffectEvent event = TestCards.onlyEvent(observed);

        assertClassLevelReached(event, sorcerer);
    }

    @Test
    void anUnlockedDoorIsAStateFlagOnTheRoom() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Card room = TestBoard.inPlay(game, a, "Bottomless Pool // Locker Room");
        SpellAbility unlock = scripted("AB$ UnlockDoor | Cost$ 0 | Mode$ ThisDoor", room, a);
        ApiEvents.ClauseObservation<?> observed = observe(unlock);

        room.unlockRoom(a, CardStateName.LeftSplit);
        EffectEvent event = TestCards.onlyEvent(observed);

        assertNotNull(event);
        assertEquals("door_unlocked", event.param(EventParam.FLAG));
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
        EffectEvent event = TestCards.onlyEvent(observed);

        assertNotNull(event);
        assertEquals("paired", event.param(EventParam.FLAG));
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
        EffectEvent event = TestCards.onlyEvent(observed);

        assertNotNull(event);
        assertEquals("merged", event.param(EventParam.FLAG));
        assertEquals(List.of(Refs.of(bears)), event.subjects());
    }
}
