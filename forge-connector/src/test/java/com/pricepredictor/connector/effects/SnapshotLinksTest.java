package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.card.CardCollection;
import forge.game.card.CardFactory;
import forge.game.combat.Combat;
import forge.game.phase.PhaseType;
import forge.game.spellability.SpellAbility;
import forge.game.staticability.StaticAbility;
import forge.game.trigger.Trigger;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A snapshot's {@code links} (FR-026, FR-050, T059): one case per family of
 * root § 10.3 the collector writes, each on a board set up the way the
 * engine leaves it, and the remembered-list rule of C8.
 *
 * <p>Each case reads the rendered state, so what is checked is what a record
 * carries: the family's spelling, the forward direction, and an ability's
 * link ending at that ability's line.
 */
@ExtendWith(ForgeExtension.class)
class SnapshotLinksTest {

    @TempDir
    Path dir;

    // ── attachment and combat ───────────────────────────────────────────

    @Test
    void anAuraIsAttachedToItsCreatureAndACurseToItsPlayer() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.opponent);
        Card pacifism = board.battlefield("Pacifism", board.you);
        pacifism.attachToEntity(bears, null);
        Card curse = board.battlefield("Curse of Death's Hold", board.you);
        curse.attachToEntity(board.opponent, null);

        List<Map<String, Object>> links = links(board, LinkFamily.ATTACHED_TO);

        assertTrue(has(links, new Ends(Refs.of(pacifism), Refs.of(bears))), links.toString());
        assertTrue(has(links, new Ends(Refs.of(curse), Refs.of(board.opponent))), links.toString());
        assertEquals(2, links.size(), links.toString());
    }

    @Test
    void anAttackerAttacksItsPlayerAndItsBlockerBlocksIt() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.COMBAT_DECLARE_BLOCKERS);
        Card bears = TestBoard.inPlay(board.game, board.you, "Grizzly Bears");
        Card giant = TestBoard.inPlay(board.game, board.opponent, "Hill Giant");
        TestBoard.blockedCombat(new TestBoard.CombatSides(board.game, board.you, board.opponent), bears, giant);

        assertEquals(List.of(link(bears, board.opponent, LinkFamily.ATTACKS)), links(board, LinkFamily.ATTACKS));
        assertEquals(List.of(link(giant, bears, LinkFamily.BLOCKS)), links(board, LinkFamily.BLOCKS));
    }

    /** A band is one link per pair of its attackers, from the lower id. */
    @Test
    void twoAttackersOfOneBandAreInBandWithEachOther() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.COMBAT_DECLARE_ATTACKERS);
        Card first = TestBoard.inPlay(board.game, board.you, "Grizzly Bears");
        Card second = TestBoard.inPlay(board.game, board.you, "Hill Giant");
        Card alone = TestBoard.inPlay(board.game, board.you, "Serra Angel");
        Combat combat = new Combat(board.you);
        combat.addAttacker(first, board.opponent);
        combat.addAttacker(second, board.opponent, combat.getBandOfAttacker(first));
        combat.addAttacker(alone, board.opponent);
        board.game.getPhaseHandler().setCombat(combat);

        assertEquals(List.of(link(first, second, LinkFamily.IN_BAND_WITH)), links(board, LinkFamily.IN_BAND_WITH));
    }

    @Test
    void aLuredBlockerMustBlockAndAGoadedCreatureIsGoadedByItsPlayer() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card lure = board.battlefield("Grizzly Bears", board.you);
        Card blocker = board.battlefield("Hill Giant", board.opponent);
        blocker.addMustBlockCard(board.game.getNextTimestamp(), lure);
        Card goaded = board.battlefield("Serra Angel", board.opponent);
        goaded.addGoad(board.game.getNextTimestamp(), board.you);

        assertEquals(List.of(link(blocker, lure, LinkFamily.MUST_BLOCK)), links(board, LinkFamily.MUST_BLOCK));
        assertEquals(List.of(link(goaded, board.you, LinkFamily.GOADED_BY)), links(board, LinkFamily.GOADED_BY));
    }

    // ── the stack ───────────────────────────────────────────────────────

    /** A spell with two targets targets each from its line, by index; its card is its source. */
    @Test
    void aSpellWithTwoTargetsTargetsEachFromItsLineByIndex() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.opponent);
        Card onStack = board.game.getAction().moveToStack(
                board.hand("Electrolyze", board.you), null);
        SpellAbility spell = TestBoard.spellOf(onStack, board.you);
        spell.resetTargets();
        spell.getTargets().add(bears);
        spell.getTargets().add(board.opponent);
        board.game.getStack().add(spell);

        List<Map<String, Object>> targets = links(board, LinkFamily.TARGETS);

        Object line = end(SnapshotLinks.line(onStack, spell));
        assertEquals("spell", traitKindOf(line));
        assertEquals(List.of(
                withIndex(link(line, Refs.of(bears), LinkFamily.TARGETS), 0),
                withIndex(link(line, Refs.of(board.opponent), LinkFamily.TARGETS), 1)), targets);
        assertEquals(List.of(link(Refs.of(onStack), line, LinkFamily.SOURCE_OF)), links(board, LinkFamily.SOURCE_OF));
    }

    /** A dies trigger on the stack is triggered by the creature that died. */
    @Test
    void aTriggerOnTheStackIsTriggeredByWhatItIsAbout() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card dissenter = board.battlefield("Doomed Dissenter", board.you);
        board.game.getTriggerHandler().resetActiveTriggers(false, null);
        board.game.copyLastState();
        dissenter.setDamage(5);
        board.game.getAction().checkStateEffects(false);
        assertTrue(board.game.getStack().addAllTriggeredAbilitiesToStack(), "the dies trigger");

        List<Map<String, Object>> triggered = links(board, LinkFamily.TRIGGERED_BY);

        assertTrue(triggered.stream().anyMatch(l -> Refs.of(dissenter).equals(l.get("to"))
                && "trigger".equals(traitKindOf(l.get("from")))), triggered.toString());
    }

    // ── statics and floating effects ────────────────────────────────────

    /** An anthem's static affects your creatures, from its line; a continuous board says nothing. */
    @Test
    void anAnthemAffectsYourCreaturesFromItsLine() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.you);
        board.battlefield("Hill Giant", board.opponent);
        Card anthem = board.battlefield("Glorious Anthem", board.you);
        board.game.getAction().checkStaticAbilities();
        StaticAbility stat = anthem.getStaticAbilities().get(0);

        assertEquals(List.of(link(end(SnapshotLinks.line(anthem, stat)), Refs.of(bears), LinkFamily.AFFECTS)),
                links(board, LinkFamily.AFFECTS));
        assertEquals(List.of(), linksIn(
                new SnapshotBuilder(board.game).ofBoardWithout(stat).render(),
                LinkFamily.AFFECTS), "the continuous record's answer");
    }

    /** Exploration's static affects the player whose land plays it raises, and no card. */
    @Test
    void explorationAffectsItsControllerFromItsLine() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        board.battlefield("Grizzly Bears", board.you);
        Card exploration = board.battlefield("Exploration", board.you);
        board.game.getAction().checkStaticAbilities();
        StaticAbility stat = exploration.getStaticAbilities().get(0);

        assertEquals(List.of(link(end(SnapshotLinks.line(exploration, stat)), board.you,
                LinkFamily.AFFECTS)), links(board, LinkFamily.AFFECTS));
    }

    /** Explore's effect card affects its caster, whose land plays it raises. */
    @Test
    void exploresEffectCardAffectsItsCaster() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        board.game.getAction().moveToLibrary(board.card("Forest", board.you), null, null);
        Card explore = board.hand("Explore", board.you);
        board.resolve(board.spell(explore, board.you), board.you);
        board.game.getAction().checkStaticAbilities();
        Card effect = commandCard(board, card -> FloatingEffects.isEffectCard(card));

        List<Map<String, Object>> affects = links(board, LinkFamily.AFFECTS);
        assertEquals(1, affects.size(), affects.toString());
        assertEquals(Refs.of(effect), entityOf(affects.get(0).get("from")));
        assertEquals(Refs.of(board.you), affects.get(0).get("to"));
    }

    /**
     * Falter's effect card is created by Falter's line and affects the
     * creatures that cannot block; it is a command-zone entity.
     */
    @Test
    void anEffectCardIsCreatedByTheLineThatMadeIt() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.opponent);
        Card falter = board.hand("Falter", board.you);
        SpellAbility spell = board.spell(falter, board.you);
        board.resolve(spell, board.you);
        board.game.getAction().checkStaticAbilities();
        Card effect = commandCard(board, card -> FloatingEffects.isEffectCard(card));

        assertEquals(List.of(link(Refs.of(effect), end(SnapshotLinks.line(falter, spell)),
                LinkFamily.CREATED_BY)), links(board, LinkFamily.CREATED_BY));
        assertTrue(links(board, LinkFamily.AFFECTS).stream().anyMatch(l -> Refs.of(bears).equals(l.get("to"))
                && Refs.of(effect).equals(entityOf(l.get("from")))), links(board, LinkFamily.AFFECTS).toString());
    }

    /**
     * Kick in the Door: its haste is a timestamped floating effect affecting
     * the creature and created by the spell's line, and its venture puts its
     * caster in a dungeon.
     */
    @Test
    void aTimestampedEffectAffectsItsCreatureAndADungeonHoldsItsPlayer() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.you);
        Card kick = board.hand("Kick in the Door", board.you);

        resolveWithHook(board, kick, bears);

        List<Map<String, Object>> affects = links(board, LinkFamily.AFFECTS);
        Map<String, Object> haste = affects.stream()
                .filter(l -> l.get("from") instanceof String id && id.startsWith("F"))
                .findFirst().orElseThrow(() -> new AssertionError(affects.toString()));
        assertEquals(Refs.of(bears), haste.get("to"));
        Map<String, Object> made = links(board, LinkFamily.CREATED_BY).stream()
                .filter(l -> haste.get("from").equals(l.get("from"))).findFirst().orElseThrow();
        assertEquals("cardsfolder/k/kick_in_the_door.txt", keyOf(made.get("to")).get("script_file"));
        Card dungeon = commandCard(board, card -> card.getType().isDungeon());
        assertEquals(List.of(link(dungeon, board.you, LinkFamily.IN_DUNGEON)), links(board, LinkFamily.IN_DUNGEON));
    }

    /** Feral Lightning's delayed exile is pending on its tokens and created by the spell's line. */
    @Test
    void aDelayedTriggerIsPendingOnTheObjectsItHolds() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Assumptions.assumeTrue(PatchHooks.PENDING_DELAYED_TRIGGERS.present(),
                "../forge has no pending-delayed-trigger accessor");
        Card feral = board.hand("Feral Lightning", board.you);
        board.resolve(board.spell(feral, board.you), board.you);

        List<Map<String, Object>> pending = links(board, LinkFamily.PENDING_ON);

        assertEquals(3, pending.size(), pending.toString());
        for (Card token : board.game.getCardsIn(ZoneType.Battlefield)) {
            assertTrue(pending.stream().anyMatch(l -> Refs.of(token).equals(l.get("to"))),
                    pending.toString());
        }
        Object delayed = pending.get(0).get("from");
        assertTrue(((String) delayed).startsWith("D"), pending.toString());
        Map<String, Object> made = links(board, LinkFamily.CREATED_BY).stream()
                .filter(l -> delayed.equals(l.get("from"))).findFirst().orElseThrow();
        assertEquals("cardsfolder/f/feral_lightning.txt", keyOf(made.get("to")).get("script_file"));
    }

    // ── rule 607, cipher, control, copies ───────────────────────────────

    /** Oblivion Ring links the card it exiled from the line that exiled it. */
    @Test
    void oblivionRingLinksItsExiledCardFromTheLineThatExiledIt() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card ring = board.battlefield("Oblivion Ring", board.you);
        Card exiled = board.game.getAction().exile(
                board.battlefield("Hill Giant", board.opponent), null, null);
        SpellAbility exiling = exileTrigger(ring);
        exiled.setExiledWith(ring);
        exiled.setExiledSA(exiling);
        ring.addExiledCard(exiled);

        Object line = end(SnapshotLinks.line(ring, exiling));
        assertEquals("trigger", traitKindOf(line));
        assertEquals(List.of(link(line, Refs.of(exiled), LinkFamily.LINKED_BY_TEXT)),
                links(board, LinkFamily.LINKED_BY_TEXT));
    }

    @Test
    void anImprintedCardIsLinkedFromItsCardWhereForgeKeepsNoAbility() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card chrome = board.battlefield("Chrome Mox", board.you);
        Card imprinted = board.game.getAction().exile(board.hand("Grizzly Bears", board.you),
                null, null);
        chrome.addImprintedCard(imprinted);

        assertEquals(List.of(link(chrome, imprinted, LinkFamily.LINKED_BY_TEXT)),
                links(board, LinkFamily.LINKED_BY_TEXT));
    }

    /** C8: a lasting remembered card is a link; while the card's ability is on the stack it is not. */
    @Test
    void aRememberedEntryIsALinkOnlyWhileNoAbilityOfItsCardIsOnTheStack() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card sorcerer = board.battlefield("Prodigal Sorcerer", board.you);
        sorcerer.setSickness(false);
        Card giant = board.battlefield("Hill Giant", board.opponent);
        sorcerer.addRemembered(giant);
        sorcerer.addRemembered("a chosen word");

        assertEquals(List.of(link(sorcerer, giant, LinkFamily.LINKED_BY_TEXT)),
                links(board, LinkFamily.LINKED_BY_TEXT));

        SpellAbility ping = board.ability(sorcerer, ApiType.DealDamage);
        ping.setActivatingPlayer(board.you);
        ping.resetTargets();
        ping.getTargets().add(board.opponent);
        board.game.getStack().add(ping);

        assertEquals(List.of(), links(board, LinkFamily.LINKED_BY_TEXT), "a transient entry");
    }

    @Test
    void aCipheredCardIsBoundToItsCreature() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card creature = board.battlefield("Grizzly Bears", board.you);
        Card encoded = board.game.getAction().exile(board.hand("Hands of Binding", board.you),
                null, null);
        creature.addEncodedCard(encoded);
        encoded.setEncodingCard(creature);

        assertEquals(List.of(link(encoded, creature, LinkFamily.BOUND_TO)), links(board, LinkFamily.BOUND_TO));
    }

    /** Control Magic: attached, its static affects, its controller controls, the owner still owns. */
    @Test
    void controlMagicGivesControlOfTheCreatureItEnchants() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card giant = board.battlefield("Hill Giant", board.opponent);
        Card magic = board.battlefield("Control Magic", board.you);
        magic.attachToEntity(giant, null);
        board.game.getAction().checkStaticAbilities();

        assertTrue(links(board, LinkFamily.CONTROLS).contains(link(board.you, giant, LinkFamily.CONTROLS)),
                links(board, LinkFamily.CONTROLS).toString());
        assertTrue(links(board, LinkFamily.OWNS).contains(link(board.opponent, giant, LinkFamily.OWNS)));
        assertTrue(links(board, LinkFamily.ATTACHED_TO).contains(link(magic, giant, LinkFamily.ATTACHED_TO)));
        StaticAbility gainControl = magic.getStaticAbilities().stream()
                .filter(stat -> stat.hasParam("GainControl")).findFirst().orElseThrow();
        assertEquals(List.of(link(end(SnapshotLinks.line(magic, gainControl)),
                Refs.of(giant), LinkFamily.AFFECTS)), links(board, LinkFamily.AFFECTS));
    }

    /** A "for as long as" control effect's card is what granted control. */
    @Test
    void aGainedControlTargetIsControlGrantedByItsCard() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card old = board.battlefield("Old Man of the Sea", board.you);
        Card giant = board.battlefield("Hill Giant", board.opponent);
        old.addGainControlTarget(giant);

        assertEquals(List.of(link(giant, old, LinkFamily.CONTROL_GRANTED_BY)),
                links(board, LinkFamily.CONTROL_GRANTED_BY));
    }

    /**
     * A token that left the game is no end: Forge keeps it in the control
     * card's gain-control list and in the damage source's history, but it
     * describes nothing on the board, and a game copy, which holds no such
     * card, cannot write it (the fork fidelity audit's control_granted_by
     * links only in the live board).
     */
    @Test
    void noLinkEndsAtACardThatLeftTheGame() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN2);
        Card old = board.battlefield("Old Man of the Sea", board.you);
        Card giant = board.battlefield("Hill Giant", board.opponent);
        Card soldier = board.token("w_1_1_soldier", board.opponent);
        old.addGainControlTarget(giant);
        old.addGainControlTarget(soldier);
        old.getDamageHistory().registerDamage(2, false, old, soldier, new HashMap<>());
        soldier.getZone().remove(soldier);

        assertEquals(List.of(link(giant, old, LinkFamily.CONTROL_GRANTED_BY)),
                links(board, LinkFamily.CONTROL_GRANTED_BY));
        assertEquals(List.of(), links(board, LinkFamily.DAMAGED_THIS_TURN));
    }

    @Test
    void aCloneIsACopyOfWhatItCopied() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card angel = board.battlefield("Serra Angel", board.opponent);
        Card clone = board.battlefield("Clone", board.you);
        clone.addCloneState(CardFactory.getCloneStates(angel, clone, clone.getFirstSpellAbility()),
                board.game.getNextTimestamp());
        clone.updateStateForView();

        assertEquals("Serra Angel", clone.getName());
        assertEquals(List.of(link(clone, angel, LinkFamily.COPY_OF)), links(board, LinkFamily.COPY_OF));
    }

    // ── pairs, choices, this turn's history ─────────────────────────────

    @Test
    void soulbondPairsAreOneLinkFromTheLowerId() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.you);
        Card paladin = board.battlefield("Silverblade Paladin", board.you);
        paladin.setPairedWith(bears);
        bears.setPairedWith(paladin);

        assertEquals(List.of(link(bears, paladin, LinkFamily.PAIRED_WITH)), links(board, LinkFamily.PAIRED_WITH));
    }

    @Test
    void aChosenCreatureAndAChosenPlayerAreChosenByTheirCard() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card doll = board.battlefield("Stuffy Doll", board.you);
        doll.setChosenPlayer(board.opponent);
        Card bears = board.battlefield("Grizzly Bears", board.opponent);
        Card choosing = board.battlefield("Hill Giant", board.you);
        choosing.setChosenCards(new CardCollection(bears));

        List<Map<String, Object>> chose = links(board, LinkFamily.CHOSE);

        assertTrue(chose.contains(link(doll, board.opponent, LinkFamily.CHOSE)), chose.toString());
        assertTrue(chose.contains(link(choosing, bears, LinkFamily.CHOSE)), chose.toString());
        assertEquals(2, chose.size(), chose.toString());
    }

    /** Sengir Vampire damaged a creature this turn; a block and a crew are this turn's too. */
    @Test
    void thisTurnsDamageBlocksAndCrewsAreLinks() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN2);
        Card sengir = board.battlefield("Sengir Vampire", board.you);
        Card bears = board.battlefield("Grizzly Bears", board.opponent);
        sengir.getDamageHistory().registerDamage(2, true, sengir, bears, new HashMap<>());
        bears.addBlockedThisTurn(sengir);
        sengir.addBlockedByThisTurn(bears);
        Card vehicle = board.battlefield("Smuggler's Copter", board.you);
        Card crew = board.battlefield("Hill Giant", board.you);
        vehicle.addCrewedByThisTurn(new CardCollection(crew));

        assertEquals(List.of(link(sengir, bears, LinkFamily.DAMAGED_THIS_TURN)),
                links(board, LinkFamily.DAMAGED_THIS_TURN));
        assertEquals(List.of(link(bears, sengir, LinkFamily.FOUGHT_THIS_TURN)),
                links(board, LinkFamily.FOUGHT_THIS_TURN));
        assertEquals(List.of(link(vehicle, crew, LinkFamily.CREWED_BY)), links(board, LinkFamily.CREWED_BY));
    }

    // ── control, ownership, the pending event ───────────────────────────

    @Test
    void everyCardIsControlledAndOwned() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.you);
        Card bolt = board.hand("Lightning Bolt", board.opponent);

        assertEquals(List.of(link(board.you, bears, LinkFamily.CONTROLS), link(board.opponent, bolt, LinkFamily.CONTROLS)),
                links(board, LinkFamily.CONTROLS));
        assertEquals(List.of(link(board.you, bears, LinkFamily.OWNS), link(board.opponent, bolt, LinkFamily.OWNS)),
                links(board, LinkFamily.OWNS));
    }

    @Test
    void thePendingSubjectsLastKnownCopyIsItsLastKnownState() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card bears = board.battlefield("Grizzly Bears", board.you);
        Card lastKnown = forge.game.card.CardCopyService.getLKICopy(bears);
        Card dead = board.game.getAction().moveTo(ZoneType.Graveyard, bears, null, null);
        EffectEvent died = new EffectEvent(EventType.ZONE_CHANGE)
                .subject(Refs.of(dead)).lastKnown(lastKnown);

        assertEquals(List.of(link("L" + bears.getId(), Refs.of(dead), LinkFamily.LAST_KNOWN_STATE_OF)),
                linksIn(new SnapshotBuilder(board.game).ofTrait(null, died).render(),
                        LinkFamily.LAST_KNOWN_STATE_OF));
    }

    /** Families the reader builds from its own layout are never written. */
    @Test
    void theReadersOwnFamiliesAreNeverWritten() {
        CharacterizationBoard board = CharacterizationBoard.full(PhaseType.MAIN1);

        List<Map<String, Object>> all = linksIn(new SnapshotBuilder(board.game).ofBoard().render(), null);

        assertFalse(all.isEmpty());
        for (LinkFamily family : LinkFamily.values()) {
            if (!family.collected()) {
                assertTrue(all.stream().noneMatch(l -> family.wireName().equals(l.get("type"))),
                        family.wireName());
            }
        }
    }

    // ── helpers ─────────────────────────────────────────────────────────

    private static List<Map<String, Object>> links(CharacterizationBoard board, LinkFamily type) {
        return linksIn(new SnapshotBuilder(board.game).ofBoard().render(), type);
    }

    /** The state's links of one type, in order; every link for a null type. */
    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> linksIn(String state, LinkFamily type) {
        List<Map<String, Object>> all = (List<Map<String, Object>>) RecordLines.parse(state).get("links");
        return all.stream().filter(l -> type == null || type.wireName().equals(l.get("type"))).toList();
    }

    private static Map<String, Object> link(Object from, Object to, LinkFamily type) {
        return Map.of("from", ref(from), "to", ref(to), "type", type.wireName());
    }

    private static Map<String, Object> withIndex(Map<String, Object> link, long index) {
        Map<String, Object> indexed = new HashMap<>(link);
        indexed.put("index", index);
        return indexed;
    }

    private static Object ref(Object end) {
        if (end instanceof Card card) {
            return Refs.of(card);
        }
        if (end instanceof forge.game.player.Player player) {
            return Refs.of(player);
        }
        return end;
    }

    /** The refs a link runs from and to. */
    private record Ends(String from, String to) {
    }

    private static boolean has(List<Map<String, Object>> links, Ends ends) {
        return links.stream()
                .anyMatch(l -> ends.from().equals(l.get("from")) && ends.to().equals(l.get("to")));
    }

    /** One end as the parsed state holds it: a ref string, or a line's map. */
    private static Object end(String json) {
        return json.startsWith("{") ? RecordLines.parse(json) : json.substring(1, json.length() - 1);
    }

    private static Map<String, Object> keyOf(Object lineEnd) {
        return RecordLines.asMap(RecordLines.asMap(lineEnd).get("key"));
    }

    private static Object traitKindOf(Object lineEnd) {
        return lineEnd instanceof Map<?, ?> ? keyOf(lineEnd).get("trait_kind") : null;
    }

    private static Object entityOf(Object end) {
        return end instanceof Map<?, ?> line ? line.get("entity") : end;
    }

    private static Card commandCard(CharacterizationBoard board,
                                    java.util.function.Predicate<Card> which) {
        return board.game.getCardsIn(ZoneType.Command).stream().filter(which).findFirst()
                .orElseThrow(() -> new AssertionError(board.game.getCardsIn(ZoneType.Command)));
    }

    /** Oblivion Ring's enters trigger's exile. */
    private static SpellAbility exileTrigger(Card ring) {
        for (Trigger trigger : ring.getTriggers()) {
            SpellAbility ability = trigger.ensureAbility();
            if (ability != null && ability.getApi() == ApiType.ChangeZone) {
                return ability;
            }
        }
        throw new AssertionError("no exile trigger on " + ring);
    }

    /**
     * Cast a card's spell at {@code target} and resolve it with every patched
     * collector installed, as {@code FloatingEffectsTest} does: a timestamped
     * floating effect names its line only through the clause hook's note.
     */
    private void resolveWithHook(CharacterizationBoard board, Card card, Card target) {
        Assumptions.assumeTrue(
                AttributionMode.detect() == AttributionMode.PATCHED
                        && PatchHooks.CLAUSE_LISTENER.present(),
                "../forge has no clause hook");
        Game game = board.game;
        RecordShardWriter writer = ShardRun.writer(dir);
        try (PatchedCollectors collectors = new PatchedCollectors(
                game, new Channels.Setup(TestCollectors.context(writer, ShardRun.GAME_ID),
                TestCaps.caps().interventions(0).probes(0).build(), 1L,
                TestCollectors.bracket(game, writer, ShardRun.GAME_ID)))) {
            collectors.install();
            Card onStack = game.getAction().moveToStack(card, null);
            SpellAbility spell = TestBoard.spellOf(onStack, board.you);
            spell.resetTargets();
            spell.getTargets().add(target);
            game.getStack().add(spell);
            game.getStack().resolveStack();
            game.getAction().checkStateEffects(false);
            game.getAction().checkStaticAbilities();
        } finally {
            writer.close();
        }
    }
}
