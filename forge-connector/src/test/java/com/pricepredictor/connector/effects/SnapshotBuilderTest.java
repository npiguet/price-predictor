package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.card.CardStateName;
import forge.card.MagicColor;
import forge.game.ability.AbilityKey;
import forge.game.card.Card;
import forge.game.card.CardCopyService;
import forge.game.card.CounterEnumType;
import forge.game.mana.Mana;
import forge.game.phase.PhaseType;
import forge.game.spellability.OptionalCost;
import forge.game.spellability.SpellAbility;
import forge.game.staticability.StaticAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The snapshot properties that a running game cannot check for you.
 *
 * <p>The tier vector is the one a corpus cannot show to be wrong: chosen per
 * call site, it would make {@code state.tiers} a perfect proxy for a
 * collection flag the schema bars from being a model input. Beside it, the
 * state's block order and the
 * resolution-time refs reading that replaces the cast-time block and nothing
 * else.
 *
 * <p>None of those needs a game: the tier vector is a constant, and the
 * blocks are assembled from already-rendered parts. The referenced tier,
 * pinned last, does: a target is a card in some zone.
 */
class SnapshotBuilderTest {

    // ── refs ────────────────────────────────────────────────────────────

    /**
     * A record with no acting spell ability still renders the whole refs shape.
     *
     * <p>Trigger, rewrite, playability and combat records pass no ability, and a
     * reader that had to tell "no refs block" from "an empty one" would be
     * reading collection wiring rather than the board.
     */
    @Test
    void refsRenderTheirFullShapeWithNoActingLine() {
        assertEquals(
                "{\"targets\":[],\"source\":null,\"modes\":[],\"x\":null,"
                        + "\"choices\":{},\"random\":null}",
                RefsJson.render(null));
    }

    // ── the state's blocks ──────────────────────────────────────────────

    /**
     * The blocks in schema order, ending in all four tiers: every snapshot
     * declares them, so an empty hand or stack reads as empty rather than as
     * uncollected. The literal is the contract: Python's {@code InclusionTier}
     * and {@code record_io} mirror it.
     */
    @Test
    void theStateBlocksKeepTheOrderTheSchemaFixes() {
        assertEquals(
                "{\"global\":G,\"players\":P,\"entities\":E,\"links\":L,\"refs\":R,"
                        + "\"pending_event\":V,\"tiers\":[1,2,3,4]}",
                new Snapshot("G", "P", "E", "L", "R", "V").render());
    }

    /**
     * A resolution-time refs reading replaces the cast-time block and leaves
     * the board it was captured with alone: re-reading the board at resolution
     * would describe what the ability had already changed.
     */
    @Test
    void laterRefsReplaceOnlyTheRefsBlock() {
        Snapshot castTime = new Snapshot("G", "P", "E", "[]", "{\"choices\":{}}", "null");
        String resolved = "{\"choices\":{\"chosen_color\":\"U\"}}";

        assertEquals(
                new Snapshot("G", "P", "E", "[]", resolved, "null"),
                castTime.withRefs(resolved));
        assertEquals("{\"choices\":{}}", castTime.refs(),
                "the cast-time snapshot is a value; a later reading makes a new one");
    }

    // ── the referenced tier ─────────────────────────────────────────────

    /**
     * What an ability's snapshot carries whatever zone it sits in: its host
     * and the cards it targets.
     *
     * <p>One list for the bracket collector and the fork collector alike: a
     * target this list leaves out and no other tier sweeps, such as an exiled
     * card, is in no tier at all, and the record names a target its state
     * does not hold.
     */
    @Nested
    @ExtendWith(ForgeExtension.class)
    class ReferencedTier {

        @Test
        void anAbilitysTargetsAreReferencedWhereverTheySit() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card host = board.hand("Pull from Eternity", board.you);
            SpellAbility pull = board.spell(host, board.you);
            Card exiled = board.game.getAction().moveTo(
                    ZoneType.Exile, board.card("Grizzly Bears", board.opponent), null, null);
            pull.getTargets().add(exiled);
            pull.getTargets().add(board.opponent);

            String state = new SnapshotBuilder(board.game).ofResolving(pull).render();

            assertTrue(state.contains("{\"id\":\"" + Refs.of(exiled)
                    + "\",\"name\":\"Grizzly Bears\",\"zone\":\"exile\""), state);
        }

        @Test
        void noAbilityReferencesNothing() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            SnapshotBuilder snapshots = new SnapshotBuilder(board.game);

            assertEquals(snapshots.ofBoard().render(), snapshots.ofResolving(null).render());
        }
    }

    // ── a card's lines (FR-021 to FR-023) ───────────────────────────────

    /**
     * Story 2's acceptance 1 to 5 and the face-down case: what a card slot's
     * {@code lines} carries, read off the rendered entity.
     */
    @Nested
    @ExtendWith(ForgeExtension.class)
    class Lines {

        /**
         * Acceptance 1: Guile's static, trigger and replacement are each a
         * line keyed into its own script, and the static carries
         * {@code active}, which holds on the battlefield.
         */
        @Test
        void aCreaturesStaticTriggerAndReplacementAreLinesWithTheirKeys() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card guile = board.battlefield("Guile", board.you);

            List<Map<String, Object>> lines = linesOf(board, guile);

            assertEquals(List.of("trigger", "static", "replacement"), kinds(lines));
            for (Map<String, Object> line : lines) {
                Map<String, Object> key = onlyKey(line);
                assertEquals("cardsfolder/g/guile.txt", key.get("script_file"), line.toString());
                assertEquals(line.get("kind"), key.get("trait_kind"), line.toString());
                assertEquals(0L, key.get("index_within_kind"), line.toString());
                assertEquals(false, line.get("granted"), line.toString());
            }
            assertEquals(true, line(lines, "static").get("active"));
            assertFalse(line(lines, "trigger").containsKey("active"),
                    "active is a static's: " + lines);
        }

        /** A static whose condition does not hold says so: no threshold, no flying. */
        @Test
        void aStaticWhoseConditionFailsIsInactive() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card enforcer = board.battlefield("Mystic Enforcer", board.you);

            assertEquals(false, line(linesOf(board, enforcer), "static").get("active"));
        }

        /**
         * Acceptance 2: under Humility a creature has lost all its abilities,
         * and the live lists Forge plays from carry none; neither does its
         * plain cast, which is no line of its own.
         */
        @Test
        void aCreatureThatLostAllAbilitiesListsNone() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card guile = board.battlefield("Guile", board.you);
            board.battlefield("Humility", board.opponent);
            board.game.getAction().checkStaticAbilities();

            assertEquals(List.of(), linesOf(board, guile));
        }

        /**
         * Acceptance 3: flying from Levitation is a keyword line on the
         * creature, granted, keyed to the static that grants it.
         */
        @Test
        void aKeywordAnotherPermanentsStaticGrantsIsListedGranted() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card bears = board.battlefield("Grizzly Bears", board.you);
            board.battlefield("Levitation", board.you);
            board.game.getAction().checkStaticAbilities();

            Map<String, Object> flying = line(linesOf(board, bears), "keyword");
            assertEquals("Flying", flying.get("fragment"));
            assertEquals(true, flying.get("granted"));
            Map<String, Object> key = onlyKey(flying);
            assertEquals("cardsfolder/l/levitation.txt", key.get("script_file"));
            assertEquals("static", key.get("trait_kind"));
        }

        /**
         * Acceptance 4: ward {2} and annihilator 2 are keyword lines carrying
         * their instance's fragment, value included, and keyed to the keyword
         * the card prints. Annihilator's trigger is that line, not one more.
         */
        @Test
        void wardAndAnnihilatorAreKeywordLinesWithTheirValues() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card spawn = board.battlefield("Aboleth Spawn", board.you);
            Card crusher = board.battlefield("Ulamog's Crusher", board.you);

            Map<String, Object> ward = keywordLine(linesOf(board, spawn), "Ward:2");
            assertEquals(false, ward.get("granted"));
            assertEquals("keyword", onlyKey(ward).get("trait_kind"));
            List<Map<String, Object>> crusherLines = linesOf(board, crusher);
            Map<String, Object> annihilator = keywordLine(crusherLines, "Annihilator:2");
            assertEquals("cardsfolder/u/ulamogs_crusher.txt",
                    onlyKey(annihilator).get("script_file"));
            assertEquals(List.of("static", "keyword"), kinds(crusherLines),
                    "the keyword's own trigger is its line: " + crusherLines);
        }

        /** Acceptance 5: the protection line is the instance's, naming its quality. */
        @Test
        void protectionFromGoblinsNamesGoblins() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card guide = board.battlefield("Foothill Guide", board.you);

            keywordLine(linesOf(board, guide), "Protection:Goblin");
        }

        /**
         * A face-down permanent's lines are its face-down state's, and its real
         * face rides beside them, every line flagged inactive.
         */
        @Test
        void aFaceDownPermanentCarriesItsRealFaceAsInactiveLines() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card guide = board.battlefield("Foothill Guide", board.you);
            guide.turnFaceDown(true);

            List<Map<String, Object>> lines = linesOf(board, guide);

            Map<String, Object> protection = keywordLine(lines, "Protection:Goblin");
            assertEquals(false, protection.get("active"), lines.toString());
            Map<String, Object> morph = keywordLine(lines, "Morph:W");
            assertEquals(false, morph.get("active"), lines.toString());
            assertEquals("cardsfolder/f/foothill_guide.txt", onlyKey(morph).get("script_file"));
        }

        /** A spell on the stack cast with its kicker says so on the kicker's line. */
        @Test
        void aKickedSpellOnTheStackMarksItsKickerLine() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card onStack = board.game.getAction().moveToStack(
                    board.hand("Burst Lightning", board.you), null);
            SpellAbility burst = board.spell(onStack, board.you);
            burst.addOptionalCost(OptionalCost.Kicker1);
            burst.resetTargets();
            burst.getTargets().add(board.opponent);
            board.game.getStack().add(burst);

            List<Map<String, Object>> lines = linesOf(board, onStack);

            assertEquals(1L, keywordLine(lines, "Kicker:4").get("cast_with"), lines.toString());
            assertFalse(line(lines, "spell").containsKey("cast_with"), lines.toString());
        }

        /**
         * A spell cast with its gift promised says so on its Gift line: the
         * optional cost is named "Promise Gift", the keyword "Gift" (B9).
         */
        @Test
        void aSpellCastWithItsGiftPromisedMarksItsGiftLine() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card bears = board.battlefield("Grizzly Bears", board.opponent);
            Card onStack = board.game.getAction().moveToStack(
                    board.hand("Blooming Blast", board.you), null);
            SpellAbility blast = board.spell(onStack, board.you);
            blast.addOptionalCost(OptionalCost.PromiseGift);
            blast.resetTargets();
            blast.getTargets().add(bears);
            board.game.getStack().add(blast);

            List<Map<String, Object>> lines = linesOf(board, onStack);

            assertEquals(1L, keywordLine(lines, "Gift").get("cast_with"), lines.toString());
        }

        /**
         * Every optional cost Forge has is mapped to its keyword or known to
         * be in no keyword line, so a cost Forge adds is not silently unread.
         */
        @Test
        void everyOptionalCostIsMappedOrKnownToHaveNoLine() {
            java.util.Set<OptionalCost> noLine = java.util.EnumSet.of(
                    OptionalCost.ReduceW, OptionalCost.ReduceU, OptionalCost.ReduceB,
                    OptionalCost.ReduceR, OptionalCost.ReduceG, OptionalCost.AltCost,
                    OptionalCost.Flash, OptionalCost.Generic);
            for (OptionalCost cost : OptionalCost.values()) {
                assertTrue(noLine.contains(cost) != CardLines.OPTIONAL_COST_KEYWORDS.containsKey(cost),
                        cost.name());
            }
        }

        /** A charm on the stack names the modes it chose on its own line. */
        @Test
        void aCharmOnTheStackNamesItsChosenModes() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card onStack = board.game.getAction().moveToStack(
                    board.hand("Boros Charm", board.you), null);
            SpellAbility charm = TestBoard.charmOf(onStack, board.you);
            // Mode 1, permanents you control gain indestructible: no target to choose.
            TestBoard.chain(charm, 1);
            board.game.getStack().add(charm);

            List<Map<String, Object>> lines = linesOf(board, onStack);

            assertEquals(List.of(1L), line(lines, "spell").get("chosen"), lines.toString());
        }

        /** Face up, the same lines are in force and carry no flag. */
        @Test
        void aFaceUpPermanentsLinesAreNotFlaggedInactive() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card guide = board.battlefield("Foothill Guide", board.you);

            for (Map<String, Object> line : linesOf(board, guide)) {
                assertFalse(line.containsKey("active"), line.toString());
            }
        }
    }

    // ── base power and toughness (FR-025) ───────────────────────────────

    @Nested
    @ExtendWith(ForgeExtension.class)
    class PowerAndToughness {

        /**
         * Ensoul Artifact sets the base (layer 7b): Ornithopter's base is
         * 5/5, not its printed 0/2, and nothing is a boost.
         */
        @Test
        void ensoulArtifactSetsTheBase() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card thopter = board.battlefield("Ornithopter", board.you);
            board.battlefield("Ensoul Artifact", board.you).attachToEntity(thopter, null);
            board.game.getAction().checkStaticAbilities();

            Map<String, Object> entity = entityOf(board, thopter);
            assertEquals(List.of(5L, 5L), entity.get("base_pt"));
            assertEquals(Map.of("temporary", List.of(0L, 0L), "counters", List.of(0L, 0L)),
                    entity.get("pt_boosts"));
        }

        /**
         * Boosts apart from the base: an anthem's +1/+1 is temporary, and
         * every counter type adds what Forge counts for it.
         */
        @Test
        void boostsAndEveryCountersContributionAreApartFromTheBase() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card bears = board.battlefield("Grizzly Bears", board.you);
            board.battlefield("Glorious Anthem", board.you);
            bears.addCounterInternal(CounterEnumType.P1P1, 1, board.you, false, null, null);
            bears.addCounterInternal(CounterEnumType.P1P2, 1, board.you, false, null, null);
            board.game.getAction().checkStaticAbilities();

            Map<String, Object> entity = entityOf(board, bears);
            assertEquals(List.of(2L, 2L), entity.get("base_pt"));
            assertEquals(Map.of("temporary", List.of(1L, 1L), "counters", List.of(2L, 3L)),
                    entity.get("pt_boosts"));
            assertEquals(2, RecordLines.asMap(entity.get("counters")).size(),
                    "both counter types, each by name: " + entity.get("counters"));
        }

        @Test
        void aNoncreatureHasNoPowerOrToughness() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Map<String, Object> entity = entityOf(board, board.battlefield("Forest", board.you));

            assertNull(entity.get("base_pt"));
            assertNull(entity.get("pt_boosts"));
        }
    }

    // ── a continuous record's board (the removed static leaves no trace) ─

    @Nested
    @ExtendWith(ForgeExtension.class)
    class WithoutTheStatic {

        /**
         * The board without Levitation lists no line of Levitation's on the
         * creature: neither the flying keyword line nor the CantBlockBy static
         * Forge builds for flying, which is the static the record is about.
         */
        @Test
        void theRemovedStaticsKeywordLeavesNoLineBehind() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card bears = board.battlefield("Grizzly Bears", board.you);
            Card levitation = board.battlefield("Levitation", board.you);
            board.game.getAction().checkStaticAbilities();
            StaticAbility levitating = levitation.getStaticAbilities().get(0);

            Map<String, Object> without = entityIn(
                    new SnapshotBuilder(board.game).ofBoardWithout(levitating).render(), bears);
            Map<String, Object> with = entityIn(
                    new SnapshotBuilder(board.game).ofBoard().render(), bears);

            assertEquals(List.of(), without.get("lines"));
            assertEquals(List.of("keyword"), kinds(lines(with)), with.toString());
        }

        /**
         * Nor does any static say which permanents it affects: that is the
         * record's answer, and hiding only the removed static's list would
         * point at it.
         */
        @Test
        void noStaticListsAffectedPermanents() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card bears = board.battlefield("Grizzly Bears", board.you);
            Card levitation = board.battlefield("Levitation", board.you);
            board.game.getAction().checkStaticAbilities();
            StaticAbility levitating = levitation.getStaticAbilities().get(0);

            Map<String, Object> without = line(lines(entityIn(
                    new SnapshotBuilder(board.game).ofBoardWithout(levitating).render(),
                    levitation)), "static");
            Map<String, Object> with = line(linesOf(board, levitation), "static");

            assertFalse(without.containsKey("affected"), without.toString());
            assertEquals(List.of(Refs.of(bears)), with.get("affected"));
        }
    }

    // ── player and global facts (FR-024, T054) ──────────────────────────

    @Nested
    @ExtendWith(ForgeExtension.class)
    class PlayerAndGlobalFacts {

        @Test
        void speedBlessingRadMonarchAndInitiative() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            board.you.setSpeed(2);
            board.you.setBlessing(true, null);
            board.opponent.addCounterInternal(
                    CounterEnumType.RAD, 3, board.you, false, null, AbilityKey.newMap());
            board.game.setMonarch(board.you);
            board.game.setHasInitiative(board.opponent);

            Map<String, Object> you = playerOf(board, 0);
            Map<String, Object> them = playerOf(board, 1);

            assertEquals(2L, you.get("speed"));
            assertEquals(true, you.get("city_blessing"));
            assertEquals(false, them.get("city_blessing"));
            assertEquals(3L, them.get("rad"));
            assertEquals(true, you.get("monarch"));
            assertEquals(false, them.get("monarch"));
            assertEquals(false, you.get("initiative"));
            assertEquals(true, them.get("initiative"));
        }

        @Test
        void theTurnsCountsBesideThisTurn() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            board.game.getAction().moveToLibrary(board.card("Forest", board.you), null, null);
            board.you.drawCard();
            board.you.gainLife(3, null, null);
            board.opponent.loseLife(2, false, false, null);
            Card bears = board.battlefield("Grizzly Bears", board.you);
            board.game.getAction().moveTo(ZoneType.Graveyard, bears, null, null);

            Map<String, Object> you = RecordLines.asMap(playerOf(board, 0).get("turn_counts"));
            Map<String, Object> them = RecordLines.asMap(playerOf(board, 1).get("turn_counts"));

            assertEquals(1L, you.get("cards_drawn"));
            assertEquals(3L, you.get("life_gained"));
            assertEquals(2L, them.get("life_lost"));
            assertEquals(1L, you.get("permanents_to_graveyard"));
            assertEquals(1L, you.get("descended"));
            assertEquals(Set.of("cards_drawn", "life_gained", "life_lost", "damage_taken",
                    "permanents_to_graveyard", "spells_cast_last_turn", "dungeons_completed",
                    "descended"), you.keySet());
        }

        /** One land play a turn, written right after the turn's tallies it bounds. */
        @Test
        void oneLandPlayWithoutARaiseRightAfterThisTurn() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);

            assertEquals(1L, playerOf(board, 0).get("max_land_plays"));
            assertTrue(PlayerJson.render(board.game, board.you)
                    .matches(".*\"this_turn\":\\{[^}]*},\"max_land_plays\":1,.*"),
                    PlayerJson.render(board.game, board.you));
        }

        /** Exploration raises its controller's land plays, not the opponent's. */
        @Test
        void explorationOnTheBattlefieldAllowsTwo() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            board.battlefield("Exploration", board.you);
            board.game.getAction().checkStaticAbilities();

            assertEquals(2L, playerOf(board, 0).get("max_land_plays"));
            assertEquals(1L, playerOf(board, 1).get("max_land_plays"));
        }

        /** Explore's raise lives on its effect card in the command zone. */
        @Test
        void exploreResolvedThisTurnAllowsTwo() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            board.game.getAction().moveToLibrary(board.card("Forest", board.you), null, null);
            Card explore = board.hand("Explore", board.you);
            board.resolve(board.spell(explore, board.you), board.you);
            board.game.getAction().checkStaticAbilities();

            assertEquals(2L, playerOf(board, 0).get("max_land_plays"));
        }

        /** Fastbond's {@code AdjustLandPlays$ Unlimited} is null, not a number. */
        @Test
        void fastbondMakesThemUnlimited() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            board.battlefield("Fastbond", board.you);
            board.game.getAction().checkStaticAbilities();

            Map<String, Object> you = playerOf(board, 0);
            assertTrue(you.containsKey("max_land_plays"), you.toString());
            assertNull(you.get("max_land_plays"));
        }

        @Test
        void dayNightIsDayNightOrNeither() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            assertNull(globalOf(board).get("day_night"));
            board.game.setDayTime(false);
            assertEquals("day", globalOf(board).get("day_night"));
            board.game.setDayTime(true);
            assertEquals("night", globalOf(board).get("day_night"));
        }

        /** Owner is not controller: a control change leaves the owner alone. */
        @Test
        void anEntityNamesItsOwnerApartFromItsController() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card bears = board.battlefield("Grizzly Bears", board.you);
            bears.addTempController(board.opponent, board.game.getNextTimestamp());

            Map<String, Object> entity = entityOf(board, bears);

            assertEquals(Refs.of(board.you), entity.get("owner"));
            assertEquals(Refs.of(board.opponent), entity.get("controller"));
        }
    }

    // ── card facts (FR-024, T055) ───────────────────────────────────────

    @Nested
    @ExtendWith(ForgeExtension.class)
    class CardFactsOfASlot {

        @Test
        void faceStateNamesTheFaceUp() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card bears = board.battlefield("Grizzly Bears", board.you);
            Card delver = board.battlefield("Delver of Secrets", board.you);
            delver.setState(CardStateName.Backside, true);
            Card bushi = board.battlefield("Bushi Tenderfoot", board.you);
            bushi.setState(CardStateName.Flipped, true);
            Card guide = board.battlefield("Foothill Guide", board.you);
            guide.turnFaceDown(true);

            assertEquals("front", entityOf(board, bears).get("face_state"));
            assertEquals("back", entityOf(board, delver).get("face_state"));
            assertEquals("flipped", entityOf(board, bushi).get("face_state"));
            assertEquals("face_down", entityOf(board, guide).get("face_state"));
        }

        @Test
        void designationsAreSortedNames() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card bears = board.battlefield("Grizzly Bears", board.you);
            bears.setRenowned(true);
            bears.setMonstrous(true);
            bears.addGoad(board.game.getNextTimestamp(), board.opponent);

            assertEquals(List.of("goaded", "monstrous", "renowned"),
                    entityOf(board, bears).get("designations"));
            assertEquals(List.of(), entityOf(board, board.battlefield("Forest", board.you))
                    .get("designations"));
        }

        @Test
        void xPaidAndCostsPaidAreTheSpells() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card mountain = board.battlefield("Mountain", board.you);
            Card fireball = board.hand("Fireball", board.you);
            Card onStack = board.game.getAction().moveToStack(fireball, null);
            onStack.setCastFrom(board.you.getZone(ZoneType.Hand));
            SpellAbility spell = board.spell(onStack, board.you);
            spell.setXManaCostPaid(2);
            for (int i = 0; i < 3; i++) {
                spell.getPayingMana().add(
                        new Mana(MagicColor.RED, mountain, null, board.you));
            }
            spell.resetTargets();
            spell.getTargets().add(board.opponent);
            board.game.getStack().add(spell);

            Map<String, Object> entity = entityOf(board, onStack);

            assertEquals(2L, entity.get("x_paid"));
            Map<String, Object> paid = RecordLines.asMap(entity.get("costs_paid"));
            assertEquals(3L, RecordLines.asMap(paid.get("mana")).get("R"));
            assertEquals("hand", paid.get("cast_from"));
            assertEquals(false, paid.get("mana_rider"));
            Map<String, Object> bears = entityOf(board, board.hand("Grizzly Bears", board.you));
            assertNull(bears.get("x_paid"));
            assertNull(bears.get("costs_paid"));
        }

        @Test
        void chosenValuesAreWhatTheCardRemembers() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card cavern = board.battlefield("Cavern of Souls", board.you);
            cavern.setChosenType("Elf");
            cavern.setChosenColors(List.of("blue"));

            Map<String, Object> chosen = RecordLines.asMap(entityOf(board, cavern).get("chosen"));

            assertEquals("Elf", chosen.get("type"));
            assertEquals(List.of("blue"), chosen.get("colors"));
            assertEquals(Map.of(), entityOf(board, board.battlefield("Forest", board.you))
                    .get("chosen"));
        }

        @Test
        void historyBitsAreThisTurns() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card fresh = board.battlefield("Grizzly Bears", board.you);
            Card old = board.battlefield("Hill Giant", board.you);
            old.setTurnInZone(0);
            old.addBlockedThisTurn(fresh);
            Card elspeth = board.battlefield("Elspeth, Knight-Errant", board.you);
            elspeth.addPlaneswalkerAbilityActivated();

            Map<String, Object> freshHistory = RecordLines.asMap(entityOf(board, fresh).get("history"));
            Map<String, Object> oldHistory = RecordLines.asMap(entityOf(board, old).get("history"));

            assertEquals(true, freshHistory.get("entered_this_turn"));
            assertEquals(false, oldHistory.get("entered_this_turn"));
            assertEquals(true, oldHistory.get("blocked_this_turn"));
            assertEquals(false, freshHistory.get("blocked_this_turn"));
            assertEquals(true, RecordLines.asMap(entityOf(board, elspeth).get("history"))
                    .get("loyalty_activated"));
        }

        @Test
        void orderRankIsTimestampOrderOnTheBattlefieldAndPositionElsewhere() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card older = board.battlefield("Grizzly Bears", board.you);
            Card newer = board.battlefield("Hill Giant", board.opponent);
            Card first = board.graveyard("Shock", board.you);
            Card second = board.graveyard("Giant Growth", board.you);

            assertEquals(0L, entityOf(board, older).get("order_rank"));
            assertEquals(1L, entityOf(board, newer).get("order_rank"));
            assertEquals(0L, entityOf(board, first).get("order_rank"));
            assertEquals(1L, entityOf(board, second).get("order_rank"));
        }

        @Test
        void aFaceDownCardCarriesItsRealFace() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card guide = board.battlefield("Foothill Guide", board.you);
            assertNull(entityOf(board, guide).get("real_face"));
            guide.turnFaceDown(true);

            Map<String, Object> real = RecordLines.asMap(entityOf(board, guide).get("real_face"));

            assertEquals("Foothill Guide", real.get("name"));
            assertEquals(List.of("creature"), real.get("types"));
            assertEquals(List.of(1L, 1L), real.get("base_pt"));
        }

        @Test
        void printedLoyaltyAndDefense() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Map<String, Object> elspeth = entityOf(board,
                    board.battlefield("Elspeth, Knight-Errant", board.you));
            Map<String, Object> invasion = entityOf(board,
                    board.hand("Invasion of Zendikar", board.you));
            Map<String, Object> bears = entityOf(board, board.hand("Grizzly Bears", board.you));

            assertEquals(4L, elspeth.get("printed_loyalty"));
            assertNull(elspeth.get("printed_defense"));
            assertEquals(3L, invasion.get("printed_defense"));
            assertNull(bears.get("printed_loyalty"));
        }

        @Test
        void manaCostCountsEachSymbolKind() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);

            assertEquals(Map.of("generic", 1L, "G/W", 2L),
                    entityOf(board, board.hand("Kitchen Finks", board.you)).get("mana_cost"));
            assertEquals(Map.of("U/P", 1L),
                    entityOf(board, board.hand("Gitaxian Probe", board.you)).get("mana_cost"));
            assertEquals(Map.of("X", 1L, "R", 1L),
                    entityOf(board, board.hand("Fireball", board.you)).get("mana_cost"));
            assertEquals(Map.of(),
                    entityOf(board, board.hand("Ornithopter", board.you)).get("mana_cost"));
            assertNull(entityOf(board, board.hand("Forest", board.you)).get("mana_cost"));
        }
    }

    // ── entities beyond the tiers (FR-024, T057) ────────────────────────

    @Nested
    @ExtendWith(ForgeExtension.class)
    class BeyondTheTiers {

        @Test
        void eachLibrarysTopCardIsALibraryCard() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card bottom = board.game.getAction().moveToLibrary(
                    board.card("Forest", board.you), null, null);
            Card top = board.game.getAction().moveToLibrary(
                    board.card("Grizzly Bears", board.you), 0, null, null);

            String state = new SnapshotBuilder(board.game).ofBoard().render();
            Map<String, Object> entity = entityIn(state, top);

            assertEquals("library_card", entity.get("role"));
            assertEquals("library", entity.get("zone"));
            assertEquals(0L, entity.get("order_rank"));
            assertFalse(idsIn(state).contains(Refs.of(bottom)), state);
        }

        /** A library card the record's events name joins at the write, where it sat. */
        @Test
        void aLibraryCardTheEventsNameJoinsWhereItSat() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            board.game.getAction().moveToLibrary(board.card("Forest", board.you), null, null);
            Card second = board.game.getAction().moveToLibrary(
                    board.card("Grizzly Bears", board.you), -1, null, null);
            Snapshot before = new SnapshotBuilder(board.game).ofBoard();
            long position = board.you.getCardsIn(ZoneType.Library).indexOf(second);
            board.game.getAction().moveTo(ZoneType.Hand, second, null, null);

            String state = before.withEventCards(List.of(
                    new EffectEvent(EventType.ZONE_CHANGE).subject(Refs.of(second)))).render();

            Map<String, Object> entity = entityIn(state, second);
            assertEquals("library_card", entity.get("role"));
            assertEquals("library", entity.get("zone"));
            assertEquals(position, entity.get("order_rank"));
            assertTrue(position > 0, "not the top, which the snapshot already holds");
            assertEquals(before.render(), before.withEventCards(List.of()).render());
        }

        @Test
        void theCardsExiledWithAPermanentAreSlots() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card ring = board.battlefield("Oblivion Ring", board.you);
            Card exiled = board.game.getAction().exile(
                    board.battlefield("Hill Giant", board.opponent), null, null);
            exiled.setExiledWith(ring);
            ring.addExiledCard(exiled);

            Map<String, Object> entity = entityIn(
                    new SnapshotBuilder(board.game).ofBoard().render(), exiled);

            assertEquals("exiled_with", entity.get("role"));
            assertEquals("exile", entity.get("zone"));
        }

        @Test
        void aMutatedPermanentsComponentsAreSlots() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card top = board.battlefield("Grizzly Bears", board.you);
            Card under = board.card("Hill Giant", board.you);
            top.addMergedCard(top);
            top.addMergedCard(under);
            top.setMutatedTimestamp(board.game.getNextTimestamp());

            String state = new SnapshotBuilder(board.game).ofBoard().render();

            assertEquals("mutate_component", entityIn(state, under).get("role"));
            assertNull(entityIn(state, top).get("role"));
        }

        /** The pending event's subject as it last existed, under an id of its own. */
        @Test
        void thePendingSubjectsLastKnownStateIsASlotOfItsOwn() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card bears = board.battlefield("Grizzly Bears", board.you);
            Card lastKnown = CardCopyService.getLKICopy(bears);
            Card dead = board.game.getAction().moveTo(ZoneType.Graveyard, bears, null, null);
            EffectEvent died = new EffectEvent(EventType.ZONE_CHANGE)
                    .subject(Refs.of(dead)).lastKnown(lastKnown);

            String state = new SnapshotBuilder(board.game).ofTrait(null, died).render();

            Map<String, Object> copy = entitiesIn(state).stream()
                    .filter(e -> ("L" + bears.getId()).equals(e.get("id")))
                    .findFirst().orElseThrow(() -> new AssertionError(state));
            assertEquals("last_known", copy.get("role"));
            assertEquals("battlefield", copy.get("zone"));
            assertEquals("graveyard", entityIn(state, dead).get("zone"));
        }

        /** Per static line, the permanents it affects. */
        @Test
        void aStaticLineListsThePermanentsItAffects() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card bears = board.battlefield("Grizzly Bears", board.you);
            board.battlefield("Hill Giant", board.opponent);
            Card anthem = board.battlefield("Glorious Anthem", board.you);
            board.game.getAction().checkStaticAbilities();

            assertEquals(List.of(Refs.of(bears)),
                    line(linesOf(board, anthem), "static").get("affected"));
        }

        /** A static that applies to a player lists the player after the permanents. */
        @Test
        void aStaticLineListsThePlayersItAffectsAfterThePermanents() {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card exploration = board.battlefield("Exploration", board.you);
            board.game.getAction().checkStaticAbilities();

            assertEquals(List.of(Refs.of(board.you)),
                    line(linesOf(board, exploration), "static").get("affected"));
        }

        /** The resolving ability's die results and coin flips, read off its events. */
        @Test
        void randomResultsAreTheResolutionsDiceAndCoins() {
            EffectEvent dice = new EffectEvent(EventType.DICE_ROLLED).param("results", List.of(
                    Map.of("natural", 3, "final", 4), Map.of("natural", 6, "final", 6)));
            EffectEvent coins = new EffectEvent(EventType.COIN_FLIPPED)
                    .param("results", List.of("win", "loss", "loss"));

            Map<String, Object> random = RecordLines.asMap(RecordLines.parse(
                    RefsJson.renderResolved(null, List.of(dice, coins), null)).get("random"));

            assertEquals(List.of(4L, 6L), random.get("dice"));
            assertEquals(Map.of("won", 1L, "lost", 2L), random.get("coin_flips"));
            assertNull(RecordLines.parse(RefsJson.render(null)).get("random"),
                    "unread before the resolution ends");
        }
    }

    // ── reading an entity ───────────────────────────────────────────────

    private static Map<String, Object> entityOf(CharacterizationBoard board, Card card) {
        return RecordLines.parse(EntityJson.render(board.game, card, StaticSuppression.NONE));
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> entitiesIn(String state) {
        return (List<Map<String, Object>>) RecordLines.parse(state).get("entities");
    }

    /** The entity of a rendered state written under this card's id. */
    private static Map<String, Object> entityIn(String state, Card card) {
        return entitiesIn(state).stream()
                .filter(entity -> Refs.of(card).equals(entity.get("id")))
                .findFirst().orElseThrow(() -> new AssertionError(card + " not in " + state));
    }

    private static List<Object> idsIn(String state) {
        return entitiesIn(state).stream().map(entity -> entity.get("id")).toList();
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> lines(Map<String, Object> entity) {
        return (List<Map<String, Object>>) entity.get("lines");
    }

    private static Map<String, Object> globalOf(CharacterizationBoard board) {
        return RecordLines.asMap(RecordLines.parse(
                new SnapshotBuilder(board.game).ofBoard().render()).get("global"));
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> playerOf(CharacterizationBoard board, int index) {
        return ((List<Map<String, Object>>) RecordLines.parse(
                new SnapshotBuilder(board.game).ofBoard().render()).get("players")).get(index);
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> linesOf(CharacterizationBoard board, Card card) {
        return (List<Map<String, Object>>) entityOf(board, card).get("lines");
    }

    private static List<Object> kinds(List<Map<String, Object>> lines) {
        return lines.stream().map(line -> line.get("kind")).toList();
    }

    /** The one line of this kind. */
    private static Map<String, Object> line(List<Map<String, Object>> lines, String kind) {
        List<Map<String, Object>> found = lines.stream()
                .filter(line -> kind.equals(line.get("kind"))).toList();
        assertEquals(1, found.size(), lines.toString());
        return found.get(0);
    }

    /** The one keyword line written with this fragment. */
    private static Map<String, Object> keywordLine(
            List<Map<String, Object>> lines, String fragment) {
        List<Map<String, Object>> found = lines.stream()
                .filter(line -> "keyword".equals(line.get("kind"))
                        && fragment.equals(line.get("fragment"))).toList();
        assertEquals(1, found.size(), "one " + fragment + " line in " + lines);
        return found.get(0);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> onlyKey(Map<String, Object> line) {
        List<Object> keys = (List<Object>) line.get("keys");
        assertEquals(1, keys.size(), line.toString());
        return RecordLines.asMap(keys.get(0));
    }
}
