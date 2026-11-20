package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.ability.AbilityFactory;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.AbilitySub;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Which clause of a line an event came from, and how long what it did
 * lasts.
 */
@ExtendWith(ForgeExtension.class)
class EventAttributionTest {

    /** A three-clause line: gain life, then draw, then scry. */
    private static SpellAbility threeClauseAbility() {
        Card host = TestCards.build("Fountain of Youth");
        host.setSVar("DBDraw",
                "DB$ Draw | Defined$ You | NumCards$ 1 | SubAbility$ DBScry");
        host.setSVar("DBScry", "DB$ Scry | Defined$ You | ScryNum$ 1");
        SpellAbility root = AbilityFactory.getAbility(
                "AB$ GainLife | Cost$ 1 | Defined$ You | LifeAmount$ 1"
                        + " | SubAbility$ DBDraw", host);
        assertNotNull(root.getSubAbility(), "the chain has to be three deep");
        assertNotNull(root.getSubAbility().getSubAbility());
        return root;
    }

    /**
     * Each clause of a multi-clause line is its own index.
     *
     * <p>A pointer set only around the line the stack resolved would attribute
     * every clause after the first to the root. "Put a counter" and "draw a
     * card" are exactly the distinction the per-entity head learns, so a whole
     * line attributed to its root teaches neither.
     */
    @Test
    void eachClauseOfAMultiClauseAbilityAttributesToItself() {
        SpellAbility root = threeClauseAbility();
        AbilitySub first = (AbilitySub) root.getSubAbility();
        AbilitySub second = (AbilitySub) first.getSubAbility();

        assertEquals("0", EventAttribution.attributedTo(first));
        assertEquals("1", EventAttribution.attributedTo(second));
    }

    /**
     * The root acting and no pointer at all are different answers.
     *
     * <p>A null for both would make the attribution rate unreadable: a
     * single-clause ability legitimately has no sub-ability and the root is
     * the right answer, and that would look exactly like the pointer failing
     * to land. Separating them is what turns the rate into a measurement
     * instead of a number.
     */
    @Test
    void theRootActingAndNoPointerAreDifferentAnswers() {
        SpellAbility root = threeClauseAbility();

        assertEquals(EventAttribution.ROOT,
                EventAttribution.attributedTo(root));
        assertEquals(EventAttribution.UNRESOLVED,
                EventAttribution.attributedTo(null));
        // A pointer that is no ability at all names nothing either.
        assertEquals(EventAttribution.UNRESOLVED,
                EventAttribution.attributedTo("not an ability"));
        assertNotEquals(EventAttribution.ROOT, EventAttribution.UNRESOLVED);
    }

    /**
     * A clause with no line above it names nothing a reader could join to.
     *
     * <p>Its index counts clauses of a chain this collector never saw, so the
     * number would address a line in some other script. Saying so is the third
     * state, not the root.
     */
    @Test
    void aDetachedClauseCannotBePlacedAndSaysSo() {
        SpellAbility root = threeClauseAbility();
        AbilitySub orphan = (AbilitySub) root.getSubAbility();
        orphan.setParent(null);

        assertEquals(EventAttribution.UNRESOLVED,
                EventAttribution.attributedTo(orphan));
    }

    /** And an event stamped inside a clause carries that clause's index. */
    @Test
    void anEventCarriesTheClauseThatProducedIt() {
        SpellAbility root = threeClauseAbility();
        AbilitySub second = (AbilitySub) root.getSubAbility().getSubAbility();

        String json = EventAttribution.stampAt(
                new EffectEvent(EventType.CARD_DRAWN).subject("P0"),
                root, second, null).toJson();

        assertTrue(json.contains("\"attributed_to\":\"1\""), json);
    }

    /**
     * An event published while no clause resolves, once a clause of the
     * resolution's line has run, is the last such clause's: the static check
     * after the last clause (finding 2.1). Before any has run it is none's,
     * and its duration and cause are read off the line either way.
     */
    @Test
    void anEventAfterTheLastClauseIsThatClauses() {
        SpellAbility root = threeClauseAbility();

        String after = EventAttribution.stamp(
                new EffectEvent(EventType.PT_CHANGE).subject("E1"), root,
                root.getSubAbility().getSubAbility(), null).toJson();
        String before = EventAttribution.stamp(
                new EffectEvent(EventType.PT_CHANGE).subject("E1"), root, null, null).toJson();

        assertTrue(after.contains("\"attributed_to\":\"1\""), after);
        assertTrue(before.contains("\"attributed_to\":\"unresolved\""), before);
        assertEquals(after.replace("\"1\"", "\"unresolved\""), before);
    }

    /**
     * The same event reads its duration off the last clause that ran, not the
     * line's root: a Pump root's end of turn gives way to an Attach
     * sub-ability's permanent (finding 2.1, task BR4).
     */
    @Test
    void anEventAfterTheLastClauseReadsItsDuration() {
        Card host = TestCards.build("Fountain of Youth");
        host.setSVar("DBPump",
                "DB$ Pump | Defined$ Self | NumAtt$ 1 | Duration$ Permanent");
        SpellAbility root = AbilityFactory.getAbility(
                "AB$ Pump | Cost$ 1 | Defined$ Self | NumAtt$ 1 | SubAbility$ DBPump", host);
        EffectEvent change = new EffectEvent(EventType.PT_CHANGE).subject("E1");

        String after = EventAttribution.stamp(change, root, root.getSubAbility(), null).toJson();
        String before = EventAttribution.stamp(
                new EffectEvent(EventType.PT_CHANGE).subject("E1"), root, null, null).toJson();

        assertTrue(after.contains("\"duration\":\"permanent\""), after);
        assertTrue(before.contains("\"duration\":\"end_of_turn\""), before);
    }

    /**
     * A clause's own {@code Duration$} outranks the line's.
     *
     * <p>Both fields come from one read of the pointer, so they cannot describe
     * different instants of a nested resolution.
     */
    @Test
    void aClauseDeclaringItsOwnDurationOutranksTheLine() {
        Card host = TestCards.build("Fountain of Youth");
        host.setSVar("DBPump",
                "DB$ Pump | Defined$ Self | NumAtt$ 1 | Duration$ Permanent");
        SpellAbility root = AbilityFactory.getAbility(
                "AB$ GainLife | Cost$ 1 | Defined$ You | LifeAmount$ 1"
                        + " | SubAbility$ DBPump", host);

        assertEquals(EventDuration.PERMANENT,
                EventAttribution.duration(root, root.getSubAbility(), EventType.PT_CHANGE));
        assertEquals(EventDuration.INSTANT,
                EventAttribution.duration(root, root, EventType.LIFE_CHANGE));
    }

    /**
     * A sub-ability with no {@code Duration$} gets Forge's default for its own
     * API, never the line's: Hypnotic Pattern's perpetual root and its
     * until-end-of-turn pump.
     */
    @Test
    void aClauseDeclaringNoDurationDoesNotInheritTheLines() {
        SpellAbility root = TestCards.build("Hypnotic Pattern").getFirstSpellAbility();
        SpellAbility sub = root.getSubAbility();

        assertEquals(EventDuration.PERMANENT,
                EventAttribution.duration(root, root, EventType.PT_CHANGE));
        assertEquals(EventDuration.END_OF_TURN,
                EventAttribution.duration(root, sub, EventType.PT_CHANGE));
    }

    /**
     * Every {@code Duration$} Forge's scripts spell, and the event
     * {@code duration} it reads: one that ends only when an object changes
     * outlasts the turn, as {@code Permanent} does; one that ends at a step
     * of some turn ends with it, as an until-end-of-turn one does.
     */
    private static final Map<String, String> DURATIONS = Map.ofEntries(
            Map.entry("Permanent", "permanent"),
            Map.entry("Perpetual", "permanent"),
            Map.entry("UntilHostLeavesPlay", "permanent"),
            Map.entry("UntilLoseControlOfHost", "permanent"),
            Map.entry("AsLongAsInPlay", "permanent"),
            Map.entry("AsLongAsControl", "permanent"),
            Map.entry("UntilUntaps", "permanent"),
            Map.entry("UntilTargetedUntaps", "permanent"),
            Map.entry("UntilUnattached", "permanent"),
            Map.entry("UntilFacedown", "permanent"),
            Map.entry("UntilAnimatedFaceup", "permanent"),
            Map.entry("UntilHostLeavesPlayOrEOT", "end_of_turn"),
            Map.entry("UntilHostLeavesPlayOrEndOfCombat", "end_of_turn"),
            Map.entry("UntilYourNextTurn", "end_of_turn"),
            Map.entry("UntilTheEndOfYourNextTurn", "end_of_turn"),
            Map.entry("UntilTheEndOfTargetedNextTurn", "end_of_turn"),
            Map.entry("UntilYourNextUntap", "end_of_turn"),
            Map.entry("UntilTheEndOfYourNextUntap", "end_of_turn"),
            Map.entry("UntilYourNextUpkeep", "end_of_turn"),
            Map.entry("UntilTheEndOfYourNextUpkeep", "end_of_turn"),
            Map.entry("UntilYourNextEndStep", "end_of_turn"),
            Map.entry("UntilNextEndStep", "end_of_turn"),
            Map.entry("UntilEndOfCombat", "end_of_turn"),
            Map.entry("UntilEndOfCombatYourNextTurn", "end_of_turn"),
            Map.entry("UntilEndOfTurn", "end_of_turn"),
            Map.entry("EndOfTurn", "end_of_turn"),
            Map.entry("EOT", "end_of_turn"),
            Map.entry("End of Turn", "end_of_turn"),
            Map.entry("UntilStateBasedActionChecked", "end_of_turn"),
            Map.entry("ThisTurnAndNextTurn", "end_of_turn"),
            Map.entry("NextThisTurn", "end_of_turn"));

    static java.util.stream.Stream<org.junit.jupiter.params.provider.Arguments> durations() {
        return DURATIONS.entrySet().stream().map(entry ->
                org.junit.jupiter.params.provider.Arguments.of(entry.getKey(), entry.getValue()));
    }

    /**
     * A declared {@code Duration$} is bucketed as the contract buckets a
     * floating effect's, {@code until_leaves} reading {@code permanent}.
     */
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.MethodSource("durations")
    void aDeclaredDurationIsBucketedAsAFloatingEffectsIs(String declared, String bucket) {
        Card host = TestCards.build("Grizzly Bears");
        SpellAbility pump = AbilityFactory.getAbility(
                "SP$ Pump | ValidTgts$ Creature | NumAtt$ +3 | Duration$ " + declared, host);

        assertEquals(bucket,
                EventAttribution.duration(pump, pump, EventType.PT_CHANGE).wireName(), declared);
        assertEquals(bucket, FloatingEffects.bucket(declared) == FloatingDuration.UNTIL_LEAVES
                ? "permanent" : FloatingEffects.bucket(declared).wireName(), declared);
    }

    /**
     * Every {@code Duration$} value a card or token script of the Forge
     * checkout spells is in the table above: a new one fails here rather
     * than falling silently into {@code end_of_turn}.
     */
    @Test
    void everyDurationForgesScriptsSpellIsBucketedExplicitly() throws java.io.IOException {
        java.nio.file.Path cards = com.pricepredictor.connector.ForgeEnvironmentInitializer
                .findCardsFolder();
        java.util.regex.Pattern declared =
                java.util.regex.Pattern.compile("Duration\\$\\s*([^|\\r\\n]+)");
        java.util.Set<String> spelled = new java.util.TreeSet<>();
        for (java.nio.file.Path tree : List.of(cards, cards.resolveSibling("tokenscripts"))) {
            if (!java.nio.file.Files.isDirectory(tree)) {
                continue;
            }
            try (var files = java.nio.file.Files.walk(tree)) {
                for (java.nio.file.Path file : files.filter(java.nio.file.Files::isRegularFile)
                        .toList()) {
                    var found = declared.matcher(new String(java.nio.file.Files.readAllBytes(
                            file), java.nio.charset.StandardCharsets.ISO_8859_1));
                    while (found.find()) {
                        spelled.add(found.group(1).strip());
                    }
                }
            }
        }
        assertFalse(spelled.isEmpty(), "no Duration$ read under " + cards);
        spelled.removeAll(DURATIONS.keySet());
        assertEquals(java.util.Set.of(), spelled,
                "Duration$ values no bucket lists: add each to DURATIONS and, where it ends"
                        + " on an object change, to FloatingEffects.bucket's until_leaves");
    }

    /**
     * An undeclared duration means what Forge does with it. A pump with no
     * {@code Duration$} ends at end of turn — Giant Growth — and a counter
     * lasts; only an outcome with no lasting state is instant.
     */
    @Test
    void anUndeclaredDurationIsTheOneForgeApplies() {
        Card host = TestCards.build("Grizzly Bears");
        SpellAbility pump = AbilityFactory.getAbility(
                "SP$ Pump | ValidTgts$ Creature | NumAtt$ +3 | NumDef$ +3", host);
        SpellAbility counter = AbilityFactory.getAbility(
                "SP$ PutCounter | Defined$ Self | CounterType$ P1P1 | CounterNum$ 1", host);

        assertEquals(EventDuration.END_OF_TURN,
                EventAttribution.duration(pump, pump, EventType.PT_CHANGE));
        assertEquals(EventDuration.PERMANENT,
                EventAttribution.duration(counter, counter, EventType.PT_CHANGE));
        assertEquals(EventDuration.INSTANT,
                EventAttribution.duration(pump, pump, EventType.DAMAGE_DEALT));

        String json = EventAttribution.stampAt(
                new EffectEvent(EventType.PT_CHANGE).subject("E1"), pump, pump, null).toJson();
        assertTrue(json.contains("\"duration\":\"end_of_turn\""), json);
    }

    /**
     * With nothing resolving and nothing named, the record says nothing.
     *
     * <p>An honest absence rather than a placeholder: Forge genuinely names no
     * causer for a turn-based action, and a reader can act on the difference.
     */
    @Test
    void anEventNothingCausedCarriesNoCause() {
        assertNull(EventAttribution.cause(null, null));

        String json = EventAttribution.stamp(
                new EffectEvent(EventType.UNTAPPED).subject("E7"), null, null)
                .toJson();

        assertFalse(json.contains("cause"), json);
    }

    // ── a damage-step probe's sources (FR-035) ──────────────────────────

    private static EffectEvent damage(String subject, int amount) {
        return new EffectEvent(EventType.DAMAGE_DEALT).subject(subject)
                .param(EventParam.AMOUNT, amount).param(EventParam.COMBAT, true);
    }

    /**
     * A probe's damage events name their dealer as {@code source}: the bus's
     * cause where it names one, else the step's assignment, which names the
     * one creature that assigned damage to the subject, or among several the
     * one that assigned exactly the amount dealt, and otherwise nothing.
     */
    @Test
    void aProbesDamageEventsNameTheirSourceFromTheAssignment() {
        Assumptions.assumeTrue(PatchHooks.ASSIGNED_DAMAGE.present(),
                "../forge does not keep the step's damage assignment");
        Game game = TestBoard.twoPlayerGame(PhaseType.COMBAT_DAMAGE);
        Player you = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card bears = TestBoard.onBattlefield(game, you, "Grizzly Bears");
        Card elves = TestBoard.onBattlefield(game, you, "Llanowar Elves");
        Card giant = TestBoard.onBattlefield(game, you, "Hill Giant");
        Card wall = TestBoard.onBattlefield(game, opponent, "Wall of Wood");
        Combat combat = TestBoard.blockedCombat(game, you, opponent, bears, wall, elves, giant);
        CombatShape shape = CombatShape.of(combat);
        assertTrue(combat.assignCombatDamage(false), "the step assigned no damage");
        shape.addAssignment(combat);

        String player = Refs.of(opponent);
        List<EffectEvent> events = EventAttribution.withCombatSources(List.of(
                damage(Refs.of(wall), 2),
                damage(player, 1),
                damage(player, 5),
                damage(player, 3).cause(Refs.of(giant))), shape);

        assertEquals(Refs.of(bears), events.get(0).param(EventParam.SOURCE), "the wall's one assigner");
        assertEquals(Refs.of(elves), events.get(1).param(EventParam.SOURCE), "the exact amount's assigner");
        assertNull(events.get(2).param(EventParam.SOURCE), "no assigner of 5: none named");
        assertEquals(Refs.of(giant), events.get(3).param(EventParam.SOURCE), "the bus's own cause");
    }

    /** Only damage gets a source; an assignment says nothing about a life change. */
    @Test
    void aProbesOtherEventsGetNoSource() {
        List<EffectEvent> events = EventAttribution.withCombatSources(List.of(
                new EffectEvent(EventType.LIFE_CHANGE).subject("P1").param(EventParam.DELTA, -1)
                        .cause("E1")), CombatShape.of(null));

        assertFalse(events.get(0).hasParam(EventParam.SOURCE), events.get(0).toJson());
    }

    /** A prevention of nothing is no prevention (FR-035); one of something is. */
    @Test
    void aPreventionOfNothingIsRecognised() {
        assertTrue(RunParamNormalizer.preventedNothing(
                new EffectEvent(EventType.DAMAGE_PREVENTED).param(EventParam.AMOUNT, 0)));
        assertFalse(RunParamNormalizer.preventedNothing(
                new EffectEvent(EventType.DAMAGE_PREVENTED).param(EventParam.AMOUNT, 2)));
        assertFalse(RunParamNormalizer.preventedNothing(
                new EffectEvent(EventType.DAMAGE_PREVENTED)), "an unread amount says nothing");
        assertFalse(RunParamNormalizer.preventedNothing(
                new EffectEvent(EventType.DAMAGE_DEALT).param(EventParam.AMOUNT, 0)));
    }

    /**
     * At Arc Trail's DamageResolve clause, the damage to the root's target is
     * the root's; the life its controller gains (a lifelink source) and any
     * event published at another clause keep the clause's own attribution.
     */
    @Test
    void onlyADamageMapsEventsOnAnEarlierLinksTargetMove() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player other = game.getPlayers().get(1);
        Card bears = TestBoard.inPlay(game, other, "Grizzly Bears");
        SpellAbility root = TestBoard.card(game, caster, "Arc Trail").getFirstSpellAbility();
        root.setActivatingPlayer(caster);
        root.getTargets().add(bears);
        SpellAbility second = root.getSubAbility();
        second.getTargets().add(other);
        SpellAbility resolve = second.getSubAbility();

        EffectEvent damage = new EffectEvent(EventType.DAMAGE_DEALT).subject(Refs.of(bears));
        EffectEvent gain = new EffectEvent(EventType.LIFE_CHANGE).subject(Refs.of(caster));
        EffectEvent died = new EffectEvent(EventType.ZONE_CHANGE).subject(Refs.of(bears));

        assertEquals("root", EventAttribution.damageLink(damage, resolve, "1"));
        assertEquals("1", EventAttribution.damageLink(gain, resolve, "1"), "the gainer is no target");
        assertEquals("0", EventAttribution.damageLink(died, second, "0"), "not a damage map's clause");
    }
}
