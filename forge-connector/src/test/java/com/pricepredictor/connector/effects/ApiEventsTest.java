package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.StaticData;
import forge.game.Direction;
import forge.game.EvenOdd;
import forge.game.Game;
import forge.game.GameRules;
import forge.game.GameType;
import forge.game.Match;
import forge.game.ability.AbilityFactory;
import forge.game.card.Card;
import forge.game.card.CardFactory;
import forge.game.combat.Combat;
import forge.game.player.Player;
import forge.game.spellability.AbilitySub;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * An effect whose outcome the engine broadcasts nowhere still describes itself:
 * its own parameters say what it did. These are the APIs that need nothing else.
 */
@ExtendWith(ForgeExtension.class)
class ApiEventsTest {

    /**
     * The failure latches are static, shared with production: without a
     * reset here, whichever test trips one first leaves every test after it --
     * including ones that never touch that latch, and any later test class in
     * the same JVM -- unable to see its own "did this print" outcome. Two
     * latches since final-fix-3.md item 4 (emitter and memo failures no
     * longer share one), so both are reset here.
     */
    @BeforeEach
    void resetEmitterFailureLatch() {
        ApiEvents.resetEmitterFailureLatchForTest();
        ApiEvents.resetMemoFailureLatchForTest();
    }

    @Test
    void anExtraTurnCarriesItsCount() {
        SpellAbility sa = TestCards.scriptedAbility("Alchemist's Gambit", "AddTurn");

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertNotNull(event);
        assertEquals(EffectEvent.TURN_ADDED, event.type());
        assertEquals(1, event.params().get("count"));
    }

    /**
     * Not Blinding Angel: its {@code SkipPhase} clause is the {@code Execute$}
     * of a triggered ability, which lives on the {@code Trigger} object, not
     * in {@code card.getCurrentState().getSpellAbilities()} or any
     * {@code SubAbility$} chain hung off it -- {@code scriptedAbility} cannot
     * reach it and never will. Moment of Silence casts {@code SP$ SkipPhase}
     * as its own root ability.
     */
    @Test
    void aSkippedPhaseNamesThePhase() {
        SpellAbility sa = TestCards.scriptedAbility("Moment of Silence", "SkipPhase");

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertEquals(EffectEvent.PHASE_SKIPPED, event.type());
        assertNotNull(event.params().get("phase"));
    }

    @Test
    void anApiWithNoEmitterIsSilent() {
        SpellAbility sa = TestCards.scriptedAbility("Lightning Bolt", "DealDamage");

        assertNull(ApiEvents.after(sa, ApiEvents.before(sa)),
                "damage already arrives on the bus; a second event would double-count it");
    }

    /** Meditate: {@code SP$ Draw | ... | SubAbility$ DBSkip}, {@code DB$ SkipTurn}. */
    @Test
    void askippedTurnCarriesItsCount() {
        SpellAbility sa = TestCards.scriptedAbility("Meditate", "SkipTurn");

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertEquals(EffectEvent.TURN_SKIPPED, event.type());
        assertEquals(1, event.params().get("count"));
    }

    /** Full Throttle: {@code SP$ AddPhase | ExtraPhase$ Combat | NumPhases$ 2}. */
    @Test
    void anAddedPhaseNamesThePhaseAndItsCount() {
        SpellAbility sa = TestCards.scriptedAbility("Full Throttle", "AddPhase");

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertEquals(EffectEvent.PHASE_ADDED, event.type());
        assertEquals("Combat", event.params().get("phase"));
        assertEquals(2, event.params().get("count"));
    }

    /**
     * Darkpact's clause carries no {@code DefinedPlayer}, so the new owner
     * falls back to the activator -- {@code null} here, since
     * {@link TestCards#game()} is shared and nothing in this suite ever
     * activates an ability as a player. What this pins is that the fallback
     * path runs end to end rather than that it names anyone in particular;
     * {@link ApiEvents} names the {@code DefinedPlayer}-driven branch's real
     * printed counter-examples (Tempest Efreet, Bronze Tablet) in its own
     * javadoc, since neither is reachable without a live target or a prior
     * {@code Remembered} player, which this harness cannot cheaply build.
     */
    @Test
    void anOwnershipChangeWithNoNamedOwnerFallsBackToTheActivator() {
        SpellAbility sa = TestCards.scriptedAbility("Darkpact", "GainOwnership");

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertEquals(EffectEvent.OWNERSHIP_CHANGE, event.type());
    }

    /**
     * {@code SpellAbilityEffect.getPlayers} ({@code SpellAbilityEffect.java:314})
     * splits a {@code " & "}-joined {@code Defined}/{@code DefinedPlayer} and
     * unions each token; a rule that read the raw string as one token would
     * hand it to {@code AbilityUtils.getDefinedPlayers} unrecognized, which
     * falls through to its catch-all ({@code AbilityUtils.java:1187-1190}:
     * {@code players.addAll(game.getPlayersInTurnOrder())}) -- empty on
     * {@link TestCards#game()}'s zero-player game, so {@code ownerOf} would
     * fall back to the activator instead.
     *
     * <p>{@code You & You} (the first version of this test) could not tell
     * that fallback apart from a correct split, because both halves name the
     * activator either way -- the test passed unchanged with the split
     * deleted. {@code Targeted & You} can: split, the owner is the targeted
     * player; unsplit, the whole string matches nothing and {@code ownerOf}
     * falls back to the activator, a different id. Mutation-tested: deleting
     * the {@code " & "} split in {@code definedPlayers} makes this fail with
     * the activator's id where the targeted player's was expected, then
     * passes again with the split restored.
     */
    @Test
    void ownershipDefinedPlayerSplitsOnAmpersand() {
        SpellAbility sa = AbilityFactory.getAbility(
                "SP$ GainOwnership | ValidTgts$ Card | DefinedPlayer$ Targeted & You",
                TestCards.build("Grizzly Bears"));
        Player activator = new Player("activator", TestCards.game(), 91501);
        Player targeted = new Player("targeted", TestCards.game(), 91502);
        sa.setActivatingPlayer(activator);
        sa.resetTargets();
        sa.getTargets().add(targeted);

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertEquals(SnapshotBuilder.playerId(targeted), event.params().get("owner"));
    }

    // ── subject is who the effect affects, not who cast it ──────────────

    /**
     * Eon Frolicker: {@code DB$ AddTurn | ValidTgts$ Opponent | NumTurns$ 1}
     * -- not reachable through {@link TestCards#scriptedAbility}, the same
     * way Blinding Angel's {@code SkipPhase} is not (it is a triggered
     * ability's {@code Execute$}), so built directly. {@code
     * AddTurnEffect.resolve} iterates {@code getTargetPlayers(sa)}, and
     * {@code ValidTgts$ Opponent} guarantees the targeted player is never the
     * activator: this is the engine's own common case, not an edge one.
     */
    @Test
    void anExtraTurnNamesTheTargetedPlayerNotTheActivator() {
        SpellAbility sa = AbilityFactory.getAbility(
                "SP$ AddTurn | ValidTgts$ Opponent | NumTurns$ 1",
                TestCards.build("Grizzly Bears"));
        Player activator = new Player("activator", TestCards.game(), 91001);
        Player targeted = new Player("targeted", TestCards.game(), 91002);
        sa.setActivatingPlayer(activator);
        sa.resetTargets();
        sa.getTargets().add(targeted);

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertEquals(List.of(SnapshotBuilder.playerId(targeted)), event.subjects());
    }

    /**
     * {@code getTargetPlayers} returns every target, not one: two players
     * targeted must read as two subjects, not the first with the second
     * silently dropped.
     */
    @Test
    void anExtraTurnNamesEveryTargetedPlayerNotJustTheFirst() {
        SpellAbility sa = AbilityFactory.getAbility(
                "SP$ AddTurn | ValidTgts$ Player | NumTurns$ 1",
                TestCards.build("Grizzly Bears"));
        Player first = new Player("first", TestCards.game(), 91101);
        Player second = new Player("second", TestCards.game(), 91102);
        sa.resetTargets();
        sa.getTargets().add(first);
        sa.getTargets().add(second);

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertEquals(
                List.of(SnapshotBuilder.playerId(first), SnapshotBuilder.playerId(second)),
                event.subjects());
    }

    /** Same rule shape as {@code AddTurn}; {@code SkipTurnEffect.resolve} matches it. */
    @Test
    void aSkippedTurnNamesTheTargetedPlayerNotTheActivator() {
        SpellAbility sa = AbilityFactory.getAbility(
                "SP$ SkipTurn | ValidTgts$ Opponent | NumTurns$ 1",
                TestCards.build("Grizzly Bears"));
        Player activator = new Player("activator", TestCards.game(), 91201);
        Player targeted = new Player("targeted", TestCards.game(), 91202);
        sa.setActivatingPlayer(activator);
        sa.resetTargets();
        sa.getTargets().add(targeted);

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertEquals(List.of(SnapshotBuilder.playerId(targeted)), event.subjects());
    }

    /** Same rule shape again; {@code SkipPhaseEffect.resolve} matches it too. */
    @Test
    void aSkippedPhaseNamesTheTargetedPlayerNotTheActivator() {
        SpellAbility sa = AbilityFactory.getAbility(
                "SP$ SkipPhase | ValidTgts$ Opponent | Phase$ BeginCombat",
                TestCards.build("Grizzly Bears"));
        Player activator = new Player("activator", TestCards.game(), 91301);
        Player targeted = new Player("targeted", TestCards.game(), 91302);
        sa.setActivatingPlayer(activator);
        sa.resetTargets();
        sa.getTargets().add(targeted);

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertEquals(List.of(SnapshotBuilder.playerId(targeted)), event.subjects());
    }

    /**
     * {@code AddPhaseEffect.resolve} reads {@code sa.getActivatingPlayer()}
     * exclusively and has no targeting or {@code Defined$} concept -- unlike
     * its three siblings above, the activator genuinely is the subject here,
     * and this pins that the fix was not generalized to a rule that never
     * needed it.
     */
    @Test
    void anAddedPhaseNamesTheActivator() {
        SpellAbility sa = TestCards.scriptedAbility("Full Throttle", "AddPhase");
        Player activator = new Player("activator", TestCards.game(), 91401);
        sa.setActivatingPlayer(activator);

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertEquals(List.of(SnapshotBuilder.playerId(activator)), event.subjects());
    }

    /**
     * {@code affectedPlayers} has two branches; the five subject tests above
     * all exercise {@code usesTargeting()}. The {@code Defined$}-only branch
     * -- the one Ral Zarek, Guest Lecturer actually uses
     * ({@code ral_zarek_guest_lecturer.txt:9}, {@code SVar:DBSkipTurn:DB$
     * SkipTurn | NumTurns$ X | Defined$ Targeted}, no {@code ValidTgts$} on
     * the clause itself) -- had no test asserting <em>who</em>, only the two
     * pre-existing {@code SkipTurn} tests' counts.
     *
     * <p>{@code Defined$ Targeted} on a non-targeting clause has to reach
     * through {@code getAllTargetChoices()} to a <em>different</em> link in
     * the same ability chain that does target -- {@code getAllTargetChoices}
     * walks {@code getRootAbility()} through {@code getSubAbility()} and
     * collects {@code getTargets()} only from links where
     * {@code usesTargeting()} is true -- so this builds the same two-link
     * shape: a targeting root ({@code Pump}, standing in for Ral Zarek's
     * {@code FlipCoin}) with a non-targeting {@code SkipTurn} wired on as its
     * sub-ability.
     *
     * <p>The real card wires its clause with {@code HeadsSubAbility$} rather
     * than {@code SubAbility$}, but {@code setAdditionalAbility} calls the same
     * {@code setParent} this test's {@code setSubAbility} does, so the parent
     * link {@code getAllTargetChoices} walks is the identical one.
     */
    @Test
    void aSkippedTurnThroughDefinedTargetedNamesTheRootsTarget() {
        SpellAbility root = AbilityFactory.getAbility(
                "SP$ Pump | ValidTgts$ Player | NumAtt$ +0 | NumDef$ +0",
                TestCards.build("Grizzly Bears"));
        SpellAbility sub = AbilityFactory.getAbility(
                "DB$ SkipTurn | Defined$ Targeted | NumTurns$ 1", root.getHostCard());
        root.setSubAbility((AbilitySub) sub);
        Player targeted = new Player("targeted", TestCards.game(), 91601);
        root.resetTargets();
        root.getTargets().add(targeted);

        EffectEvent event = ApiEvents.after(sub, ApiEvents.before(sub));

        assertEquals(EffectEvent.TURN_SKIPPED, event.type());
        assertEquals(List.of(SnapshotBuilder.playerId(targeted)), event.subjects());
    }

    // ── Task 8: choices, healed damage, granted abilities, retargets ────

    @Test
    void aChosenColourIsAChoice() {
        SpellAbility sa = TestCards.scriptedAbility("Akroma's Blessing", "ChooseColor");
        Object memo = ApiEvents.before(sa);
        sa.getHostCard().setChosenColors(com.google.common.collect.ImmutableList.of("red"));

        EffectEvent event = ApiEvents.after(sa, memo);

        assertEquals(EffectEvent.CHOICE_MADE, event.type());
        assertEquals("color", event.params().get("choice_kind"));
        assertEquals("red", event.params().get("value"));
    }

    /**
     * {@code choice()}'s first branch checks {@code host.hasChosenColor()}.
     * The brief's original draft checked {@code host.getChosenColor() !=
     * null} instead; {@code Card.getChosenColor()} never returns {@code
     * null} for an unset colour -- it returns {@code ""} -- so that check is
     * always true and would report <em>every</em> clause routed through
     * {@code choice()} as a colour choice, this one included, regardless of
     * which of the {@code choice_made} APIs actually ran.
     *
     * <p>This is also the only single-API test in the suite that exercises a
     * {@code choice_kind} other than {@code "color"}, so it is what would
     * catch that regression on its own (the broader {@code
     * everyChoiceMadeApiHasAReadableChoice} guard below would too, across
     * every branch at once).
     *
     * <p>Originally pointed at {@code ChooseSector}: fix round 1 removed that
     * rule (see the comment in {@code ApiEvents.RULES} above {@code
     * HealDamage} -- {@code event_schema.py} excludes it as unreachable in
     * this corpus's sealed/draft pools), so this now points at {@code
     * ChooseDirection} instead, which stayed. {@code
     * ChooseDirectionEffect.resolve()} always calls {@code
     * setChosenDirection}, verified directly against {@code
     * ChooseDirectionEffect.java} rather than assumed.
     */
    @Test
    void aChosenDirectionIsAChoiceNotAMiscolouredOne() {
        SpellAbility sa = AbilityFactory.getAbility("SP$ ChooseDirection", TestCards.build("Grizzly Bears"));
        sa.getHostCard().setChosenDirection(Direction.Left);

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertEquals(EffectEvent.CHOICE_MADE, event.type());
        assertEquals("direction", event.params().get("choice_kind"));
        assertEquals("Left", event.params().get("value"));
    }

    /**
     * The general form of the bug class in the test above and in Ruling
     * R18's missed branches: for every {@code RULES} entry the schema
     * declares as {@code choice_made} (fixture keys asserted equal to {@code
     * ApiEvents.choiceMadeApiKeysForTest()}, read live off the table rather
     * than hand-counted), simulate that API's own effect class writing its
     * one host field, and require {@code choice()} to read it back as that
     * API's own kind. An invariant, not a count: it keeps holding as rules
     * are added, and it fails loudly -- either on the set comparison, if a
     * new {@code choice_made} API has no fixture yet, or on the per-API
     * assertions, if a fixture exists but the matching {@code choice()}
     * branch does not -- the moment either half drifts from the other.
     *
     * <p>{@code ChooseSector} is deliberately not a key here: fix round 1
     * removed it from {@code RULES} entirely (see the comment there), so it
     * is correctly absent from {@code choiceMadeApiKeysForTest()} too, not
     * an exemption carved out of this test.
     *
     * <p>Each fixture calls the exact setter that API's real effect class
     * calls: {@code ChooseColorEffect.setChosenColors}, {@code
     * ChooseTypeEffect.setChosenType}, {@code ChooseCardEffect}/{@code
     * ChooseSourceEffect.setChosenCards} (both share the one accessor, so
     * both correctly read back as {@code "cards"}), {@code
     * ChooseCardNameEffect.addNamedCard}, {@code
     * ChooseNumberEffect.setChosenNumber}, {@code
     * ChoosePlayerEffect.setChosenPlayer}, {@code
     * ChooseDirectionEffect.setChosenDirection}, {@code
     * ChooseEvenOddEffect.setChosenEvenOdd}, and {@code
     * ChooseGenericEffect}'s {@code SetChosenMode$ True} path's {@code
     * setChosenMode} -- on a fresh host each time, so one API's fixture
     * cannot leak into another's read.
     */
    @Test
    void everyChoiceMadeApiHasAReadableChoice() {
        record ChoiceFixture(String expectedKind, Consumer<Card> mutate) {
        }

        Map<String, ChoiceFixture> fixtures = new TreeMap<>();
        fixtures.put("ChooseColor", new ChoiceFixture("color",
                host -> host.setChosenColors(List.of("blue"))));
        fixtures.put("ChooseType", new ChoiceFixture("type",
                host -> host.setChosenType("Goblin")));
        fixtures.put("ChooseCard", new ChoiceFixture("cards",
                host -> host.setChosenCards(List.of(TestCards.build("Grizzly Bears")))));
        fixtures.put("ChooseSource", new ChoiceFixture("cards",
                host -> host.setChosenCards(List.of(TestCards.build("Grizzly Bears")))));
        fixtures.put("NameCard", new ChoiceFixture("card_name",
                host -> host.addNamedCard("Grizzly Bears")));
        fixtures.put("ChooseNumber", new ChoiceFixture("number",
                host -> host.setChosenNumber(3)));
        fixtures.put("ChoosePlayer", new ChoiceFixture("player",
                host -> host.setChosenPlayer(new Player("chooser", TestCards.game(), 93001))));
        fixtures.put("ChooseDirection", new ChoiceFixture("direction",
                host -> host.setChosenDirection(Direction.Left)));
        fixtures.put("ChooseEvenOdd", new ChoiceFixture("even_odd",
                host -> host.setChosenEvenOdd(EvenOdd.Odd)));
        fixtures.put("GenericChoice", new ChoiceFixture("mode",
                host -> host.setChosenMode("Abzan")));

        assertEquals(fixtures.keySet(), ApiEvents.choiceMadeApiKeysForTest(),
                "the fixture table above and RULES's live choice_made keys have "
                        + "drifted apart -- add a fixture (and, if it is new, a "
                        + "choice() branch to read it) for whichever API changed");

        for (Map.Entry<String, ChoiceFixture> fixture : fixtures.entrySet()) {
            Card host = TestCards.build("Grizzly Bears");
            SpellAbility sa = AbilityFactory.getAbility("SP$ " + fixture.getKey(), host);
            fixture.getValue().mutate().accept(host);

            EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

            assertNotNull(event, fixture.getKey() + " -- choice() reported nothing");
            assertEquals(EffectEvent.CHOICE_MADE, event.type(), fixture.getKey());
            assertEquals(
                    fixture.getValue().expectedKind(), event.params().get("choice_kind"),
                    fixture.getKey());
            assertNotNull(event.params().get("value"), fixture.getKey() + " -- empty value");
        }
    }

    /**
     * HealDamageEffect calls healDamage() and leaves zero behind, so the amount
     * exists only before the clause. This is the case the before-half of the
     * hook was added for.
     *
     * <p>Rewritten in fix round 2 (task-8-fix-2.md, Finding 1). The original
     * version called {@code sa.resetTargets(); sa.getTargets().add(target);}
     * directly on the {@code HealDamage} clause -- a state the real engine
     * never produces for this API: {@code HealDamageEffect.resolve()} reads
     * its cards through {@code SpellAbilityEffect.getTargetCards(sa)}, which
     * only consults {@code sa.getTargets()} when {@code sa.usesTargeting()}
     * is true, and neither real cardsfolder usage declares any targeting on
     * the {@code HealDamage} clause itself -- {@code pyramids.txt}'s is
     * {@code Defined$ ReplacedCard}, {@code wolverine_fierce_fighter.txt}'s
     * is {@code Defined$ ReplacedTarget}. The old test was internally
     * consistent and would still have passed with {@code DAMAGE_BEFORE}
     * reading {@code sa.getTargets()} unconditionally, which is exactly the
     * bug: confirmed by reverting {@code ApiEvents}'s fix and re-running
     * this rewritten test first -- it failed with {@code amount=0} against
     * {@code assertEquals(3, ...)}, the same silent-wrong-zero the finding
     * describes, before the fix was applied.
     *
     * <p>Builds the shape production actually uses instead: no {@code
     * ValidTgts$}/{@code Tgt$} on the {@code HealDamage} clause itself
     * (matching both real cards), resolving its card through {@code
     * Defined$ Targeted} instead -- the same {@code getAllTargetChoices()}
     * mechanism {@link #aSkippedTurnThroughDefinedTargetedNamesTheRootsTarget}
     * already exercises and pins for players, mirrored here for a card, so
     * the healed creature is a different object than the sub-ability's own
     * host and a hardcoded {@code sa.getHostCard()} shortcut could not pass
     * this by coincidence.
     */
    @Test
    void healedDamageIsReadFromBeforeTheClause() {
        SpellAbility root = AbilityFactory.getAbility(
                "SP$ Pump | ValidTgts$ Creature | NumAtt$ +0 | NumDef$ +0",
                TestCards.build("Grizzly Bears"));
        SpellAbility sub = AbilityFactory.getAbility(
                "DB$ HealDamage | Defined$ Targeted", root.getHostCard());
        root.setSubAbility((AbilitySub) sub);
        Card target = TestCards.build("Runeclaw Bear");
        target.setDamage(3);
        root.resetTargets();
        root.getTargets().add(target);

        Object memo = ApiEvents.before(sub);
        target.setDamage(0);

        EffectEvent event = ApiEvents.after(sub, memo);

        assertEquals(EffectEvent.DAMAGE_HEALED, event.type());
        assertEquals(3, event.params().get("amount"));
    }

    /**
     * Dragonshift: {@code SP$ Animate | ... | RemoveAllAbilities$ True | ...},
     * a root ability. No reachable root-level Animate/AnimateAll script in
     * the cardsfolder pairs with an {@code Abilities$} grant list (the
     * common shapes use {@code Keywords$}, {@code staticAbilities$} or
     * {@code Triggers$} instead), so {@code abilities} here is the {@code ""}
     * fallback; {@code removed} is the distinct, identifiable value this
     * card actually pins.
     */
    @Test
    void anAnimateClauseNamesWhetherItStrippedAbilities() {
        SpellAbility sa = TestCards.scriptedAbility("Dragonshift", "Animate");

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertEquals(EffectEvent.ABILITY_CHANGE, event.type());
        assertEquals(true, event.params().get("removed"));
    }

    /** Impractical Joke: {@code SP$ Effect | StaticAbilities$ STCantPrevent | ...}. */
    @Test
    void anEffectClauseNamesItsStaticAbilities() {
        SpellAbility sa = TestCards.scriptedAbility("Impractical Joke", "Effect");

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertEquals(EffectEvent.CONTINUOUS_EFFECT_CREATED, event.type());
        assertEquals("STCantPrevent", event.params().get("layers"));
    }

    /**
     * Wild Ricochet: {@code SP$ ChangeTargets | TargetType$ Spell | ...}, a
     * root ability -- built and inspected directly rather than resolved,
     * since resolution needs a live {@code SpellAbilityStackInstance} this
     * harness does not construct. The retarget helper reads {@code
     * sa.getTargets().getTargetSpells()} (the spell(s) the clause itself
     * targets) and then each one's own, post-change targets, so wiring a
     * hand-built "changed" ability directly onto {@code sa}'s targets and
     * giving that ability its own new target exercises the same read a real
     * resolution would leave behind.
     */
    @Test
    void changeTargetsNamesWhatTheRetargetedSpellPointsAtNow() {
        SpellAbility sa = TestCards.scriptedAbility("Wild Ricochet", "ChangeTargets");
        SpellAbility changed = AbilityFactory.getAbility(
                "SP$ Pump | ValidTgts$ Creature | NumAtt$ +0 | NumDef$ +0",
                TestCards.build("Grizzly Bears"));
        Card newTarget = TestCards.build("Runeclaw Bear");
        changed.resetTargets();
        changed.getTargets().add(newTarget);
        sa.resetTargets();
        sa.getTargets().add(changed);

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertEquals(EffectEvent.TARGETS_CHANGED, event.type());
        assertEquals(List.of(SnapshotBuilder.entityId(newTarget)), event.params().get("targets"));
    }

    /**
     * Finding 2 (task-8-fix-2.md): redirecting a burn spell at a
     * <em>player</em> is the classic use of {@code ChangeTargets} --
     * Deflection, Misdirection and Bolt Bend are all real, and all reach
     * {@code ChangeTargetsEffect}'s default "choose any new legal target"
     * branch (no {@code RandomTarget}/{@code DefinedMagnet}/{@code
     * ChangeSingleTarget} param), which can retarget at a player as readily
     * as a card. The original {@code retargeted()} iterated only {@code
     * getTargetCards()}, which {@code TargetChoices} filters to {@code Card}
     * instances -- a {@code Player} target was silently dropped, and the
     * event still fired with {@code targets: []}, so the channel looked
     * wired while reporting nothing distinguishable from "retargeted at
     * nothing."
     *
     * <p>Confirmed this failed against the pre-fix code before fixing it:
     * {@code event.params().get("targets")} was {@code []}, not the
     * targeted player's id.
     */
    @Test
    void changeTargetsNamesARetargetedPlayerNotJustCards() {
        SpellAbility sa = TestCards.scriptedAbility("Deflection", "ChangeTargets");
        SpellAbility changed = AbilityFactory.getAbility(
                "SP$ DealDamage | ValidTgts$ Any | NumDmg$ 3",
                TestCards.build("Grizzly Bears"));
        Player newTarget = new Player("redirect-target", TestCards.game(), 94001);
        changed.resetTargets();
        changed.getTargets().add(newTarget);
        sa.resetTargets();
        sa.getTargets().add(changed);

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertEquals(EffectEvent.TARGETS_CHANGED, event.type());
        assertEquals(List.of(SnapshotBuilder.playerId(newTarget)), event.params().get("targets"));
    }

    /**
     * Finding 3, minor (task-8-fix-2.md): {@code ChooseNumberEffect.java:90}
     * calls {@code setChosenNumber(chosen, true)} -- the secret variant that
     * skips the trackable view update -- when {@code Secretly$} is set and
     * {@code KeepSecret$} is not. The general guard's {@code ChooseNumber}
     * fixture only exercises the plain setter; both variants write the same
     * {@code chosenNumber} field
     * ({@code setChosenNumber(int, boolean)}), so {@code choice()} was
     * already correct either way -- this is coverage for that claim, not a
     * fix for a bug, and it passes unchanged on both sides of this round's
     * other two fixes.
     */
    @Test
    void aSecretlyChosenNumberIsStillAChoice() {
        Card host = TestCards.build("Grizzly Bears");
        host.setChosenNumber(7, true);
        SpellAbility sa = AbilityFactory.getAbility("SP$ ChooseNumber", host);

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertEquals(EffectEvent.CHOICE_MADE, event.type());
        assertEquals("number", event.params().get("choice_kind"));
        assertEquals("7", event.params().get("value"));
    }

    // ── the once-latched failure report ──────────────────────────────────

    /**
     * Mirrors {@code EffectRecordClauseHookTest
     * .testAThrowingListenerIsReportedExactlyOnceOnStderr} in the forge repo:
     * capture {@code System.err}, call the package-private seam directly with
     * a synthetic exception rather than contrive a rule that reliably throws,
     * and assert both halves of the promise -- one line naming the event type
     * and the exception, and silence on every call after the first, which is
     * the half inspection of the {@code compareAndSet} alone cannot establish.
     */
    @Test
    void anEmitterFailureIsReportedExactlyOnceOnStderr() {
        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        PrintStream original = System.err;
        System.setErr(new PrintStream(captured, true));
        try {
            ApiEvents.reportEmitterFailure("turn_added", new RuntimeException("boom"));
            ApiEvents.reportEmitterFailure("turn_added", new RuntimeException("boom"));
            ApiEvents.reportEmitterFailure("x_changed", new RuntimeException("boom again"));
        } finally {
            System.setErr(original);
        }

        String output = captured.toString();
        long lineCount = output.isBlank() ? 0 : output.strip().lines().count();
        assertEquals(1, lineCount, "exactly one report despite three throws: " + output);
        assertTrue(output.contains("turn_added"), output);
        assertTrue(output.contains("boom"), output);
    }

    @Test
    void anEmitterThatNeverThrowsReportsNothing() {
        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        PrintStream original = System.err;
        System.setErr(new PrintStream(captured, true));
        try {
            SpellAbility sa = TestCards.scriptedAbility("Alchemist's Gambit", "AddTurn");
            ApiEvents.after(sa, ApiEvents.before(sa));
        } finally {
            System.setErr(original);
        }

        assertEquals("", captured.toString());
    }

    // ── final-fix-3.md item 4: the memo half's own report ────────────────

    /** As {@link #anEmitterFailureIsReportedExactlyOnceOnStderr}, for the memo half. */
    @Test
    void aMemoFailureIsReportedExactlyOnceOnStderr() {
        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        PrintStream original = System.err;
        System.setErr(new PrintStream(captured, true));
        try {
            ApiEvents.reportMemoFailure("turn_added", new RuntimeException("boom"));
            ApiEvents.reportMemoFailure("turn_added", new RuntimeException("boom"));
            ApiEvents.reportMemoFailure("x_changed", new RuntimeException("boom again"));
        } finally {
            System.setErr(original);
        }

        String output = captured.toString();
        long lineCount = output.isBlank() ? 0 : output.strip().lines().count();
        assertEquals(1, lineCount, "exactly one report despite three throws: " + output);
        assertTrue(output.contains("turn_added"), output);
        assertTrue(output.contains("boom"), output);
    }

    /**
     * The wording fix itself: a memo failure must name the memo, not the
     * emitter -- before this fix, {@link ApiEvents#before} routed its catch
     * through {@link ApiEvents#reportEmitterFailure}, which unconditionally
     * printed "the emitter for ... threw" regardless of which half actually
     * failed, sending an operator to the wrong function.
     */
    @Test
    void aMemoFailureNamesTheMemoNotTheEmitter() {
        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        PrintStream original = System.err;
        System.setErr(new PrintStream(captured, true));
        try {
            ApiEvents.reportMemoFailure("turn_added", new RuntimeException("boom"));
        } finally {
            System.setErr(original);
        }

        String output = captured.toString();
        assertTrue(output.contains("the memo for"), output);
        assertFalse(output.contains("the emitter for"), output);
    }

    /**
     * The latch-independence fix: before this, both halves shared {@code
     * EMITTER_FAILURE_REPORTED}, so a memo failure on one clause silenced a
     * later, unrelated, genuine emitter failure on a different one for the
     * rest of the JVM's run. Reversed here: a memo failure first must not
     * stop a later emitter failure from printing its own line, and vice
     * versa.
     */
    @Test
    void aMemoFailureDoesNotSilenceALaterEmitterFailure() {
        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        PrintStream original = System.err;
        System.setErr(new PrintStream(captured, true));
        try {
            ApiEvents.reportMemoFailure("turn_added", new RuntimeException("memo boom"));
            ApiEvents.reportEmitterFailure("x_changed", new RuntimeException("emitter boom"));
        } finally {
            System.setErr(original);
        }

        String output = captured.toString();
        long lineCount = output.isBlank() ? 0 : output.strip().lines().count();
        assertEquals(2, lineCount, "one report from each latch, independently: " + output);
        assertTrue(output.contains("memo boom"), output);
        assertTrue(output.contains("emitter boom"), output);
    }

    /**
     * The wiring, not just the message: {@link ApiEvents#before} itself,
     * when a real rule's memo throws, must report through {@link
     * ApiEvents#reportMemoFailure} rather than {@link
     * ApiEvents#reportEmitterFailure} -- the actual pre-fix defect, which a
     * test that only calls {@code reportMemoFailure} directly (the two tests
     * above) cannot see, since it never exercises {@code before}'s own catch
     * block. Same throwing-memo shape {@code
     * ClauseContractTest.aThrowingMemoDoesNotDesyncTheSiblingsPop} uses.
     */
    @Test
    void beforeReportsAThrowingMemoAsAMemoNotAnEmitter() {
        SpellAbility sa = AbilityFactory.getAbility(
                "SP$ SkipTurn | ValidTgts$ Player | NumTurns$ 1",
                TestCards.build("Runeclaw Bear"));
        ApiEvents.setRuleOverrideForTest("SkipTurn", new ApiEvents.Rule(
                EffectEvent.TURN_SKIPPED,
                (s, host) -> {
                    throw new IllegalStateException("boom (memo)");
                },
                (s, host, memo) -> new EffectEvent(EffectEvent.TURN_SKIPPED)));

        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        PrintStream original = System.err;
        System.setErr(new PrintStream(captured, true));
        try {
            Object memo = ApiEvents.before(sa);
            assertNull(memo, "a failed memo still reads back as null");
        } finally {
            System.setErr(original);
            ApiEvents.clearRuleOverridesForTest();
        }

        String output = captured.toString();
        assertTrue(output.contains("the memo for"), output);
        assertFalse(output.contains("the emitter for"), output);
    }

    /** The same, in the other order, since neither latch is checked first by construction. */
    @Test
    void anEmitterFailureDoesNotSilenceALaterMemoFailure() {
        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        PrintStream original = System.err;
        System.setErr(new PrintStream(captured, true));
        try {
            ApiEvents.reportEmitterFailure("x_changed", new RuntimeException("emitter boom"));
            ApiEvents.reportMemoFailure("turn_added", new RuntimeException("memo boom"));
        } finally {
            System.setErr(original);
        }

        String output = captured.toString();
        long lineCount = output.isBlank() ? 0 : output.strip().lines().count();
        assertEquals(2, lineCount, "one report from each latch, independently: " + output);
        assertTrue(output.contains("memo boom"), output);
        assertTrue(output.contains("emitter boom"), output);
    }

    // ── Task 12: the eight highest-reach unemitted types ─────────────────

    /**
     * A fresh {@code Game}, not {@link TestCards#game()} -- combat state
     * ({@code PhaseHandler.setCombat}) is per-{@code Game} and this suite's
     * shared game is used by every other test in this class; mutating combat
     * on it would leak across tests that never asked for it.
     */
    private static Game freshGame() {
        GameRules rules = new GameRules(GameType.Constructed);
        return new Game(List.of(), rules, new Match(rules, List.of(), "ApiEventsCombatTest"));
    }

    private static Card cardIn(Game game, String name) {
        return CardFactory.getCard(
                StaticData.instance().getCommonCards().getCard(name), null,
                TestCards.nextCardId(), game);
    }

    // ── delayed_trigger_created ───────────────────────────────────────────

    @Test
    void aDelayedTriggerNamesTheSchedulingCard() {
        SpellAbility sa = AbilityFactory.getAbility(
                "DB$ DelayedTrigger | Mode$ Phase | Phase$ EndCombat | ValidPlayer$ Player | "
                        + "TriggerDescription$ test",
                TestCards.build("Grizzly Bears"));

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertEquals(EffectEvent.DELAYED_TRIGGER_CREATED, event.type());
        assertEquals(List.of(SnapshotBuilder.entityId(sa.getHostCard())), event.subjects());
    }

    @Test
    void anImmediateTriggerNamesTheSchedulingCard() {
        SpellAbility sa = AbilityFactory.getAbility(
                "DB$ ImmediateTrigger | TriggerDescription$ test",
                TestCards.build("Grizzly Bears"));

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertEquals(EffectEvent.DELAYED_TRIGGER_CREATED, event.type());
        assertEquals(List.of(SnapshotBuilder.entityId(sa.getHostCard())), event.subjects());
    }

    /**
     * CR 603.12a: "once for each of those times" can be zero times.
     * {@code ImmediateTriggerEffect.resolve()} returns before registering
     * anything when {@code TriggerAmount$} calculates to <= 0. The plausible
     * wrong implementation this falsifies is a rule that always fires
     * regardless of the amount.
     */
    @Test
    void anImmediateTriggerWithZeroAmountRegistersNothing() {
        SpellAbility sa = AbilityFactory.getAbility(
                "DB$ ImmediateTrigger | TriggerAmount$ 0 | TriggerDescription$ test",
                TestCards.build("Grizzly Bears"));

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertNull(event, "TriggerAmount$ 0 registers no delayed trigger at all");
    }

    // ── replacement_applied ───────────────────────────────────────────────

    /**
     * The card whose replacement applied -- the host, not the activator (no
     * activator is even set here, which is the point: this rule reads
     * nothing but the host).
     */
    @Test
    void aReplaceCounterNamesTheHostCard() {
        SpellAbility sa = AbilityFactory.getAbility(
                "DB$ ReplaceCounter | ValidCounterType$ ENERGY | ChooseCounter$ True | Amount$ 1",
                TestCards.build("Grizzly Bears"));

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertEquals(EffectEvent.REPLACEMENT_APPLIED, event.type());
        assertEquals(List.of(SnapshotBuilder.entityId(sa.getHostCard())), event.subjects());
    }

    @Test
    void aReplaceDamageNamesTheHostCard() {
        SpellAbility sa = AbilityFactory.getAbility(
                "DB$ ReplaceDamage | Amount$ 0", TestCards.build("Grizzly Bears"));

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertEquals(EffectEvent.REPLACEMENT_APPLIED, event.type());
        assertEquals(List.of(SnapshotBuilder.entityId(sa.getHostCard())), event.subjects());
    }

    /**
     * All six {@code Replace*} APIs share one emitter; this pins that every
     * one of them is actually keyed in {@code ApiEvents.RULES} rather than
     * trusting the six hand-written entries agree with each other. A missing
     * or misspelled key here would make {@code ruleFor} return null and this
     * would fail with a {@code NullPointerException} on {@code event.type()}.
     */
    @Test
    void allSixReplaceApisMapToReplacementApplied() {
        Map<String, String> minimalParams = new TreeMap<>();
        minimalParams.put("ReplaceEffect", "VarName$ DamageAmount | VarValue$ X");
        minimalParams.put("ReplaceCounter", "ValidCounterType$ ENERGY | ChooseCounter$ True | Amount$ 1");
        minimalParams.put("ReplaceDamage", "Amount$ 0");
        minimalParams.put("ReplaceMana", "ReplaceType$ C");
        minimalParams.put("ReplaceSplitDamage", "DamageTarget$ Remembered | VarName$ Y");
        minimalParams.put("ReplaceToken", "Type$ ReplaceToken | TokenScript$ c_a_clue_draw");

        for (Map.Entry<String, String> entry : minimalParams.entrySet()) {
            SpellAbility sa = AbilityFactory.getAbility(
                    "DB$ " + entry.getKey() + " | " + entry.getValue(),
                    TestCards.build("Grizzly Bears"));

            EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

            assertEquals(EffectEvent.REPLACEMENT_APPLIED, event.type(), entry.getKey());
            assertEquals(List.of(SnapshotBuilder.entityId(sa.getHostCard())), event.subjects(),
                    entry.getKey());
        }
    }

    // ── monarch_changed ────────────────────────────────────────────────────

    /**
     * {@code GameAction.becomeMonarch} directly, not a hand-rolled {@code
     * game.setMonarch(...)} -- the same public method {@code
     * BecomeMonarchEffect.resolve()} itself calls, guards included, so this
     * exercises the real production code path around the rule under test
     * rather than a state the engine never produces.
     */
    @Test
    void becomingMonarchNamesTheNewMonarch() {
        SpellAbility sa = AbilityFactory.getAbility(
                "SP$ BecomeMonarch | ValidTgts$ Player", TestCards.build("Grizzly Bears"));
        Player target = new Player("target", TestCards.game(), 93001);
        sa.resetTargets();
        sa.getTargets().add(target);

        Object memo = ApiEvents.before(sa);
        target.getGame().getAction().becomeMonarch(target, "M12");
        EffectEvent event = ApiEvents.after(sa, memo);

        assertEquals(EffectEvent.MONARCH_CHANGED, event.type());
        assertEquals(List.of(SnapshotBuilder.playerId(target)), event.subjects());
    }

    /**
     * {@code becomeMonarch} is a real no-op (no state change, no trigger)
     * when the named player already is the monarch. The plausible wrong
     * implementation this falsifies is a rule that always names the clause's
     * own target instead of checking whether the monarchy actually changed.
     */
    @Test
    void reaffirmingTheSameMonarchNamesNoOne() {
        SpellAbility sa = AbilityFactory.getAbility(
                "SP$ BecomeMonarch | ValidTgts$ Player", TestCards.build("Grizzly Bears"));
        Player target = new Player("already-monarch", TestCards.game(), 93002);
        sa.resetTargets();
        sa.getTargets().add(target);
        target.getGame().getAction().becomeMonarch(target, "M12");

        Object memo = ApiEvents.before(sa);
        target.getGame().getAction().becomeMonarch(target, "M12");
        EffectEvent event = ApiEvents.after(sa, memo);

        assertNull(event, "the monarch did not change, so nothing should be reported");
    }

    // ── ring_tempts ────────────────────────────────────────────────────────

    @Test
    void ringTemptsNamesTheActivatorAndTheChosenRingBearer() {
        SpellAbility sa = AbilityFactory.getAbility("DB$ RingTemptsYou", TestCards.build("Grizzly Bears"));
        Player activator = new Player("tempted", TestCards.game(), 93101);
        sa.setActivatingPlayer(activator);
        Card bearer = TestCards.build("Runeclaw Bear");
        activator.setRingBearer(bearer);

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertEquals(EffectEvent.RING_TEMPTS, event.type());
        assertEquals(
                List.of(SnapshotBuilder.playerId(activator), SnapshotBuilder.entityId(bearer)),
                event.subjects());
    }

    /**
     * {@code chooseSingleEntityForEffect} returns {@code null} on an empty
     * candidate list (no creatures to become the Ring-bearer), and {@code
     * RingTemptsYouEffect.resolve()} calls {@code setRingBearer(null)}
     * unconditionally either way -- so the tempted player is still reported,
     * alone.
     */
    @Test
    void ringTemptsWithNoRingBearerNamesOnlyTheActivator() {
        SpellAbility sa = AbilityFactory.getAbility("DB$ RingTemptsYou", TestCards.build("Grizzly Bears"));
        Player activator = new Player("tempted-no-bearer", TestCards.game(), 93102);
        sa.setActivatingPlayer(activator);

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertEquals(EffectEvent.RING_TEMPTS, event.type());
        assertEquals(List.of(SnapshotBuilder.playerId(activator)), event.subjects());
    }

    // ── removed_from_combat: RemoveFromCombat ───────────────────────────────

    /**
     * Two real attacking creatures, both removed -- not a single-entry
     * fixture where "first only" and "both" would agree (Task 6's shipped
     * defect shape).
     */
    @Test
    void removeFromCombatNamesEveryCreatureActuallyRemoved() {
        Game game = freshGame();
        Player attackerCtrl = new Player("attacker", game, 93201);
        Player defender = new Player("defender", game, 93202);
        Card c1 = cardIn(game, "Grizzly Bears");
        Card c2 = cardIn(game, "Runeclaw Bear");
        Combat combat = new Combat(attackerCtrl);
        game.getPhaseHandler().setCombat(combat);
        combat.addAttacker(c1, defender);
        combat.addAttacker(c2, defender);

        SpellAbility sa = AbilityFactory.getAbility(
                "DB$ RemoveFromCombat | ValidTgts$ Creature", c1);
        sa.resetTargets();
        sa.getTargets().add(c1);
        sa.getTargets().add(c2);

        Object memo = ApiEvents.before(sa);
        combat.removeFromCombat(c1);
        combat.removeFromCombat(c2);
        EffectEvent event = ApiEvents.after(sa, memo);

        assertEquals(EffectEvent.REMOVED_FROM_COMBAT, event.type());
        assertEquals(
                List.of(SnapshotBuilder.entityId(c1), SnapshotBuilder.entityId(c2)),
                event.subjects());
    }

    /**
     * Only one of the two targets is genuinely removed -- {@code
     * RemoveFromCombatEffect}'s own per-target guards (stale LKI, not in
     * play) leave the other untouched, and this must report exactly the one
     * that actually left, not both and not neither.
     */
    @Test
    void removeFromCombatNamesOnlyTheCreatureThatWasActuallyRemoved() {
        Game game = freshGame();
        Player attackerCtrl = new Player("attacker", game, 93203);
        Player defender = new Player("defender", game, 93204);
        Card c1 = cardIn(game, "Grizzly Bears");
        Card c2 = cardIn(game, "Runeclaw Bear");
        Combat combat = new Combat(attackerCtrl);
        game.getPhaseHandler().setCombat(combat);
        combat.addAttacker(c1, defender);
        combat.addAttacker(c2, defender);

        SpellAbility sa = AbilityFactory.getAbility(
                "DB$ RemoveFromCombat | ValidTgts$ Creature", c1);
        sa.resetTargets();
        sa.getTargets().add(c1);
        sa.getTargets().add(c2);

        Object memo = ApiEvents.before(sa);
        combat.removeFromCombat(c1);
        // c2 stays attacking -- its own removal was skipped by a guard this
        // rule does not replicate.
        EffectEvent event = ApiEvents.after(sa, memo);

        assertEquals(List.of(SnapshotBuilder.entityId(c1)), event.subjects());
    }

    // ── removed_from_combat: ChangeCombatants ───────────────────────────────

    /**
     * {@code addToCombat}'s reselection branch calls {@code
     * combat.removeFromCombat(c)} immediately before re-adding the same
     * creature against the new defender (SpellAbilityEffect.java:762) --
     * simulated here with the same two public {@code Combat} calls, not a
     * hand-rolled approximation.
     */
    @Test
    void changeCombatantsNamesTheReselectedAttacker() {
        Game game = freshGame();
        Player attackerCtrl = new Player("attacker", game, 93301);
        Player oldDefender = new Player("old-defender", game, 93302);
        Player newDefender = new Player("new-defender", game, 93303);
        Card c = cardIn(game, "Grizzly Bears");
        Combat combat = new Combat(attackerCtrl);
        game.getPhaseHandler().setCombat(combat);
        combat.addAttacker(c, oldDefender);

        SpellAbility sa = AbilityFactory.getAbility(
                "DB$ ChangeCombatants | Attacking$ True", c);
        sa.resetTargets();
        sa.getTargets().add(c);

        Object memo = ApiEvents.before(sa);
        combat.removeFromCombat(c);
        combat.addAttacker(c, newDefender);
        EffectEvent event = ApiEvents.after(sa, memo);

        assertEquals(EffectEvent.REMOVED_FROM_COMBAT, event.type());
        assertEquals(List.of(SnapshotBuilder.entityId(c)), event.subjects());
    }

    /**
     * The player declines the optional reselection (or reselects the same
     * defender): {@code addToCombat} never calls {@code removeFromCombat} at
     * all in that case, so nothing should be reported. The plausible wrong
     * implementation this falsifies is a rule that reports every clause
     * targeting an attacker, whether or not a reselection actually happened.
     */
    @Test
    void changeCombatantsNamesNoOneWhenTheDefenderDidNotChange() {
        Game game = freshGame();
        Player attackerCtrl = new Player("attacker", game, 93304);
        Player defender = new Player("defender", game, 93305);
        Card c = cardIn(game, "Grizzly Bears");
        Combat combat = new Combat(attackerCtrl);
        game.getPhaseHandler().setCombat(combat);
        combat.addAttacker(c, defender);

        SpellAbility sa = AbilityFactory.getAbility(
                "DB$ ChangeCombatants | Attacking$ True", c);
        sa.resetTargets();
        sa.getTargets().add(c);

        Object memo = ApiEvents.before(sa);
        // Nothing changes: declined, or reselected the same defender.
        EffectEvent event = ApiEvents.after(sa, memo);

        assertNull(event, "the defender never changed, so no removal happened");
    }

    // ── initiative_taken ─────────────────────────────────────────────────

    /**
     * {@code GameAction.takeInitiative} directly, the same public method
     * {@code TakeInitiativeEffect.resolve()} calls.
     */
    @Test
    void takingInitiativeNamesThePlayer() {
        SpellAbility sa = AbilityFactory.getAbility(
                "SP$ TakeInitiative | ValidTgts$ Player", TestCards.build("Grizzly Bears"));
        Player target = new Player("initiative-taker", TestCards.game(), 93401);
        sa.resetTargets();
        sa.getTargets().add(target);

        target.getGame().getAction().takeInitiative(target, "BRO");
        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertEquals(EffectEvent.INITIATIVE_TAKEN, event.type());
        assertEquals(List.of(SnapshotBuilder.playerId(target)), event.subjects());
    }

    /**
     * Unlike {@code becomeMonarch}, {@code takeInitiative} fires its trigger
     * even when the named player already has the initiative -- "You can take
     * the initiative even if you already have it" is in the engine's own
     * comment. The plausible wrong implementation this falsifies is a
     * before/after diff copied from {@code BecomeMonarch}, which would wrongly
     * suppress this re-affirmation.
     */
    @Test
    void takingInitiativeAgainStillNamesThePlayer() {
        SpellAbility sa = AbilityFactory.getAbility(
                "SP$ TakeInitiative | ValidTgts$ Player", TestCards.build("Grizzly Bears"));
        Player target = new Player("already-has-it", TestCards.game(), 93402);
        sa.resetTargets();
        sa.getTargets().add(target);
        target.getGame().getAction().takeInitiative(target, "BRO");

        Object memo = ApiEvents.before(sa);
        target.getGame().getAction().takeInitiative(target, "BRO");
        EffectEvent event = ApiEvents.after(sa, memo);

        assertEquals(EffectEvent.INITIATIVE_TAKEN, event.type());
        assertEquals(List.of(SnapshotBuilder.playerId(target)), event.subjects());
    }

    /** No one processed by this clause ever took the initiative here. */
    @Test
    void noInitiativeTakenNamesNoOne() {
        SpellAbility sa = AbilityFactory.getAbility(
                "SP$ TakeInitiative | ValidTgts$ Player", TestCards.build("Grizzly Bears"));
        Player target = new Player("never-took-it", TestCards.game(), 93403);
        sa.resetTargets();
        sa.getTargets().add(target);

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertNull(event, "nobody targeted by this clause holds the initiative");
    }

    // ── text_change: ChangeText ──────────────────────────────────────────

    @Test
    void changeTextNamesEveryTargetedCard() {
        SpellAbility sa = AbilityFactory.getAbility(
                "SP$ ChangeText | ValidTgts$ Card | ChangeColorWord$ Choose Choose",
                TestCards.build("Grizzly Bears"));
        Card c1 = TestCards.build("Runeclaw Bear");
        Card c2 = TestCards.build("Grizzly Bears");
        sa.resetTargets();
        sa.getTargets().add(c1);
        sa.getTargets().add(c2);

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertEquals(EffectEvent.TEXT_CHANGE, event.type());
        assertEquals(
                List.of(SnapshotBuilder.entityId(c1), SnapshotBuilder.entityId(c2)),
                event.subjects());
    }

    /**
     * {@code ChangeTextEffect.resolve()}'s target loop still fires {@code
     * GameEventCardStatsChanged} even when the clause names no word to
     * replace at all -- the plausible wrong implementation this falsifies is
     * a rule that reports every target regardless.
     */
    @Test
    void changeTextWithNoWordToReplaceNamesNoOne() {
        SpellAbility sa = AbilityFactory.getAbility(
                "SP$ ChangeText | ValidTgts$ Card", TestCards.build("Grizzly Bears"));
        sa.resetTargets();
        sa.getTargets().add(TestCards.build("Runeclaw Bear"));

        EffectEvent event = ApiEvents.after(sa, ApiEvents.before(sa));

        assertNull(event, "no ChangeColorWord$/ChangeTypeWord$ means nothing textual changed");
    }
}
