package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Direction;
import forge.game.EvenOdd;
import forge.game.ability.AbilityFactory;
import forge.game.card.Card;
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
     * The failure latch is static, shared with production: without a reset
     * here, whichever test trips it first leaves every test after it --
     * including ones that never touch the latch, and any later test class in
     * the same JVM -- unable to see its own "did this print" outcome.
     */
    @BeforeEach
    void resetEmitterFailureLatch() {
        ApiEvents.resetEmitterFailureLatchForTest();
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
}
