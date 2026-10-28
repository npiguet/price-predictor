package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.StdStreams;
import forge.game.Direction;
import forge.game.EvenOdd;
import forge.game.Game;
import forge.game.ability.AbilityFactory;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.player.Player;
import forge.game.spellability.AbilitySub;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.function.BiConsumer;
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
     * the same JVM -- unable to see its own "did this print" outcome. The
     * emitter and the memo each have one, so both are reset here.
     */
    @BeforeEach
    void resetFailureLatches() {
        TestLatches.reset(ApiEvents.class, "EMITTER_FAILURE");
        TestLatches.reset(ApiEvents.class, "MEMO_FAILURE");
    }

    /**
     * A clause's rule and what its memo takes before the clause resolves,
     * through the real rules; {@code finish()} builds the event after it.
     */
    private static ApiEvents.ClauseObservation<?> observe(SpellAbility sa) {
        return ApiEvents.observe(sa, ApiEvents::ruleFor);
    }

    @Test
    void anExtraTurnCarriesItsCount() {
        SpellAbility sa = TestCards.scriptedAbility("Alchemist's Gambit", "AddTurn");

        EffectEvent event = observe(sa).finish();

        assertNotNull(event);
        assertEquals(EventType.TURN_ADDED, event.type());
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

        EffectEvent event = observe(sa).finish();

        assertEquals(EventType.PHASE_SKIPPED, event.type());
        assertEquals("BeginCombat", event.params().get("phase"));
    }

    @Test
    void anApiWithNoEmitterIsSilent() {
        SpellAbility sa = TestCards.scriptedAbility("Lightning Bolt", "DealDamage");

        assertNull(observe(sa).finish(),
                "damage already arrives on the bus; a second event would double-count it");
    }

    /** Meditate: {@code SP$ Draw | ... | SubAbility$ DBSkip}, {@code DB$ SkipTurn}. */
    @Test
    void askippedTurnCarriesItsCount() {
        SpellAbility sa = TestCards.scriptedAbility("Meditate", "SkipTurn");

        EffectEvent event = observe(sa).finish();

        assertEquals(EventType.TURN_SKIPPED, event.type());
        assertEquals(1, event.params().get("count"));
    }

    /** Full Throttle: {@code SP$ AddPhase | ExtraPhase$ Combat | NumPhases$ 2}. */
    @Test
    void anAddedPhaseNamesThePhaseAndItsCount() {
        SpellAbility sa = TestCards.scriptedAbility("Full Throttle", "AddPhase");

        EffectEvent event = observe(sa).finish();

        assertEquals(EventType.PHASE_ADDED, event.type());
        assertEquals("Combat", event.params().get("phase"));
        assertEquals(2, event.params().get("count"));
    }

    /**
     * Darkpact's clause carries no {@code DefinedPlayer}, so the new owner
     * falls back to the activator -- {@code null} here, since
     * {@link TestCards#game()} is shared and nothing in this suite ever
     * activates an ability as a player. What this pins is that the fallback
     * path still yields the event rather than that it names anyone in particular;
     * {@link ApiEvents} names the {@code DefinedPlayer}-driven branch's real
     * printed counter-examples (Tempest Efreet, Bronze Tablet) in its own
     * javadoc, since neither is reachable without a live target or a prior
     * {@code Remembered} player, which this harness cannot cheaply build.
     */
    @Test
    void anOwnershipChangeWithNoNamedOwnerIsStillAnOwnershipChange() {
        SpellAbility sa = TestCards.scriptedAbility("Darkpact", "GainOwnership");

        EffectEvent event = observe(sa).finish();

        assertEquals(EventType.OWNERSHIP_CHANGE, event.type());
    }

    /**
     * {@code SpellAbilityEffect.getPlayers} splits a {@code " & "}-joined
     * {@code Defined}/{@code DefinedPlayer} and
     * unions each token; a rule that read the raw string as one token would
     * hand it to {@code AbilityUtils.getDefinedPlayers} unrecognized, which
     * falls through to its catch-all ({@code
     * players.addAll(game.getPlayersInTurnOrder())}) -- empty on
     * {@link TestCards#game()}'s zero-player game, so {@code ownerOf} would
     * fall back to the activator instead.
     *
     * <p>{@code You & You} could not tell that fallback apart from a correct
     * split, because both halves name the activator either way.
     * {@code Targeted & You} can: split, the owner is the targeted player;
     * unsplit, the whole string matches nothing and {@code ownerOf} falls back
     * to the activator, a different id. Deleting the {@code " & "} split in
     * {@code definedPlayers} makes this fail with the activator's id where the
     * targeted player's was expected.
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

        EffectEvent event = observe(sa).finish();

        assertEquals(Refs.of(targeted), event.params().get("owner"));
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
     * {@code SkipTurnEffect} and {@code SkipPhaseEffect} resolve the same way,
     * so they share the rule shape and this test.
     */
    @ParameterizedTest(name = "{0}")
    @CsvSource(delimiter = ';', value = {
            "an extra turn;     SP$ AddTurn | ValidTgts$ Opponent | NumTurns$ 1;           91001",
            "a skipped turn;    SP$ SkipTurn | ValidTgts$ Opponent | NumTurns$ 1;          91201",
            "a skipped phase;   SP$ SkipPhase | ValidTgts$ Opponent | Phase$ BeginCombat;  91301",
    })
    void aTurnOrPhaseEffectNamesTheTargetedPlayerNotTheActivator(
            String ignoredCase, String script, int firstPlayerId) {
        SpellAbility sa = AbilityFactory.getAbility(script, TestCards.build("Grizzly Bears"));
        Player activator = new Player("activator", TestCards.game(), firstPlayerId);
        Player targeted = new Player("targeted", TestCards.game(), firstPlayerId + 1);
        sa.setActivatingPlayer(activator);
        sa.resetTargets();
        sa.getTargets().add(targeted);

        EffectEvent event = observe(sa).finish();

        assertEquals(List.of(Refs.of(targeted)), event.subjects());
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

        EffectEvent event = observe(sa).finish();

        assertEquals(
                List.of(Refs.of(first), Refs.of(second)),
                event.subjects());
    }

    /**
     * {@code AddPhaseEffect.resolve} reads {@code sa.getActivatingPlayer()}
     * exclusively and has no targeting or {@code Defined$} concept -- unlike
     * its three siblings above, the activator genuinely is the subject here,
     * and this pins that their targeting rule does not reach an API that has
     * none.
     */
    @Test
    void anAddedPhaseNamesTheActivator() {
        SpellAbility sa = TestCards.scriptedAbility("Full Throttle", "AddPhase");
        Player activator = new Player("activator", TestCards.game(), 91401);
        sa.setActivatingPlayer(activator);

        EffectEvent event = observe(sa).finish();

        assertEquals(List.of(Refs.of(activator)), event.subjects());
    }

    /**
     * {@code affectedPlayers} has two branches; the five subject tests above
     * all exercise {@code usesTargeting()}. This one asserts <em>who</em> on
     * the {@code Defined$}-only branch -- the one Ral Zarek, Guest Lecturer
     * uses ({@code SVar:DBSkipTurn:DB$ SkipTurn | NumTurns$ X | Defined$
     * Targeted}, no {@code ValidTgts$} on the clause itself).
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

        EffectEvent event = observe(sub).finish();

        assertEquals(EventType.TURN_SKIPPED, event.type());
        assertEquals(List.of(Refs.of(targeted)), event.subjects());
    }

    // ── choices, healed damage, granted abilities, retargets ────────────

    @Test
    void aChosenColourIsAChoice() {
        SpellAbility sa = TestCards.scriptedAbility("Akroma's Blessing", "ChooseColor");
        ApiEvents.ClauseObservation<?> observed = observe(sa);
        sa.getHostCard().setChosenColors(com.google.common.collect.ImmutableList.of("red"));

        EffectEvent event = observed.finish();

        assertEquals(EventType.CHOICE_MADE, event.type());
        assertEquals("color", event.params().get("choice_kind"));
        assertEquals("red", event.params().get("value"));
    }

    /**
     * A rule keyed by a name that is no {@code ApiType} member -- an effect
     * class's stem, say, or a member Forge renamed -- matches nothing and
     * fails silently, since {@code ruleFor} looks rules up by {@code name()}.
     */
    @Test
    void everyRuleIsKeyedByAnApiTypeMember() {
        Set<String> members = new TreeSet<>();
        for (ApiType api : ApiType.values()) {
            members.add(api.name());
        }
        Set<String> unknown = new TreeSet<>(ApiEvents.apiNames());
        unknown.removeAll(members);

        assertFalse(ApiEvents.apiNames().isEmpty());
        assertEquals(Set.of(), unknown, "RULES keys that name no ApiType member");
    }

    /**
     * For every {@code RULES} entry the schema
     * declares as {@code choice_made} (fixture keys asserted equal to every
     * {@code ApiType} whose rule emits {@code choice_made}, read live off the
     * table rather than hand-counted), simulate that API's own effect class writing its
     * one host field, and require {@code choice()} to read it back as that
     * API's own kind. An invariant, not a count: it keeps holding as rules
     * are added, and it fails loudly -- either on the set comparison, if a
     * new {@code choice_made} API has no fixture yet, or on the per-API
     * assertions, if a fixture exists but the matching {@code choice()}
     * branch does not -- the moment either half drifts from the other.
     *
     * <p>{@code choice()}'s first branch checks {@code host.hasChosenColor()},
     * not {@code host.getChosenColor() != null}: {@code
     * Card.getChosenColor()} never returns {@code null} for an unset colour --
     * it returns {@code ""} -- so that check would report <em>every</em>
     * clause routed through {@code choice()} as a colour choice. Every fixture
     * but {@code ChooseColor} catches that on its kind, and {@code
     * ChooseDirection} on its value as well. {@code ChooseDirectionEffect.resolve()}
     * always calls {@code setChosenDirection}.
     *
     * <p>{@code ChooseSector} is deliberately not a key here: it has no rule
     * in {@code RULES} (see the comment there), so no rule emits {@code
     * choice_made} for it either -- not an exemption carved out of this test.
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
        record ChoiceFixture(String expectedKind, Consumer<Card> mutate, Object expectedValue) {
            /** A fixture whose value is only required to be there. */
            ChoiceFixture(String expectedKind, Consumer<Card> mutate) {
                this(expectedKind, mutate, null);
            }
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
                host -> host.setChosenDirection(Direction.Left), "Left"));
        fixtures.put("ChooseEvenOdd", new ChoiceFixture("even_odd",
                host -> host.setChosenEvenOdd(EvenOdd.Odd)));
        fixtures.put("GenericChoice", new ChoiceFixture("mode",
                host -> host.setChosenMode("Abzan")));

        Set<String> choiceMadeApis = new TreeSet<>();
        for (ApiType api : ApiType.values()) {
            ApiEvents.Rule<?> rule = ApiEvents.ruleFor(api);
            if (rule != null && EventType.CHOICE_MADE.equals(rule.eventType())) {
                choiceMadeApis.add(api.name());
            }
        }
        assertEquals(fixtures.keySet(), choiceMadeApis,
                "the fixture table above and RULES's live choice_made keys have "
                        + "drifted apart -- add a fixture (and, if it is new, a "
                        + "choice() branch to read it) for whichever API changed");

        for (Map.Entry<String, ChoiceFixture> fixture : fixtures.entrySet()) {
            Card host = TestCards.build("Grizzly Bears");
            SpellAbility sa = AbilityFactory.getAbility("SP$ " + fixture.getKey(), host);
            fixture.getValue().mutate().accept(host);

            EffectEvent event = observe(sa).finish();

            assertNotNull(event, fixture.getKey() + " -- choice() reported nothing");
            assertEquals(EventType.CHOICE_MADE, event.type(), fixture.getKey());
            assertEquals(
                    fixture.getValue().expectedKind(), event.params().get("choice_kind"),
                    fixture.getKey());
            assertNotNull(event.params().get("value"), fixture.getKey() + " -- empty value");
            if (fixture.getValue().expectedValue() != null) {
                assertEquals(fixture.getValue().expectedValue(), event.params().get("value"),
                        fixture.getKey());
            }
        }
    }

    /**
     * HealDamageEffect calls healDamage() and leaves zero behind, so the amount
     * exists only before the clause. This is the case the before-half of the
     * hook exists for.
     *
     * <p>The {@code HealDamage} clause itself carries no targets, because the
     * real engine never gives it any: {@code HealDamageEffect.resolve()} reads
     * its cards through {@code SpellAbilityEffect.getTargetCards(sa)}, which
     * only consults {@code sa.getTargets()} when {@code sa.usesTargeting()}
     * is true, and neither real cardsfolder usage declares any targeting on
     * the {@code HealDamage} clause -- {@code pyramids.txt}'s is {@code
     * Defined$ ReplacedCard}, {@code wolverine_fierce_fighter.txt}'s is
     * {@code Defined$ ReplacedTarget}. Targets added straight onto the clause
     * would let a {@code damageBefore} memo that reads {@code sa.getTargets()}
     * unconditionally pass.
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

        ApiEvents.ClauseObservation<?> observed = observe(sub);
        target.setDamage(0);

        EffectEvent event = observed.finish();

        assertEquals(EventType.DAMAGE_HEALED, event.type());
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

        EffectEvent event = observe(sa).finish();

        assertEquals(EventType.ABILITY_CHANGE, event.type());
        assertEquals(true, event.params().get("removed"));
    }

    /** Impractical Joke: {@code SP$ Effect | StaticAbilities$ STCantPrevent | ...}. */
    @Test
    void anEffectClauseNamesItsStaticAbilities() {
        SpellAbility sa = TestCards.scriptedAbility("Impractical Joke", "Effect");

        EffectEvent event = observe(sa).finish();

        assertEquals(EventType.CONTINUOUS_EFFECT_CREATED, event.type());
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

        EffectEvent event = observe(sa).finish();

        assertEquals(EventType.TARGETS_CHANGED, event.type());
        assertEquals(List.of(Refs.of(newTarget)), event.params().get("targets"));
    }

    /**
     * Redirecting a burn spell at a
     * <em>player</em> is the classic use of {@code ChangeTargets} --
     * Deflection, Misdirection and Bolt Bend are all real, and all reach
     * {@code ChangeTargetsEffect}'s default "choose any new legal target"
     * branch (no {@code RandomTarget}/{@code DefinedMagnet}/{@code
     * ChangeSingleTarget} param), which can retarget at a player as readily
     * as a card. A {@code retargeted()} iterating only {@code
     * getTargetCards()}, which {@code TargetChoices} filters to {@code Card}
     * instances, would drop a {@code Player} target silently, and the event
     * would still fire with {@code targets: []}, so the channel would look
     * wired while reporting nothing distinguishable from "retargeted at
     * nothing."
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

        EffectEvent event = observe(sa).finish();

        assertEquals(EventType.TARGETS_CHANGED, event.type());
        assertEquals(List.of(Refs.of(newTarget)), event.params().get("targets"));
    }

    /**
     * {@code ChooseNumberEffect.resolve}
     * calls {@code setChosenNumber(chosen, true)} -- the secret variant that
     * skips the trackable view update -- when {@code Secretly$} is set and
     * {@code KeepSecret$} is not. The general guard's {@code ChooseNumber}
     * fixture only exercises the plain setter; both variants write the same
     * {@code chosenNumber} field
     * ({@code setChosenNumber(int, boolean)}), so {@code choice()} reads
     * either -- this is coverage for that claim.
     */
    @Test
    void aSecretlyChosenNumberIsStillAChoice() {
        Card host = TestCards.build("Grizzly Bears");
        host.setChosenNumber(7, true);
        SpellAbility sa = AbilityFactory.getAbility("SP$ ChooseNumber", host);

        EffectEvent event = observe(sa).finish();

        assertEquals(EventType.CHOICE_MADE, event.type());
        assertEquals("number", event.params().get("choice_kind"));
        assertEquals("7", event.params().get("value"));
    }

    // ── the once-latched failure reports ─────────────────────────────────

    /** The two halves of a rule that can throw, each reported on its own latch. */
    enum Report {
        EMITTER("the emitter for", ApiEvents::reportEmitterFailure),
        MEMO("the memo for", ApiEvents::reportMemoFailure);

        /** How the report names the half that failed. */
        final String names;
        final BiConsumer<EventType, RuntimeException> report;

        Report(String names, BiConsumer<EventType, RuntimeException> report) {
            this.names = names;
            this.report = report;
        }

        Report other() {
            return this == EMITTER ? MEMO : EMITTER;
        }
    }

    /**
     * Mirrors {@code OnceOnlyWarningTest.testRaisedThreeTimesPrintsOneLine}
     * in the forge repo:
     * call the report directly with a synthetic exception rather than contrive
     * a rule that reliably throws, and assert both halves of the promise --
     * one line naming the half that failed, the event type and the exception,
     * and silence on every call after the first, which is the half inspection
     * of the {@code compareAndSet} alone cannot establish.
     */
    @ParameterizedTest
    @EnumSource(Report.class)
    void aFailureIsReportedExactlyOnceOnStderr(Report half) throws Throwable {
        String output = StdStreams.captureStderr(() -> {
            half.report.accept(EventType.TURN_ADDED, new RuntimeException("boom"));
            half.report.accept(EventType.TURN_ADDED, new RuntimeException("boom"));
            half.report.accept(EventType.X_CHANGED, new RuntimeException("boom again"));
        });

        long lineCount = output.isBlank() ? 0 : output.strip().lines().count();
        assertEquals(1, lineCount, "exactly one report despite three throws: " + output);
        assertTrue(output.contains(half.names), output);
        assertTrue(output.contains("turn_added"), output);
        assertTrue(output.contains("boom"), output);
    }

    @Test
    void anEmitterThatNeverThrowsReportsNothing() throws Throwable {
        SpellAbility sa = TestCards.scriptedAbility("Alchemist's Gambit", "AddTurn");

        assertEquals("", StdStreams.captureStderr(() -> observe(sa).finish()));
    }

    /**
     * Each half latches its own report: a memo failure on one clause must not
     * silence a later, unrelated emitter failure on another for the rest of the
     * JVM's run, nor the reverse.
     *
     * <p>Both orders, because neither is the natural one: a report that checked
     * the other half's latch but set its own would pass one order and fail the
     * other.
     */
    @ParameterizedTest
    @EnumSource(Report.class)
    void oneHalfsFailureDoesNotSilenceALaterFailureOfTheOther(Report first) throws Throwable {
        String output = StdStreams.captureStderr(() -> {
            first.report.accept(EventType.TURN_ADDED, new RuntimeException("first boom"));
            first.other().report.accept(EventType.X_CHANGED, new RuntimeException("second boom"));
        });

        long lineCount = output.isBlank() ? 0 : output.strip().lines().count();
        assertEquals(2, lineCount, "one report from each latch, independently: " + output);
        assertTrue(output.contains("first boom"), output);
        assertTrue(output.contains("second boom"), output);
    }

    /**
     * The wiring, not just the message: {@link ApiEvents#observe} itself,
     * when a real rule's memo throws, must report through {@link
     * ApiEvents#reportMemoFailure} rather than {@link
     * ApiEvents#reportEmitterFailure}, so the line names the memo -- the
     * function to look at -- rather than the emitter. A test that only calls
     * the report directly cannot see this, since it never exercises
     * {@code observe}'s own catch block. Same throwing-memo shape {@code
     * ClauseContractTest.aThrowingMemoDoesNotDesyncTheSiblingsPop} uses.
     */
    @Test
    void observeReportsAThrowingMemoAsAMemoNotAnEmitter() throws Throwable {
        SpellAbility sa = AbilityFactory.getAbility(
                "SP$ SkipTurn | ValidTgts$ Player | NumTurns$ 1",
                TestCards.build("Runeclaw Bear"));
        ApiEvents.Rule<Object> throwingMemo = new ApiEvents.Rule<>(
                EventType.TURN_SKIPPED,
                (s, host) -> {
                    throw new IllegalStateException("boom (memo)");
                },
                (s, host, memo) -> new EffectEvent(EventType.TURN_SKIPPED));

        String output = StdStreams.captureStderr(() -> assertNull(
                ApiEvents.observe(sa, api -> throwingMemo).memo(),
                "a failed memo still reads back as null"));

        assertTrue(output.contains("the memo for"), output);
        assertFalse(output.contains("the emitter for"), output);
    }

    // ── the eight highest-reach types the bus does not emit ──────────────
    //
    // The combat tests below build their cards in a game of their own, not
    // TestCards.game(): combat state (PhaseHandler.setCombat) is per-Game and
    // the shared game is used by every other test in this class, so mutating
    // combat on it would leak across tests that never asked for it.

    // ── delayed_trigger_created ───────────────────────────────────────────

    @Test
    void aDelayedTriggerNamesTheSchedulingCard() {
        SpellAbility sa = AbilityFactory.getAbility(
                "DB$ DelayedTrigger | Mode$ Phase | Phase$ EndCombat | ValidPlayer$ Player | "
                        + "TriggerDescription$ test",
                TestCards.build("Grizzly Bears"));

        EffectEvent event = observe(sa).finish();

        assertEquals(EventType.DELAYED_TRIGGER_CREATED, event.type());
        assertEquals(List.of(Refs.of(sa.getHostCard())), event.subjects());
    }

    @Test
    void anImmediateTriggerNamesTheSchedulingCard() {
        SpellAbility sa = AbilityFactory.getAbility(
                "DB$ ImmediateTrigger | TriggerDescription$ test",
                TestCards.build("Grizzly Bears"));

        EffectEvent event = observe(sa).finish();

        assertEquals(EventType.DELAYED_TRIGGER_CREATED, event.type());
        assertEquals(List.of(Refs.of(sa.getHostCard())), event.subjects());
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

        EffectEvent event = observe(sa).finish();

        assertNull(event, "TriggerAmount$ 0 registers no delayed trigger at all");
    }

    // ── replacement_applied ───────────────────────────────────────────────

    /**
     * Every {@code Replace*} API shares one emitter, which names the card whose
     * replacement applied -- the host, not the activator (no activator is even
     * set here, which is the point: this rule reads nothing but the host). This
     * pins that each of the six is actually keyed in {@code ApiEvents.RULES}
     * rather than trusting the six hand-written entries agree with each other;
     * a missing or misspelled key would make {@code ruleFor} return null and
     * fail with a {@code NullPointerException} on {@code event.type()}.
     */
    @ParameterizedTest(name = "{0}")
    @CsvSource(delimiter = ';', value = {
            "ReplaceEffect;      VarName$ DamageAmount | VarValue$ X",
            "ReplaceCounter;     ValidCounterType$ ENERGY | ChooseCounter$ True | Amount$ 1",
            "ReplaceDamage;      Amount$ 0",
            "ReplaceMana;        ReplaceType$ C",
            "ReplaceSplitDamage; DamageTarget$ Remembered | VarName$ Y",
            "ReplaceToken;       Type$ ReplaceToken | TokenScript$ c_a_clue_draw",
    })
    void aReplaceApiNamesTheHostCardAsReplacementApplied(String api, String minimalParams) {
        SpellAbility sa = AbilityFactory.getAbility(
                "DB$ " + api + " | " + minimalParams, TestCards.build("Grizzly Bears"));

        EffectEvent event = observe(sa).finish();

        assertEquals(EventType.REPLACEMENT_APPLIED, event.type());
        assertEquals(List.of(Refs.of(sa.getHostCard())), event.subjects());
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

        ApiEvents.ClauseObservation<?> observed = observe(sa);
        target.getGame().getAction().becomeMonarch(target, "M12");
        EffectEvent event = observed.finish();

        assertEquals(EventType.MONARCH_CHANGED, event.type());
        assertEquals(List.of(Refs.of(target)), event.subjects());
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

        ApiEvents.ClauseObservation<?> observed = observe(sa);
        target.getGame().getAction().becomeMonarch(target, "M12");
        EffectEvent event = observed.finish();

        assertNull(event, "the monarch did not change, so nothing should be reported");
    }

    // ── ring_tempts ────────────────────────────────────────────────────────

    /**
     * With no creature to choose, {@code RingTemptsYouEffect.resolve()} still
     * calls {@code setRingBearer(null)}, which {@code Player.setRingBearer}
     * makes a no-op (it returns early on null) -- so an existing Ring-bearer
     * survives a re-temptation and is reported again as the second subject.
     * This end state is therefore also that shape's. The plausible wrong
     * implementation this falsifies is one that reads the no-op as an
     * unconditional clear.
     */
    @Test
    void ringTemptsNamesTheActivatorAndTheChosenRingBearer() {
        SpellAbility sa = AbilityFactory.getAbility("DB$ RingTemptsYou", TestCards.build("Grizzly Bears"));
        Player activator = new Player("tempted", TestCards.game(), 93101);
        sa.setActivatingPlayer(activator);
        Card bearer = TestCards.build("Runeclaw Bear");
        activator.setRingBearer(bearer);

        EffectEvent event = observe(sa).finish();

        assertEquals(EventType.RING_TEMPTS, event.type());
        assertEquals(
                List.of(Refs.of(activator), Refs.of(bearer)),
                event.subjects());
    }

    /**
     * {@code chooseSingleEntityForEffect} returns {@code null} on an empty
     * candidate list (no creatures to become the Ring-bearer), and {@code
     * RingTemptsYouEffect.resolve()} calls {@code setRingBearer(null)}
     * either way -- but {@code Player.setRingBearer} is {@code if
     * (bearer == null) return;}, a no-op, not an unconditional clear. This
     * fresh {@code Player} never had a Ring-bearer to begin with, so it
     * reports only the activator for a reason this test can actually
     * demonstrate -- {@code getRingBearer()} was null before {@code resolve()}
     * ran and stays null after a no-op call. It does NOT demonstrate "no
     * creatures clears an existing Ring-bearer", which is false; see {@link
     * #ringTemptsNamesTheActivatorAndTheChosenRingBearer} for the shape a
     * fresh player cannot reach.
     */
    @Test
    void ringTemptsWithNoRingBearerNamesOnlyTheActivator() {
        SpellAbility sa = AbilityFactory.getAbility("DB$ RingTemptsYou", TestCards.build("Grizzly Bears"));
        Player activator = new Player("tempted-no-bearer", TestCards.game(), 93102);
        sa.setActivatingPlayer(activator);

        EffectEvent event = observe(sa).finish();

        assertEquals(EventType.RING_TEMPTS, event.type());
        assertEquals(List.of(Refs.of(activator)), event.subjects());
    }

    // ── removed_from_combat: RemoveFromCombat ───────────────────────────────

    /** A combat on {@code game} in which every one of {@code attackers} attacks {@code defender}. */
    private static Combat combatOf(Game game, int attackerControllerId, Player defender, Card... attackers) {
        Combat combat = new Combat(new Player("attacker", game, attackerControllerId));
        game.getPhaseHandler().setCombat(combat);
        for (Card attacker : attackers) {
            combat.addAttacker(attacker, defender);
        }
        return combat;
    }

    /** {@code script} built on {@code host}, with exactly {@code targets} chosen. */
    private static SpellAbility targeting(String script, Card host, Card... targets) {
        SpellAbility sa = AbilityFactory.getAbility(script, host);
        sa.resetTargets();
        for (Card target : targets) {
            sa.getTargets().add(target);
        }
        return sa;
    }

    @Test
    void endingTheCombatPhaseNamesTheAttackingPlayer() {
        Game game = TestBoard.anotherGame();
        Card attacker = TestBoard.cardIn(game, "Grizzly Bears");
        Combat combat = combatOf(game, 93205, new Player("defender", game, 93206), attacker);
        Player attacking = combat.getAttackingPlayer();
        SpellAbility sa = AbilityFactory.getAbility("SP$ EndCombatPhase", attacker);

        ApiEvents.ClauseObservation<?> observed = observe(sa);
        game.getPhaseHandler().setCombat(null);
        EffectEvent event = observed.finish();

        assertEquals(EventType.COMBAT_ENDED, event.type());
        assertEquals(List.of(Refs.of(attacking)), event.subjects());
    }

    @Test
    void endingTheCombatPhaseOutsideCombatEmitsNothing() {
        Game game = TestBoard.anotherGame();
        Card host = TestBoard.cardIn(game, "Grizzly Bears");
        game.getPhaseHandler().setCombat(null);
        SpellAbility sa = AbilityFactory.getAbility("SP$ EndCombatPhase", host);

        assertNull(observe(sa).finish());
    }

    /**
     * Two real attacking creatures, both removed -- not a single-entry
     * fixture where "first only" and "both" would agree.
     */
    @Test
    void removeFromCombatNamesEveryCreatureActuallyRemoved() {
        Game game = TestBoard.anotherGame();
        Card c1 = TestBoard.cardIn(game, "Grizzly Bears");
        Card c2 = TestBoard.cardIn(game, "Runeclaw Bear");
        Combat combat = combatOf(game, 93201, new Player("defender", game, 93202), c1, c2);
        SpellAbility sa = targeting("DB$ RemoveFromCombat | ValidTgts$ Creature", c1, c1, c2);

        ApiEvents.ClauseObservation<?> observed = observe(sa);
        combat.removeFromCombat(c1);
        combat.removeFromCombat(c2);
        EffectEvent event = observed.finish();

        assertEquals(EventType.REMOVED_FROM_COMBAT, event.type());
        assertEquals(
                List.of(Refs.of(c1), Refs.of(c2)),
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
        Game game = TestBoard.anotherGame();
        Card c1 = TestBoard.cardIn(game, "Grizzly Bears");
        Card c2 = TestBoard.cardIn(game, "Runeclaw Bear");
        Combat combat = combatOf(game, 93203, new Player("defender", game, 93204), c1, c2);
        SpellAbility sa = targeting("DB$ RemoveFromCombat | ValidTgts$ Creature", c1, c1, c2);

        ApiEvents.ClauseObservation<?> observed = observe(sa);
        combat.removeFromCombat(c1);
        // c2 stays attacking -- its own removal was skipped by a guard this
        // rule does not replicate.
        EffectEvent event = observed.finish();

        assertEquals(List.of(Refs.of(c1)), event.subjects());
    }

    // ── removed_from_combat: ChangeCombatants ───────────────────────────────

    /**
     * {@code portal_manipulator.txt} is {@code DB$ ChangeCombatants |
     * ValidTgts$ Creature.attacking | TargetMin$ 0 | TargetMax$ MaxTargets} --
     * any number of attackers, not just one, so two are reselected here: a
     * single-entry fixture would not tell "every attacker" from "the first".
     * The plausible wrong implementation this falsifies is a {@code
     * reselectedAttackers} that returns on the first matching entry instead of
     * continuing the loop.
     *
     * <p>{@code addToCombat}'s reselection branch calls {@code
     * combat.removeFromCombat(c)} immediately before re-adding the same
     * creature against the new defender --
     * simulated here with the same two public {@code Combat} calls, not a
     * hand-rolled approximation.
     *
     * <p>{@code ValidTgts$ Creature.attacking} on the ability text, and a
     * host distinct from the attackers, are load-bearing, not decoration:
     * {@code SpellAbility.usesTargeting()} is exactly {@code
     * targetRestrictions != null}, which {@code AbilityFactory} only sets when
     * the script declares {@code ValidTgts$}. Without it, {@code
     * affectedCards} falls through to its {@code Defined$} branch (default
     * {@code "Self"} -- the host), so a host that is itself an attacker would
     * make the host-fallback and the manually-added targets name the same card
     * and the test pass for a reason that had nothing to do with targeting.
     */
    @Test
    void changeCombatantsNamesEveryReselectedAttackerNotJustTheFirst() {
        Game game = TestBoard.anotherGame();
        Player oldDefender = new Player("old-defender", game, 93307);
        Player newDefender = new Player("new-defender", game, 93308);
        Card host = TestBoard.cardIn(game, "Mountain");
        Card c1 = TestBoard.cardIn(game, "Grizzly Bears");
        Card c2 = TestBoard.cardIn(game, "Runeclaw Bear");
        Combat combat = combatOf(game, 93306, oldDefender, c1, c2);
        SpellAbility sa = targeting(
                "DB$ ChangeCombatants | ValidTgts$ Creature.attacking | Attacking$ True", host, c1, c2);

        ApiEvents.ClauseObservation<?> observed = observe(sa);
        combat.removeFromCombat(c1);
        combat.addAttacker(c1, newDefender);
        combat.removeFromCombat(c2);
        combat.addAttacker(c2, newDefender);
        EffectEvent event = observed.finish();

        assertEquals(EventType.REMOVED_FROM_COMBAT, event.type());
        assertEquals(
                List.of(Refs.of(c1), Refs.of(c2)),
                event.subjects());
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
        Game game = TestBoard.anotherGame();
        Card host = TestBoard.cardIn(game, "Mountain");
        Card c = TestBoard.cardIn(game, "Grizzly Bears");
        combatOf(game, 93304, new Player("defender", game, 93305), c);
        SpellAbility sa = targeting(
                "DB$ ChangeCombatants | ValidTgts$ Creature.attacking | Attacking$ True", host, c);

        ApiEvents.ClauseObservation<?> observed = observe(sa);
        // Nothing changes: declined, or reselected the same defender.
        EffectEvent event = observed.finish();

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
        EffectEvent event = observe(sa).finish();

        assertEquals(EventType.INITIATIVE_TAKEN, event.type());
        assertEquals(List.of(Refs.of(target)), event.subjects());
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

        ApiEvents.ClauseObservation<?> observed = observe(sa);
        target.getGame().getAction().takeInitiative(target, "BRO");
        EffectEvent event = observed.finish();

        assertEquals(EventType.INITIATIVE_TAKEN, event.type());
        assertEquals(List.of(Refs.of(target)), event.subjects());
    }

    /** No one processed by this clause ever took the initiative here. */
    @Test
    void noInitiativeTakenNamesNoOne() {
        SpellAbility sa = AbilityFactory.getAbility(
                "SP$ TakeInitiative | ValidTgts$ Player", TestCards.build("Grizzly Bears"));
        Player target = new Player("never-took-it", TestCards.game(), 93403);
        sa.resetTargets();
        sa.getTargets().add(target);

        EffectEvent event = observe(sa).finish();

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

        EffectEvent event = observe(sa).finish();

        assertEquals(EventType.TEXT_CHANGE, event.type());
        assertEquals(
                List.of(Refs.of(c1), Refs.of(c2)),
                event.subjects());
    }

    /**
     * {@code artificial_evolution}/{@code sleight_of_mind}/{@code
     * magical_hack}-shaped: {@code ValidTgts$ Card | TgtZone$
     * Stack,Battlefield}, targeting a spell on the stack rather than a
     * permanent. {@code TargetChoices} stores that as the targeted {@code
     * SpellAbility} itself, not as a {@code Card} -- {@code getTargetCards()}
     * filters to {@code Card.class} instances and would see nothing -- so
     * {@code cardsFromTargets}'s {@code getTargetSpells()} union is the only
     * path that can name a subject here, so the union is not dead code. Built the way {@code
     * changeTargetsNamesWhatTheRetargetedSpellPointsAtNow} already builds a
     * "spell on the stack" target: a real, distinct {@code SpellAbility}
     * added to {@code sa}'s own targets via {@code TargetChoices.add(GameObject)}'s
     * {@code SpellAbility} overload, the same way real targeting leaves it.
     * The plausible wrong implementation this falsifies is {@link
     * #affectedCards} alone (no {@code getTargetSpells()} union at all),
     * which would report zero subjects here instead of one.
     */
    @Test
    void changeTextTargetingASpellOnTheStackNamesItsHostCard() {
        SpellAbility sa = AbilityFactory.getAbility(
                "SP$ ChangeText | ValidTgts$ Card | TgtZone$ Stack,Battlefield | "
                        + "ChangeColorWord$ Choose Choose",
                TestCards.build("Grizzly Bears"));
        SpellAbility spellOnStack = AbilityFactory.getAbility(
                "SP$ Pump | ValidTgts$ Creature | NumAtt$ +0 | NumDef$ +0",
                TestCards.build("Runeclaw Bear"));
        sa.resetTargets();
        sa.getTargets().add(spellOnStack);

        EffectEvent event = observe(sa).finish();

        assertEquals(EventType.TEXT_CHANGE, event.type());
        assertEquals(
                List.of(Refs.of(spellOnStack.getHostCard())),
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

        EffectEvent event = observe(sa).finish();

        assertNull(event, "no ChangeColorWord$/ChangeTypeWord$ means nothing textual changed");
    }
}
