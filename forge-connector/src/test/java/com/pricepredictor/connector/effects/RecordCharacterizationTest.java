package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.card.MagicColor;
import forge.game.ability.AbilityFactory;
import forge.game.ability.AbilityKey;
import forge.game.ability.ApiType;
import forge.game.ability.SpellAbilityEffect;
import forge.game.ability.effects.CharmEffect;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.event.GameEventCardDamaged;
import forge.game.event.GameEventPlayerDamaged;
import forge.game.event.GameEventPlayerLivesChanged;
import forge.game.event.GameEventTurnPhase;
import forge.game.mana.Mana;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.replacement.ReplacementEffect;
import forge.game.replacement.ReplacementResult;
import forge.game.spellability.AbilitySub;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.lang.reflect.Field;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Every record kind's rendered line, built the way the collectors build it.
 *
 * <p>Each test drives the production collectors — the bracket on the game's
 * bus, the patched collectors' hook handlers called through a proxy as the
 * engine calls them, the fork collector's probe — over a fresh game, and
 * compares the shard's lines with an expected file. Only the envelope's
 * wall-clock timestamp is replaced.
 *
 * <p>Fork records (the damage-step probe and the interventional resolution)
 * sit in expected files of their own, so a change to what a fork writes
 * regenerates exactly those.
 */
@ExtendWith({ForgeExtension.class, Characterization.PinnedForgeIds.class})
class RecordCharacterizationTest {

    private static final String GAME_ID = "run.0-l1.0";

    /** Every sampled channel at full rate, no forks unless a test adds them. */
    private static final CollectionCaps CAPS =
            new CollectionCaps(1, 1.0, 0, 0, List.of(), 0.0, 0.0);

    @TempDir
    Path tempDir;

    @BeforeEach
    void patchedCheckout() {
        Assumptions.assumeTrue(AttributionMode.detect() == AttributionMode.PATCHED,
                "the expected records were captured against the patched ../forge");
    }

    // ── resolution ──────────────────────────────────────────────────────

    /**
     * Both moments, through the engine: a two-colour spell that kills its
     * target, a spell that fizzles, a tap ability, and an ability hosted on an
     * engine-built effect card, which keys to the line that made the effect.
     */
    @Test
    void resolutionRecords() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Player you = board.you;
        Player opponent = board.opponent;
        Card mountain = board.battlefield("Mountain", you);
        Card plains = board.battlefield("Plains", you);
        Card pyromancer = board.battlefield("Prodigal Pyromancer", you);
        pyromancer.setSickness(false);
        Card bears = board.battlefield("Grizzly Bears", opponent);
        Card giant = board.battlefield("Hill Giant", opponent);

        List<String> records = records(board, CAPS, (bracket, collectors) -> {
            board.game.subscribeToEvents(bracket);

            SpellAbility helix = board.spell(
                    board.game.getAction().moveToStack(board.card("Lightning Helix", you), null),
                    you);
            helix.resetTargets();
            helix.getTargets().add(bears);
            helix.getPayingMana().add(new Mana(MagicColor.RED, mountain, null, you));
            helix.getPayingMana().add(new Mana(MagicColor.WHITE, plains, null, you));
            resolveAll(board, helix);

            board.cast(board.card("Shock", you), you, giant);
            board.game.getAction().exile(giant, null, null);
            board.game.getStack().resolveStack();
            board.game.getAction().checkStateEffects(false);

            SpellAbility ping = board.ability(pyromancer, ApiType.DealDamage);
            ping.setActivatingPlayer(you);
            ping.resetTargets();
            ping.getTargets().add(opponent);
            resolveAll(board, ping);

            Card effect = SpellAbilityEffect.createEffect(
                    ping, you, "Characterization Effect", null);
            board.game.getAction().moveToCommand(effect, null);
            SpellAbility gain = AbilityFactory.getAbility(
                    "AB$ GainLife | Cost$ 0 | Defined$ You | LifeAmount$ 2", effect);
            gain.setActivatingPlayer(you);
            resolveAll(board, gain);

            nextStep(board);
        });

        Characterization.assertMatches(
                "record-resolution.jsonl", Characterization.recordLines(records));
    }

    /**
     * A modal spell through the engine with the clause hook installed: one
     * cost half through the root key and one effect half per chosen mode,
     * each through its {@code option} key.
     */
    @Test
    void aModalResolutionRecordsOneHalfPerMode() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Player you = board.you;
        board.battlefield("Llanowar Elves", board.opponent);
        board.battlefield("Hill Giant", board.opponent);
        SpellAbility charm = board.ability(
                board.game.getAction().moveToStack(board.card("Fiery Confluence", you), null),
                ApiType.Charm);
        charm.setActivatingPlayer(you);
        List<AbilitySub> choices = charm.getAdditionalAbilityList("Choices");
        List<AbilitySub> chosen = new ArrayList<>(
                List.of(choices.get(0), choices.get(1), choices.get(1)));
        charm.setChosenList(chosen);
        CharmEffect.chainAbilities(charm, chosen);

        List<String> records = records(board, CAPS, (bracket, collectors) -> {
            board.game.subscribeToEvents(bracket);
            Assumptions.assumeTrue(
                    collectors.install() > 0
                            && PatchHooks.CLAUSE_LISTENER.present(),
                    "../forge has no clause hook to split the modes");
            resolveAll(board, charm);
            nextStep(board);
        });

        Characterization.assertMatches(
                "record-resolution-modal.jsonl", Characterization.recordLines(records));
    }

    // ── rewrite ─────────────────────────────────────────────────────────

    /**
     * A substitution that ran another ability, and an in-place edit of the
     * damage amount, through the replacement hook's handler.
     */
    @Test
    void rewriteRecords() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Player you = board.you;
        Card rest = board.battlefield("Rest in Peace", you);
        Card armor = board.battlefield("Urza's Armor", you);
        Card bears = board.battlefield("Grizzly Bears", board.opponent);
        Card angel = board.battlefield("Serra Angel", board.opponent);

        List<String> records = records(board, CAPS, (bracket, collectors) -> {
            HookShapes.Replacement listener =
                    HookShapes.proxy(HookShapes.Replacement.class, collectors.rewriteHandler());

            ReplacementEffect exile = rest.getReplacementEffects().iterator().next();
            SpellAbility ran = exile.ensureAbility();
            ran.setReplacementEffect(exile);
            Map<AbilityKey, Object> dying = AbilityKey.newMap();
            dying.put(AbilityKey.Card, bears);
            dying.put(AbilityKey.Origin, ZoneType.Battlefield.name());
            dying.put(AbilityKey.Destination, ZoneType.Graveyard.name());
            listener.onReplacement(exile, dying, AbilityKey.newMap(dying),
                    ReplacementResult.Replaced, ran);

            ReplacementEffect prevent = armor.getReplacementEffects().iterator().next();
            Map<AbilityKey, Object> before = AbilityKey.newMap();
            before.put(AbilityKey.Affected, you);
            before.put(AbilityKey.DamageSource, angel);
            before.put(AbilityKey.DamageAmount, 4);
            before.put(AbilityKey.IsCombat, true);
            Map<AbilityKey, Object> after = AbilityKey.newMap(before);
            after.put(AbilityKey.DamageAmount, 3);
            listener.onReplacement(prevent, before, after, ReplacementResult.Updated, null);
        });

        Characterization.assertMatches(
                "record-rewrite.jsonl", Characterization.recordLines(records));
    }

    // ── trigger ─────────────────────────────────────────────────────────

    /**
     * A fired evaluation, the non-fired one it buys, and the Monarch's
     * engine-built trigger, whose line no script holds.
     */
    @Test
    void triggerRecords() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Player you = board.you;
        Card warden = board.battlefield("Soul Warden", you);
        Card bears = board.battlefield("Grizzly Bears", board.opponent);
        board.game.getAction().becomeMonarch(you, "M12");
        Trigger monarch = board.game.getCardsIn(ZoneType.Command).get(0)
                .getTriggers().iterator().next();

        List<String> records = records(board, CAPS, (bracket, collectors) -> {
            HookShapes.TriggerFire listener =
                    HookShapes.proxy(HookShapes.TriggerFire.class, collectors.triggerFireHandler());
            Trigger entered = warden.getTriggers().iterator().next();

            Map<AbilityKey, Object> bearsEntered = AbilityKey.newMap();
            bearsEntered.put(AbilityKey.Card, bears);
            bearsEntered.put(AbilityKey.Destination, ZoneType.Battlefield.name());
            listener.onConditionEvaluated(entered, bearsEntered, true);

            Map<AbilityKey, Object> itselfEntered = AbilityKey.newMap();
            itselfEntered.put(AbilityKey.Card, warden);
            itselfEntered.put(AbilityKey.Destination, ZoneType.Battlefield.name());
            listener.onConditionEvaluated(entered, itselfEntered, false);

            Map<AbilityKey, Object> endStep = AbilityKey.newMap();
            endStep.put(AbilityKey.Phase, PhaseType.END_OF_TURN.nameForScripts);
            endStep.put(AbilityKey.Player, you);
            listener.onConditionEvaluated(monarch, endStep, true);
        });

        Characterization.assertMatches(
                "record-trigger.jsonl", Characterization.recordLines(records));
    }

    // ── continuous ──────────────────────────────────────────────────────

    /** One record per static: a P/T boost, a granted keyword, a type and a colour change. */
    @Test
    void continuousRecords() throws Throwable {
        Assumptions.assumeTrue(PatchedCollectors.CAN_SUPPRESS,
                "../forge cannot remove one static's contributions");
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        board.battlefield("Glorious Anthem", board.you);
        board.battlefield("Levitation", board.you);
        board.battlefield("Grizzly Bears", board.you);
        board.battlefield("Mycosynth Lattice", board.opponent);
        board.battlefield("Serra Angel", board.opponent);
        board.game.getAction().checkStaticAbilities();

        List<String> records = records(board, CAPS,
                (bracket, collectors) -> collectors.collectContinuous());

        Characterization.assertMatches(
                "record-continuous.jsonl", Characterization.recordLines(records));
    }

    // ── playability ─────────────────────────────────────────────────────

    /**
     * The three subkinds: a {@code decision} with no {@code what_if}, and the
     * {@code attackers} and {@code blockers} legality records, which carry it.
     * The defending player is the random seat, so the blockers record is the
     * one stamped {@code random_seat}.
     */
    @Test
    void playabilityRecords() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Player you = board.you;
        Player opponent = board.opponent;
        board.battlefield("Mountain", you);
        Card bears = board.battlefield("Grizzly Bears", you);
        bears.setSickness(false);
        Card elves = board.battlefield("Llanowar Elves", you);
        elves.setSickness(false);
        board.battlefield("Pacifism", you).attachToEntity(elves, null);
        Card angel = board.battlefield("Serra Angel", opponent);
        Card giant = board.battlefield("Hill Giant", opponent);
        giant.setTapped(true);
        board.game.getAction().checkStaticAbilities();
        SpellAbility bolt = board.spell(board.hand("Lightning Bolt", you), you);

        List<String> records = records(board, CAPS, (bracket, collectors) -> {
            collectors.setRandomSeatPlayerId(SnapshotBuilder.playerId(opponent));
            HookShapes.proxy(HookShapes.Playability.class, collectors.playabilityHandler())
                    .onCandidate(bolt);

            HookShapes.CombatLegality listener = HookShapes.proxy(
                    HookShapes.CombatLegality.class, collectors.combatLegalityHandler());
            listener.onAttackersDeclarable(
                    you, opponent, List.of(bears, elves), List.of(bears));
            listener.onBlockersDeclarable(
                    opponent, bears, List.of(angel, giant), List.of(angel), 1);
        });

        Characterization.assertMatches(
                "record-playability.jsonl", Characterization.recordLines(records));
    }

    /**
     * A legality record answering an AI what-if query: {@code what_if} true,
     * right after {@code random_seat}.
     *
     * <p>No collector writes one any more: the engine's declarations are the
     * only legality source, and they are stamped false. The value stays part
     * of the record schema, and gen-2 shards carry it, so its rendering is
     * pinned through the record builder the legality path uses, over the
     * board's own snapshot.
     */
    @Test
    void aWhatIfLegalityRecord() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        board.battlefield("Grizzly Bears", board.you);
        board.battlefield("Serra Angel", board.opponent);

        String record = EffectRecord.playability(
                "run.0-l1.0", "run", RecordShardWriter.timestamp(), GAME_ID,
                PlayabilitySubkind.BLOCKERS, AttributionMode.detect())
                .actor(SnapshotBuilder.playerId(board.opponent))
                .whatIf(true)
                .state(new SnapshotBuilder(board.game).toJson(null, List.of()))
                .toJson();

        Characterization.assertMatches(
                "record-what-if.jsonl", Characterization.recordLines(List.of(record)));
    }

    // ── degraded ────────────────────────────────────────────────────────

    /**
     * A modal spell as a worker without the patch records it: {@code mode}
     * degraded, one cost half, and one effect half through the root line
     * holding every mode's bus events, since no clause hook splits them.
     *
     * <p>The collector reads its mode from {@link AttributionMode#detect},
     * which probes the sub-ability hook once per JVM; the probe's answer is
     * set to degraded while the collector is built, as on a stock checkout,
     * and put back after. The events are published outside any resolution,
     * where the sub-ability pointer is empty as it always is unpatched.
     */
    @Test
    void aDegradedWorkersModalResolution() throws Exception {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Player you = board.you;
        Player opponent = board.opponent;
        Card elves = board.battlefield("Llanowar Elves", opponent);
        Card confluence = board.game.getAction().moveToStack(
                board.card("Fiery Confluence", you), null);
        SpellAbility charm = board.ability(confluence, ApiType.Charm);
        charm.setActivatingPlayer(you);
        List<AbilitySub> choices = charm.getAdditionalAbilityList("Choices");
        List<AbilitySub> chosen = new ArrayList<>(List.of(choices.get(0), choices.get(1)));
        charm.setChosenList(chosen);
        CharmEffect.chainAbilities(charm, chosen);

        writer = new RecordShardWriter(tempDir, "run", 0, "l1");
        try {
            BusBracketCollector bracket = degraded(
                    () -> new BusBracketCollector(board.game, writer, GAME_ID));
            board.game.subscribeToEvents(bracket);
            bracket.beginBracket(charm);
            board.game.fireEvent(new GameEventCardDamaged(
                    elves.getView(), confluence.getView(), 1,
                    GameEventCardDamaged.DamageType.Normal));
            board.game.fireEvent(new GameEventPlayerDamaged(
                    opponent.getView(), confluence.getView(), 2, false, false));
            board.game.fireEvent(new GameEventPlayerLivesChanged(opponent, 20, 18));
            bracket.endBracket(charm.getId(), false);
            bracket.flushHeldResolution();
        } finally {
            writer.close();
        }

        Characterization.assertMatches("record-degraded.jsonl",
                Characterization.recordLines(ShardRun.readShard(writer.path())));
    }

    /** Build {@code collector} while the JVM's attribution probe answers degraded. */
    private static <T> T degraded(Callable<T> collector) throws Exception {
        Field detected = AttributionMode.class.getDeclaredField("detected");
        detected.setAccessible(true);
        Object probed = detected.get(null);
        detected.set(null, AttributionMode.DEGRADED);
        try {
            return collector.call();
        } finally {
            detected.set(null, probed);
        }
    }

    // ── combat ──────────────────────────────────────────────────────────

    /**
     * A trampler blocked by a smaller creature, the observed damage step's
     * record, and the probe branch that re-ran the step without trample and
     * mirrors it, wired as a worker wires them.
     */
    @Test
    void combatRecordAndItsDamageStepProbe() throws Throwable {
        Assumptions.assumeTrue(PatchHooks.ASSIGNED_DAMAGE.present(),
                "../forge does not keep the step's damage assignment");
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.COMBAT_DAMAGE);
        Player you = board.you;
        Player opponent = board.opponent;
        Card dreadmaw = board.battlefield("Colossal Dreadmaw", you);
        Card soldier = board.token("w_1_1_soldier", you);
        Card bears = board.battlefield("Grizzly Bears", opponent);
        Combat combat = new Combat(you);
        combat.addAttacker(dreadmaw, opponent);
        combat.addAttacker(soldier, opponent);
        combat.addBlocker(dreadmaw, bears);
        combat.setBlocked(dreadmaw, true);
        combat.setBlocked(soldier, false);
        combat.orderBlockersForDamageAssignment();
        combat.orderAttackersForDamageAssignment();
        board.game.getPhaseHandler().setCombat(combat);
        CollectionCaps caps = new CollectionCaps(1, 1.0, 0, 1, List.of("trample"), 0.0, 0.0);

        List<String> records = records(board, caps, (bracket, collectors) -> {
            board.game.subscribeToEvents(bracket);
            collectors.withForks(new ForkCollector(board.game, writer, GAME_ID, caps, 1L));
            bracket.onCombatRecord(collectors::writeHeldProbes);
            collectors.collectProbes();
            assertTrue(combat.assignCombatDamage(false), "the step assigned no damage");
            combat.dealAssignedDamage();
            board.game.getAction().checkStateEffects(false);
            board.game.fireEvent(new GameEventTurnPhase(you, PhaseType.COMBAT_END, ""));
        });

        List<String> observed = new ArrayList<>();
        List<String> probes = new ArrayList<>();
        for (String record : records) {
            (record.contains("\"fork\":true") ? probes : observed).add(record);
        }
        assertTrue(!probes.isEmpty(), "no probe branch was written: " + records);
        Characterization.assertMatches(
                "record-combat.jsonl", Characterization.recordLines(observed));
        Characterization.assertMatches(
                "record-combat-probe.jsonl", Characterization.recordLines(probes));
    }

    // ── an interventional fork ──────────────────────────────────────────

    /** A forced resolution of a spell nobody cast, in a copy of the game. */
    @Test
    void anInterventionalResolution() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Player you = board.you;
        board.battlefield("Mountain", you);
        board.battlefield("Grizzly Bears", board.opponent);
        SpellAbility pyroclasm = board.spell(board.hand("Pyroclasm", you), you);
        CollectionCaps caps = new CollectionCaps(1, 1.0, 1, 0, List.of(), 0.0, 0.0);

        List<String> records = records(board, caps, (bracket, collectors) -> {
            ForkCollector forks = new ForkCollector(board.game, writer, GAME_ID, caps, 1L);
            collectors.withForks(forks);
            assertTrue(forks.intervene(pyroclasm, 1), "the fork observed nothing");
        });

        Characterization.assertMatches(
                "record-intervention.jsonl", Characterization.recordLines(records));
    }

    // ── plumbing ────────────────────────────────────────────────────────

    /** What a test does with a game's two collectors. */
    @FunctionalInterface
    private interface Body {
        void run(BusBracketCollector bracket, PatchedCollectors collectors) throws Throwable;
    }

    private RecordShardWriter writer;

    /**
     * Run {@code body} with a game's bracket and patched collectors writing
     * one shard, wired to each other as a worker wires them, and return the
     * shard's lines once both are closed.
     */
    private List<String> records(CharacterizationBoard board, CollectionCaps caps, Body body)
            throws Throwable {
        writer = new RecordShardWriter(tempDir, "run", 0, "l1");
        try {
            BusBracketCollector bracket = new BusBracketCollector(board.game, writer, GAME_ID);
            try (PatchedCollectors collectors = new PatchedCollectors(
                    board.game, writer, GAME_ID, caps, 1L)) {
                collectors.withBracket(bracket);
                body.run(bracket, collectors);
            }
        } finally {
            writer.close();
        }
        return ShardRun.readShard(writer.path());
    }

    /** Cast or activate onto the stack, resolve it, and run the check after. */
    private static void resolveAll(CharacterizationBoard board, SpellAbility ability) {
        board.game.getStack().add(ability);
        board.game.getStack().resolveStack();
        board.game.getAction().checkStateEffects(false);
    }

    /** The next action, which closes the held resolution's window. */
    private static void nextStep(CharacterizationBoard board) {
        board.game.fireEvent(new GameEventTurnPhase(board.you, PhaseType.MAIN2, ""));
    }

}
