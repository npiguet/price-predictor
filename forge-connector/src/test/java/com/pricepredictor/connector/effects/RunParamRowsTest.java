package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.card.MagicColor;
import forge.game.CardTraitBase;
import forge.game.Game;
import forge.game.ability.AbilityFactory;
import forge.game.card.Card;
import forge.game.card.CardDamageTable;
import forge.game.card.CardZoneTable;
import forge.game.mana.Mana;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The rows of {@link RunParamRows} that read a batch, a table or nothing,
 * each driven through the run-parameter map the engine itself builds and
 * hands its trigger or replacement hook: the subjects and the parameters the
 * row declares, and no others (findings 9.1, 9.2).
 */
@ExtendWith(ForgeExtension.class)
class RunParamRowsTest {

    private static Game game() {
        return TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
    }

    private static Player you(Game game) {
        return game.getPlayers().get(0);
    }

    private static Player opponent(Game game) {
        return game.getPlayers().get(1);
    }

    // ── what the hooks were handed ──────────────────────────────────────

    /** Whether {@code trait} is one of {@code host}'s, of {@code mode}. */
    private static boolean isOf(Object trait, Card host, String mode) {
        return trait instanceof CardTraitBase on
                && on.getHostCard().getId() == host.getId()
                && mode.equals(RunParamNormalizer.modeOf(trait));
    }

    /** A hook and the callback of it that hands over a trait and its run-parameter map. */
    private record Hearing(PatchHooks.Hook hook, PatchHooks.Callback callback) {
        static final Hearing CONDITIONS =
                new Hearing(PatchHooks.TRIGGER_LISTENER, PatchHooks.Callback.ON_CONDITION_EVALUATED);
        static final Hearing REPLACEMENTS =
                new Hearing(PatchHooks.REPLACEMENT_LISTENER, PatchHooks.Callback.ON_REPLACEMENT);
    }

    /**
     * Every condition the engine evaluated for {@code host}'s triggers of
     * {@code mode} while {@code body} ran, read as the trigger channel reads
     * it: off the run-parameter map the hook was handed. The triggers of the
     * cards put on the board by hand are made active first, as the engine's
     * next state check would make them.
     */
    private static List<EffectEvent> evaluated(Card host, String mode, Runnable body) {
        host.getGame().getTriggerHandler().resetActiveTriggers();
        return heard(Hearing.CONDITIONS, host, mode, body);
    }

    /**
     * The event every replacement of {@code host}'s of {@code mode} received
     * while {@code body} ran, read as a rewrite record's {@code incoming} is.
     */
    private static List<EffectEvent> replaced(Card host, String mode, Runnable body) {
        return heard(Hearing.REPLACEMENTS, host, mode, body);
    }

    private static List<EffectEvent> heard(Hearing hearing, Card host, String mode, Runnable body) {
        List<EffectEvent> events = new ArrayList<>();
        PatchHooks.Listener listener = hearing.hook().listener((callback, args) -> {
            if (callback == hearing.callback() && isOf(args.get(0), host, mode)) {
                events.add(RunParamNormalizer.describeParams(args.get(0), args.get(1)));
            }
        });
        assertTrue(listener.install(), "the sibling Forge checkout has no " + hearing.hook());
        try {
            body.run();
        } finally {
            listener.uninstall();
        }
        assertFalse(events.isEmpty(), "the engine handed its hook no " + mode);
        return events;
    }

    private static Set<String> refsOf(Object... entities) {
        Set<String> refs = new LinkedHashSet<>();
        for (Object entity : entities) {
            refs.add(Refs.ofAny(entity).orElseThrow());
        }
        return refs;
    }

    // ── damage ──────────────────────────────────────────────────────────

    /** Grizzly Bears dealing 2 to the opponent and 3 to their creature, as {@code watcher} saw it. */
    private record BearsDamage(Card bears, Set<String> damaged, EffectEvent event) {

        static BearsDamage seenBy(String watcher, String mode) {
            Game game = game();
            Card host = TestBoard.inPlay(game, you(game), watcher);
            Card bears = TestBoard.inPlay(game, you(game), "Grizzly Bears");
            Card theirs = TestBoard.inPlay(game, opponent(game), "Grizzly Bears");
            EffectEvent event = evaluated(host, mode, () -> {
                CardDamageTable damage = new CardDamageTable();
                damage.put(bears, opponent(game), 2);
                damage.put(bears, theirs, 3);
                damage.triggerDamageDoneOnce(false, game);
            }).get(0);
            return new BearsDamage(bears, refsOf(opponent(game), theirs), event);
        }

        Map<String, Object> params() {
            return Map.of("mode", event.params().get("mode"),
                    "amount", 5, "combat", false, "source", Refs.of(bears));
        }
    }

    /** DamageDealtOnce: subjects the DamageMap's keys, amount its sum, source DamageSource. */
    @Test
    void damageDealtOnceReadsItsDamageMap() {
        BearsDamage damage = BearsDamage.seenBy("Armadillo Cloak", "DamageDealtOnce");

        assertEquals(EventType.DAMAGE_DEALT, damage.event().type(), damage.event().toJson());
        // A HashMap of target to amount: its order is its own.
        assertEquals(damage.damaged(), new LinkedHashSet<>(damage.event().subjects()),
                damage.event().toJson());
        assertEquals(damage.params(), damage.event().params(), damage.event().toJson());
    }

    /** DamageAll: subjects the table's damaged, amount its sum, source its one source. */
    @Test
    void damageAllReadsItsDamageTable() {
        BearsDamage damage = BearsDamage.seenBy("Contaminant Grafter", "DamageAll");

        assertEquals(List.copyOf(damage.damaged()), damage.event().subjects(), damage.event().toJson());
        assertEquals(damage.params(), damage.event().params(), damage.event().toJson());
    }

    /** DamageAll from two sources names none: no one source dealt it. */
    @Test
    void damageAllFromTwoSourcesNamesNoSource() {
        Game game = game();
        Card grafter = TestBoard.inPlay(game, you(game), "Contaminant Grafter");
        Card bears = TestBoard.inPlay(game, you(game), "Grizzly Bears");
        Card other = TestBoard.inPlay(game, you(game), "Grizzly Bears");

        EffectEvent event = evaluated(grafter, "DamageAll", () -> {
            CardDamageTable damage = new CardDamageTable();
            damage.put(bears, opponent(game), 2);
            damage.put(other, opponent(game), 2);
            damage.triggerDamageDoneOnce(true, game);
        }).get(0);

        assertEquals(List.of(Refs.of(opponent(game))), event.subjects(), event.toJson());
        assertEquals(Map.of("mode", "DamageAll", "amount", 4, "combat", true), event.params(),
                event.toJson());
    }

    // ── looking at cards, mana ──────────────────────────────────────────

    /** Surveil: Forge puts no count in the map, so the row reads none; its subject is the player. */
    @Test
    void surveilReadsItsPlayerAndNoCount() {
        Game game = game();
        Card spybug = TestBoard.inPlay(game, you(game), "Dimir Spybug");
        for (int i = 0; i < 3; i++) {
            game.getAction().moveToLibrary(TestBoard.card(game, you(game), "Forest"), null, null);
        }
        Card host = TestBoard.inPlay(game, you(game), "Grizzly Bears");
        SpellAbility surveil = AbilityFactory.getAbility("AB$ Surveil | Cost$ 0 | Amount$ 2", host);
        surveil.setActivatingPlayer(you(game));

        EffectEvent event = evaluated(spybug, "Surveil", surveil::resolve).get(0);

        assertEquals(EventType.CARD_LOOKED_AT, event.type(), event.toJson());
        assertEquals(List.of(Refs.of(you(game))), event.subjects(), event.toJson());
        assertEquals(Map.of("mode", "Surveil"), event.params(), event.toJson());
    }

    /** LoseMana: the map's Mana is the replacement's default answer, not the mana lost. */
    @Test
    void loseManaReadsNoMana() {
        Game game = game();
        Card stone = TestBoard.inPlay(game, you(game), "Horizon Stone");
        you(game).getManaPool().addMana(new Mana(MagicColor.GREEN, stone, null, you(game)));
        you(game).getManaPool().addMana(new Mana(MagicColor.GREEN, stone, null, you(game)));

        EffectEvent event = replaced(stone, "LoseMana",
                () -> you(game).getManaPool().clearPool(true)).get(0);

        assertEquals(EventType.MANA_LOST, event.type(), event.toJson());
        assertEquals(List.of(Refs.of(you(game))), event.subjects(), event.toJson());
        assertEquals(Map.of("mode", "LoseMana"), event.params(), event.toJson());
    }

    // ── tokens ──────────────────────────────────────────────────────────

    /**
     * CreateToken: the tokens the affected player's row makes, not the whole
     * table's. Forge runs the replacement once per player with one table, so
     * a token for each player, doubled for one of them, is 1 in each record.
     */
    @Test
    void aTokenReplacementCountsTheAffectedPlayersRow() {
        Game game = game();
        Card lives = TestBoard.inPlay(game, you(game), "Parallel Lives");
        Card host = TestBoard.inPlay(game, you(game), "Grizzly Bears");
        SpellAbility clues = AbilityFactory.getAbility(
                "AB$ Token | Cost$ 0 | TokenOwner$ Player | TokenScript$ c_a_clue_draw", host);
        clues.setActivatingPlayer(you(game));

        List<EffectEvent> events = replaced(lives, "CreateToken", clues::resolve);

        for (EffectEvent event : events) {
            assertEquals(EventType.TOKEN_CREATED, event.type(), event.toJson());
            assertEquals(1, event.subjects().size(), event.toJson());
            assertEquals(1, event.params().get("count"), event.toJson());
        }
    }

    /** TokenCreatedOnce: subjects the tokens made, count how many. */
    @Test
    void tokensCreatedAtOnceAreTheirSubjects() {
        Game game = game();
        Card akim = TestBoard.inPlay(game, you(game), "Akim, the Soaring Wind");
        Card host = TestBoard.inPlay(game, you(game), "Grizzly Bears");
        SpellAbility clues = AbilityFactory.getAbility(
                "AB$ Token | Cost$ 0 | TokenAmount$ 2 | TokenScript$ c_a_clue_draw", host);
        clues.setActivatingPlayer(you(game));

        EffectEvent event = evaluated(akim, "TokenCreatedOnce", clues::resolve).get(0);

        Object[] made = you(game).getCardsIn(ZoneType.Battlefield).stream()
                .filter(Card::isToken).toArray();
        assertEquals(List.copyOf(refsOf(made)), event.subjects(), event.toJson());
        assertEquals(2, event.params().get("count"), event.toJson());
    }

    // ── batches of moves ────────────────────────────────────────────────

    /**
     * ChangesZoneAll: the batch's cards in the order of their moves' zone
     * names, whatever order the moves were made in; two moves, so no one
     * move's zones.
     */
    @Test
    void aBatchOfMovesNamesItsCardsInZoneNameOrder() {
        Game game = game();
        Card finale = TestBoard.inPlay(game, you(game), "Dramatic Finale");
        Card fromHand = TestBoard.inHand(game, you(game), "Forest");
        Card fromPlay = TestBoard.inPlay(game, you(game), "Grizzly Bears");

        EffectEvent event = evaluated(finale, "ChangesZoneAll", () -> {
            CardZoneTable moves = new CardZoneTable();
            moves.put(ZoneType.Hand, ZoneType.Graveyard, fromHand);
            moves.put(ZoneType.Battlefield, ZoneType.Graveyard, fromPlay);
            moves.triggerChangesZoneAll(game, null);
        }).get(0);

        assertEquals(EventType.ZONE_CHANGE, event.type(), event.toJson());
        assertEquals(List.of(Refs.of(fromPlay), Refs.of(fromHand)), event.subjects(), event.toJson());
        assertEquals(Map.of("mode", "ChangesZoneAll"), event.params(), event.toJson());
    }
}
