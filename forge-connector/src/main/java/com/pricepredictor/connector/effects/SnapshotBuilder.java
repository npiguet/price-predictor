package com.pricepredictor.connector.effects;

import forge.card.ColorSet;
import forge.card.MagicColor;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CardCollectionView;
import forge.game.card.CounterEnumType;
import forge.game.card.CounterType;
import forge.game.keyword.KeywordInterface;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.StringJoiner;

/**
 * Builds a record's pre-event state snapshot.
 *
 * <p>Three rules shape every snapshot, and each exists because getting it wrong
 * is invisible until the model is trained:
 *
 * <ul>
 *   <li><b>Characteristics are computed, not printed.</b> The model has to
 *       predict what happens to a 4/4 that an anthem made a 5/5; recording the
 *       4/4 would ask it to predict from a board that does not exist.</li>
 *   <li><b>Perspective is not stored.</b> Controllers are absolute player ids;
 *       mine and opponent derive at training time from the record's actor. One
 *       snapshot then serves records with different actors.</li>
 *   <li><b>An absent inclusion tier means uncollected, not empty.</b> The
 *       collected tiers are written into every record, because an empty entity
 *       list and an uncollected tier are otherwise identical.</li>
 * </ul>
 *
 * <p>Stage one writes tiers 1 and 2: every referenced object in whatever zone it
 * sits, plus the global state, the battlefield, and command-zone effect cards.
 */
public final class SnapshotBuilder {

    /** Inclusion tiers, in the order the stages add them. */
    public static final int TIER_REFERENCED = 1;
    public static final int TIER_CORE = 2;
    public static final int TIER_UNREFERENCED_STACK = 3;
    public static final int TIER_UNREFERENCED_HAND_GRAVEYARD = 4;

    private static final int[] STAGE_ONE_TIERS = {TIER_REFERENCED, TIER_CORE};

    private static final char[] COLORS = {'W', 'U', 'B', 'R', 'G'};

    private final Game game;
    private final int[] tiers;

    public SnapshotBuilder(Game game) {
        this(game, STAGE_ONE_TIERS);
    }

    public SnapshotBuilder(Game game, int[] tiers) {
        this.game = game;
        this.tiers = tiers.clone();
    }

    /** The entity id an event's subject and the snapshot both use. */
    public static String entityId(Card card) {
        return "E" + card.getId();
    }

    public static String playerId(Player player) {
        return "P" + player.getId();
    }

    /**
     * Render the snapshot for one record.
     *
     * @param acting the ability being recorded, or null for a combat record
     * @param referenced entities the record points at, carried whatever zone
     *                   they sit in (tier 1)
     */
    public String toJson(SpellAbility acting, Iterable<Card> referenced) {
        Set<Card> entities = new LinkedHashSet<>();
        if (referenced != null) {
            for (Card card : referenced) {
                if (card != null) {
                    entities.add(card);
                }
            }
        }
        // Tier 2: the board and the command zone's effect cards.
        for (Card card : game.getCardsIn(EnumSet.of(
                ZoneType.Battlefield, ZoneType.Command))) {
            entities.add(card);
        }
        if (contains(tiers, TIER_UNREFERENCED_STACK)) {
            for (Card card : game.getCardsIn(ZoneType.Stack)) {
                entities.add(card);
            }
        }
        if (contains(tiers, TIER_UNREFERENCED_HAND_GRAVEYARD)) {
            for (Card card : game.getCardsIn(EnumSet.of(
                    ZoneType.Hand, ZoneType.Graveyard))) {
                entities.add(card);
            }
        }

        StringJoiner entityJson = new StringJoiner(",", "[", "]");
        for (Card card : entities) {
            entityJson.add(entityToJson(card));
        }
        StringJoiner playerJson = new StringJoiner(",", "[", "]");
        for (Player player : game.getPlayers()) {
            playerJson.add(playerToJson(player));
        }
        StringJoiner tierJson = new StringJoiner(",", "[", "]");
        for (int tier : tiers) {
            tierJson.add(String.valueOf(tier));
        }

        return "{\"global\":" + globalToJson()
                + ",\"players\":" + playerJson
                + ",\"entities\":" + entityJson
                + ",\"refs\":" + refsToJson(acting)
                + ",\"pending_event\":null"
                + ",\"tiers\":" + tierJson + "}";
    }

    private String globalToJson() {
        var phase = game.getPhaseHandler();
        Player active = phase.getPlayerTurn();
        Player priority = phase.getPriorityPlayer();
        return "{\"turn\":" + phase.getTurn()
                + ",\"phase\":" + Json.string(phaseName(phase.getPhase()))
                + ",\"active\":" + Json.string(active == null ? null : playerId(active))
                + ",\"priority\":"
                + Json.string(priority == null ? null : playerId(priority))
                + ",\"stack_size\":" + game.getStack().size()
                + ",\"combat_substep\":" + Json.string(combatSubstep())
                + ",\"emblems\":[]}";
    }

    private static String phaseName(Object phase) {
        return phase == null
                ? null
                : phase.toString().toLowerCase(java.util.Locale.ROOT);
    }

    /**
     * Which damage step is running, or null outside combat damage.
     *
     * <p>Named because a first-strike combat produces two damage steps and
     * therefore two combat records; without the substep they would be
     * indistinguishable.
     */
    private String combatSubstep() {
        var phase = game.getPhaseHandler();
        if (phase.getPhase() == null) {
            return null;
        }
        String name = phase.getPhase().toString();
        if (name.contains("FIRST_STRIKE")) {
            return "first_strike";
        }
        if (name.contains("COMBAT_DAMAGE")) {
            return "regular";
        }
        return null;
    }

    private String playerToJson(Player player) {
        StringJoiner floating = new StringJoiner(",", "{", "}");
        StringJoiner production = new StringJoiner(",", "{", "}");
        for (char color : COLORS) {
            byte shard = MagicColor.fromName(String.valueOf(color));
            floating.add(
                    Json.string(String.valueOf(color)) + ":"
                            + player.getManaPool().getAmountOfColor(shard));
            production.add(Json.string(String.valueOf(color)) + ":0");
        }
        return "{\"id\":" + Json.string(playerId(player))
                + ",\"life\":" + player.getLife()
                + ",\"hand\":" + player.getCardsIn(ZoneType.Hand).size()
                + ",\"library\":" + player.getCardsIn(ZoneType.Library).size()
                + ",\"graveyard\":" + player.getCardsIn(ZoneType.Graveyard).size()
                + ",\"poison\":" + player.getPoisonCounters()
                + ",\"energy\":" + player.getCounters(
                        CounterEnumType.ENERGY)
                + ",\"this_turn\":{\"creatures_died\":0,\"spells_cast\":"
                + player.getSpellsCastThisTurn()
                + ",\"lands_played\":" + player.getLandsPlayedThisTurn() + "}"
                + ",\"floating_mana\":" + floating
                + ",\"untapped_production\":" + production + "}";
    }

    private String entityToJson(Card card) {
        Card attached = card.getAttachedTo();
        return "{\"id\":" + Json.string(entityId(card))
                + ",\"name\":" + Json.string(card.getName())
                + ",\"zone\":" + Json.string(zoneName(card))
                + ",\"controller\":" + Json.string(
                        card.getController() == null
                                ? null : playerId(card.getController()))
                + ",\"face\":" + 0
                + ",\"copy_source\":null"
                + ",\"token_script_id\":"
                + Json.string(card.isToken() ? card.getName() : null)
                + ",\"types\":" + typeJson(card)
                + ",\"subtypes\":" + subtypeJson(card)
                + ",\"supertypes\":" + supertypeJson(card)
                + ",\"colors\":" + colorJson(card)
                + ",\"mana_value\":" + card.getCMC()
                + ",\"pt\":" + ptJson(card)
                + ",\"tapped\":" + card.isTapped()
                + ",\"sick\":" + card.isSick()
                + ",\"damage\":" + card.getDamage()
                + ",\"counters\":" + counterJson(card)
                + ",\"combat\":" + combatJson(card)
                + ",\"attached_to\":"
                + Json.string(attached == null ? null : entityId(attached))
                + ",\"face_down\":" + card.isFaceDown()
                + ",\"granted_attached\":" + keyJson(grantedAttached(card))
                + ",\"granted_temporary\":" + grantedTemporaryJson(card)
                + ",\"printed\":" + keyJson(printedKeys(card))
                + ",\"stack_extras\":null}";
    }

    private String zoneName(Card card) {
        var zone = card.getZone();
        return zone == null
                ? null
                : zone.getZoneType().name().toLowerCase(java.util.Locale.ROOT);
    }

    /**
     * Computed power and toughness, decomposed into the three sources.
     *
     * <p>Base plus boosts plus counters, kept apart because an ability that
     * adds a boost and one that adds a counter are different effects with the
     * same total, and the head has to predict which happened.
     */
    private String ptJson(Card card) {
        if (!card.isCreature()) {
            return "null";
        }
        int counterPower = card.getCounters(CounterEnumType.P1P1)
                - card.getCounters(CounterEnumType.M1M1);
        int basePower = card.getBasePower();
        int baseToughness = card.getBaseToughness();
        int boostPower = card.getNetPower() - basePower - counterPower;
        int boostToughness = card.getNetToughness() - baseToughness - counterPower;
        return "{\"base\":[" + basePower + "," + baseToughness + "]"
                + ",\"boosts\":[" + boostPower + "," + boostToughness + "]"
                + ",\"counters\":[" + counterPower + "," + counterPower + "]}";
    }

    private String counterJson(Card card) {
        StringJoiner joiner = new StringJoiner(",", "{", "}");
        for (com.google.common.collect.Multiset.Entry<CounterType> entry
                : card.getCounters().entrySet()) {
            joiner.add(
                    Json.string(entry.getElement().getName().toUpperCase(
                            java.util.Locale.ROOT))
                            + ":" + entry.getCount());
        }
        return joiner.toString();
    }

    private String combatJson(Card card) {
        var combat = game.getCombat();
        if (combat == null || !combat.isAttacking(card) && !combat.isBlocking(card)) {
            return "null";
        }
        StringJoiner blockedBy = new StringJoiner(",", "[", "]");
        if (combat.isAttacking(card)) {
            CardCollectionView blockers = combat.getBlockers(card);
            if (blockers != null) {
                for (Card blocker : blockers) {
                    blockedBy.add(Json.string(entityId(blocker)));
                }
            }
        }
        Object defender = combat.isAttacking(card)
                ? combat.getDefenderByAttacker(card) : null;
        return "{\"attacking\":"
                + Json.string(defender == null ? null : String.valueOf(defender))
                + ",\"blocking\":[]"
                + ",\"blocked_by\":" + blockedBy
                + ",\"became_blocked\":" + combat.isBlocked(card) + "}";
    }

    private String grantedTemporaryJson(Card card) {
        StringJoiner keywords = new StringJoiner(",", "[", "]");
        for (KeywordInterface keyword : card.getKeywords()) {
            if (!keyword.getOriginal().isEmpty()) {
                keywords.add(Json.string(
                        keyword.getOriginal().toLowerCase(java.util.Locale.ROOT)));
            }
        }
        return "{\"keywords\":" + keywords + ",\"abilities\":[]}";
    }

    /** Printed lines, as provenance keys into the card's own script. */
    private List<ProvenanceKey> printedKeys(Card card) {
        List<ProvenanceKey> keys = new ArrayList<>();
        for (SpellAbility sa : card.getSpellAbilities()) {
            ProvenanceKey key = ProvenanceKey.of(sa);
            if (key != null && !keys.contains(key)) {
                keys.add(key);
            }
        }
        return keys;
    }

    /**
     * Attachment-granted lines only.
     *
     * <p>Temporary grants ride the overlay instead, and the two channels stay
     * separate because the model reads them differently: an entity's ability
     * tokens are its printed and attachment-granted lines, while a grant until
     * end of turn is part of what happened to it rather than what it is.
     */
    private List<ProvenanceKey> grantedAttached(Card card) {
        List<ProvenanceKey> keys = new ArrayList<>();
        for (Card attachment : card.getAttachedCards()) {
            for (SpellAbility sa : attachment.getSpellAbilities()) {
                ProvenanceKey key = ProvenanceKey.of(sa);
                if (key != null && !keys.contains(key)) {
                    keys.add(key);
                }
            }
        }
        return keys;
    }

    private static String keyJson(List<ProvenanceKey> keys) {
        StringJoiner joiner = new StringJoiner(",", "[", "]");
        for (ProvenanceKey key : keys) {
            joiner.add(key.toJson());
        }
        return joiner.toString();
    }

    private String typeJson(Card card) {
        StringJoiner joiner = new StringJoiner(",", "[", "]");
        for (var type : card.getType().getCoreTypes()) {
            joiner.add(Json.string(typeName(type)));
        }
        return joiner.toString();
    }

    private String subtypeJson(Card card) {
        StringJoiner joiner = new StringJoiner(",", "[", "]");
        for (String subtype : card.getType().getSubtypes()) {
            joiner.add(Json.string(typeName(subtype)));
        }
        return joiner.toString();
    }

    private String supertypeJson(Card card) {
        StringJoiner joiner = new StringJoiner(",", "[", "]");
        for (var supertype : card.getType().getSupertypes()) {
            joiner.add(Json.string(typeName(supertype)));
        }
        return joiner.toString();
    }

    private String colorJson(Card card) {
        StringJoiner joiner = new StringJoiner(",", "[", "]");
        for (String color : colorLetters(card.getColor())) {
            joiner.add(Json.string(color));
        }
        return joiner.toString();
    }

    /**
     * One type-line name as the snapshot spells it.
     *
     * <p>Core types and supertypes arrive as enums and subtypes as strings, and
     * a continuous record's contributions mix all three into one list. Shared so
     * that a static's share of an entity's types is spelled the way the entity's
     * own types are — a model that saw {@code Creature} in one and
     * {@code creature} in the other would learn them as different types.
     */
    static String typeName(Object type) {
        String name = type instanceof Enum<?> value ? value.name() : String.valueOf(type);
        return name.toLowerCase(java.util.Locale.ROOT);
    }

    /**
     * A colour set as the snapshot's letters, in WUBRG order.
     *
     * <p>Shared with the continuous record's contributions for the same reason
     * {@link #typeName} is.
     */
    static List<String> colorLetters(ColorSet colors) {
        List<String> letters = new ArrayList<>();
        if (colors == null) {
            return letters;
        }
        for (char color : COLORS) {
            if (colors.hasAnyColor(MagicColor.fromName(String.valueOf(color)))) {
                letters.add(String.valueOf(color));
            }
        }
        return letters;
    }

    /** Targets, source and announced values for the acting ability. */
    private String refsToJson(SpellAbility acting) {
        StringJoiner targets = new StringJoiner(",", "[", "]");
        String source = null;
        Integer x = null;
        if (acting != null) {
            if (acting.getHostCard() != null) {
                source = entityId(acting.getHostCard());
            }
            if (acting.getTargets() != null) {
                for (var target : acting.getTargets()) {
                    if (target instanceof Card card) {
                        targets.add(Json.string(entityId(card)));
                    } else if (target instanceof Player player) {
                        targets.add(Json.string(playerId(player)));
                    }
                }
            }
            if (acting.hasParam("X") || acting.getXManaCostPaid() != null) {
                x = acting.getXManaCostPaid();
            }
        }
        return "{\"targets\":" + targets
                + ",\"source\":" + Json.string(source)
                + ",\"modes\":[]"
                + ",\"x\":" + (x == null ? "null" : x)
                + ",\"choices\":{}}";
    }

    private static boolean contains(int[] values, int value) {
        for (int candidate : values) {
            if (candidate == value) {
                return true;
            }
        }
        return false;
    }
}
