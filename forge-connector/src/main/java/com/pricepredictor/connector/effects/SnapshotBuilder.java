package com.pricepredictor.connector.effects;

import com.google.common.collect.Table;
import forge.card.CardStateName;
import forge.card.CardTypeView;
import forge.card.ColorSet;
import forge.card.MagicColor;
import forge.game.Game;
import forge.game.ability.AbilityUtils;
import forge.game.card.Card;
import forge.game.GameEntity;
import forge.game.card.CardCollectionView;

import forge.game.card.CounterEnumType;
import forge.game.card.CounterType;
import forge.game.keyword.KeywordInterface;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.apache.commons.lang3.tuple.Pair;

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
 * <p>The tier vector is a property of the collection run, not of the caller.
 * Tiers 1 and 2 -- every referenced object in whatever zone it sits, plus the
 * global state, the battlefield and command-zone effect cards -- are what a
 * snapshot is; tiers 3 and 4 widen it to the unreferenced stack and to hand and
 * graveyard. Which of them a run collects is decided once and passed in, for the
 * reason the constructor gives.
 */
public final class SnapshotBuilder {

    /** Inclusion tiers, in the order the stages add them. */
    public static final int TIER_REFERENCED = 1;
    public static final int TIER_CORE = 2;
    public static final int TIER_UNREFERENCED_STACK = 3;
    public static final int TIER_UNREFERENCED_HAND_GRAVEYARD = 4;

    private static final char[] COLORS = {'W', 'U', 'B', 'R', 'G'};

    /**
     * The key the refs block opens with, and the key that follows it.
     *
     * <p>Published because a collector that re-reads refs at a later moment than
     * the rest of the snapshot splices the block back in by these two markers.
     * They are constants here, and the state is assembled from them in
     * {@link #stateJson}, so a key reorder that compiles cannot silently break
     * the splice.
     */
    public static final String REFS_KEY = ",\"refs\":";
    public static final String AFTER_REFS_KEY = ",\"pending_event\":";

    /**
     * A rendered state with its refs block replaced by a later reading.
     *
     * <p>The board must be captured before the ability resolves, but the modes
     * a charm chose, the X it announced and the card, colour or type its
     * controller named are set by the engine <b>during</b> resolution — so a
     * snapshot taken at the cast carries an empty refs block for exactly the
     * abilities whose choice is the interesting part of the effect. Re-rendering
     * the whole snapshot at resolution is not the alternative: that would
     * describe the board the ability had already changed, which is the one thing
     * a pre-event snapshot must not do.
     *
     * <p>Lives here rather than in a collector because the markers it splices
     * between are this class's, and a second collector doing the same string
     * edit by hand is how two readings drift. A state missing either marker is
     * returned unchanged: the splice widens a record, and a caller left with
     * cast-time refs is better off than one handed a corrupt state.
     */
    public static String spliceRefs(String state, String refs) {
        if (state == null || refs == null) {
            return state;
        }
        int open = state.indexOf(REFS_KEY);
        if (open < 0) {
            return state;
        }
        int close = state.indexOf(AFTER_REFS_KEY, open + REFS_KEY.length());
        if (close < 0) {
            return state;
        }
        return state.substring(0, open) + REFS_KEY + refs + state.substring(close);
    }

    private final Game game;
    private final int[] tiers;

    /**
     * @param tiers the inclusion tiers this <b>run</b> collects
     *
     * <p>There is deliberately no constructor that picks a default depth. The
     * tier vector is a run-level property -- the feature spec adds tier 3 and
     * then tier 4 by stage, not by call site -- and a convenience constructor is
     * how that turned into a per-call-site choice: interventional forks asked for
     * four tiers while every other collector asked for two or three, so
     * {@code state.tiers} containing 4 identified {@code interventional} exactly,
     * and a flag the schema bars from ever being a model input had a perfect
     * proxy sitting in the state. Making every construction state its depth is
     * what stops the next call site from reintroducing it.
     */
    public SnapshotBuilder(Game game, int[] tiers) {
        this.game = game;
        this.tiers = tiers.clone();
        requirePrefix(this.tiers);
    }

    /**
     * Reject a tier vector that is not a prefix of the four tiers.
     *
     * <p>Tiers are cumulative -- 3 widens 2, 4 widens 3 -- so a set with a hole
     * in it describes no collection anyone could run, and the renderer would
     * still write it into {@code state.tiers} as though it had. Tier 2 is
     * required as well as merely declared: the battlefield and the command zone
     * go into every snapshot unconditionally, so a vector omitting 2 would tell
     * a reader the board was uncollected while the board sits in
     * {@code entities}.
     */
    private static void requirePrefix(int[] tiers) {
        boolean prefix = tiers.length >= 2 && tiers.length <= 4;
        for (int i = 0; prefix && i < tiers.length; i++) {
            prefix = tiers[i] == i + 1;
        }
        if (!prefix) {
            throw new IllegalArgumentException(
                    "snapshot tiers must be a prefix of [1,2,3,4] holding at"
                            + " least 1 and 2, got "
                            + java.util.Arrays.toString(tiers));
        }
    }

    /**
     * The depth this builder renders at.
     *
     * <p>Readable so a collector's wiring can be checked where the vector lands
     * rather than where it was read: the tier-4 leak was a value that looked
     * right at every call site and was overridden inside the constructor.
     */
    int[] tiers() {
        return tiers.clone();
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
        return toJson(acting, referenced, null, null, null);
    }

    /**
     * Render the snapshot for one record, minus one static's contributions.
     *
     * @param withoutStatic the id of a static ability whose changes are removed
     *                      from every entity, or null to show the board as it
     *                      stands
     */
    public String toJson(
            SpellAbility acting, Iterable<Card> referenced, Long withoutStatic) {
        return toJson(acting, referenced, withoutStatic, null, null);
    }

    /**
     * Render the snapshot, naming the event a record is about to handle.
     *
     * @param pending the incoming event on a {@code rewrite} or {@code trigger}
     *                record. The event itself lives in the payload; the snapshot
     *                carries it so the entities it is about to affect are
     *                nameable from the board the model is shown
     */
    public String toJson(
            SpellAbility acting, Iterable<Card> referenced, Long withoutStatic,
            EffectEvent pending) {
        return toJson(acting, referenced, withoutStatic, pending, null);
    }

    /**
     * Render the snapshot for a record whose acting line is not a spell ability.
     *
     * <p>A trigger and a replacement effect are traits, not {@link SpellAbility}
     * instances, so neither can be the {@code acting} argument -- but both have a
     * host card, and that is what {@code refs.source} names: the join that lets
     * the record say which permanent's text acted. Neither has targets or
     * announced values at the moment it is recorded, so the rest of the refs
     * block is legitimately empty.
     *
     * <p>Deliberately not an overload of {@code toJson}: a {@code null} literal
     * at one of the existing call sites would become ambiguous between a
     * {@link SpellAbility} and a {@link Card}.
     */
    public String toJsonForTrait(
            Card sourceCard, Iterable<Card> referenced, EffectEvent pending) {
        return toJson(null, referenced, null, pending, sourceCard);
    }

    private String toJson(
            SpellAbility acting, Iterable<Card> referenced, Long withoutStatic,
            EffectEvent pending, Card sourceOverride) {
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
            entityJson.add(entityToJson(card, Suppressed.of(card, withoutStatic)));
        }
        StringJoiner playerJson = new StringJoiner(",", "[", "]");
        for (Player player : game.getPlayers()) {
            playerJson.add(playerToJson(player));
        }
        StringJoiner tierJson = new StringJoiner(",", "[", "]");
        for (int tier : tiers) {
            tierJson.add(String.valueOf(tier));
        }

        return stateJson(
                globalToJson(), playerJson.toString(), entityJson.toString(),
                refsJson(acting, sourceOverride),
                pending == null ? "null" : pending.toJson(),
                tierJson.toString());
    }

    /**
     * The snapshot's blocks, in the order the record schema fixes.
     *
     * <p>Assembled here rather than inline so that the one ordering another
     * collector depends on -- {@link #REFS_KEY} immediately before
     * {@link #AFTER_REFS_KEY}, with nothing between them but the refs block --
     * is stated once and can be asserted without a running game. A splice that
     * finds those two markers missing falls back to the unspliced state, which
     * is a silent degradation, so the adjacency is the part worth pinning.
     */
    static String stateJson(
            String global, String players, String entities, String refs,
            String pendingEvent, String tiers) {
        return "{\"global\":" + global
                + ",\"players\":" + players
                + ",\"entities\":" + entities
                + REFS_KEY + refs
                + AFTER_REFS_KEY + pendingEvent
                + ",\"tiers\":" + tiers + "}";
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
                + ",\"emblems\":" + keyJson(emblems()) + "}";
    }

    /**
     * Command-zone emblems, as the key of the line that made each.
     *
     * <p>An emblem is a continuous effect with no permanent to hang on, so it
     * appears nowhere else in the snapshot; without this the board says nothing
     * about an effect that changes every turn of the rest of the game.
     */
    private List<ProvenanceKey> emblems() {
        List<ProvenanceKey> keys = new ArrayList<>();
        for (Card card : game.getCardsIn(ZoneType.Command)) {
            if (!card.isEmblem()) {
                continue;
            }
            for (SpellAbility sa : card.getSpellAbilities()) {
                ProvenanceKey key = ProvenanceKey.of(sa);
                if (key != null && !keys.contains(key)) {
                    keys.add(key);
                }
            }
        }
        return keys;
    }

    /**
     * How much of each colour this player could still make, per colour.
     *
     * <p>Counted over untapped permanents' mana abilities, one per ability that
     * can produce the colour. It is what separates a board that can answer a
     * threat from one that only looks like it can, and the field had been
     * written as zero for every colour since the snapshot was first built.
     *
     * <p>An ability that makes one mana of any colour counts once for each, so
     * the numbers are an upper bound rather than a sum that can be spent.
     */
    private Map<Character, Integer> untappedProduction(Player player) {
        Map<Character, Integer> production = new java.util.HashMap<>();
        for (Card card : player.getCardsIn(ZoneType.Battlefield)) {
            if (card.isTapped()) {
                continue;
            }
            for (SpellAbility ability : card.getManaAbilities()) {
                var part = ability.getManaPart();
                if (part == null) {
                    continue;
                }
                for (char color : COLORS) {
                    if (part.canProduce(String.valueOf(color), ability)) {
                        production.merge(color, 1, Integer::sum);
                    }
                }
            }
        }
        return production;
    }

    /**
     * Creatures this player lost from the battlefield this turn.
     *
     * <p>The engine keeps what left the battlefield, not what died, so a
     * bounced or exiled creature counts here too. It is a close reading of a
     * condition many triggers ask about and the field had been a constant zero;
     * an exact count would need a hook where the death is decided.
     */
    private int creaturesLost(Player player) {
        int count = 0;
        for (Card card : game.getLeftBattlefieldThisTurn()) {
            if (card.isCreature() && card.getController() == player) {
                count++;
            }
        }
        return count;
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
        Map<Character, Integer> untapped = untappedProduction(player);
        for (char color : COLORS) {
            byte shard = MagicColor.fromName(String.valueOf(color));
            floating.add(
                    Json.string(String.valueOf(color)) + ":"
                            + player.getManaPool().getAmountOfColor(shard));
            production.add(Json.string(String.valueOf(color)) + ":"
                    + untapped.getOrDefault(color, 0));
        }
        return "{\"id\":" + Json.string(playerId(player))
                + ",\"life\":" + player.getLife()
                + ",\"hand\":" + player.getCardsIn(ZoneType.Hand).size()
                + ",\"library\":" + player.getCardsIn(ZoneType.Library).size()
                + ",\"graveyard\":" + player.getCardsIn(ZoneType.Graveyard).size()
                + ",\"poison\":" + player.getPoisonCounters()
                + ",\"energy\":" + player.getCounters(
                        CounterEnumType.ENERGY)
                + ",\"this_turn\":{\"creatures_died\":"
                + creaturesLost(player)
                + ",\"spells_cast\":"
                + player.getSpellsCastThisTurn()
                + ",\"lands_played\":" + player.getLandsPlayedThisTurn() + "}"
                + ",\"floating_mana\":" + floating
                + ",\"untapped_production\":" + production + "}";
    }

    /**
     * One entity's characteristics with an acting static's changes removed.
     *
     * <p>A {@code continuous} record's label is what its static does to the
     * board, so a snapshot that still contains the static's own work hands the
     * model the answer with the question. Every channel that static wrote has
     * to come back out.
     *
     * <p>Recombined by the engine rather than subtracted here, because
     * subtraction gets two cases wrong: another static granting the same
     * keyword or type would go with it, and a static that overwrites a type
     * line or strips a whole class of type cannot be inverted from what it
     * contributed. Power and toughness are the exception — boosts are summed
     * per static, so dropping one term is already exact.
     */
    private record Suppressed(
            CardTypeView type, ColorSet colors,
            Iterable<KeywordInterface> keywords, int power, int toughness) {

        /**
         * What the card would be without this static, or null for no change.
         *
         * <p>Null when the patch is absent, which is also when a continuous
         * record is never written — {@link PatchedCollectors#collectContinuous}
         * checks for these three before it collects, so a leaking snapshot is
         * not reachable by falling back to here.
         */
        static Suppressed of(Card card, Long staticId) {
            if (staticId == null) {
                return null;
            }
            if (!(PatchHooks.read(card, "getTypeWithout", staticId)
                            instanceof CardTypeView type)
                    || !(PatchHooks.read(card, "getColorWithout", staticId)
                            instanceof ColorSet colors)
                    || !(PatchHooks.read(card, "getKeywordsWithout", staticId)
                            instanceof Iterable<?> keywords)) {
                return null;
            }
            int power = 0;
            int toughness = 0;
            for (Table.Cell<Long, Long, Pair<Integer, Integer>> cell
                    : card.getPTBoostTable().cellSet()) {
                if (staticId.equals(cell.getColumnKey())) {
                    power += cell.getValue().getLeft();
                    toughness += cell.getValue().getRight();
                }
            }
            @SuppressWarnings("unchecked")
            Iterable<KeywordInterface> typed = (Iterable<KeywordInterface>) keywords;
            return new Suppressed(type, colors, typed, power, toughness);
        }
    }

    private String entityToJson(Card card, Suppressed without) {
        Card attached = card.getAttachedTo();
        return "{\"id\":" + Json.string(entityId(card))
                + ",\"name\":" + Json.string(card.getName())
                + ",\"zone\":" + Json.string(zoneName(card))
                + ",\"controller\":" + Json.string(
                        card.getController() == null
                                ? null : playerId(card.getController()))
                + ",\"face\":" + faceOf(card)
                + ",\"copy_source\":" + Json.string(copySourceOf(card))
                + ",\"token_script_id\":"
                + Json.string(tokenScriptIdOf(card))
                + ",\"types\":" + typeJson(card, without)
                + ",\"subtypes\":" + subtypeJson(card, without)
                + ",\"supertypes\":" + supertypeJson(card, without)
                + ",\"colors\":" + colorJson(card, without)
                + ",\"mana_value\":" + card.getCMC()
                + ",\"pt\":" + ptJson(card, without)
                + ",\"tapped\":" + card.isTapped()
                + ",\"sick\":" + card.isSick()
                + ",\"damage\":" + card.getDamage()
                + ",\"counters\":" + counterJson(card)
                + ",\"combat\":" + combatJson(card)
                + ",\"attached_to\":"
                + Json.string(attached == null ? null : entityId(attached))
                + ",\"face_down\":" + card.isFaceDown()
                + ",\"granted_attached\":" + keyJson(grantedAttached(card))
                + ",\"granted_temporary\":" + grantedTemporaryJson(card, without)
                + ",\"printed\":" + keyJson(printedKeys(card))
                + ",\"stack_extras\":" + stackExtrasJson(card) + "}";
    }

    /**
     * The token's script stem — what FR-078 means by a token-script id — or
     * the printed name when no script can be named (an engine-built token with
     * no image key), so a token is never reported as a non-token.
     */
    private static String tokenScriptIdOf(Card card) {
        if (!card.isToken()) {
            return null;
        }
        String stem = ProvenanceKey.tokenScriptStemOf(card);
        return stem != null ? stem : card.getName();
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
    private String ptJson(Card card, Suppressed without) {
        if (!card.isCreature()) {
            return "null";
        }
        int counterPower = card.getCounters(CounterEnumType.P1P1)
                - card.getCounters(CounterEnumType.M1M1);
        int basePower = card.getBasePower();
        int baseToughness = card.getBaseToughness();
        int boostPower = card.getNetPower() - basePower - counterPower;
        int boostToughness = card.getNetToughness() - baseToughness - counterPower;
        if (without != null) {
            boostPower -= without.power();
            boostToughness -= without.toughness();
        }
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
        StringJoiner blocking = new StringJoiner(",", "[", "]");
        if (combat.isBlocking(card)) {
            // The other half of blocked_by. Without it a blocker's own entry
            // says only that it is somewhere in a combat, so the flag the head
            // reads for "is this creature blocking" was always false — and the
            // blocking relationship is what trample and deathtouch act through.
            for (Card attacker : combat.getAttackersBlockedBy(card)) {
                blocking.add(Json.string(entityId(attacker)));
            }
        }
        Object defender = combat.isAttacking(card)
                ? combat.getDefenderByAttacker(card) : null;
        return "{\"attacking\":"
                + Json.string(defender == null ? null : String.valueOf(defender))
                + ",\"blocking\":" + blocking
                + ",\"blocked_by\":" + blockedBy
                + ",\"became_blocked\":" + combat.isBlocked(card) + "}";
    }

    private String grantedTemporaryJson(Card card, Suppressed without) {
        StringJoiner keywords = new StringJoiner(",", "[", "]");
        for (KeywordInterface keyword
                : without == null ? card.getKeywords() : without.keywords()) {
            if (!keyword.getOriginal().isEmpty()) {
                keywords.add(Json.string(
                        keyword.getOriginal().toLowerCase(java.util.Locale.ROOT)));
            }
        }
        return "{\"keywords\":" + keywords
                + ",\"abilities\":" + keyJson(grantedTemporaryAbilities(card)) + "}";
    }

    /**
     * Lines granted by a timestamped change rather than printed or attached.
     *
     * <p>"Gains flying and 'T: draw a card' until end of turn" puts the second
     * half here. Attachment grants ride {@code granted_attached} instead, and
     * the two stay apart because the model reads them differently: an entity's
     * ability tokens are its printed and attached lines, while a temporary
     * grant is part of what happened to it.
     */
    private List<ProvenanceKey> grantedTemporaryAbilities(Card card) {
        List<ProvenanceKey> keys = new ArrayList<>();
        Object changes = PatchHooks.read(card, "getChangedCardTraits");
        if (!(changes instanceof com.google.common.collect.Table<?, ?, ?> table)) {
            return keys;
        }
        for (Object value : table.values()) {
            Object abilities = PatchHooks.read(value, "getAbilities");
            if (!(abilities instanceof Iterable<?> granted)) {
                continue;
            }
            for (Object ability : granted) {
                if (ability instanceof SpellAbility sa) {
                    ProvenanceKey key = ProvenanceKey.of(sa);
                    if (key != null && !keys.contains(key)) {
                        keys.add(key);
                    }
                }
            }
        }
        return keys;
    }

    /** Which face is up, as an index; 0 is the printed front. */
    private static int faceOf(Card card) {
        var state = card.getCurrentStateName();
        return state == null || state == CardStateName.Original ? 0 : 1;
    }

    /** What this permanent is copying, or null when it is itself. */
    private static String copySourceOf(Card card) {
        Card copied = card.getCopiedPermanent();
        if (copied != null) {
            return entityId(copied);
        }
        Card cloner = card.getCloner();
        return cloner == null ? null : entityId(cloner);
    }

    /**
     * What a stack object announced, or null when the entity is not on the stack.
     *
     * <p>Divided damage and "up to N" counts are choices the caster made that no
     * other field records: a Fireball for 3 split two ways and one aimed at a
     * single creature have the same cost and the same text.
     */
    private String stackExtrasJson(Card card) {
        SpellAbility onStack = null;
        for (var instance : game.getStack()) {
            SpellAbility candidate = instance.getSpellAbility();
            if (candidate != null && candidate.getHostCard() == card) {
                onStack = candidate;
                break;
            }
        }
        if (onStack == null || onStack.getTargets() == null) {
            return "null";
        }
        StringJoiner targets = new StringJoiner(",", "[", "]");
        StringJoiner amounts = new StringJoiner(",", "{", "}");
        int chosen = 0;
        for (GameEntity target : onStack.getTargets().getTargetEntities()) {
            chosen++;
            String id = target instanceof Card targeted
                    ? entityId(targeted)
                    : target instanceof Player player ? playerId(player) : null;
            if (id == null) {
                continue;
            }
            targets.add(Json.string(id));
            Integer divided = onStack.getDividedValue(target);
            if (divided != null) {
                amounts.add(Json.string(id) + ":" + divided);
            }
        }
        StringJoiner upTo = new StringJoiner(",", "{", "}");
        if (onStack.getTargetRestrictions() != null) {
            upTo.add("\"chosen\":" + chosen);
        }
        return "{\"targets\":" + targets
                + ",\"per_target_amounts\":" + amounts
                + ",\"up_to_counts\":" + upTo + "}";
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

    private static CardTypeView typeOf(Card card, Suppressed without) {
        return without == null ? card.getType() : without.type();
    }

    private String typeJson(Card card, Suppressed without) {
        StringJoiner joiner = new StringJoiner(",", "[", "]");
        for (var type : typeOf(card, without).getCoreTypes()) {
            joiner.add(Json.string(typeName(type)));
        }
        return joiner.toString();
    }

    private String subtypeJson(Card card, Suppressed without) {
        StringJoiner joiner = new StringJoiner(",", "[", "]");
        for (String subtype : typeOf(card, without).getSubtypes()) {
            joiner.add(Json.string(typeName(subtype)));
        }
        return joiner.toString();
    }

    private String supertypeJson(Card card, Suppressed without) {
        StringJoiner joiner = new StringJoiner(",", "[", "]");
        for (var supertype : typeOf(card, without).getSupertypes()) {
            joiner.add(Json.string(typeName(supertype)));
        }
        return joiner.toString();
    }

    private String colorJson(Card card, Suppressed without) {
        StringJoiner joiner = new StringJoiner(",", "[", "]");
        for (String color
                : colorLetters(without == null ? card.getColor() : without.colors())) {
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

    /**
     * Targets, source and announced values for the acting ability.
     *
     * <p>Public because the refs block is the one part of a snapshot that is not
     * read at the same moment as the rest. The board has to be captured before
     * the ability resolves, but a resolution-time choice -- a named card, a
     * chosen colour -- does not exist until the engine asks for it, so a
     * collector re-renders this block alone at resolution and splices it back
     * into the state it already holds.
     */
    public String refsJson(SpellAbility acting) {
        return refsJson(acting, null);
    }

    /**
     * @param sourceOverride the host card of an acting line that is not a spell
     *                       ability; used only when {@code acting} is null, so
     *                       an acting ability still names its own host
     */
    private String refsJson(SpellAbility acting, Card sourceOverride) {
        StringJoiner targets = new StringJoiner(",", "[", "]");
        String source = sourceOverride == null ? null : entityId(sourceOverride);
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
            x = acting.getXManaCostPaid();
            if (x == null) {
                x = announcedX(acting);
            }
        }
        return "{\"targets\":" + targets
                + ",\"source\":" + Json.string(source)
                + ",\"modes\":" + modesJson(acting)
                + ",\"x\":" + (x == null ? "null" : x)
                + ",\"choices\":" + choicesJson(acting) + "}";
    }

    /**
     * X when it was not paid into the mana cost.
     *
     * <p>{@code getXManaCostPaid()} answers only for an X the caster paid as
     * mana, and that is the minority form. The common one is
     * {@code SVar:X:Count$...} with the parameter key {@code NumDmg} or
     * {@code Amount} holding the literal {@code X}: the ability has no {@code X}
     * parameter at all and the announced value lives only in the SVar. So an X
     * read from the mana payment alone is absent on exactly the cards whose X is
     * the interesting part of the effect.
     *
     * <p>Computed rather than read, because a {@code Count$} SVar is an
     * expression over the board and only the engine knows what it comes to. One
     * that cannot be answered from this record's moment is recorded as absent
     * rather than as zero -- a zero would say the caster announced nothing,
     * which is a different claim.
     */
    private static Integer announcedX(SpellAbility acting) {
        Card host = acting.getHostCard();
        if (host == null) {
            return null;
        }
        String amount = acting.getSVar("X");
        if (amount == null || amount.isEmpty()) {
            return null;
        }
        try {
            return AbilityUtils.calculateAmount(host, amount, acting);
        } catch (RuntimeException e) {
            return null;
        }
    }

    /**
     * The modes chosen on a modal spell.
     *
     * <p>A charm's three modes are one card and three effects, and which was
     * chosen is nowhere else in the record: the ability key is the same line
     * either way.
     */
    private String modesJson(SpellAbility acting) {
        StringJoiner modes = new StringJoiner(",", "[", "]");
        if (acting != null && acting.getChosenList() != null) {
            // Forge keeps the chosen modes as the sub-abilities they select,
            // so the mode is named by the clause it turns on.
            for (var mode : acting.getChosenList()) {
                modes.add(Json.string(
                        mode.getApi() == null ? "?" : mode.getApi().name()));
            }
        }
        return modes.toString();
    }

    /**
     * Resolution-time choices the engine asked the controller to make.
     *
     * <p>A named card, a chosen colour, a chosen type or number — each turns
     * one line into a different effect, and the head reads them from
     * {@code [ACT]} alongside the announced values.
     */
    private String choicesJson(SpellAbility acting) {
        StringJoiner choices = new StringJoiner(",", "{", "}");
        if (acting == null || acting.getHostCard() == null) {
            return choices.toString();
        }
        Card host = acting.getHostCard();
        addChoice(choices, "named_card", host.getNamedCard());
        addChoice(choices, "chosen_color", host.getChosenColors() == null
                ? null : String.join("|", host.getChosenColors()));
        addChoice(choices, "chosen_type", host.getChosenType());
        if (host.hasChosenNumber()) {
            choices.add("\"chosen_number\":" + host.getChosenNumber());
        }
        return choices.toString();
    }

    private static void addChoice(StringJoiner into, String name, String value) {
        if (value != null && !value.isEmpty()) {
            into.add(Json.string(name) + ":" + Json.string(value));
        }
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
