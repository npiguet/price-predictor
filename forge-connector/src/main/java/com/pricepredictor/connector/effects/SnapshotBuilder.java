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
import forge.game.card.CardTraitChanges;
import forge.game.card.ICardTraitChanges;

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
 * <p>Every snapshot renders all four inclusion tiers: every referenced object
 * in whatever zone it sits (1), the global state, the battlefield and
 * command-zone effect cards (2), the unreferenced stack (3), and unreferenced
 * hands and graveyards (4). The depth is the same on every record, whatever
 * collector or run wrote it, because a depth that varied made
 * {@code state.tiers} a proxy for how a record was collected: interventional
 * forks once rendered four tiers while every other collector rendered two or
 * three, so tier 4 identified {@code interventional} exactly, and a flag the
 * schema bars from ever being a model input had a perfect proxy sitting in the
 * state.
 */
public final class SnapshotBuilder {

    private static final char[] COLORS = {'W', 'U', 'B', 'R', 'G'};

    private final Game game;

    public SnapshotBuilder(Game game) {
        this.game = game;
    }

    /** The entity id an event's subject and the snapshot both use. */
    public static String entityId(Card card) {
        return Refs.of(card);
    }

    public static String playerId(Player player) {
        return Refs.of(player);
    }

    /**
     * An ability's tier-1 referenced list: its host and the cards it targets.
     *
     * <p>One list for every collector that snapshots an ability, observed or
     * forked, because a target outside the zones the other tiers sweep — an
     * exiled card, say — is in a snapshot only through this tier, and a
     * collector that left it out wrote a record naming a target its own state
     * did not hold.
     */
    static List<Card> referencedOf(SpellAbility ability) {
        List<Card> referenced = new ArrayList<>();
        if (ability == null) {
            return referenced;
        }
        if (ability.getHostCard() != null) {
            referenced.add(ability.getHostCard());
        }
        if (ability.getTargets() != null) {
            for (Object target : ability.getTargets()) {
                if (target instanceof Card card) {
                    referenced.add(card);
                }
            }
        }
        return referenced;
    }

    /**
     * The snapshot for a spell ability that acts: its refs name the ability's
     * host, targets and announced values.
     *
     * @param acting     the ability, or null — a bracket opened for nothing, a
     *                   legality question with no candidate — whose refs then
     *                   render empty, as {@link #ofBoard}'s do
     * @param referenced the cards the record points at, carried whatever zone
     *                   they sit in (tier 1) — {@link #referencedOf} for an
     *                   ability being cast or forced, the host alone for a
     *                   playability candidate, which has no targets yet
     */
    Snapshot ofAbility(SpellAbility acting, Iterable<Card> referenced) {
        return snapshot(referenced, null, acting, null, null);
    }

    /**
     * The snapshot for a record no single line acts in — a damage step, a
     * legality question, a mana activation, an orphaned outcome — so its refs
     * block is the empty shape.
     */
    Snapshot ofBoard() {
        return snapshot(List.of(), null, null, null, null);
    }

    /**
     * The board-only snapshot with one static's contributions removed from
     * every entity: a {@code continuous} record's state, which must not
     * already contain the answer the record labels.
     */
    Snapshot ofBoardWithout(long staticId) {
        return snapshot(List.of(), staticId, null, null, null);
    }

    /**
     * The snapshot for a trait — a trigger or a replacement — about to handle
     * an event.
     *
     * <p>A trait is not a {@link SpellAbility}, so it cannot be the acting
     * ability -- but it has a host card, and that is what {@code refs.source}
     * names: the join that lets the record say which permanent's text acted.
     * Neither kind has targets or announced values at the moment it is
     * recorded, so the rest of the refs block is legitimately empty.
     *
     * <p>The host is the tier-1 referenced list too. Without it a host outside
     * every zone the other tiers sweep — one in exile, say — is in no
     * {@code state.entities}, and {@code refs.source} would name an entity id
     * the reader cannot follow.
     *
     * @param source  the trait's host, or null where the trait has none
     * @param pending the event as the trait received it
     */
    Snapshot ofTrait(Card source, EffectEvent pending) {
        return snapshot(source == null ? List.of() : List.of(source), null, null, source, pending);
    }

    /**
     * The one assembly behind every call shape, private so that no caller
     * passes the combination of nulls that picks its shape.
     *
     * @param withoutStatic the id of a static whose changes are removed from
     *                      every entity, or null to show the board as it stands
     * @param source        the host of an acting line that is not a spell
     *                      ability, or null
     * @param pending       the event a trait is about to handle, or null
     */
    private Snapshot snapshot(
            Iterable<Card> referenced, Long withoutStatic, SpellAbility acting, Card source,
            EffectEvent pending) {
        StringJoiner entityJson = new StringJoiner(",", "[", "]");
        for (Card card : collectEntities(referenced)) {
            entityJson.add(entityToJson(card, Suppressed.of(card, withoutStatic)));
        }
        StringJoiner playerJson = new StringJoiner(",", "[", "]");
        for (Player player : game.getPlayers()) {
            playerJson.add(playerToJson(player));
        }
        return new Snapshot(
                globalToJson(), playerJson.toString(), entityJson.toString(),
                refsJson(acting, source), pending == null ? "null" : pending.toJson());
    }

    /**
     * Every entity the snapshot carries, tier by tier, each once and in the
     * first tier that reaches it: the referenced cards wherever they sit (1),
     * the battlefield and the command zone's effect cards (2), the stack (3),
     * then hands and graveyards (4).
     */
    private Set<Card> collectEntities(Iterable<Card> referenced) {
        Set<Card> entities = new LinkedHashSet<>();
        for (Card card : referenced) {
            if (card != null) {
                entities.add(card);
            }
        }
        entities.addAll(game.getCardsIn(EnumSet.of(ZoneType.Battlefield, ZoneType.Command)));
        entities.addAll(game.getCardsIn(ZoneType.Stack));
        entities.addAll(game.getCardsIn(EnumSet.of(ZoneType.Hand, ZoneType.Graveyard)));
        return entities;
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
                + ",\"emblems\":" + ProvenanceKey.toJsonArray(emblems()) + "}";
    }

    /**
     * Command-zone emblems, as the key of the line that made each.
     *
     * <p>An emblem is a continuous effect with no permanent to hang on, so it
     * appears nowhere else in the snapshot; without this the board says nothing
     * about an effect that changes every turn of the rest of the game.
     */
    private List<ProvenanceKey> emblems() {
        Set<ProvenanceKey> keys = new LinkedHashSet<>();
        for (Card card : game.getCardsIn(ZoneType.Command)) {
            if (card.isEmblem()) {
                addKeys(keys, card.getSpellAbilities());
            }
        }
        return List.copyOf(keys);
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
        DamageStep step = DamageStep.of(game.getPhaseHandler());
        return step == null ? null : step.wireName();
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
         * record is never written — {@link ContinuousChannel#collect}
         * checks for these three before it collects, so a leaking snapshot is
         * not reachable by falling back to here.
         */
        static Suppressed of(Card card, Long staticId) {
            if (staticId == null) {
                return null;
            }
            if (!(PatchHooks.TYPE_WITHOUT.read(card, staticId) instanceof CardTypeView type)
                    || !(PatchHooks.COLOR_WITHOUT.read(card, staticId) instanceof ColorSet colors)
                    || !(PatchHooks.KEYWORDS_WITHOUT.read(card, staticId)
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
        CardTypeView type = without == null ? card.getType() : without.type();
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
                + ",\"types\":" + typeNamesJson(type.getCoreTypes())
                + ",\"subtypes\":" + typeNamesJson(type.getSubtypes())
                + ",\"supertypes\":" + typeNamesJson(type.getSupertypes())
                + ",\"colors\":" + Json.stringArray(
                        colorLetters(without == null ? card.getColor() : without.colors()))
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
                + ",\"granted_attached\":" + ProvenanceKey.toJsonArray(grantedAttached(card))
                + ",\"granted_temporary\":" + grantedTemporaryJson(card, without)
                + ",\"printed\":" + ProvenanceKey.toJsonArray(printedKeys(card))
                + ",\"stack_extras\":" + stackExtrasJson(card) + "}";
    }

    /**
     * The token's script stem — what FR-078 means by a token-script id — or
     * the printed name when no script can be named (an engine-built token with
     * no image key), so a token is never reported as a non-token.
     */
    static String tokenScriptIdOf(Card card) {
        if (!card.isToken()) {
            return null;
        }
        String stem = ScriptLocator.tokenScriptStemOf(card);
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
        return "{\"keywords\":" + Json.stringArray(keywordNames(
                        without == null ? card.getKeywords() : without.keywords()))
                + ",\"abilities\":" + ProvenanceKey.toJsonArray(grantedTemporaryAbilities(card)) + "}";
    }

    /**
     * An entity's keywords as the snapshot spells them: each keyword's
     * original text, lower-cased, skipping the ones with none.
     *
     * <p>Shared with the {@code token_created} outcome, whose description of
     * a token has to read the same as that token's entry on the board.
     */
    static List<String> keywordNames(Iterable<KeywordInterface> keywords) {
        List<String> names = new ArrayList<>();
        for (KeywordInterface keyword : keywords) {
            if (!keyword.getOriginal().isEmpty()) {
                names.add(keyword.getOriginal().toLowerCase(java.util.Locale.ROOT));
            }
        }
        return names;
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
        Set<ProvenanceKey> keys = new LinkedHashSet<>();
        for (ICardTraitChanges change : card.getChangedCardTraits().values()) {
            // Only the record carries granted lines; a keyword change sharing
            // the table has none to name.
            if (change instanceof CardTraitChanges granted) {
                addKeys(keys, granted.getAbilities());
            }
        }
        return List.copyOf(keys);
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
            String id = Refs.ofAny(target).orElse(null);
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
        Set<ProvenanceKey> keys = new LinkedHashSet<>();
        addKeys(keys, card.getSpellAbilities());
        return List.copyOf(keys);
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
        Set<ProvenanceKey> keys = new LinkedHashSet<>();
        for (Card attachment : card.getAttachedCards()) {
            addKeys(keys, attachment.getSpellAbilities());
        }
        return List.copyOf(keys);
    }

    /** Add every line's key that has one and is not already listed. */
    private static void addKeys(Set<ProvenanceKey> into, Iterable<?> lines) {
        for (Object line : lines) {
            if (line instanceof SpellAbility sa) {
                ProvenanceKey key = ProvenanceResolver.of(sa);
                if (key != null) {
                    into.add(key);
                }
            }
        }
    }

    /** One part of a type line, as the snapshot's names. */
    private static String typeNamesJson(Iterable<?> types) {
        List<String> names = new ArrayList<>();
        for (Object type : types) {
            names.add(typeName(type));
        }
        return Json.stringArray(names);
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
     * <p>Readable on its own because the refs block is the one part of a
     * snapshot that is not read at the same moment as the rest: a collector
     * re-reads it at resolution and hands it to {@link Snapshot#withRefs}.
     */
    String refsJson(SpellAbility acting) {
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
                    Refs.ofAny(target).ifPresent(ref -> targets.add(Json.string(ref)));
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
}
