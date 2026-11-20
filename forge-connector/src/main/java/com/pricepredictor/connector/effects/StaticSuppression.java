package com.pricepredictor.connector.effects;

import com.google.common.collect.Iterables;
import com.google.common.collect.Table;
import forge.card.CardTypeView;
import forge.card.ColorSet;
import forge.card.mana.ManaCost;
import forge.game.CardTraitBase;
import forge.game.card.Card;
import forge.game.card.CardState;
import forge.game.card.CardTraitChanges;
import forge.game.card.ICardTraitChanges;
import forge.game.keyword.Keyword;
import forge.game.keyword.KeywordInterface;
import forge.game.player.Player;
import forge.game.replacement.ReplacementEffect;
import forge.game.spellability.SpellAbility;
import forge.game.staticability.StaticAbility;
import forge.game.trigger.Trigger;
import forge.util.collect.FCollectionView;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * The board a snapshot shows: as it stands, or without one static — a
 * {@code continuous} record's board, which must not already hold the answer
 * the record labels (base record schema, "Rules").
 *
 * <p>The one place every entity channel is read, so no reader can show one
 * channel without the static and the next with it: type, colour, keywords,
 * set base power and toughness and the boosts on top, controller, summoning
 * sickness, the lines the static granted and the lines it took away, and
 * whether a static says which permanents it affects. Recombined by the engine
 * rather than subtracted here: another static granting the same keyword or
 * type keeps it, and a static that overwrites a type line or strips every
 * ability cannot be inverted from what it contributed. Forge's
 * {@code ...Without} reads run the layer system with the static's entries
 * left out and touch no cached value.
 *
 * <p>A static's control change is filed under its timestamp, not its id
 * ({@code Card.addTempController}), so control is read without that
 * timestamp.
 */
final class StaticSuppression {

    /** The board as it stands. */
    static final StaticSuppression NONE = new StaticSuppression(null, 0L);

    /** The suppressed static's layer-table id, or null for none. */
    private final Long staticId;
    /** The suppressed static's timestamp, which its control change is filed under. */
    private final long timestamp;

    private StaticSuppression(Long staticId, long timestamp) {
        this.staticId = staticId;
        this.timestamp = timestamp;
    }

    /** The board without {@code stat}. */
    static StaticSuppression of(StaticAbility stat) {
        return new StaticSuppression((long) stat.getId(), stat.getTimestamp());
    }

    /** The board without the static of this layer-table id, its control change kept. */
    static StaticSuppression ofId(long staticId) {
        return new StaticSuppression(staticId, Long.MIN_VALUE);
    }

    /** Whether a static is suppressed: a {@code continuous} record's board. */
    boolean active() {
        return staticId != null;
    }

    /** Whether a layer-table entry under {@code column} is the suppressed static's. */
    boolean wrote(Long column) {
        return staticId != null && staticId.equals(column);
    }

    // ── characteristics ─────────────────────────────────────────────────

    CardTypeView type(Card card) {
        return staticId == null ? card.getType() : card.getTypeWithout(staticId);
    }

    ColorSet colors(Card card) {
        return staticId == null ? card.getColor() : card.getColorWithout(staticId);
    }

    Iterable<KeywordInterface> keywords(Card card) {
        return staticId == null ? card.getKeywords() : card.getKeywordsWithout(staticId);
    }

    /**
     * The card's hidden extrinsic keywords as Forge spells them ("CARDNAME
     * can't block."), in timestamp order, the suppressed static's left out.
     * Forge keeps them apart from its keyword list: a resolved {@code KW$
     * HIDDEN} grant and a static's {@code AddHiddenKeyword$}.
     */
    List<String> hiddenKeywords(Card card) {
        List<String> hidden = new ArrayList<>();
        var table = ForgeFields.hiddenExtrinsicKeywords(card);
        if (table == null) {
            return hidden;
        }
        for (var cell : table.cellSet()) {
            if (!suppressed(cell)) {
                addKeywords(cell.getValue(), hidden);
            }
        }
        return hidden;
    }

    /** One hidden-keyword entry's keywords, each as Forge spells it. */
    private static void addKeywords(Object entry, List<String> into) {
        if (entry instanceof Iterable<?> keywords) {
            for (Object keyword : keywords) {
                into.add(String.valueOf(keyword));
            }
        }
    }

    /**
     * The card's name on this board: without the suppressed static's name
     * change (layer 3, a {@code SetName$}, Volrath's Shapeshifter's text),
     * the other changes still applied in order.
     */
    String name(Card card) {
        var names = changedNames(card);
        if (keepsForgeReading(names)) {
            return card.getName();
        }
        String name = card.getCurrentState().getName();
        for (var cell : names.cellSet()) {
            Card.CardChangedName change = overwriteIn(cell);
            if (change != null) {
                name = change.newName();
            }
        }
        return name;
    }

    /** The card's name changes ({@code Card.changedCardNames}), or null where unread. */
    private static Table<?, ?, ?> changedNames(Card card) {
        return PatchHooks.CHANGED_NAMES.read(card) instanceof Table<?, ?, ?> table ? table : null;
    }

    /** A name-table entry's overwriting name change, or null: the suppressed static's, or none. */
    private Card.CardChangedName overwriteIn(Table.Cell<?, ?, ?> cell) {
        if (suppressed(cell)) {
            return null;
        }
        return cell.getValue() instanceof Card.CardChangedName change && change.isOverwrite()
                ? change : null;
    }

    /**
     * The card's mana cost on this board: without the suppressed static's
     * change ({@code ManaCost$}, {@code Incorporate$}), else Forge's own.
     */
    ManaCost manaCost(Card card) {
        var changes = ForgeFields.changedCardManaCost(card);
        if (keepsForgeReading(changes)) {
            return card.getManaCost();
        }
        ManaCost cost = card.getOriginalManaCost();
        for (var cell : changes.cellSet()) {
            if (suppressed(cell)) {
                continue;
            }
            Map.Entry<ManaCost, Boolean> change = ForgeFields.manaCostChange(cell.getValue());
            if (change == null) {
                return card.getManaCost();
            }
            cost = applied(cost, change);
        }
        return cost;
    }

    /** {@code cost} after one mana-cost change: added to it, or in its place. */
    private static ManaCost applied(ManaCost cost, Map.Entry<ManaCost, Boolean> change) {
        return change.getValue() ? ManaCost.combine(cost, change.getKey()) : change.getKey();
    }

    /** The card's mana value on this board: Forge's own unless the static changed its cost. */
    int manaValue(Card card) {
        if (keepsForgeReading(ForgeFields.changedCardManaCost(card))) {
            return card.getCMC();
        }
        ManaCost cost = manaCost(card);
        return cost == null ? 0 : cost.getCMC();
    }

    /**
     * Whether Forge's own reading is this board's for a characteristic kept
     * in {@code changes}: no static is suppressed, the table is unread, or
     * the suppressed static wrote nothing in it.
     */
    private boolean keepsForgeReading(Table<?, ?, ?> changes) {
        return staticId == null || changes == null || !changes.containsColumn(staticId);
    }

    /** Whether a layer-table cell is the suppressed static's. */
    private boolean suppressed(Table.Cell<?, ?, ?> cell) {
        return cell.getColumnKey() instanceof Long column && wrote(column);
    }

    /** Whether the card is a creature on this board: whether it has power and toughness. */
    boolean isCreature(Card card) {
        return staticId == null ? card.isCreature() : type(card).isCreature();
    }

    /** Forge's breakdown of the power the card fights with; zeros on a noncreature permanent. */
    Card.StatBreakdown power(Card card) {
        return staticId == null
                ? card.getNetPowerBreakdown() : card.getNetPowerBreakdownWithout(staticId);
    }

    /** As {@link #power}, for toughness. */
    Card.StatBreakdown toughness(Card card) {
        return staticId == null
                ? card.getNetToughnessBreakdown() : card.getNetToughnessBreakdownWithout(staticId);
    }

    Player controller(Card card) {
        return staticId == null ? card.getController() : card.getControllerWithout(timestamp);
    }

    /**
     * Gives the live card this board's controller, for a read Forge makes of
     * the live board only, by taking off the control change the suppressed
     * static filed. Where the two controllers differ that change is the one
     * in force, so putting it back under its timestamp restores the card
     * exactly. Returns what puts it back, or null when the controllers agree.
     */
    Runnable lendController(Card card) {
        Player live = card.getController();
        if (live.equals(controller(card))) {
            return null;
        }
        card.removeTempController(timestamp);
        return () -> card.addTempController(live, timestamp);
    }

    /**
     * Gives the live card this board's type line, for a read Forge makes of
     * the live board only (a static's condition counting artifacts), by
     * lending it as the type line Forge caches for the card's state
     * ({@code CardState.lendEffectRecordType}). The layer tables the line is
     * computed from are not touched, so giving the cached line back restores
     * the card exactly. Returns what gives it back, or null when the
     * suppressed static changed no type of the card.
     */
    Runnable lendType(Card card) {
        if (staticId == null || !changedType(card)) {
            return null;
        }
        return PatchHooks.LEND_TYPE.lend(card.getCurrentState(), CardTypeView.class, type(card));
    }

    /**
     * Gives the live card this board's colours, for a read Forge makes of
     * the live board only ("each colour among permanents you control"), by
     * setting the suppressed static's colour changes aside from the tables
     * Forge computes the card's colours from
     * ({@code Card.setAsideEffectRecordColors}); putting the same changes
     * back restores the card exactly. Returns what puts them back, or null
     * when the suppressed static changed no colour of the card.
     */
    Runnable lendColors(Card card) {
        return staticId == null ? null : PatchHooks.SET_ASIDE_COLORS.lend(card, long.class, staticId);
    }

    /** Whether the suppressed static has a type change on the card. */
    private boolean changedType(Card card) {
        return card.getChangedCardTypesWithout(staticId).size()
                != Iterables.size(card.getChangedCardTypes());
    }

    /**
     * Summoning sickness as Forge reads it — controlled since the turn
     * began, no haste, a creature — from this board's controller, keywords
     * and type.
     */
    boolean sick(Card card) {
        if (staticId == null) {
            return card.isSick();
        }
        if (!firstTurnControlled(card) || !type(card).isCreature()) {
            return false;
        }
        for (KeywordInterface keyword : keywords(card)) {
            if (keyword.getKeyword() == Keyword.HASTE) {
                return false;
            }
        }
        return true;
    }

    /**
     * Whether the card's controller on this board has not controlled it
     * continuously since their most recent turn began (302.6). Forge keeps
     * one flag, which the zone correction of a control change sets for the
     * new controller: for a permanent the suppressed static took from its
     * controller here, the flag the patch kept for that player at the change
     * ({@code Card.isEffectRecordSickForPriorController}), cleared once a
     * turn of theirs has begun since.
     */
    private boolean firstTurnControlled(Card card) {
        Player board = controller(card);
        if (!movedAwayFrom(card, board)) {
            return card.isFirstTurnControlled();
        }
        return Boolean.TRUE.equals(PatchHooks.PRIOR_SICKNESS.read(card));
    }

    /**
     * Whether the card's most recent control change moved it from
     * {@code board}'s battlefield to another player's, where it sits: the
     * zone is never lent, so this reads the same inside a lend.
     */
    private static boolean movedAwayFrom(Card card, Player board) {
        if (card.getZone() == null || board.equals(card.getZone().getPlayer())) {
            return false;
        }
        return board.equals(PatchHooks.PRIOR_CONTROLLER.read(card));
    }

    // ── lines ───────────────────────────────────────────────────────────

    FCollectionView<SpellAbility> spellAbilities(CardState state) {
        return staticId == null ? state.getSpellAbilities() : state.getSpellAbilitiesWithout(staticId);
    }

    FCollectionView<Trigger> triggers(CardState state) {
        return staticId == null ? state.getTriggers() : state.getTriggersWithout(staticId);
    }

    FCollectionView<StaticAbility> staticAbilities(CardState state) {
        return staticId == null
                ? state.getStaticAbilities() : state.getStaticAbilitiesWithout(staticId);
    }

    FCollectionView<ReplacementEffect> replacementEffects(CardState state) {
        return staticId == null
                ? state.getReplacementEffects() : state.getReplacementEffectsWithout(staticId);
    }

    /**
     * The lines a layer-6 change granted the card, the suppressed static's
     * left out: a static's, an attachment's, a resolved effect's grant. Each
     * maps to the id of the static that granted it, 0 for a resolved effect.
     */
    Map<CardTraitBase, Long> grantedLines(Card card) {
        Map<CardTraitBase, Long> granted = new IdentityHashMap<>();
        for (Table.Cell<Long, Long, ICardTraitChanges> cell : card.getChangedCardTraits().cellSet()) {
            if (cell.getValue() instanceof CardTraitChanges changes && !wrote(cell.getColumnKey())) {
                for (CardTraitBase line : linesOf(changes)) {
                    granted.put(line, cell.getColumnKey());
                }
            }
        }
        return granted;
    }

    private static List<CardTraitBase> linesOf(CardTraitChanges changes) {
        List<CardTraitBase> lines = new ArrayList<>(changes.getAbilities());
        lines.addAll(changes.getTriggers());
        lines.addAll(changes.getReplacements());
        lines.addAll(changes.getStaticAbilities());
        return lines;
    }

    /**
     * Whether a keyword's trait comes from a keyword the suppressed static
     * granted: such a trait is not on this board.
     */
    boolean grantedKeyword(KeywordInterface keyword) {
        // A long against Forge's int id: Long.equals(Integer) is always
        // false, which once let the removed static's flying back in as the
        // CantBlockBy static it builds.
        return staticId != null && keyword != null && keyword.getStatic() != null
                && staticId == keyword.getStatic().getId();
    }

    /**
     * Whether a static's line says which permanents and players it affects:
     * never on a continuous record's board, whose answer that is — hiding
     * only the suppressed static's list would point at it.
     */
    boolean showsAffected() {
        return staticId == null;
    }
}
