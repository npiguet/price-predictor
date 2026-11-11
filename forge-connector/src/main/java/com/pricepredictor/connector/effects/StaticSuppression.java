package com.pricepredictor.connector.effects;

import com.google.common.collect.Table;
import forge.card.CardTypeView;
import forge.card.ColorSet;
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

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Set;

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
     * Summoning sickness as Forge reads it — controlled since the turn
     * began, no haste, a creature — from this board's keywords and type.
     */
    boolean sick(Card card) {
        if (staticId == null) {
            return card.isSick();
        }
        if (!card.isFirstTurnControlled() || !type(card).isCreature()) {
            return false;
        }
        for (KeywordInterface keyword : keywords(card)) {
            if (keyword.getKeyword() == Keyword.HASTE) {
                return false;
            }
        }
        return true;
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
     * left out: a static's, an attachment's, a resolved effect's grant.
     */
    Set<CardTraitBase> grantedLines(Card card) {
        Set<CardTraitBase> granted = Collections.newSetFromMap(new IdentityHashMap<>());
        for (Table.Cell<Long, Long, ICardTraitChanges> cell : card.getChangedCardTraits().cellSet()) {
            if (cell.getValue() instanceof CardTraitChanges changes && !wrote(cell.getColumnKey())) {
                granted.addAll(changes.getAbilities());
                granted.addAll(changes.getTriggers());
                granted.addAll(changes.getReplacements());
                granted.addAll(changes.getStaticAbilities());
            }
        }
        return granted;
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
