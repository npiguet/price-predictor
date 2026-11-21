package com.pricepredictor.connector.effects;

import com.google.common.collect.Table;
import forge.card.ICardChangedType;
import forge.game.CardTraitBase;
import forge.game.Game;
import forge.game.StaticEffect;
import forge.game.card.Card;
import forge.game.card.CardState;
import forge.game.card.CardTraitChanges;
import forge.game.card.ICardTraitChanges;
import forge.game.keyword.KeywordsChange;
import forge.game.player.Player;
import forge.game.staticability.StaticAbility;
import forge.game.zone.ZoneType;
import org.apache.commons.lang3.tuple.Pair;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.BiConsumer;

/**
 * Each static's share of the board, read off the layer tables.
 *
 * <p>The tables are keyed by (timestamp, applying static's id), so each
 * entity's share of an effect is a read rather than an inference — an
 * anthem's +1/+1 is recorded against the anthem, on each creature it touches,
 * without diffing anything. Grouped by static, then by entity, both in the
 * order the board first names them.
 */
final class LayerContributions {

    private final Map<Long, Map<String, Contribution>> byStatic = new LinkedHashMap<>();

    /** Static id to entity to that entity's share of the static's effect. */
    Map<Long, Map<String, Contribution>> byStatic() {
        return byStatic;
    }

    /** File every layer change {@code card} receives under the static that made it. */
    void addCard(Card card) {
        new CardLayers(card).addAll();
    }

    /** File {@code card}'s name changes alone; see {@link CardLayers#addNames}. */
    void addNames(Card card) {
        new CardLayers(card).addNames();
    }

    /**
     * File each permanent a static gives control of under that static, with
     * the controller it gives.
     *
     * <p>Forge files a static's control change under the static's timestamp
     * rather than its id ({@code Card.addTempController}), so the static is
     * found from its side: each {@code GainControl$} static in force and the
     * permanents its effect applies to. A permanent whose controller is the
     * same without the static — a later control change overrides it — gets
     * nothing from it.
     */
    void addControl(Game game) {
        List<Card> hosts = new ArrayList<>(
                game.getCardsIn(EnumSet.of(ZoneType.Battlefield, ZoneType.Command)));
        for (Card host : hosts) {
            for (StaticAbility stat : host.getStaticAbilities()) {
                if (stat.hasParam("GainControl")) {
                    addControlBy(game, stat);
                }
            }
        }
    }

    /** The permanents one {@code GainControl$} static's effect applies to, each under it. */
    private void addControlBy(Game game, StaticAbility stat) {
        StaticEffect effect = ForgeFields.staticEffectOf(game, stat);
        if (effect == null) {
            return;
        }
        for (Card affected : effect.getAffectedCards()) {
            Player now = affected.getController();
            if (affected.isInPlay() && now != affected.getControllerWithout(stat.getTimestamp())) {
                new CardLayers(affected).under((long) stat.getId()).controller(now);
            }
        }
    }

    /** One card's layer tables, each change filed under the static that made it. */
    private final class CardLayers {
        private final Card card;
        private final String entity;

        CardLayers(Card card) {
            this.card = card;
            this.entity = Refs.of(card);
        }

        /** Every layer, in the order the contributions name them. */
        void addAll() {
            addBoosts();
            addSetPt();
            addNames();
            addKeywords();
            addTraits();
            addHiddenKeywords();
            addLayerChanges(PatchHooks.CHANGED_TYPES_BY_STATIC, ICardChangedType.class, LayerTokens::typeTokens);
            addLayerChanges(PatchHooks.CHANGED_COLORS_BY_STATIC, Object.class, LayerTokens::colorTokens);
        }

        private Contribution under(Long staticId) {
            return byStatic
                    .computeIfAbsent(staticId, id -> new LinkedHashMap<>())
                    .computeIfAbsent(entity, Contribution::new);
        }

        private void addBoosts() {
            for (Table.Cell<Long, Long, Pair<Integer, Integer>> cell
                    : card.getPTBoostTable().cellSet()) {
                under(cell.getColumnKey()).boost(cell.getValue());
            }
        }

        /** Layer 7a then 7b, so a static writing both ends on the base it set last. */
        private void addSetPt() {
            List<Table.Cell<Long, Long, Pair<Integer, Integer>>> cells = new ArrayList<>(
                    card.getSetPTCharacterDefiningTable().cellSet());
            cells.addAll(card.getSetPTTable().cellSet());
            for (Table.Cell<Long, Long, Pair<Integer, Integer>> cell : cells) {
                under(cell.getColumnKey()).setPt(cell.getValue());
            }
        }

        private void addKeywords() {
            for (Table.Cell<Long, Long, KeywordsChange> cell
                    : card.getChangedCardKeywords().cellSet()) {
                LayerTokens.keywordTokens(cell.getValue(), under(cell.getColumnKey()));
            }
        }

        /**
         * Layer 6's ability changes, filed under the static that made each: an
         * ability it grants as its line's key ({@code ProvenanceKey.describe},
         * {@code runtime} where it keys to none), and an ability it takes away
         * as {@code -} and that key, found as the abilities the card has without
         * the static and lacks with it. A keyword's own traits are its keyword's.
         */
        private void addTraits() {
            card.getChangedCardTraits().cellSet().forEach(this::addTrait);
        }

        /** One cell's ability changes, and the abilities it takes away when it takes any and names its static. */
        private void addTrait(Table.Cell<Long, Long, ICardTraitChanges> cell) {
            if (!(cell.getValue() instanceof CardTraitChanges changes)) {
                return;
            }
            Long staticId = cell.getColumnKey();
            Contribution into = under(staticId);
            LayerTokens.grantedTokens(changes, into);
            if (changes.remove() != null && staticId != 0L) {
                LayerTokens.lostTokens(removedBy(StaticSuppression.ofId(staticId)), into);
            }
        }

        /** The abilities, no keyword's, the card has without one static and lacks with it. */
        private List<CardTraitBase> removedBy(StaticSuppression without) {
            CardState state = card.getCurrentState();
            Set<CardTraitBase> now = Collections.newSetFromMap(new IdentityHashMap<>());
            List<CardTraitBase> lost = new ArrayList<>();
            now.addAll(state.getSpellAbilities());
            now.addAll(state.getTriggers());
            now.addAll(state.getStaticAbilities());
            now.addAll(state.getReplacementEffects());
            for (Iterable<? extends CardTraitBase> had : List.of(
                    without.spellAbilities(state), without.triggers(state),
                    without.staticAbilities(state), without.replacementEffects(state))) {
                for (CardTraitBase trait : had) {
                    if (trait.getKeyword() == null && !now.contains(trait)) {
                        lost.add(trait);
                    }
                }
            }
            return lost;
        }

        /**
         * A static's {@code AddHiddenKeyword$} grants ("CARDNAME can't block."),
         * in Forge's hidden-keyword table under the static's id.
         */
        private void addHiddenKeywords() {
            Table<?, ?, ?> hidden = ForgeFields.hiddenExtrinsicKeywords(card);
            if (hidden == null) {
                return;
            }
            hidden.cellSet().forEach(this::addHiddenKeyword);
        }

        /** One cell's hidden keywords, when it is keyed by a static id and holds keywords. */
        private void addHiddenKeyword(Table.Cell<?, ?, ?> cell) {
            Long staticId = staticIdOf(cell);
            if (staticId != null && cell.getValue() instanceof Iterable<?> keywords) {
                Contribution into = under(staticId);
                keywords.forEach(keyword -> into.hiddenKeyword(LayerToken.added(String.valueOf(keyword))));
            }
        }

        /**
         * A patched layer accessor's answer, each change of the {@code entries}
         * type filed as tokens under the static that wrote it.
         *
         * <p>Reflective because the accessor is the patch's, and the P/T and keyword
         * accessors beside it are not: those exist in stock Forge, which is what the
         * connector compiles against. An unpatched checkout answers an empty map,
         * though the continuous channel has already declined by then.
         */
        private <T> void addLayerChanges(
                InstanceAccessor accessor, Class<T> entries, BiConsumer<T, Contribution> tokens) {
            for (Map.Entry<Long, List<Object>> changed : changesBy(accessor).entrySet()) {
                Contribution into = under(changed.getKey());
                changed.getValue().stream()
                        .filter(entries::isInstance)
                        .forEach(change -> tokens.accept(entries.cast(change), into));
            }
        }

        @SuppressWarnings("unchecked")
        private Map<Long, List<Object>> changesBy(InstanceAccessor accessor) {
            Object answer = accessor.read(card);
            return answer instanceof Map ? (Map<Long, List<Object>>) answer : Map.of();
        }

        /**
         * A card's name contribution, filed under the static that wrote it.
         *
         * <p>Reflective because {@code getChangedCardNames} is the patch's own
         * accessor. Unlike the type and colour "ByStatic" readers above, that
         * table is not split across layers -- it is keyed (timestamp, static id)
         * exactly like {@code boostPT} -- so its shape here is the P/T loop's
         * rather than theirs: the raw table, filtered by column key directly
         * instead of pre-grouped on the Forge side.
         */
        private void addNames() {
            if (PatchHooks.CHANGED_NAMES.read(card) instanceof Table<?, ?, ?> names) {
                names.cellSet().forEach(this::addName);
            }
        }

        /** One cell's name, when it is keyed by a static id. */
        private void addName(Table.Cell<?, ?, ?> cell) {
            Long staticId = staticIdOf(cell);
            if (staticId != null) {
                LayerTokens.nameTokens(cell.getValue(), under(staticId));
            }
        }
    }

    /** The static id {@code cell} is filed under, or null for a cell keyed by anything else. */
    private static Long staticIdOf(Table.Cell<?, ?, ?> cell) {
        return cell.getColumnKey() instanceof Long staticId ? staticId : null;
    }
}
