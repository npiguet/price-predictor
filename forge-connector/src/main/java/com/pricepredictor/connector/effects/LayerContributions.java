package com.pricepredictor.connector.effects;

import com.google.common.collect.Table;
import forge.card.CardChangedType;
import forge.card.CardTypeView;
import forge.card.ColorSet;
import forge.card.RemoveType;
import forge.card.StateChangedType;
import forge.card.WordChangedType;
import forge.game.card.Card;
import forge.game.keyword.KeywordInterface;
import forge.game.keyword.KeywordsChange;
import org.apache.commons.lang3.tuple.Pair;

import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
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
        String entity = SnapshotBuilder.entityId(card);
        for (Table.Cell<Long, Long, Pair<Integer, Integer>> cell
                : card.getPTBoostTable().cellSet()) {
            contribution(cell.getColumnKey(), entity)
                    .boost(cell.getValue().getLeft(), cell.getValue().getRight());
        }
        addNames(card, entity);
        for (Table.Cell<Long, Long, KeywordsChange> cell
                : card.getChangedCardKeywords().cellSet()) {
            Contribution into = contribution(cell.getColumnKey(), entity);
            for (KeywordInterface keyword : cell.getValue().getKeywords()) {
                into.keyword(keyword.getOriginal());
            }
            // The same channel's other half. A static that takes a keyword
            // away had no path to keywords_lost, which is a head field, so
            // "loses flying" and "does nothing" recorded identically.
            for (String removed : cell.getValue().getRemoveKeywords()) {
                into.keyword("-" + removed);
            }
        }
        addLayerChanges(card, entity, PatchHooks.CHANGED_TYPES_BY_STATIC,
                LayerContributions::typeTokens);
        addLayerChanges(card, entity, PatchHooks.CHANGED_COLORS_BY_STATIC,
                LayerContributions::colorTokens);
    }

    private Contribution contribution(Long staticId, String entity) {
        return byStatic
                .computeIfAbsent(staticId, id -> new LinkedHashMap<>())
                .computeIfAbsent(entity, Contribution::new);
    }

    /**
     * A patched layer accessor's answer, each change filed as tokens under the
     * static that wrote it.
     *
     * <p>Reflective because the accessor is the patch's, and the P/T and keyword
     * accessors beside it are not: those exist in stock Forge, which is what the
     * connector compiles against. An unpatched checkout answers an empty map,
     * though the continuous channel has already declined by then.
     */
    private void addLayerChanges(
            Card card, String entity, PatchHooks.Hook accessor,
            BiConsumer<Object, Contribution> tokens) {
        for (Map.Entry<Long, List<Object>> changed : changesByStatic(card, accessor).entrySet()) {
            Contribution into = contribution(changed.getKey(), entity);
            for (Object change : changed.getValue()) {
                tokens.accept(change, into);
            }
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<Long, List<Object>> changesByStatic(Card card, PatchHooks.Hook accessor) {
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
    void addNames(Card card, String entity) {
        if (!(PatchHooks.CHANGED_NAMES.read(card) instanceof Table<?, ?, ?> names)) {
            return;
        }
        for (Table.Cell<?, ?, ?> cell : names.cellSet()) {
            if (cell.getColumnKey() instanceof Long staticId) {
                nameTokens(cell.getValue(), contribution(staticId, entity));
            }
        }
    }

    /**
     * What one type-layer entry does, as tokens.
     *
     * <p>Read off the change object rather than diffed out of the applied type
     * line, for the reason the P/T and keyword channels are: two statics that
     * both grant {@code Creature} are indistinguishable in the result, and the
     * record has to say which one this is.
     *
     * <p>A bare token is a type the static adds, {@code -x} one it removes, and
     * a lone {@code =} marks the rest as a type line the static sets outright
     * rather than adds to. Removing every type of a kind has no name to give, so
     * it gets {@code -all-creature-types} and its siblings.
     */
    static void typeTokens(Object change, Contribution into) {
        if (change instanceof CardChangedType typed) {
            addTypes(into, "", typed.addType());
            addTypes(into, "-", typed.removeType());
            if (typed.addAllCreatureTypes()) {
                into.type("all-creature-types");
            }
            for (RemoveType removed : typed.remove()) {
                into.type("-all-" + kebab(removed.name()));
            }
        } else if (change instanceof StateChangedType state) {
            into.type("=");
            addTypes(into, "", state.type());
        } else if (change instanceof WordChangedType word) {
            // A text change swaps one subtype word for another, which is a
            // removal and an addition in the same entry.
            into.type("-" + SnapshotBuilder.typeName(word.oldWord()));
            into.type(SnapshotBuilder.typeName(word.newWord()));
        }
    }

    private static void addTypes(Contribution into, String prefix, CardTypeView types) {
        if (types == null) {
            return;
        }
        for (String name : TypeLine.names(types)) {
            into.type(prefix + name);
        }
    }

    /** {@code CreatureTypes} as {@code creature-types}. */
    private static String kebab(String name) {
        return name.replaceAll("(?<=.)(?=\\p{Upper})", "-")
                .toLowerCase(java.util.Locale.ROOT);
    }

    /**
     * What one colour-layer entry does, as tokens.
     *
     * <p>The letters are the snapshot's, so a static's share of an entity's
     * colours reads the same as the entity's own. A lone {@code =} carries the
     * same meaning it does for types: the static sets the colour rather than
     * adding to it, and a {@code =} with nothing after it turns the entity
     * colourless.
     *
     * <p>Reflective on both components, because the record they belong to is
     * private in the stock Forge this side compiles against; the patch makes it
     * public.
     */
    static void colorTokens(Object change, Contribution into) {
        // Both components or neither: a half-read entry would report a colour
        // change as a replacement, which is the one reading that is never safe
        // to guess at.
        if (!(PatchHooks.COLOR_CHANGE_ADDITIONAL.read(change) instanceof Boolean additional)
                || !(PatchHooks.COLOR_CHANGE_COLORS.read(change) instanceof ColorSet colors)) {
            return;
        }
        List<String> letters = SnapshotBuilder.colorLetters(colors);
        if (!additional) {
            into.color("=");
            if (letters.isEmpty()) {
                // Setting the colour to nothing is what makes a permanent
                // colourless, which the vocabulary spells C rather than as an
                // empty list.
                into.color("C");
            }
        }
        for (String letter : letters) {
            into.color(letter);
        }
    }

    /**
     * What one name-layer entry does: the whole contribution is the name
     * itself, unlike a type or colour token, and a card either has a static
     * overwriting its name or it does not -- {@code newName} is null on
     * every entry that only ever set {@code addNonLegendaryCreatureNames}.
     *
     * <p>Reflective for the reason {@link #colorTokens} is: the record is
     * private in stock Forge.
     */
    static void nameTokens(Object change, Contribution into) {
        if (PatchHooks.NAME_CHANGE_NAME.read(change) instanceof String name) {
            into.name(name);
        }
    }

    /** One entity's share of one static's effect, accumulated across layers. */
    static final class Contribution {
        private final String entity;
        private int power;
        private int toughness;
        private final Set<String> keywords = new LinkedHashSet<>();
        // Sets rather than lists: a static that writes to two type layers can
        // name the same type twice, and the second says nothing the first did
        // not. Insertion-ordered so a "=" marker stays ahead of what it marks.
        private final Set<String> types = new LinkedHashSet<>();
        private final Set<String> colors = new LinkedHashSet<>();
        // Unlike the token sets above, a name contribution is a single value:
        // a static either overwrites the name or it does not, so the last
        // write standing is the answer rather than an accumulation. Null for
        // the statics that are not one of the handful that rename anything --
        // that is the correct, meaningful answer, not a missing one.
        private String name;

        Contribution(String entity) {
            this.entity = entity;
        }

        void boost(int addPower, int addToughness) {
            power += addPower;
            toughness += addToughness;
        }

        void keyword(String keyword) {
            if (keyword != null && !keyword.isEmpty()) {
                keywords.add(keyword.toLowerCase(java.util.Locale.ROOT));
            }
        }

        void type(String token) {
            if (token != null && !token.isEmpty()) {
                types.add(token);
            }
        }

        void color(String token) {
            if (token != null && !token.isEmpty()) {
                colors.add(token);
            }
        }

        void name(String value) {
            if (value != null && !value.isEmpty()) {
                name = value;
            }
        }

        String toJson() {
            return "{\"entity\":" + Json.string(entity)
                    + ",\"pt_boost\":[" + power + "," + toughness + "]"
                    + ",\"keywords\":" + Json.stringArray(keywords)
                    + ",\"types\":" + Json.stringArray(types)
                    + ",\"colors\":" + Json.stringArray(colors)
                    + ",\"name\":" + Json.string(name) + "}";
        }
    }
}
