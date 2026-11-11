package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.json.Json;
import forge.card.CardStateName;
import forge.game.CardTraitBase;
import forge.game.Game;
import forge.game.StaticEffect;
import forge.game.card.Card;
import forge.game.card.CardState;
import forge.game.keyword.Keyword;
import forge.game.keyword.KeywordInterface;
import forge.game.spellability.OptionalCost;
import forge.game.spellability.SpellAbility;
import forge.game.spellability.SpellPermanent;
import forge.game.staticability.StaticAbility;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.EnumMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * A card slot's {@code lines}: every line the card carries now, each with the
 * provenance key the sidecar joins and the flags the model reads beside it.
 *
 * <p>Read off Forge's live trait lists rather than the card's script, because
 * the live lists are what the engine plays: a creature that lost all its
 * abilities has none, a creature an anthem gave a line has it, and a line a
 * text change rewrote is there as rewritten. A line not printed on the card
 * itself is flagged {@code granted}.
 *
 * <p>Keywords are lines of their own, one per instance, written with the
 * instance's own fragment ({@code Ward:2}, {@code Protection:Goblin}): a
 * granted keyword names no printed line on the card that received it, and the
 * fragment is then the only place its value is said. The triggers, statics and
 * abilities a keyword instance builds are that keyword's line, not lines of
 * their own.
 *
 * <p>A static of the face up that Forge applies as a continuous effect lists
 * the permanents, then the players, it affects ({@code affected}), from
 * Forge's own static effect; no static of a {@code continuous} record's board lists them, since
 * which permanents a static touches is that record's answer.
 *
 * <p>The faces that are not up are listed too, every line flagged inactive:
 * that is how a face-down permanent carries its real face, and how a
 * double-faced card carries the face it would turn to.
 */
final class CardLines {

    private CardLines() {
    }

    /**
     * The card's lines as the JSON array a card slot carries.
     *
     * @param board the board they are read on: as it stands, or without one
     *              static, whose grants are gone and whose removals are back
     */
    static String render(Game game, Card card, StaticSuppression board) {
        return Json.array(new CardLines.Builder(game, card, board).build(), Line::toJson);
    }

    /** One line of a card slot. Null members are absent from the JSON. */
    private record Line(
            List<ProvenanceKey> keys, String kind, boolean granted, Boolean active,
            Integer castWith, List<Integer> chosen, String fragment, List<String> affected) {

        String toJson() {
            Json.ObjectWriter json = Json.object()
                    .raw("keys", ProvenanceKey.toJsonArray(keys))
                    .value("kind", kind)
                    .value("granted", granted);
            if (active != null) {
                json.value("active", active);
            }
            if (castWith != null) {
                json.value("cast_with", castWith);
            }
            if (chosen != null) {
                json.value("chosen", chosen);
            }
            if (fragment != null) {
                json.value("fragment", fragment);
            }
            if (affected != null) {
                json.raw("affected", Json.stringArray(affected));
            }
            return json.render();
        }
    }

    private static final class Builder {
        private final Game game;
        private final Card card;
        private final StaticSuppression board;
        private final String ownScript;
        /**
         * Lines a layer-6 change appended — a static's, an attachment's, a
         * pump's grant — on this board.
         */
        private final Set<CardTraitBase> layerGranted;
        /** The ability of this card on the stack, or null. */
        private final SpellAbility onStack;
        /** The spell this card was cast as, for the cost lines it was cast with. */
        private final SpellAbility cast;
        private final List<Line> lines = new ArrayList<>();
        /** Keys already listed, so a face not up repeats none of the face that is. */
        private final Set<ProvenanceKey> listed = new HashSet<>();

        Builder(Game game, Card card, StaticSuppression board) {
            this.game = game;
            this.card = card;
            this.board = board;
            this.ownScript = ScriptLocator.scriptFileOf(card);
            this.layerGranted = board.grantedLines(card);
            this.onStack = stackAbilityOf(game, card);
            this.cast = castSpellOf(card, onStack);
        }

        List<Line> build() {
            CardState up = card.getCurrentState();
            List<KeywordInterface> shown = sorted(board.keywords(card));
            addTraitLines(up, shown, true);
            addKeywordLines(up, shown, true);
            addSplicedLines();
            for (CardState face : facesNotUp()) {
                List<KeywordInterface> printed = sorted(face.getIntrinsicKeywords());
                addTraitLines(face, printed, false);
                addKeywordLines(face, printed, false);
            }
            return lines;
        }

        // ── keyword lines ───────────────────────────────────────────────

        private void addKeywordLines(CardState state, List<KeywordInterface> keywords,
                                     boolean faceUp) {
            Set<KeywordInterface> printed = intrinsicKeywordsOf(state);
            for (KeywordInterface keyword : keywords) {
                if (keyword.getOriginal() == null || keyword.getOriginal().isEmpty()) {
                    continue;
                }
                ProvenanceKey key = ProvenanceResolver.keyOfKeyword(state, keyword);
                int castWith = faceUp ? castWithCount(keyword) : 0;
                add(new Line(keysOf(key), "keyword",
                        !printed.contains(keyword) || notOwn(key),
                        faceUp ? null : Boolean.FALSE,
                        castWith > 0 ? castWith : null, null, keyword.getOriginal(), null),
                        key, faceUp);
            }
        }

        /**
         * How many times the spell this card was cast as paid this keyword's
         * cost: one for an alternative cost or a kicker, the count for a
         * multikicker; zero when it was not cast with it.
         */
        private int castWithCount(KeywordInterface keyword) {
            if (cast == null || keyword.getKeyword() == null) {
                return 0;
            }
            SpellAbility root = cast.getRootAbility();
            int count = sameKeyword(root.getKeyword(), keyword) ? 1 : 0;
            if (root.hasOptionalKeywordAmount(keyword)) {
                count = Math.max(count, root.getOptionalKeywordAmount(keyword));
            }
            int optional = 0;
            for (OptionalCost cost : root.getOptionalCosts()) {
                if (OPTIONAL_COST_KEYWORDS.get(cost) == keyword.getKeyword()) {
                    optional++;
                }
            }
            return Math.max(count, optional);
        }

        // ── trait lines ─────────────────────────────────────────────────

        private void addTraitLines(CardState state, List<KeywordInterface> keywords,
                                   boolean faceUp) {
            Set<KeywordInterface> keywordLines = identitySet();
            keywordLines.addAll(keywords);
            Set<String> keywordTexts = new HashSet<>();
            for (KeywordInterface keyword : keywords) {
                keywordTexts.add(keyword.getOriginal());
            }
            ProvenanceKey stackKey = onStack == null ? null : ProvenanceResolver.of(onStack);
            for (SpellAbility ability : board.spellAbilities(state)) {
                if (isRulesSpell(ability)) {
                    continue;
                }
                addTrait(ability, ability.isSpell() ? "spell" : "activated",
                        faceUp, keywordLines, keywordTexts, stackKey);
            }
            for (CardTraitBase trigger : board.triggers(state)) {
                addTrait(trigger, "trigger", faceUp, keywordLines, keywordTexts, stackKey);
            }
            for (CardTraitBase stat : board.staticAbilities(state)) {
                addTrait(stat, "static", faceUp, keywordLines, keywordTexts, stackKey);
            }
            for (CardTraitBase replacement : board.replacementEffects(state)) {
                addTrait(replacement, "replacement", faceUp, keywordLines, keywordTexts, stackKey);
            }
        }

        private void addTrait(CardTraitBase trait, String kind, boolean faceUp,
                              Set<KeywordInterface> keywordLines, Set<String> keywordTexts,
                              ProvenanceKey stackKey) {
            if (!faceUp && layerGranted.contains(trait)) {
                return;
            }
            KeywordInterface keyword = trait.getKeyword();
            if (keyword != null) {
                if (keywordLines.contains(keyword) || keywordTexts.contains(keyword.getOriginal())) {
                    // The keyword's own line already stands for it.
                    return;
                }
                if (board.grantedKeyword(keyword)) {
                    // A trait of the removed static's keyword grant.
                    return;
                }
            }
            ProvenanceKey key = ProvenanceResolver.of(trait);
            Boolean active = faceUp
                    ? (trait instanceof StaticAbility stat ? holds(stat) : null)
                    : Boolean.FALSE;
            List<Integer> chosen = faceUp && key != null && key.equals(stackKey)
                    ? chosenModes(onStack) : null;
            List<String> affected = faceUp && trait instanceof StaticAbility stat
                    ? affectedBy(stat) : null;
            add(new Line(keysOf(key), kind, layerGranted.contains(trait) || notOwn(key),
                    active, null, chosen, null, affected), key, faceUp);
        }

        /**
         * The permanents, then the players, a static applies to now, as
         * Forge's own static effect lists them; null for a static Forge applies as no continuous effect
         * (a restriction, a cost change), and on every static of a continuous
         * record's board. That record's answer is which permanents its static
         * touches, and hiding only its own static's list would both point at
         * that static and set apart two records that otherwise say the same.
         */
        private List<String> affectedBy(StaticAbility stat) {
            if (!board.showsAffected()) {
                return null;
            }
            StaticEffect effect = ForgeFields.staticEffectOf(game, stat);
            if (effect == null) {
                return null;
            }
            return Refs.affectedBy(effect);
        }

        /**
         * Whether an ability is one every card of its type has rather than a
         * line: a land's play, a permanent's plain cast. Forge's own
         * {@code hasNoAbilities} passes over the same two, and the sidecar
         * maps them to no line ({@code dropped_keys}), so a creature that
         * lost all its abilities lists none.
         */
        private static boolean isRulesSpell(SpellAbility ability) {
            return ability.isLandAbility()
                    || (ability instanceof SpellPermanent && ability.isBasicSpell());
        }

        /**
         * Forge's answer to whether a static's condition holds now; false when
         * the question cannot be asked of this card, such as a static on a
         * card no player controls.
         */
        private static Boolean holds(StaticAbility stat) {
            try {
                return stat.checkConditions();
            } catch (RuntimeException e) {
                return Boolean.FALSE;
            }
        }

        /** The {@code Choices$} positions of the modes a modal ability chose, or null. */
        private static List<Integer> chosenModes(SpellAbility ability) {
            List<Integer> chosen = new ArrayList<>();
            for (SpellAbility sub = ability.getSubAbility(); sub != null; sub = sub.getSubAbility()) {
                CharmModes.ModeStart start = CharmModes.modeStartOf(sub);
                if (start != null && start.option() != null) {
                    chosen.add(start.option());
                }
            }
            return chosen.isEmpty() ? null : List.copyOf(chosen);
        }

        /**
         * The lines spliced onto this card's spell on the stack: each spliced
         * card's spell, granted to this one for the resolution.
         */
        private void addSplicedLines() {
            if (onStack == null || onStack.getSplicedCards() == null) {
                return;
            }
            for (Card spliced : onStack.getSplicedCards()) {
                SpellAbility spell = spliced.getFirstSpellAbility();
                ProvenanceKey key = spell == null ? null : ProvenanceResolver.of(spell);
                add(new Line(keysOf(key), "spell", true, null, null, null, null, null), key, true);
            }
        }

        private void add(Line line, ProvenanceKey key, boolean faceUp) {
            if (!faceUp && key != null && listed.contains(key)) {
                return;
            }
            if (key != null) {
                listed.add(key);
            }
            lines.add(line);
        }

        /** Whether a line's key names a script other than this card's own. */
        private boolean notOwn(ProvenanceKey key) {
            return key != null && !key.scriptFile().equals(ownScript);
        }

        /**
         * The printed faces that are not up. None for a split card, whose
         * printed state already carries both halves.
         */
        private List<CardState> facesNotUp() {
            List<CardStateName> order = FaceOrder.faceOrder(card);
            if (order.get(0) == CardStateName.LeftSplit) {
                return List.of();
            }
            CardStateName up = card.getCurrentStateName();
            List<CardState> faces = new ArrayList<>();
            for (CardStateName name : order) {
                if (name != up && name != CardStateName.FaceDown && card.hasState(name)) {
                    faces.add(card.getState(name));
                }
            }
            return faces;
        }

        /** The keywords a state prints, its split halves' included. */
        private Set<KeywordInterface> intrinsicKeywordsOf(CardState state) {
            Set<KeywordInterface> printed = identitySet();
            printed.addAll(state.getIntrinsicKeywords());
            if (state.getStateName() == CardStateName.Original) {
                for (CardStateName half : List.of(CardStateName.LeftSplit, CardStateName.RightSplit)) {
                    if (card.hasState(half)) {
                        printed.addAll(card.getState(half).getIntrinsicKeywords());
                    }
                }
            }
            return printed;
        }
    }

    /**
     * The keyword whose line an optional cost is paid through (B9). By the
     * enum rather than by name: {@code PromiseGift} is named "Promise Gift"
     * and its keyword "Gift", {@code Jumpstart} "Jump-start" and its keyword
     * {@code JUMP_START}. A cost no keyword line offers — a colour
     * reduction, an alternative cost, a "pay extra for flash", a generic
     * pseudo-kicker — is in no line.
     */
    static final Map<OptionalCost, Keyword> OPTIONAL_COST_KEYWORDS = optionalCostKeywords();

    private static Map<OptionalCost, Keyword> optionalCostKeywords() {
        Map<OptionalCost, Keyword> byCost = new EnumMap<>(OptionalCost.class);
        byCost.put(OptionalCost.Bargain, Keyword.BARGAIN);
        byCost.put(OptionalCost.Buyback, Keyword.BUYBACK);
        byCost.put(OptionalCost.Entwine, Keyword.ENTWINE);
        byCost.put(OptionalCost.Kicker1, Keyword.KICKER);
        byCost.put(OptionalCost.Kicker2, Keyword.KICKER);
        byCost.put(OptionalCost.PromiseGift, Keyword.GIFT);
        byCost.put(OptionalCost.Retrace, Keyword.RETRACE);
        byCost.put(OptionalCost.Jumpstart, Keyword.JUMP_START);
        byCost.put(OptionalCost.Offering, Keyword.OFFERING);
        byCost.put(OptionalCost.Teamwork, Keyword.TEAMWORK);
        return Collections.unmodifiableMap(byCost);
    }

    /** The ability hosted by this card that is on the stack, or null. */
    static SpellAbility stackAbilityOf(Game game, Card card) {
        for (var instance : game.getStack()) {
            SpellAbility candidate = instance.getSpellAbility();
            if (candidate != null && candidate.getHostCard() == card) {
                return candidate;
            }
        }
        return null;
    }

    /**
     * The spell this card was cast as: its spell on the stack, or the spell a
     * permanent resolved from, which Forge keeps while it is on the
     * battlefield for "if it was kicked".
     */
    static SpellAbility castSpellOf(Card card, SpellAbility onStack) {
        if (onStack != null && onStack.isSpell()) {
            return onStack;
        }
        SpellAbility castSA = card.isInPlay() || card.isInZone(ZoneType.Stack) ? card.getCastSA() : null;
        return castSA != null && castSA.isSpell() ? castSA : null;
    }

    private static boolean sameKeyword(KeywordInterface a, KeywordInterface b) {
        return a != null && (a == b || (a.getKeyword() == b.getKeyword()
                && a.getOriginal() != null && a.getOriginal().equals(b.getOriginal())));
    }

    /**
     * Keywords in a fixed order: by fragment, ties kept as Forge lists them.
     * Forge's own order is hash order over an enum whose hash is the identity
     * hash, so it differs from one JVM to the next.
     */
    private static List<KeywordInterface> sorted(Iterable<KeywordInterface> keywords) {
        List<KeywordInterface> list = new ArrayList<>();
        keywords.forEach(list::add);
        list.sort(Comparator.comparing(k -> k.getOriginal() == null ? "" : k.getOriginal()));
        return list;
    }

    private static List<ProvenanceKey> keysOf(ProvenanceKey key) {
        return key == null ? List.of() : List.of(key);
    }

    private static <T> Set<T> identitySet() {
        return Collections.newSetFromMap(new IdentityHashMap<>());
    }
}
