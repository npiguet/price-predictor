package com.pricepredictor.connector.effects;

import com.google.common.collect.Iterables;
import forge.game.CardTraitBase;
import forge.game.card.Card;
import forge.game.keyword.KeywordInterface;
import forge.game.staticability.StaticAbility;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Why a line of a card slot is not printed on the card: a granted line's
 * {@code grant}, and null for a printed line.
 *
 * <p>A card prints the lines of the script it prints
 * ({@link ScriptLocator#printedScriptOf}: a copy's is what it copies) and of
 * its own script, which for a copy holds the lines its copy effect adds, a
 * copiable exception (707.9b). A mutated permanent prints every merged
 * card's lines. A card also prints a line it grants itself: a
 * Class level's, a "has ... as long as" trigger, a manland's animated
 * trigger, a keyword its own static gives it. Since A05 such a line keys to
 * the line that granted it, in the card's own script.
 *
 * <p>Every other line is granted, by one of:
 * <ul>
 *   <li>{@link #STATIC}: a permanent's static ability (an anthem, an
 *       attachment, an emblem, a type change such as Blood Moon's, the
 *       suspected designation's);</li>
 *   <li>{@link #EFFECT}: a resolved effect (a pump, an animation, a copy
 *       effect another card made), which lasts as long as its duration;</li>
 *   <li>{@link #COUNTER}: a keyword counter on the card (a trample counter);</li>
 *   <li>{@link #SPLICE}: a spliced card's spell, for one resolution.</li>
 * </ul>
 */
final class LineGrant {

    static final String STATIC = "static";
    static final String EFFECT = "effect";
    static final String COUNTER = "counter";
    static final String SPLICE = "splice";

    /** The column a resolved effect's change is filed under, where a static's id would be. */
    private static final Long NO_STATIC = 0L;

    private final Card card;
    private final Set<String> printedScripts;
    /** Each line a layer-6 change granted, to its static's id; 0 for a resolved effect. */
    private final Map<CardTraitBase, Long> layerGranted;

    LineGrant(Card card, Map<CardTraitBase, Long> layerGranted) {
        this.card = card;
        this.printedScripts = printedScriptsOf(card);
        this.layerGranted = layerGranted;
    }

    /**
     * The scripts a card prints the lines of: the one it prints and its own,
     * and for a mutated permanent each merged card's.
     */
    private static Set<String> printedScriptsOf(Card card) {
        Set<String> scripts = new HashSet<>();
        Iterable<Card> printers = card.hasMergedCard() ? card.getMergedCards() : List.of(card);
        for (Card printer : printers) {
            scripts.add(ScriptLocator.printedScriptOf(printer));
            scripts.add(ScriptLocator.scriptFileOf(printer));
        }
        return scripts;
    }

    /** Whether a layer-6 change granted this line. */
    boolean byLayerChange(CardTraitBase trait) {
        return layerGranted.containsKey(trait);
    }

    /** Why a trigger, static, replacement or ability line is granted; null when printed. */
    String ofTrait(CardTraitBase trait, ProvenanceKey key) {
        Long grantor = layerGranted.get(trait);
        if (grantor != null) {
            return ofLayerChange(grantor, key);
        }
        if (ProvenanceResolver.addedByTypeChange(trait)) {
            return ofTypeChange();
        }
        return elsewhere(key) ? EFFECT : null;
    }

    /**
     * Why a keyword line is granted; null when printed. A keyword the state
     * prints is granted only when a copy effect another card made added it:
     * it keys elsewhere, or, on a token or spell copy, nowhere.
     */
    String ofKeyword(KeywordInterface keyword, boolean intrinsic, ProvenanceKey key) {
        if (intrinsic) {
            return elsewhere(key) || addedToACopy(key) ? EFFECT : null;
        }
        StaticAbility granting = keyword.getStatic();
        return granting == null ? EFFECT : ofStatic(granting, key);
    }

    /**
     * A layer-6 grant: a resolved effect's unless the line keys into what
     * the card prints (its own animation), else its static's.
     */
    private String ofLayerChange(Long staticId, ProvenanceKey key) {
        if (NO_STATIC.equals(staticId)) {
            return printed(key) ? null : EFFECT;
        }
        StaticAbility own = ownStatic(staticId);
        return own == null ? STATIC : ofStatic(own, key);
    }

    private String ofStatic(StaticAbility stat, ProvenanceKey key) {
        if (isCounterStatic(stat)) {
            return COUNTER;
        }
        return grantsItself(stat, key) ? null : STATIC;
    }

    /** Whether a static of the card's own gives it a line it prints. */
    private boolean grantsItself(StaticAbility stat, ProvenanceKey key) {
        if (stat.getHostCard() != card || isHidden(stat)) {
            return false;
        }
        return printed(key);
    }

    /**
     * Whether a static is one a keyword counter on the card builds: Forge's
     * hidden statics are those and the suspected designation's.
     */
    private boolean isCounterStatic(StaticAbility stat) {
        return stat != card.getSuspectedStatic() && isHidden(stat);
    }

    private boolean isHidden(StaticAbility stat) {
        return card.getHiddenStaticAbilities().contains(stat);
    }

    /** The card's own static of this id, hidden ones included; null for another card's. */
    private StaticAbility ownStatic(long id) {
        for (StaticAbility stat : Iterables.concat(card.getStaticAbilities(), card.getHiddenStaticAbilities())) {
            if (stat.getId() == id) {
                return stat;
            }
        }
        return null;
    }

    /**
     * A type change's grant (a basic land type's mana ability): a static's
     * when any static changes the card's types (Blood Moon, Spreading Seas,
     * Urborg), even under a later animation; else a resolved effect's.
     */
    private String ofTypeChange() {
        var changes = ForgeFields.changedCardTypes(card);
        if (changes == null) {
            return STATIC;
        }
        for (var change : changes.cellSet()) {
            if (!NO_STATIC.equals(change.getColumnKey())) {
                return STATIC;
            }
        }
        return EFFECT;
    }

    /** Whether a key names a script the card prints. */
    private boolean printed(ProvenanceKey key) {
        return key != null && printedScripts.contains(key.scriptFile());
    }

    /**
     * Whether a keyword the state prints with no key is one a token or spell
     * copy's copy effect added (Kiki-Jiki's haste), which keys to no line
     * ({@link KeywordPrinters#indexOf}).
     */
    private boolean addedToACopy(ProvenanceKey key) {
        return key == null && KeywordPrinters.isStateCopy(card);
    }

    /** Whether a key names a script the card does not print. */
    private boolean elsewhere(ProvenanceKey key) {
        return key != null && !printedScripts.contains(key.scriptFile());
    }
}
