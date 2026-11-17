package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.KeywordRendering;
import com.pricepredictor.connector.ability.OptionAbility;
import forge.game.CardTraitBase;
import forge.game.keyword.KeywordInterface;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Collects, while a card is parsed, which runtime trait produced which
 * {@link Ability}.
 *
 * <p>Recording happens at the parse sites rather than inside the fifteen
 * {@code Ability} implementations, because that is the only place both the
 * trait and its index within its kind are in scope. Abilities are keyed by
 * identity: two abilities can render identical text and still come from
 * different traits, which is exactly the case the converter deduplicates and
 * the sidecar has to report.
 *
 * <p>A recorder is per face. Keys declared but attached to no surviving ability
 * end up in the sidecar's {@code dropped_keys}, which holds the traits no
 * line states — a permanent's implicit cast spell, say, or a charm whose
 * modes are its only lines, and a trait whose text the converter does not
 * render. A trait the converter deduplicated, merged or
 * derived from a keyword keeps its attribution on the line that carries its
 * text, because it is still live at runtime and a record naming it must reach
 * the words it produced. A mode key is never declared: it is claimed by the
 * mode's line or it does not exist, so it can never be dropped.
 *
 * <p>A trait a keyword built is named through its keyword's key, never by its
 * kind-indexed key, which is declared nowhere: a keyword printing one line
 * declares its bare key there, one printing several declares on each line the
 * keys with {@code built} of the traits that line prints.
 */
public final class ProvenanceRecorder {

    /**
     * What one parsed ability came from: the script of the trait that first
     * claimed it, and every key claiming it since.
     *
     * <p>A line merged from several traits carries several keys, so the join
     * is many-to-one and never assumed one-to-one. Keys keep the order they
     * were claimed in, which is the order the sidecar writes them.
     */
    public static final class Source {
        private final Set<ProvenanceKey> keys = new LinkedHashSet<>();
        private final TraitScript script;

        private Source(ProvenanceKey first, TraitScript script) {
            this.keys.add(first);
            this.script = script;
        }

        /** The keys claiming this ability, in claim order; read-only. */
        public Set<ProvenanceKey> keys() {
            return Collections.unmodifiableSet(keys);
        }

        public TraitScript script() {
            return script;
        }

        /** Claim this ability under {@code key} too; a key already held is not repeated. */
        private void addKey(ProvenanceKey key) {
            keys.add(key);
        }
    }

    private final Map<Ability, Source> byAbility = new IdentityHashMap<>();
    private final Set<ProvenanceKey> declared = new LinkedHashSet<>();
    /** Where a chain's undefined SVars go; shared across every face of a run. */
    private final MissingSVarReport missingSVars;

    public ProvenanceRecorder(MissingSVarReport missingSVars) {
        this.missingSVars = missingSVars;
    }

    /**
     * Note that a trait exists on this face, whether or not it produced a line.
     *
     * <p>Called for every trait the parser walks, so the dropped set can be the
     * difference between what the card has and what the file shows.
     */
    public void declare(ProvenanceKey key) {
        if (key != null) declared.add(key);
    }

    /**
     * Attribute one or more parsed abilities to the trait that produced them.
     *
     * <p>A charm's mode lines are attributed here too, to the trait's key
     * extended by the mode's {@code Choices$} position and to the mode's own
     * chain: they arrive either as the charm line's children or, for a charm
     * with no description of its own, as top-level lines in this list. The
     * root key is then claimed by nobody and lands in {@code dropped_keys},
     * which is the honest answer for a line the file does not show. A mode
     * belonging to some other trait — a chapter's options being attributed to
     * a sibling chapter's trigger — is left alone, so it keys to its own.
     */
    public void attribute(List<Ability> abilities, ProvenanceKey key,
                          CardTraitBase trait) {
        if (key == null) return;
        declare(key);
        TraitScript script = TraitScript.of(trait, key, missingSVars);
        for (Ability ability : abilities) {
            if (ability instanceof OptionAbility option) {
                attributeMode(option, key, trait);
                continue;
            }
            claim(ability, key, script);
            attributeModesBelow(ability, key, trait);
        }
    }

    private void attributeModesBelow(Ability ability, ProvenanceKey key, CardTraitBase trait) {
        for (Ability sub : ability.subAbilities()) {
            if (sub instanceof OptionAbility option) attributeMode(option, key, trait);
        }
    }

    private void attributeMode(OptionAbility option, ProvenanceKey rootKey,
                               CardTraitBase trait) {
        if (!option.belongsTo(trait)) return;
        ProvenanceKey modeKey = rootKey.withOption(option.modeIndex());
        claim(option, modeKey, TraitScript.ofMode(
                option.mode(), option.label(), modeKey, missingSVars));
    }

    private void claim(Ability ability, ProvenanceKey key, TraitScript script) {
        Source existing = byAbility.get(ability);
        if (existing == null) {
            byAbility.put(ability, new Source(key, script));
        } else {
            existing.addKey(key);
        }
    }

    /** Attribute a single parsed ability. */
    public void attribute(Ability ability, ProvenanceKey key, CardTraitBase trait) {
        if (ability != null) attribute(List.of(ability), key, trait);
    }

    /**
     * Attribute the lines a keyword rendered to it, under {@code key}, the
     * keyword's own key.
     *
     * <p>A keyword rendering one line declares {@code key} on it, and the
     * traits it built join that line through it. A keyword rendering several
     * declares {@code key} on none: each trait it built is claimed under
     * {@code key} with its {@code built}, by the line the rendering names for
     * it, or by the keyword's one ability. A trait with no such line is
     * declared nowhere, and a record naming it falls back to the bare key.
     * Where no built trait has a line, the keyword's first line declares the
     * bare key, as if the keyword printed that line alone; a keyword
     * rendering no line has its bare key in {@code dropped_keys}.
     */
    public void attributeKeyword(KeywordRendering rendering, ProvenanceKey key,
                                 KeywordInterface keyword) {
        if (key == null) return;
        List<Ability> abilities = rendering.lines();
        if (rendersSeveralLines(abilities) && attributeBuiltTraits(rendering, key, keyword)) return;
        declare(key);
        if (abilities.isEmpty()) return;
        Ability line = abilities.get(0);
        claim(line, key, lineScript(rendering, line, key, keyword));
        // Each built trait again under the bare key: the line holds that key
        // already, and a charm's mode lines (an Attraction's visit) take it
        // with the mode's option.
        for (TraitKind kind : TraitKind.values()) {
            for (CardTraitBase trait : kind.builtBy(keyword)) {
                attribute(List.of(line), key, trait);
            }
        }
    }

    /**
     * The script of a keyword's one line: the chain of the traits the line
     * is the text of (a chapter's trigger, an ETB replacement, a Visit's or
     * Backup's trigger, an opening-hand effect; a Class level's level-up
     * ability then its granting static), or the
     * keyword's own script text where the line states the keyword itself
     * ({@code Flying}, {@code Ward:2}).
     */
    private TraitScript lineScript(KeywordRendering rendering, Ability line, ProvenanceKey key,
                                   KeywordInterface keyword) {
        List<CardTraitBase> traits = rendering.scriptTraitsOf(line);
        return traits.isEmpty() ? TraitScript.ofKeyword(keyword.getOriginal())
                : TraitScript.ofTraits(traits, key, missingSVars);
    }

    private static boolean rendersSeveralLines(List<Ability> abilities) {
        return abilities.size() > 1;
    }

    /** Claim each built trait's key on its line; whether any trait had one. */
    private boolean attributeBuiltTraits(KeywordRendering rendering, ProvenanceKey key,
                                         KeywordInterface keyword) {
        boolean claimed = false;
        for (TraitKind kind : TraitKind.values()) {
            int index = 0;
            for (CardTraitBase trait : kind.builtBy(keyword)) {
                Ability line = lineOf(rendering, trait);
                ProvenanceKey builtKey = key.withBuilt(kind, index);
                claimScriptOf(rendering, line, builtKey);
                claimed |= attributeBuiltTrait(line, builtKey, trait);
                index++;
            }
        }
        return claimed;
    }

    /**
     * Claim a several-line keyword's {@code line} first with the script of the
     * traits the rendering names for it ({@link KeywordRendering#scriptTraitsOf}),
     * so the line carries the chain of the trait whose text it prints and not
     * that of whichever trait claims it first in index order (haunt's line:
     * the haunting trigger's, not a description-less trigger ending the
     * haunting). The script reports no missing SVar: each trait's own claim
     * reports its own.
     */
    private void claimScriptOf(KeywordRendering rendering, Ability line, ProvenanceKey builtKey) {
        if (line == null || byAbility.containsKey(line)) return;
        List<CardTraitBase> traits = rendering.scriptTraitsOf(line);
        if (!traits.isEmpty()) claim(line, builtKey, TraitScript.ofTraits(traits, builtKey, null));
    }

    /** The line a built trait printed: the keyword's one line, or the one named for it. */
    private static Ability lineOf(KeywordRendering rendering, CardTraitBase trait) {
        List<Ability> lines = rendering.lines();
        return lines.size() == 1 ? lines.get(0) : rendering.lineOf(trait);
    }

    private boolean attributeBuiltTrait(Ability line, ProvenanceKey builtKey, CardTraitBase trait) {
        if (line == null) return false;
        attribute(List.of(line), builtKey, trait);
        return true;
    }

    /**
     * Attribute {@code entry}, the line a keyword-built trait rendered of its
     * own (Tribute's unpaid-tribute trigger), to the trait's keyword key
     * with its {@code built}. A trait that rendered no line of its own has
     * its text on its keyword's lines, which carry its key already.
     */
    public void attributeBuilt(Ability entry, ProvenanceKey keywordKey, CardTraitBase trait) {
        if (entry == null || keywordKey == null) return;
        ProvenanceKey.Built built = ProvenanceResolver.builtOf(trait);
        if (built != null) attribute(entry, keywordKey.withBuilt(built.kind(), built.index()), trait);
    }

    /**
     * Fold a discarded duplicate's attribution into the ability that stays.
     *
     * <p>The converter deduplicates two traits that render one description. The
     * survivor carries the text of both, so it has to carry both keys: dropping
     * the duplicate's key would leave every record fired by that live trait
     * with no line to join to.
     */
    public void merge(Ability survivor, Ability duplicate) {
        // Tested before the removal, so a merge that cannot land leaves the
        // duplicate's attribution where it was rather than discarding it.
        if (survivor == null || duplicate == null) return;
        Source from = byAbility.remove(duplicate);
        if (from == null) return;
        Source into = byAbility.get(survivor);
        if (into == null) {
            byAbility.put(survivor, from);
            return;
        }
        for (ProvenanceKey key : from.keys()) {
            into.addKey(key);
        }
    }

    /**
     * Move an ability's attribution to the object that replaces it.
     *
     * <p>Post-processing rebuilds some abilities as fresh objects — a Class
     * card's level lines, for one — and the recorder keys by identity, so
     * without this the replacement would be textless to the model and the
     * original's key would look dropped.
     */
    public void transfer(Ability from, Ability to) {
        Source source = byAbility.remove(from);
        if (source != null && to != null) byAbility.put(to, source);
    }

    /**
     * Give each mode line its mode under every root key its charm line carries.
     *
     * <p>A charm line can carry several keys: two triggers sharing one charm,
     * which the converter merged, or the triggers a keyword generated. Forge
     * resolves a mode through whichever of them fired, so a mode line keyed
     * under one root alone leaves the others' records with no line. Run once
     * on the face's finished lines, every merge and keyword attribution
     * having landed, which no single attribution site can know; each line is
     * a root before the lines below it, as the file prints them.
     */
    public void claimModesUnderEveryRoot(Iterable<Ability> faceLines) {
        for (Ability ability : faceLines) {
            Source root = byAbility.get(ability);
            if (root != null) claimModesBelow(ability, root.keys());
            claimModesUnderEveryRoot(ability.subAbilities());
        }
    }

    private void claimModesBelow(Ability parent, Set<ProvenanceKey> rootKeys) {
        for (Ability sub : parent.subAbilities()) {
            Source mode = byAbility.get(sub);
            if (sub instanceof OptionAbility option && mode != null) {
                claimModeUnder(mode, option, rootKeys);
            } else if (mode == null) {
                // A line of the charm's own rendering, between root and modes.
                claimModesBelow(sub, rootKeys);
            }
        }
    }

    private static void claimModeUnder(Source mode, OptionAbility option,
                                       Set<ProvenanceKey> rootKeys) {
        for (ProvenanceKey key : rootKeys) {
            if (key.option() == null) mode.addKey(key.withOption(option.modeIndex()));
        }
    }

    /** What produced this ability, or null if it came from no runtime trait. */
    public Source sourceOf(Ability ability) {
        return byAbility.get(ability);
    }

    /** Keys that no surviving ability claims — the deduplicated traits. */
    public List<ProvenanceKey> droppedKeys(Iterable<Ability> surviving) {
        Set<ProvenanceKey> claimed = new LinkedHashSet<>();
        for (Ability ability : surviving) {
            collectClaimed(ability, claimed);
        }
        List<ProvenanceKey> dropped = new ArrayList<>();
        for (ProvenanceKey key : declared) {
            if (!claimed.contains(key)) dropped.add(key);
        }
        return dropped;
    }

    private void collectClaimed(Ability ability, Set<ProvenanceKey> claimed) {
        Source source = byAbility.get(ability);
        if (source != null) claimed.addAll(source.keys());
        for (Ability sub : ability.subAbilities()) {
            collectClaimed(sub, claimed);
        }
    }
}
