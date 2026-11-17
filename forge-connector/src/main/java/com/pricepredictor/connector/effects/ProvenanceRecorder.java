package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.Ability;
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
 * end up in the sidecar's {@code dropped_keys}, which holds only traits with no
 * text of their own — a permanent's implicit cast spell, say, or a charm whose
 * modes are its only lines. A trait the converter deduplicated, merged or
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
    /**
     * The one line a keyword rendered for one of the traits it generates,
     * where the keyword renders a line per trait (a saga's chapters, haunt's
     * two halves): that line alone declares the trait's {@code built} key.
     */
    private final Map<CardTraitBase, Ability> renderedFor = new IdentityHashMap<>();
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
            for (Ability sub : ability.subAbilities()) {
                if (sub instanceof OptionAbility option) {
                    attributeMode(option, key, trait);
                }
            }
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
     * — several abilities, or one whose text runs onto continuation lines —
     * declares {@code key} on none: each trait it built is claimed under
     * {@code key} with its {@code built}, by the line {@link #rendered} named
     * for it, or by the keyword's one ability. A trait with no such line is
     * declared nowhere, and a record naming it falls back to the bare key.
     * Where no built trait has a line, the keyword's first line declares the
     * bare key, as if the keyword printed that line alone; a keyword
     * rendering no line has its bare key in {@code dropped_keys}.
     */
    public void attributeKeyword(List<Ability> abilities, ProvenanceKey key,
                                 KeywordInterface keyword) {
        if (key == null) return;
        if (rendersSeveralLines(abilities) && attributeBuiltTraits(abilities, key, keyword)) return;
        declare(key);
        if (abilities.isEmpty()) return;
        Ability line = abilities.get(0);
        claim(line, key, TraitScript.ofKeyword(keyword.getOriginal()));
        // Each built trait again under the bare key: the line holds that key
        // already, and a charm's mode lines (an Attraction's visit) take it
        // with the mode's option.
        for (TraitKind kind : TraitKind.values()) {
            for (CardTraitBase trait : kind.builtBy(keyword)) {
                attribute(List.of(line), key, trait);
            }
        }
    }

    private static boolean rendersSeveralLines(List<Ability> abilities) {
        return abilities.size() > 1 || (abilities.size() == 1 && spansLines(abilities.get(0)));
    }

    private static boolean spansLines(Ability ability) {
        return String.valueOf(ability.descriptionText()).contains("\n");
    }

    /** Claim each built trait's key on its line; whether any trait had one. */
    private boolean attributeBuiltTraits(List<Ability> abilities, ProvenanceKey key,
                                         KeywordInterface keyword) {
        boolean claimed = false;
        for (TraitKind kind : TraitKind.values()) {
            int index = 0;
            for (CardTraitBase trait : kind.builtBy(keyword)) {
                Ability line = abilities.size() == 1 ? abilities.get(0) : renderedFor.get(trait);
                if (line != null) {
                    attribute(List.of(line), key.withBuilt(kind, index), trait);
                    claimed = true;
                }
                index++;
            }
        }
        return claimed;
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
     * Note that a keyword rendered {@code line} for {@code trait}, one of the
     * traits it generates.
     *
     * <p>A keyword rendering several lines, one per trait or group of traits,
     * says so here while it renders: without it, every trait the keyword
     * generated would be claimed by every one of its lines, and the join,
     * which reads the first line holding a key, would give a saga's chapter II
     * record chapter I's text.
     */
    public void rendered(CardTraitBase trait, Ability line) {
        if (trait != null && line != null) renderedFor.put(trait, line);
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
     * every merge and keyword attribution has landed, which no single
     * attribution site can know.
     */
    public void claimModesUnderEveryRoot(Iterable<Ability> surviving) {
        for (Ability ability : surviving) {
            Source root = ability == null ? null : byAbility.get(ability);
            if (root != null) claimModesBelow(ability, root.keys());
        }
    }

    private void claimModesBelow(Ability parent, Set<ProvenanceKey> rootKeys) {
        for (Ability sub : parent.subAbilities()) {
            Source mode = byAbility.get(sub);
            if (sub instanceof OptionAbility option && mode != null) {
                for (ProvenanceKey key : rootKeys) {
                    if (key.option() != null) continue;
                    mode.addKey(key.withOption(option.modeIndex()));
                }
            } else if (mode == null) {
                // A line of the charm's own rendering, between root and modes.
                claimModesBelow(sub, rootKeys);
            }
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
