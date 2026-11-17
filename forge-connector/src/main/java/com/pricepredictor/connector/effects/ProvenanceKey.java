package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.json.Json;
import java.util.List;
import java.util.Objects;

/**
 * Printed provenance of one runtime trait: which script file, which face, which
 * kind of trait, and which position within that kind.
 *
 * <p>This is the join every effect record uses to name the ability it is about.
 * Converted-line ordinals cannot serve: the converter runs one per-face bracket
 * counter across five mixed line kinds, emits keyword-derived lines first though
 * Forge appends them last in its runtime lists, and merges, deduplicates and
 * splits abilities relative to the runtime objects. An ordinal names different
 * things on the two sides; printed provenance is stable.
 *
 * <p>{@link ProvenanceResolver} finds the key for a live trait.
 *
 * <p>{@code indexWithinKind} indexes that kind's slice of the face's raw trait
 * list, not a global ordinal.
 *
 * <p>{@code option} names one mode of a modal trait: the mode's 0-based
 * position in the root's {@code Choices$}. Forge gives a mode no identity of
 * its own — a chosen mode is a clone with no back-reference, keyed to the
 * charm it came from — so the position is the only stable name, and a key
 * carrying it is a different key from the root's. It is null on every other
 * trait, and absent from the JSON then, which is what keeps a gen-1 key
 * readable unchanged.
 *
 * <p>{@code built} is set only on a {@link TraitKind#KEYWORD} key, and names
 * which of the traits that keyword instance built the key is about: a saga's
 * chapter triggers, haunt's two triggers, a Class level's statics. A keyword
 * that renders several lines gives each line one of its traits, so the bare
 * keyword key cannot say which line acted. It is null on every other key, and
 * absent from the JSON then.
 */
public record ProvenanceKey(
        String scriptFile,
        int face,
        TraitKind traitKind,
        int indexWithinKind,
        Integer option,
        Built built
) {

    public ProvenanceKey {
        Objects.requireNonNull(scriptFile, "scriptFile");
        Objects.requireNonNull(traitKind, "traitKind");
        if (option != null && option < 0) {
            throw new IllegalArgumentException("option must be a Choices$ position: " + option);
        }
        if (built != null && traitKind != TraitKind.KEYWORD) {
            throw new IllegalArgumentException("only a keyword key has a built trait: " + traitKind);
        }
    }

    /** A root key: the trait itself rather than one of its modes. */
    public ProvenanceKey(String scriptFile, int face, TraitKind traitKind, int indexWithinKind) {
        this(scriptFile, face, traitKind, indexWithinKind, null, null);
    }

    /** A key with a mode, or none, and no built trait. */
    public ProvenanceKey(
            String scriptFile, int face, TraitKind traitKind, int indexWithinKind, Integer option) {
        this(scriptFile, face, traitKind, indexWithinKind, option, null);
    }

    /**
     * The trait a keyword instance built: its kind, and its position in that
     * instance's list of that kind ({@code KeywordInterface.getTriggers()},
     * {@code getReplacements()}, {@code getStaticAbilities()},
     * {@code getAbilities()}). Never itself a keyword.
     */
    public record Built(TraitKind kind, int index) {

        public Built {
            Objects.requireNonNull(kind, "kind");
            if (kind == TraitKind.KEYWORD || index < 0) {
                throw new IllegalArgumentException("a built trait is a non-keyword kind at a position: "
                        + kind + " " + index);
            }
        }

        String toJson() {
            return Json.object().value("kind", kind).value("index", index).render();
        }
    }

    /** This key's mode at {@code option}, the 0-based {@code Choices$} position. */
    public ProvenanceKey withOption(int option) {
        return new ProvenanceKey(scriptFile, face, traitKind, indexWithinKind, option, built);
    }

    /** This keyword key naming the trait of {@code kind} at {@code index} it built. */
    public ProvenanceKey withBuilt(TraitKind kind, int index) {
        return new ProvenanceKey(scriptFile, face, traitKind, indexWithinKind, option, new Built(kind, index));
    }

    /** The keyword's own key for a built key; this key otherwise. */
    public ProvenanceKey withoutBuilt() {
        return built == null ? this : new ProvenanceKey(scriptFile, face, traitKind, indexWithinKind, option, null);
    }

    /**
     * The key for a report a person reads, such as an undefined SVar's:
     * {@code (cardsfolder/a/abrade.txt) face0/spell[0]/option1}, or
     * {@code (cardsfolder/b/binding_the_old_gods.txt) face0/keyword[0]/built trigger[2]}.
     */
    public String describe() {
        StringBuilder sb = new StringBuilder()
                .append('(').append(scriptFile).append(") face").append(face)
                .append('/').append(traitKind.wireName()).append('[').append(indexWithinKind).append(']');
        if (option != null) {
            sb.append("/option").append(option);
        }
        if (built != null) {
            sb.append("/built ").append(built.kind().wireName()).append('[').append(built.index()).append(']');
        }
        return sb.toString();
    }

    /** The root this key names: itself, or the charm a mode key belongs to. */
    ProvenanceKey root() {
        return option == null ? this : new ProvenanceKey(scriptFile, face, traitKind, indexWithinKind, null, built);
    }

    /**
     * A key, or the reason there is none. Exactly one of the two is set.
     *
     * <p>The reason is the whole point of the type. An empty {@code ability}
     * array on a record cannot separate "no printed line exists, correctly" —
     * the Monarch, a dungeon, an engine-built effect card — from "the resolver
     * regressed", and a corpus with many records naming no line could not be
     * asked which it was. The reason is written to the record's
     * {@code ability_unresolved}, which is collection metadata and never a
     * model input.
     */
    public record Resolved(ProvenanceKey key, UnresolvedReason reason) {

        /**
         * No line, for a trait that is none of the kinds a printed line can be,
         * or no trait at all.
         */
        public static final Resolved UNKNOWN = new Resolved(null, UnresolvedReason.UNKNOWN_KIND);

        public Resolved {
            if ((key == null) == (reason == null)) {
                throw new IllegalArgumentException(
                        "exactly one of a key and a reason: " + key + ", " + reason);
            }
        }

        /**
         * The same line's mode at {@code option}, or this when there is no
         * option or no line to give it to: a mode nothing could place keeps
         * the root key, and a reason stays a reason.
         */
        public Resolved withOption(Integer option) {
            return option == null || key == null
                    ? this : new Resolved(key.withOption(option), null);
        }

        /** The acting-line list a record carries: the key, or nothing. */
        public List<ProvenanceKey> keys() {
            return key == null ? List.of() : List.of(key);
        }
    }

    /** Keys as the JSON array a record carries them in. */
    public static String toJsonArray(List<ProvenanceKey> keys) {
        return Json.array(keys, ProvenanceKey::toJson);
    }

    /** The JSON object form used inside a record's {@code ability} array. */
    public String toJson() {
        return withFields(Json.object().value("script_file", scriptFile)).render();
    }

    /** The JSON object form used inside a sidecar, where the file is in the header. */
    public String toSidecarJson() {
        return withFields(Json.object()).render();
    }

    /**
     * The members both forms share, after whatever {@code json} already holds.
     *
     * <p>The {@code option} member is written only on a mode key: the Python
     * reader spells the member the same way and omits it on a root key, so a
     * gen-1 sidecar and a gen-2 root key serialize byte for byte alike. The
     * {@code built} member, likewise, only on a key naming a built trait.
     */
    private Json.ObjectWriter withFields(Json.ObjectWriter json) {
        json.value("face", face)
                .value("trait_kind", traitKind)
                .value("index_within_kind", indexWithinKind);
        if (option != null) {
            json.value("option", option);
        }
        if (built != null) {
            json.raw("built", built.toJson());
        }
        return json;
    }
}
