package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityType;
import com.pricepredictor.connector.MultiCard;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.StringJoiner;

/**
 * The {@code <name>.provenance.json} written beside each converted card.
 *
 * <p>It is the join between runtime Forge trait objects and converted ability
 * lines: every effect record names an ability through it, and every ability
 * cache row is aligned by it. Writing it never alters the converted text —
 * {@link MultiCard#renderLines} produces both from one pass, so the line
 * indices always describe the file that was actually written.
 */
public record ProvenanceSidecar(
        String card,
        String scriptFile,
        List<Line> lines,
        List<ProvenanceKey> droppedKeys
) {

    /** One rendered ability line and everything that produced it. */
    public record Line(
            int lineIndex,
            String lineKind,
            List<ProvenanceKey> provenance,
            TraitScript script,
            List<RoleSpans.Span> roleSpans
    ) {
        public String toJson() {
            StringJoiner keys = new StringJoiner(",", "[", "]");
            for (ProvenanceKey key : provenance) keys.add(key.toSidecarJson());
            StringJoiner links = new StringJoiner(",", "[", "]");
            for (TraitScript.SubAbilityLink link : script.subAbilityLinks()) {
                links.add(link.toJson());
            }
            StringJoiner spans = new StringJoiner(",", "[", "]");
            for (RoleSpans.Span span : roleSpans) spans.add(span.toJson());
            return "{\"line_index\":" + lineIndex
                    + ",\"line_kind\":" + Json.string(lineKind)
                    + ",\"provenance\":" + keys
                    + ",\"sub_ability_links\":" + links
                    + ",\"script_api_type\":" + Json.string(script.apiType())
                    + ",\"script_param_keys\":" + Json.stringArray(script.paramKeys())
                    + ",\"script_text\":" + Json.string(script.scriptText())
                    + ",\"role_spans\":" + spans + "}";
        }
    }

    public String toJson() {
        StringJoiner renderedLines = new StringJoiner(",", "[", "]");
        for (Line line : lines) renderedLines.add(line.toJson());
        StringJoiner dropped = new StringJoiner(",", "[", "]");
        for (ProvenanceKey key : droppedKeys) dropped.add(key.toSidecarJson());
        return "{\"card\":" + Json.string(card)
                + ",\"script_file\":" + Json.string(scriptFile)
                + ",\"lines\":" + renderedLines
                + ",\"dropped_keys\":" + dropped + "}";
    }

    /**
     * Build the sidecar for one converted card.
     *
     * @param renderedLines the file's lines, exactly as written
     * @param owners        the ability that produced each line, null for headers
     * @param recorders     one recorder per face, in face order
     */
    public static ProvenanceSidecar build(
            String cardName,
            String scriptFile,
            MultiCard card,
            List<String> renderedLines,
            List<Ability> owners,
            List<Integer> faceOfLine,
            List<ProvenanceRecorder> recorders
    ) {
        List<Line> lines = new ArrayList<>();
        Set<Ability> seen = new LinkedHashSet<>();
        for (int i = 0; i < renderedLines.size(); i++) {
            Ability owner = owners.get(i);
            if (owner == null) continue;
            seen.add(owner);
            int face = faceOfLine.get(i);
            ProvenanceRecorder recorder = face >= 0 && face < recorders.size()
                    ? recorders.get(face) : null;
            ProvenanceRecorder.Source source =
                    recorder == null ? null : recorder.sourceOf(owner);
            lines.add(new Line(
                    i,
                    // The prefix the line was rendered with, so line_kind and
                    // the file agree word for word.
                    owner.type().getOutputPrefix(),
                    source == null ? List.of() : List.copyOf(source.keys()),
                    source == null ? TraitScript.of(null) : source.script(),
                    RoleSpans.of(
                            prose(renderedLines.get(i)),
                            owner.type() == AbilityType.ACTIVATED)));
        }

        List<ProvenanceKey> dropped = new ArrayList<>();
        for (ProvenanceRecorder recorder : recorders) {
            dropped.addAll(recorder.droppedKeys(seen));
        }
        return new ProvenanceSidecar(cardName, scriptFile, lines, dropped);
    }

    /**
     * The prose of a rendered line, without its {@code kind[n]: } prefix.
     *
     * <p>Role-span offsets are relative to this, because the prefix is
     * structure rather than card text and the tokenizer does not see it. The
     * prefix is a bare word plus an optional bracketed number, so the first
     * {@code ": "} in the line always ends it — an activated ability's own cost
     * colon comes later.
     */
    static String prose(String line) {
        int separator = line.indexOf(": ");
        return separator < 0 ? line : line.substring(separator + 2);
    }
}
