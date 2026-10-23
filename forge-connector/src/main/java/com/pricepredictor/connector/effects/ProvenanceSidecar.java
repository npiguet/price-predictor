package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.Ability;
import com.pricepredictor.connector.AbilityType;
import com.pricepredictor.connector.MultiCard;
import com.pricepredictor.connector.RenderedLine;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

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

    /**
     * One rendered ability line and everything that produced it.
     *
     * @param proseOffset where the line's prose starts, after its
     *                    {@code kind[n]: } prefix: zero on a continuation line
     *                    of a multi-line description, which has none. Written
     *                    so a reader never re-derives the split from the text,
     *                    which a continuation line holding a {@code ": "} of
     *                    its own would defeat
     */
    public record Line(
            int lineIndex,
            String lineKind,
            int proseOffset,
            List<ProvenanceKey> provenance,
            TraitScript script,
            List<RoleSpans.Span> roleSpans
    ) {
        public String toJson() {
            return Json.object()
                    .value("line_index", lineIndex)
                    .value("line_kind", lineKind)
                    .value("prose_offset", proseOffset)
                    .raw("provenance", Json.array(provenance, ProvenanceKey::toSidecarJson))
                    .raw("sub_ability_links",
                            Json.array(script.subAbilityLinks(), TraitScript.SubAbilityLink::toJson))
                    .value("script_api_type", script.apiType())
                    .raw("script_param_keys", Json.stringArray(script.paramKeys()))
                    .value("script_text", script.scriptText())
                    .raw("role_spans", Json.array(roleSpans, RoleSpans.Span::toJson))
                    .render();
        }
    }

    public String toJson() {
        return Json.object()
                .value("card", card)
                .value("script_file", scriptFile)
                .raw("lines", Json.array(lines, Line::toJson))
                .raw("dropped_keys", Json.array(droppedKeys, ProvenanceKey::toSidecarJson))
                .render();
    }

    /**
     * Build the sidecar for one converted card.
     *
     * @param renderedLines the file's lines, exactly as written, with what
     *                      produced each
     * @param recorders     one recorder per face, in face order
     */
    public static ProvenanceSidecar build(
            String cardName,
            String scriptFile,
            List<RenderedLine> renderedLines,
            List<ProvenanceRecorder> recorders
    ) {
        List<Ability> owners = renderedLines.stream().map(RenderedLine::owner).toList();
        for (ProvenanceRecorder recorder : recorders) {
            recorder.claimModesUnderEveryRoot(owners);
        }
        List<Line> lines = new ArrayList<>();
        // By identity, as the recorders key abilities: the faces of a card
        // render equal abilities (both faces' flying) as distinct objects,
        // and an equality set would keep only the first face's.
        Set<Ability> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        for (int i = 0; i < renderedLines.size(); i++) {
            RenderedLine rendered = renderedLines.get(i);
            Ability owner = rendered.owner();
            if (owner == null) continue;
            seen.add(owner);
            int face = rendered.face();
            ProvenanceRecorder recorder = face >= 0 && face < recorders.size()
                    ? recorders.get(face) : null;
            ProvenanceRecorder.Source source =
                    recorder == null ? null : recorder.sourceOf(owner);
            lines.add(new Line(
                    i,
                    // The prefix the line was rendered with, so line_kind and
                    // the file agree word for word.
                    owner.type().getOutputPrefix(),
                    rendered.prefix().length(),
                    source == null ? List.of() : List.copyOf(source.keys()),
                    source == null ? TraitScript.EMPTY : source.script(),
                    // Over the prose alone: the prefix is structure rather
                    // than card text, and the tokenizer does not see it.
                    RoleSpans.of(rendered.prose(), owner.type() == AbilityType.ACTIVATED)));
        }

        List<ProvenanceKey> dropped = new ArrayList<>();
        for (ProvenanceRecorder recorder : recorders) {
            dropped.addAll(recorder.droppedKeys(seen));
        }
        return new ProvenanceSidecar(cardName, scriptFile, lines, dropped);
    }
}
