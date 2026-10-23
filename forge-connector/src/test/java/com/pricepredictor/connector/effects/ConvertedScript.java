package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.RenderedLine;
import com.pricepredictor.connector.RulesParser;

import java.util.List;

/**
 * A parsed script rendered and given its sidecar, as {@code BatchConverter}
 * writes a converted card and its {@code .provenance.json}.
 */
record ConvertedScript(
        RulesParser.ParsedCard parsed, List<RenderedLine> rendered, ProvenanceSidecar sidecar) {

    /** Render {@code parsed} and build its sidecar under {@code scriptFile}. */
    static ConvertedScript of(RulesParser.ParsedCard parsed, String scriptFile) {
        List<RenderedLine> rendered = parsed.card().renderLines();
        ProvenanceSidecar sidecar = ProvenanceSidecar.build(
                parsed.card().faces().get(0).name(), scriptFile,
                rendered, parsed.recorders());
        return new ConvertedScript(parsed, rendered, sidecar);
    }

    /** The rendered lines' text, in order. */
    List<String> texts() {
        return rendered.stream().map(RenderedLine::text).toList();
    }
}
