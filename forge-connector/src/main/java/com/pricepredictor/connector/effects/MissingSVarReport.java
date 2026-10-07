package com.pricepredictor.connector.effects;

import java.util.ArrayList;
import java.util.List;

/**
 * The SVars a card's script chains name but never define, collected over a
 * conversion run.
 *
 * <p>A chain parameter — {@code SubAbility$ DBFoo} with no {@code SVar:DBFoo}
 * line — is the one script defect Forge tolerates at parse: the accessor comes
 * back null and the card loads with a hole where the sub-ability should be. An
 * undefined {@code Execute$} or {@code ReplaceWith$} fails the whole card
 * instead, and surfaces as a conversion warning. The chain renderer leaves the
 * segment out and notes it here, so the operator sees every hole once, as one
 * block at the end of the run, rather than one line per card lost in Forge's
 * own parse chatter.
 */
public final class MissingSVarReport {

    /** One undefined reference: which card, which line, which parameter named it. */
    public record Entry(String card, ProvenanceKey line, String parameter, String label) {

        /** One report line, readable without the sidecar beside it. */
        public String format() {
            StringBuilder sb = new StringBuilder();
            sb.append(card == null ? "?" : card);
            if (line != null) {
                sb.append(" (").append(line.scriptFile()).append(") face")
                        .append(line.face()).append('/')
                        .append(line.traitKind()).append('[')
                        .append(line.indexWithinKind()).append(']');
                if (line.option() != null) {
                    sb.append("/option").append(line.option());
                }
            }
            return sb.append(": ").append(parameter).append("$ ").append(label).toString();
        }
    }

    private final List<Entry> entries = new ArrayList<>();

    public void add(String card, ProvenanceKey line, String parameter, String label) {
        entries.add(new Entry(card, line, parameter, label));
    }

    public List<Entry> entries() {
        return List.copyOf(entries);
    }

    public boolean isEmpty() {
        return entries.isEmpty();
    }

    /** Take every entry collected so far, leaving the report empty for the next tree. */
    public List<Entry> drain() {
        List<Entry> taken = List.copyOf(entries);
        entries.clear();
        return taken;
    }

    /** The block a converter prints once per run, or null when there is nothing to say. */
    public static String formatBlock(List<Entry> entries) {
        if (entries.isEmpty()) return null;
        StringBuilder sb = new StringBuilder();
        sb.append("Undefined SVars (").append(entries.size())
                .append("): each chain parameter below names an SVar its script"
                        + " does not define; the segment is left out of script_text\n");
        for (Entry entry : entries) {
            sb.append("  ").append(entry.format()).append('\n');
        }
        return sb.toString();
    }
}
