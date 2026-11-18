package com.pricepredictor.connector;

import com.pricepredictor.connector.ability.TextAbility;
import com.pricepredictor.connector.effects.FaceOrder;
import com.pricepredictor.connector.effects.MissingSVarReport;
import com.pricepredictor.connector.effects.ProvenanceRecorder;
import forge.card.CardRules;
import forge.card.CardSplitType;
import forge.card.CardStateName;
import forge.card.ICardFace;
import forge.game.card.Card;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Parses Forge card scripts into domain objects (MultiCard/CardFace/Ability).
 * Acts as a router: each face goes to a {@link FaceParser}, which delegates to
 * the variant Ability implementations in the ability sub-package.
 */
public class RulesParser {

    private final CardRules.Reader reader = new CardRules.Reader();
    /**
     * The undefined SVars every recorder of this parser reports into: a run's
     * worth, drained by whoever converts a tree, so the operator sees one block
     * rather than a line per card.
     */
    private final MissingSVarReport missingSVars = new MissingSVarReport();

    /** The undefined-SVar report the recorded faces have filled so far. */
    public MissingSVarReport missingSVars() {
        return missingSVars;
    }

    /** A parsed card together with the per-face provenance recorders. */
    public record ParsedCard(MultiCard card, List<ProvenanceRecorder> recorders) {
    }

    /**
     * Parse a card script, optionally recording provenance.
     *
     * <p>Recording is additive: the rendered text is identical whether or not
     * a script file is supplied, which is what lets the sidecar ship without
     * reconverting a corpus.
     *
     * @param scriptFile the tree-prefixed script path keys should name, or null
     *                   to skip recording
     */
    public ParsedCard parseScript(List<String> scriptLines, String scriptFile) {
        List<ProvenanceRecorder> recorders = new ArrayList<>();
        MultiCard card = parseScriptInternal(scriptLines, recorders, scriptFile);
        return new ParsedCard(card, scriptFile == null ? List.of() : recorders);
    }

    private MultiCard parseScriptInternal(List<String> scriptLines,
                                          List<ProvenanceRecorder> recorders,
                                          String scriptFile) {
        reader.reset();

        ScriptHeader header = ScriptHeader.of(scriptLines);

        for (String line : scriptLines) {
            if (line.isEmpty() || line.charAt(0) == '#') {
                continue;
            }
            reader.parseLine(line);
        }
        CardRules rules = reader.getCard();
        CopiedFaces.supply(rules, scriptLines);
        MultiCard result = parseRules(rules, recorders, scriptFile);

        // Apply oracle fallback: if the primary face produced no ability lines and no
        // non-ability text (e.g. Draft/conspiracy cards whose game-engine abilities are
        // not represented as standard Forge abilities), emit the Oracle text directly.
        // Skip oracle fallback when draft lines are present: draft-only cards (e.g. Cogwork
        // Librarian) produce their content via applyDraftLines(); firing the oracle fallback
        // too would duplicate those lines as both draft: and text: entries. An instant's or
        // sorcery's spell takes its own keyed line from the oracle (FaceParser), so the
        // fallback never fires for one with text.
        if (header.draftLines().isEmpty()) {
            result = applyOracleFallbackIfNeeded(result, header.oracle(), header.name());
        }
        // Prepend Draft: lines (if any) as TEXT abilities on the primary face.
        // The Forge game engine ignores Draft fields, so they never appear in game abilities,
        // but they are part of the card's oracle text and must be included in the output.
        if (!header.draftLines().isEmpty()) {
            result = applyDraftLines(result, header.draftLines(), header.name());
        }
        return result;
    }

    /**
     * Parse CardRules into a MultiCard, optionally recording provenance.
     *
     * @param recorders one recorder per face is appended here, in face order
     * @param scriptFile the tree-prefixed script path keys should name, or
     *                   null to record nothing
     */
    private MultiCard parseRules(CardRules rules, List<ProvenanceRecorder> recorders,
                                String scriptFile) {
        CardSplitType splitType = rules.getSplitType();
        Card card = buildFullCard(rules);

        if (splitType == CardSplitType.None) {
            ICardFace main = rules.getMainPart();
            FaceParser parser = new FaceParser(card, main, nextRecorder(recorders), 0, scriptFile);
            return MultiCard.singleFace(OwnNameScope.within(
                    OwnName.of(main.getName(), main.getName()), parser::parse));
        }

        String layout = splitType.name().toLowerCase(Locale.ROOT);
        Map<CardStateName, ICardFace> parts = printedFaces(rules, card);
        String wholeName = String.join(" // ",
                parts.values().stream().map(ICardFace::getName).toList());
        List<CardFace> faces = new ArrayList<>();
        for (Map.Entry<CardStateName, ICardFace> part : parts.entrySet()) {
            card.setState(part.getKey(), false);
            faces.add(OwnNameScope.within(OwnName.of(part.getValue().getName(), wholeName),
                    new FaceParser(card, part.getValue(), nextRecorder(recorders),
                            faces.size(), scriptFile)::parse));
        }

        return MultiCard.multiFace(layout, faces);
    }

    /**
     * The printed faces of a multi-face card by state, in the order the
     * collectors number faces, read from the same method, because a key's
     * face ordinal is joined by value: CardRules holds the specialize parts in
     * a HashMap keyed by enum constants, whose order changes from one JVM to
     * the next.
     */
    private static Map<CardStateName, ICardFace> printedFaces(CardRules rules, Card card) {
        Map<CardStateName, ICardFace> parts = new LinkedHashMap<>();
        for (CardStateName state : FaceOrder.faceOrder(card)) {
            ICardFace part = FaceOrder.printedFace(rules, state);
            if (part != null) parts.put(state, part);
        }
        return parts;
    }

    /** Append and return a fresh recorder for the next face. */
    private ProvenanceRecorder nextRecorder(List<ProvenanceRecorder> recorders) {
        ProvenanceRecorder recorder = new ProvenanceRecorder(missingSVars);
        recorders.add(recorder);
        return recorder;
    }

    private Card buildFullCard(CardRules rules) {
        return OfflineGame.card(rules);
    }

    /**
     * If the primary face of {@code card} produced no ability lines and no non-ability
     * text, re-emit it with the front-face Oracle text split into {@code TEXT} ability
     * lines.  This handles Draft/conspiracy cards whose game-engine representation in
     * Forge carries no standard spell/trigger/static abilities.
     */
    private static MultiCard applyOracleFallbackIfNeeded(
            MultiCard card, String rawOracle, String cardName) {
        if (rawOracle == null || rawOracle.isEmpty()) return card;
        CardFace first = card.faces().get(0);
        if (!first.abilities().isEmpty() || first.text() != null) {
            return card; // already has content — fallback not needed
        }
        // The script writes the oracle's line breaks as literal \n escapes.
        List<String> oracleLines = List.of(rawOracle.replace("\\n", "\n").split("\n"));
        List<Ability> fallback = textAbilities(oracleLines, cardName, AbilityType.TEXT);
        if (fallback.isEmpty()) return card;

        return card.withFirstFace(first.withAbilities(fallback));
    }

    /**
     * Prepend {@code Draft:} script lines as TEXT abilities on the primary face.
     * Draft instructions are part of the card's oracle text but are invisible to the
     * Forge game engine, so they must be injected from the raw script.
     */
    private static MultiCard applyDraftLines(
            MultiCard card, List<String> draftLines, String cardName) {
        CardFace first = card.faces().get(0);
        List<Ability> draftAbilities = textAbilities(draftLines, cardName, AbilityType.DRAFT);
        if (draftAbilities.isEmpty()) return card;

        List<Ability> combined = new ArrayList<>(draftAbilities);
        combined.addAll(first.abilities());
        return card.withFirstFace(first.withAbilities(combined));
    }

    /**
     * Raw script text lines as {@code type} abilities, written the way every
     * other line of the output is: the card's name as the CARDNAME placeholder,
     * reminder text stripped, cased. A line left empty by the stripping renders
     * nothing.
     */
    private static List<Ability> textAbilities(
            List<String> lines, String cardName, AbilityType type) {
        List<Ability> abilities = new ArrayList<>();
        for (String raw : lines) {
            String line = AbilityDescription.scriptLine(raw, cardName);
            if (line != null) abilities.add(new TextAbility(type, line));
        }
        return abilities;
    }
}
