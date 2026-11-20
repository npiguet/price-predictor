package com.pricepredictor.connector.effects;

import forge.game.CardTraitBase;
import forge.game.card.CardState;
import forge.game.keyword.KeywordInterface;
import forge.util.FileSection;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * The printed line that granted a trait out of one of its card's SVars.
 *
 * <p>An aura's or an equipment's {@code AddTrigger$}, an {@code Animate}'s
 * {@code Triggers$} and an {@code Effect}'s traits are all parsed from an
 * SVar of the donor, and Forge points the result at the donor's
 * {@code CardState} — so the card is known, and only the line is missing.
 * Nothing in the engine records which line named the SVar, but the state
 * still holds both halves, so the lookup Forge did on the way out can be
 * done again on the way back: find the SVar whose text parses to exactly
 * these parameters, then find the line that names that SVar.
 *
 * <p>Both halves refuse when more than one candidate matches
 * ({@link SoleMatch}), unless the naming lines all key to one line
 * ({@link #oneLineOf}). The parameter comparison is equality between two
 * runs of Forge's own {@code FileSection} parse, not a resemblance test, so
 * a match is the SVar and not something like it.
 */
final class GrantingLineSearch {

    /** How a line separates the SVar names it grants: spaces, commas, ampersands. */
    private static final Pattern SVAR_LIST = Pattern.compile("[\\s,&]+");

    private GrantingLineSearch() {
    }

    /**
     * The key of the line that granted {@code trait}, or null; {@code depth}
     * granting lines into a search already (see
     * {@link ProvenanceResolver#keyFor(CardTraitBase, int)}).
     */
    static ProvenanceKey keyOf(CardTraitBase trait, int depth) {
        CardState state = trait.getCardState();
        if (state == null) return null;
        Map<String, String> want = trait.getOriginalMapParams();
        if (want == null || want.isEmpty()) return null;
        String svar = soleSVarParsingTo(state, want);
        // The SVar may itself be named only by another SVar: a Clone's
        // AddAbilities$ names the granted ability's SVar from the copy
        // ability's own SVar, which the replacement line names (Gigantoplasm,
        // FR-039). Climbed one SVar at a time, each step refusing two
        // candidates, until a line names one.
        Set<String> climbed = new HashSet<>();
        for (int hop = 0; svar != null && hop < ProvenanceResolver.MAX_HOPS && climbed.add(svar); hop++) {
            List<CardTraitBase> naming = linesNaming(state, svar);
            if (!naming.isEmpty()) return oneLineOf(naming, depth);
            svar = soleSVarNaming(state, svar);
        }
        return null;
    }

    /**
     * The key of the one line every trait in {@code naming} keys to: their
     * key when they all key alike; the bare keyword key when they differ only
     * in {@code built}; else null, refusing as {@link SoleMatch} refuses.
     *
     * <p>A keyword names its SVars in its own text, so each trait it built
     * names them: a Class level's static, which grants the level's lines,
     * and its level-up ability both name each SVar the level grants. They
     * are one line, the {@code Class:N} keyword's, which declares the bare
     * key; a granted line is none of the traits the keyword built, so no
     * {@code built} names it.
     */
    private static ProvenanceKey oneLineOf(List<CardTraitBase> naming, int depth) {
        Set<ProvenanceKey> keys = new HashSet<>();
        Set<ProvenanceKey> bare = new HashSet<>();
        for (CardTraitBase line : naming) {
            ProvenanceKey key = ProvenanceResolver.keyFor(line, depth + 1);
            keys.add(key);
            bare.add(key == null ? null : key.withoutBuilt());
        }
        if (keys.size() == 1) return keys.iterator().next();
        return bare.size() == 1 ? bare.iterator().next() : null;
    }

    /** The one other SVar of {@code state} whose text names {@code svar}, or null. */
    private static String soleSVarNaming(CardState state, String svar) {
        Map.Entry<String, String> hit = SoleMatch.of(state.getSVars().entrySet(), other ->
                !other.getKey().equals(svar) && textNames(other.getValue(), svar));
        return hit == null ? null : hit.getKey();
    }

    /** The one SVar of {@code state} whose text parses to {@code want}, or null. */
    private static String soleSVarParsingTo(CardState state, Map<String, String> want) {
        Map.Entry<String, String> hit = SoleMatch.of(state.getSVars().entrySet(),
                svar -> parsesTo(svar.getValue(), want));
        return hit == null ? null : hit.getKey();
    }

    /** Whether an SVar's text is a script line whose parameters are {@code want}. */
    private static boolean parsesTo(String text, Map<String, String> want) {
        return text != null && text.indexOf('$') >= 0 && want.equals(
                FileSection.parseToMap(text, FileSection.DOLLAR_SIGN_KV_SEPARATOR));
    }

    /** The printed lines of {@code state} that name {@code svar}. */
    private static List<CardTraitBase> linesNaming(CardState state, String svar) {
        List<CardTraitBase> naming = new ArrayList<>();
        for (TraitKind kind : TraitKind.values()) {
            kind.slice(state).forEach(line -> {
                if (namesSVar(line, svar)) naming.add(line);
            });
        }
        return naming;
    }

    /**
     * Whether a line names an SVar, as a whole word.
     *
     * <p>{@code AddTrigger$ TrigA & TrigB} and {@code Triggers$ TrigA} both
     * name {@code TrigA}; a substring test would also match {@code TrigAttack}.
     */
    private static boolean namesSVar(CardTraitBase trait, String svar) {
        // A keyword-built line names its SVars in the keyword's own text,
        // colon-separated: ETBReplacement:Copy:DBCopy:Optional.
        KeywordInterface keyword = trait.getKeyword();
        if (keyword != null && textNames(keyword.getOriginal().replace(':', ' '), svar)) return true;
        Map<String, String> params = trait.getOriginalMapParams();
        if (params == null) return false;
        for (String value : params.values()) {
            if (textNames(value, svar)) return true;
        }
        return false;
    }

    /** Whether a parameter value or an SVar's text names {@code svar}, as a whole word. */
    private static boolean textNames(String text, String svar) {
        if (text == null) return false;
        for (String token : SVAR_LIST.split(text)) {
            if (token.equals(svar)) return true;
        }
        return false;
    }
}
