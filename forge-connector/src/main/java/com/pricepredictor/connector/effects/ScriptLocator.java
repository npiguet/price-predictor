package com.pricepredictor.connector.effects;

import forge.ImageKeys;
import forge.StaticData;
import forge.card.CardRules;
import forge.game.card.Card;
import forge.item.IPaperCard;
import forge.item.PaperToken;

import java.util.Optional;

/**
 * The converted script path a live card's traits belong to: the
 * {@code script_file} of every provenance key the collectors write.
 *
 * <p>The path names a tree as well as a file — {@code cardsfolder},
 * {@code tokenscripts} or {@code variant-scripts} — and the Python side looks
 * for the sidecar under exactly that path, so a card filed in the wrong tree
 * joins to nothing.
 */
final class ScriptLocator {

    private ScriptLocator() {
    }

    /**
     * The converted script path a live card's traits belong to.
     *
     * <p>A staged variant is checked first: once loaded it is an ordinary
     * non-token card, and only {@link VariantRegistry} knows it belongs under
     * {@code variant-scripts}.
     *
     * <p>Otherwise Forge is asked first and the name is only a fallback. The
     * strongest answer is {@code CardRules.getPath()}, the file the card reader
     * actually opened: the converter keys each sidecar by that same path
     * relativised against the same root, so taking it here is not a rule that
     * agrees with the tree but the tree's own record, and the two cannot drift.
     * It is also the only answer that is right for the 86 cards the tree does
     * not file under their initial — {@code +2 Mace} is
     * {@code cardsfolder/p/+2_mace.txt} and no derivation reaches that.
     *
     * <p>Failing that, {@code CardRules.getNormalizedName()} is the stem of the
     * file Forge read the card from, and that still beats any sanitizer:
     * Forge's filenames disagree with its card names often enough to matter — a
     * Fallaji Archaeologist lives in {@code fallaji_archeologist.txt} — and a
     * misspelling cannot be derived. It also settles the two-faced cards for
     * free, where {@code getName()} answers whichever face is up while the
     * script is {@code westvale_abbey_ormendahl_profane_prince.txt} either way.
     *
     * <p>A token is filed under what it <em>is</em>, not what it is called:
     * {@code c_1_1_eldrazi_scion_sac.txt} is the Eldrazi Scion. The token tree
     * is tried ahead of the card tree: a {@link PaperToken}'s image
     * filename, or failing that a token host's image key ({@code t:<stem>...},
     * which {@code TokenInfo.toCard} preserves through a {@code GameCopier}
     * fork), checked against the token database. Only when neither yields a
     * known stem does the card tree take over — {@code CardRules.getPath()},
     * then the normalized name, then the printed name. A {@code t:} key that
     * names a stem the token database does not know falls through the same
     * way a token copy of a printed card does, landing on a name-derived
     * {@code cardsfolder} path rather than an unresolved result; that is a
     * known limitation (see the token-access audit note).
     */
    static String scriptFileOf(Card host) {
        return variantScriptOf(host)
                .or(() -> tokenScriptStem(host).map(
                        stem -> CardFilenames.scriptFileForStem(SourceTree.TOKENSCRIPTS, stem)))
                .or(() -> readScriptOf(host.getRules()))
                .or(() -> normalizedStemOf(host.getRules()).map(
                        stem -> CardFilenames.scriptFileForStem(SourceTree.CARDSFOLDER, stem)))
                .orElseGet(() -> CardFilenames.scriptFile(SourceTree.CARDSFOLDER, host.getName()))
                .key();
    }

    /** A staged variant's script, named from its card name: nothing else knows it. */
    private static Optional<ScriptFile> variantScriptOf(Card host) {
        return VariantRegistry.isVariant(host.getName())
                ? Optional.of(CardFilenames.scriptFile(SourceTree.VARIANT_SCRIPTS, host.getName()))
                : Optional.empty();
    }

    /** The card-tree file the card reader actually opened, when it says so. */
    private static Optional<ScriptFile> readScriptOf(CardRules rules) {
        return Optional.ofNullable(rules)
                .flatMap(r -> SourceTree.relativePathIn(SourceTree.CARDSFOLDER, r.getPath()))
                .map(relative -> new ScriptFile(SourceTree.CARDSFOLDER, relative));
    }

    /** The stem of the file Forge read the card from, when it recorded one. */
    private static Optional<String> normalizedStemOf(CardRules rules) {
        return Optional.ofNullable(rules)
                .map(CardRules::getNormalizedName)
                .filter(normalized -> !normalized.isEmpty());
    }

    /**
     * {@link #tokenScriptStem}, or null — for a caller that reports a token's
     * script id and falls back to its printed name.
     */
    static String tokenScriptStemOf(Card host) {
        return tokenScriptStem(host).orElse(null);
    }

    /**
     * A token's script-file stem, or empty when the card did not come from a
     * token script — including a token copy of a printed card.
     *
     * <p>Two sources, tried in order. A {@link PaperToken} is the token as
     * Forge read it from {@code tokenscripts}, and its image filename starts
     * with the script stem. A token that was rebuilt by {@code TokenInfo} —
     * every token in a {@code GameCopier} fork — has no paper card at all, but
     * {@code TokenInfo.toCard} copies the image key, and a token image key is
     * {@code t:<stem>[|edition|…]}. The stem read either way is checked
     * against the token database, so a key that names no script (a token copy
     * of a printed card carries the printed card's image key, which is not a
     * token key) yields nothing rather than a fabricated path.
     */
    static Optional<String> tokenScriptStem(Card host) {
        try {
            IPaperCard paper = host.getPaperCard();
            String image = null;
            if (paper instanceof PaperToken token) {
                image = token.getImageFilename(1);
            } else if (host.isToken()) {
                image = ImageKeys.getTokenImageName(host.getImageKey());
            }
            if (image == null || image.isEmpty()) {
                return Optional.empty();
            }
            int bar = image.indexOf('|');
            String stem = (bar < 0 ? image : image.substring(0, bar)).replace(' ', '_');
            if (stem.isEmpty() || !StaticData.instance().getAllTokens().containsRule(stem)) {
                return Optional.empty();
            }
            return Optional.of(stem);
        } catch (RuntimeException e) {
            return Optional.empty();
        }
    }
}
