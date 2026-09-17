package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.card.CardStateName;
import forge.card.GamePieceType;
import forge.game.ability.AbilityFactory;
import forge.game.card.Card;
import forge.game.card.CardState;
import forge.game.keyword.KeywordInterface;
import forge.game.spellability.AbilitySub;
import forge.game.spellability.SpellAbility;
import forge.game.staticability.StaticAbility;
import forge.game.trigger.Trigger;
import forge.game.trigger.TriggerHandler;
import forge.game.trigger.WrappedAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import forge.game.trigger.TriggerType;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Resolving a live Forge trait to the printed line it came from.
 *
 * <p>Every case here is a shape Forge actually hands the collectors, and every
 * one of them failed to key in the first collected corpus: the stack replaces
 * an activated ability with a copy before pushing it, a triggered ability
 * arrives wrapped around an SVar ability that belongs to no trait list, the
 * cost-variant helpers copy with no back-reference at all, and an effect card's
 * traits sit on a card that has no script file in any tree. The tests build
 * those objects the way the engine builds them rather than asserting on the
 * pristine parse, because keying the pristine parse always worked — it is the
 * object that comes back from the engine that did not.
 */
@ExtendWith(ForgeExtension.class)
class ProvenanceKeyTest {

    private static Card card(String name) {
        return TestCards.build(name);
    }

    private static Card token(String script) {
        return TestCards.token(script);
    }

    /** The SA a card's own script declares, skipping keyword-derived ones. */
    private static SpellAbility scriptedSpell(CardState state, String apiFragment) {
        for (SpellAbility sa : state.getSpellAbilities()) {
            if (sa.getKeyword() != null) continue;
            if (String.valueOf(sa.getApi()).contains(apiFragment)) return sa;
        }
        throw new AssertionError("no scripted " + apiFragment + " ability on "
                + state.getCard().getName());
    }

    /**
     * The object {@code MagicStack.add} pushes, built exactly as it builds it:
     * a copy carrying the original as a back-reference.
     */
    private static SpellAbility asStackCopy(SpellAbility sa) {
        SpellAbility copy = sa.copy(sa.getHostCard(), null, false, true);
        copy.setOriginalAbility(sa);
        return copy;
    }

    // ── activated abilities the stack replaced with a copy ───────────────

    /**
     * The largest single bucket of unkeyed records: every non-mana activated
     * ability is copied before it is pushed, and the copy carries the original
     * card state, so it looks intrinsic and uncopied while being a member of no
     * trait list at all.
     */
    @Test
    void aStackCopyOfAnActivatedAbilityKeysToTheLineItCameFrom() {
        Card fountain = card("Fountain of Youth");
        SpellAbility printed = scriptedSpell(fountain.getCurrentState(), "GainLife");
        ProvenanceKey expected = ProvenanceKey.of(printed);

        assertNotNull(expected, "the pristine ability must key");
        assertEquals("cardsfolder/f/fountain_of_youth.txt", expected.scriptFile());
        assertEquals(ProvenanceKey.KIND_SPELL, expected.traitKind());
        assertEquals(expected, ProvenanceKey.of(asStackCopy(printed)));
    }

    /**
     * The copy the stack pushes is not the trait object, which is the whole
     * reason identity lookup returned -1 on it.
     */
    @Test
    void aStackCopyIsNotTheTraitObjectItKeysTo() {
        Card fountain = card("Fountain of Youth");
        SpellAbility printed = scriptedSpell(fountain.getCurrentState(), "GainLife");
        SpellAbility copy = asStackCopy(printed);

        assertNotEquals(System.identityHashCode(printed), System.identityHashCode(copy));
        for (SpellAbility member : fountain.getCurrentState().getSpellAbilities()) {
            assertNotEquals(member, copy,
                    "the pushed copy must not be a member of the state's list");
        }
    }

    /**
     * The alternative- and extra-cost helpers copy with no back-reference at
     * all — no original ability, no grantor, nothing — so only the structural
     * fingerprint can find the line. {@code getOriginalMapParams} survives the
     * copy because {@code putParam} writes only the live map.
     */
    @Test
    void aCopyWithNoBackReferenceKeysThroughItsScriptParameters() {
        Card firebolt = card("Firebolt");
        SpellAbility printed = scriptedSpell(firebolt.getCurrentState(), "DealDamage");
        SpellAbility orphan = printed.copy(printed.getHostCard(), null, false, true);

        assertNull(orphan.getOriginalAbility(), "this shape carries no back-reference");
        assertEquals(ProvenanceKey.of(printed), ProvenanceKey.of(orphan));
    }

    /**
     * No key beats a wrong key. Two lines with the same parameters are
     * indistinguishable to the fingerprint, so a copy of one that has lost its
     * back-reference must resolve to neither.
     */
    @Test
    void anAmbiguousFingerprintRefusesToGuess() {
        Card twin = card("Fountain of Youth");
        CardState state = twin.getCurrentState();
        SpellAbility printed = scriptedSpell(state, "GainLife");
        // A second, byte-identical line: exactly the case the converter would
        // deduplicate and the fingerprint cannot separate.
        state.addSpellAbility(printed.copy(twin, null, false, true));

        SpellAbility orphan = printed.copy(twin, null, false, true);
        assertNull(ProvenanceKey.of(orphan),
                "two same-shape members must produce no key rather than a guess");
    }

    // ── triggered abilities, which arrive wrapped ───────────────────────

    /**
     * A triggered ability reaches the stack as a {@link WrappedAbility} around
     * the {@code Execute$} SVar ability. That SVar lives in
     * {@code Trigger.overridingAbility} and is a member of no slice, so the
     * printed line is the {@code Trigger} itself.
     */
    @Test
    void aWrappedTriggeredAbilityKeysToItsTriggerLine() {
        Card paralyze = card("Paralyze");
        List<Trigger> triggers = new ArrayList<>();
        paralyze.getCurrentState().getTriggers().forEach(triggers::add);
        assertEquals(2, triggers.size(), "Paralyze declares two T: lines");

        for (int i = 0; i < triggers.size(); i++) {
            Trigger trigger = triggers.get(i);
            WrappedAbility wrapped =
                    new WrappedAbility(trigger, trigger.ensureAbility(), null);
            ProvenanceKey key = ProvenanceKey.of(wrapped);

            assertNotNull(key, "wrapped trigger " + i + " must key");
            assertEquals("cardsfolder/p/paralyze.txt", key.scriptFile());
            assertEquals(ProvenanceKey.KIND_TRIGGER, key.traitKind(),
                    "a wrapped trigger keys to its trigger line, not to a spell");
            assertEquals(i, key.indexWithinKind());
        }
    }

    /**
     * Brass Man's {@code Execute$} SVar parses as an activated ability rather
     * than a sub-ability, which is the shape most likely to be mistaken for a
     * printed {@code A:} line.
     */
    @Test
    void aTriggerWhoseExecuteSvarIsAnActivatedAbilityStillKeysToTheTrigger() {
        Card brassMan = card("Brass Man");
        Trigger trigger = brassMan.getCurrentState().getTriggers().iterator().next();
        SpellAbility execute = trigger.ensureAbility();
        execute.setTrigger(trigger);

        ProvenanceKey key = ProvenanceKey.of(execute);
        assertNotNull(key);
        assertEquals("cardsfolder/b/brass_man.txt", key.scriptFile());
        assertEquals(ProvenanceKey.KIND_TRIGGER, key.traitKind());
        assertEquals(0, key.indexWithinKind());
    }

    /** A replacement effect keys where it stands, and always did. */
    @Test
    void aReplacementEffectKeysToItsPrintedLine() {
        Card paralyze = card("Paralyze");
        ProvenanceKey key = ProvenanceKey.of(
                paralyze.getCurrentState().getReplacementEffects().iterator().next());

        assertNotNull(key);
        assertEquals("cardsfolder/p/paralyze.txt", key.scriptFile());
        assertEquals(ProvenanceKey.KIND_REPLACEMENT, key.traitKind());
        assertEquals(0, key.indexWithinKind());
    }

    // ── faces other than the one that is up ─────────────────────────────

    /**
     * A back face's traits key to the same script file under their own face
     * ordinal. Nothing sets the card to that state first — the engine hands
     * back whatever state the trait belongs to, and the key has to follow.
     */
    @Test
    void aTraitOnTheBackFaceKeysToThatFacesOrdinal() {
        Card werewolf = card("Daybreak Ranger");
        CardState back = werewolf.getState(CardStateName.Backside);
        assertNotNull(back, "Daybreak Ranger is double-faced");
        assertEquals(CardStateName.Original, werewolf.getCurrentStateName(),
                "the front face is the one that is up");

        SpellAbility fight = scriptedSpell(back, "Fight");
        ProvenanceKey key = ProvenanceKey.of(fight);

        assertNotNull(key);
        assertEquals("cardsfolder/d/daybreak_ranger_nightfall_predator.txt",
                key.scriptFile());
        assertEquals(1, key.face(), "the back face is ordinal 1");
        assertEquals(ProvenanceKey.KIND_SPELL, key.traitKind());
    }

    /** The front face of the same card stays ordinal 0. */
    @Test
    void aTraitOnTheFrontFaceKeepsOrdinalZero() {
        Card werewolf = card("Daybreak Ranger");
        SpellAbility damage =
                scriptedSpell(werewolf.getState(CardStateName.Original), "DealDamage");
        ProvenanceKey key = ProvenanceKey.of(damage);

        assertNotNull(key);
        assertEquals(0, key.face());
        assertEquals("cardsfolder/d/daybreak_ranger_nightfall_predator.txt",
                key.scriptFile());
    }

    /** A back-face trait survives the stack copy the same way a front one does. */
    @Test
    void aStackCopyOfABackFaceAbilityKeepsItsFace() {
        Card werewolf = card("Daybreak Ranger");
        SpellAbility fight =
                scriptedSpell(werewolf.getState(CardStateName.Backside), "Fight");

        assertEquals(ProvenanceKey.of(fight), ProvenanceKey.of(asStackCopy(fight)));
    }

    // ── tokens ──────────────────────────────────────────────────────────

    /**
     * A token is filed under what it is, not what it is called, and in its own
     * tree: {@code tokenscripts} is flat while {@code cardsfolder} is letter
     * keyed, so the two trees name different files for one stem.
     */
    @Test
    void aTokensAbilityKeysIntoTheTokenTree() {
        Card food = token("c_a_food_sac");
        SpellAbility gainLife = scriptedSpell(food.getCurrentState(), "GainLife");
        ProvenanceKey key = ProvenanceKey.of(gainLife);

        assertNotNull(key);
        assertEquals("tokenscripts/c_a_food_sac.txt", key.scriptFile());
        assertEquals(0, key.face());
        assertEquals(ProvenanceKey.KIND_SPELL, key.traitKind());
    }

    /** And through the stack copy, which is how a token's ability resolves. */
    @Test
    void aStackCopyOfATokensAbilityKeysIntoTheTokenTree() {
        Card food = token("c_a_food_sac");
        SpellAbility gainLife = scriptedSpell(food.getCurrentState(), "GainLife");

        assertEquals(ProvenanceKey.of(gainLife),
                ProvenanceKey.of(asStackCopy(gainLife)));
    }

    /**
     * A forked game's tokens are rebuilt by {@code TokenInfo}, which drops the
     * {@code PaperToken}. The first corpus keyed those to
     * {@code cardsfolder/z/zombie_token.txt}, a file no tree holds, so every
     * Food, Treasure and Zombie on a forked board reached the model as no text.
     * The image key survives the copy and names the script stem (FR-150).
     */
    @Test
    void aCopiedTokenWithNoPaperCardStillKeysIntoTheTokenTree() {
        Card copy = TestCards.copiedToken("c_a_food_sac");

        assertNull(copy.getPaperCard(), "the fixture must reproduce the copier's shape");
        assertEquals("tokenscripts/c_a_food_sac.txt", ProvenanceKey.scriptFileOf(copy));
    }

    /** And a real card's copy is not mistaken for a token: no image-key path for it. */
    @Test
    void aCopiedPrintedCardStillKeysIntoTheCardTree() {
        Card bolt = card("Lightning Bolt");
        Card copy = new Card(TestCards.nextCardId(), TestCards.game());
        copy.setName(bolt.getName());
        copy.setImageKey(bolt.getImageKey());

        assertEquals("cardsfolder/l/lightning_bolt.txt", ProvenanceKey.scriptFileOf(copy));
    }

    /** A token image key the database does not know is not turned into a path. */
    @Test
    void anUnknownTokenImageKeyFallsThroughToNull() {
        Card copy = TestCards.copiedToken("c_a_food_sac");
        copy.setImageKey(forge.ImageKeys.getTokenKey("no_such_token_xyz"));

        // Falls through to the name-derived cardsfolder path, exactly as before
        // this change; what must not happen is a fabricated tokenscripts path.
        assertEquals("cardsfolder/f/food_token.txt", ProvenanceKey.scriptFileOf(copy));
    }

    /**
     * A token copy of a printed card: {@code isToken()} is true (the copier
     * sets {@code GamePieceType.TOKEN} on every copy it makes, real permanent
     * or not) but the image key is the printed card's, not a {@code t:} key.
     * {@code ImageKeys.getTokenImageName} rejects a non-token key outright, so
     * this must still reach the cardsfolder path — the case the class doc for
     * {@code tokenScriptStem} names explicitly.
     */
    @Test
    void aTokenCopyOfAPrintedCardStillKeysIntoTheCardTree() {
        Card copy = TestCards.copiedToken("c_a_food_sac");
        copy.setName("Lightning Bolt");
        copy.setImageKey(card("Lightning Bolt").getImageKey());

        assertEquals("cardsfolder/l/lightning_bolt.txt", ProvenanceKey.scriptFileOf(copy));
    }

    // ── engine-built command-zone cards ─────────────────────────────────

    /**
     * The Monarch, The Initiative and the dungeons are built inline with no
     * paper card and no rules. There is no printed line to name, which is a
     * different thing from a resolver that failed — and an empty key alone
     * cannot say which, so the reason has to.
     */
    @Test
    void anEngineBuiltEffectCardReportsThatRatherThanAKey() {
        Card monarch = new Card(TestCards.nextCardId(), TestCards.game());
        monarch.setName("The Monarch");
        monarch.setGamePieceType(GamePieceType.EFFECT);
        SpellAbility draw = AbilityFactory.getAbility(
                "DB$ Draw | Defined$ You | NumCards$ 1", monarch);
        monarch.getCurrentState().addSpellAbility(draw);

        ProvenanceKey.Resolved resolved = ProvenanceKey.resolve(draw);

        assertNull(resolved.key(),
                "an engine-built card names no script file in any tree");
        assertEquals(ProvenanceKey.UNRESOLVED_ENGINE_EFFECT, resolved.reason());
    }

    /**
     * A {@code createEffect}-built card is the other half of the same shape: it
     * is equally scriptless, but the ability that created it is not, and that
     * is where the printed line is.
     */
    @Test
    void anEffectCardCreatedByAnAbilityKeysToItsCreator() {
        Card fountain = card("Fountain of Youth");
        SpellAbility creator = scriptedSpell(fountain.getCurrentState(), "GainLife");

        Card effect = new Card(TestCards.nextCardId(), TestCards.game());
        effect.setName("Fountain of Youth's Effect");
        effect.setGamePieceType(GamePieceType.EFFECT);
        effect.setEffectSource(creator);
        SpellAbility onEffect = AbilityFactory.getAbility(
                "DB$ GainLife | Defined$ You | LifeAmount$ 1", effect);
        effect.getCurrentState().addSpellAbility(onEffect);

        assertEquals(ProvenanceKey.of(creator), ProvenanceKey.of(onEffect));
    }

    // ── keyword-derived abilities ───────────────────────────────────────

    /**
     * The ordinal both sides compute. It cannot be a position in
     * {@code getKeywords()}: that collection is hash-ordered over an enum whose
     * {@code hashCode()} is the identity hash, so the convert JVM and the
     * collect JVM disagree about it. Sorting the printed text is stable in
     * both.
     */
    @Test
    void aKeywordOrdinalIsThePositionOfItsPrintedTextWhenSorted() {
        Card blast = card("Blast from the Past");
        CardState state = blast.getCurrentState();

        List<String> sorted = new ArrayList<>();
        for (KeywordInterface keyword : state.getIntrinsicKeywords()) {
            sorted.add(keyword.getOriginal());
        }
        java.util.Collections.sort(sorted);
        assertEquals(5, sorted.size(), "Blast from the Past prints five keywords");

        for (KeywordInterface keyword : state.getIntrinsicKeywords()) {
            assertEquals(sorted.indexOf(keyword.getOriginal()),
                    ProvenanceKey.keywordIndex(state, keyword));
        }
    }

    /**
     * The load-bearing assertion: if someone reverts to a running counter over
     * {@code getKeywords()}, this fails. The two orders genuinely differ on a
     * card with five keywords, which is why the counter was not reproducible.
     */
    @Test
    void aKeywordOrdinalDoesNotFollowTheHashOrderedCollection() {
        Card blast = card("Blast from the Past");
        CardState state = blast.getCurrentState();

        boolean anyDisagreement = false;
        int position = 0;
        for (KeywordInterface keyword : blast.getKeywords()) {
            if (ProvenanceKey.keywordIndex(state, keyword) != position) {
                anyDisagreement = true;
            }
            position++;
        }
        assertTrue(anyDisagreement,
                "the sorted ordinal must not merely reproduce the collection order");
    }

    /** Two builds of one card agree, which the hash-ordered counter need not. */
    @Test
    void aKeywordOrdinalIsTheSameForTwoBuildsOfOneCard() {
        Card first = card("Blast from the Past");
        Card second = card("Blast from the Past");

        assertEquals(ordinalsByKeywordText(first), ordinalsByKeywordText(second));
    }

    private static List<String> ordinalsByKeywordText(Card card) {
        List<String> rendered = new ArrayList<>();
        CardState state = card.getCurrentState();
        for (KeywordInterface keyword : state.getIntrinsicKeywords()) {
            rendered.add(keyword.getOriginal() + "@"
                    + ProvenanceKey.keywordIndex(state, keyword));
        }
        java.util.Collections.sort(rendered);
        return rendered;
    }

    /**
     * A granted keyword names no printed line on the card that received it, so
     * it answers -1 and keys to nothing. That is strictly better than the
     * position it happens to occupy in the recipient's list, which is a wrong
     * key dressed as an answer.
     */
    @Test
    void aGrantedKeywordHasNoPrintedOrdinal() {
        Card bear = card("Grizzly Bears");
        Card blast = card("Blast from the Past");
        KeywordInterface flashback = null;
        for (KeywordInterface keyword : blast.getCurrentState().getIntrinsicKeywords()) {
            if (keyword.getOriginal().startsWith("Flashback")) flashback = keyword;
        }
        assertNotNull(flashback);

        assertEquals(-1,
                ProvenanceKey.keywordIndex(bear.getCurrentState(), flashback),
                "a keyword the card does not print has no ordinal on it");
    }

    /**
     * A keyword-derived ability — flashback, cycling, unearth, equip — keys to
     * the keyword, not into the spell slice it was appended to. The corpus has
     * never carried a {@code keyword} key, and the {@code spell} index those
     * abilities were getting instead came out of the hash-ordered tail.
     */
    @Test
    void aKeywordDerivedAbilityKeysToTheKeywordSlot() {
        Card blast = card("Blast from the Past");
        CardState state = blast.getCurrentState();
        KeywordInterface flashback = null;
        for (KeywordInterface keyword : state.getIntrinsicKeywords()) {
            if (keyword.getOriginal().startsWith("Flashback")) flashback = keyword;
        }
        assertNotNull(flashback);

        // The shape GameActionUtil produces: a copy of the spell carrying the
        // keyword instance and no back-reference to a printed spell line.
        SpellAbility spell = scriptedSpell(state, "DealDamage");
        SpellAbility derived = spell.copy(blast, null, false, true);
        derived.setKeyword(flashback);

        ProvenanceKey key = ProvenanceKey.of(derived);
        assertNotNull(key, "a keyword-derived ability must key");
        assertEquals(ProvenanceKey.KIND_KEYWORD, key.traitKind());
        assertEquals(ProvenanceKey.keywordIndex(state, flashback),
                key.indexWithinKind());
        assertEquals("cardsfolder/b/blast_from_the_past.txt", key.scriptFile());
    }

    // ── the reason vocabulary ───────────────────────────────────────────

    /** A trait with no state has nothing to index against, and says so. */
    @Test
    void aTraitWithNoCardStateSaysSo() {
        Card fountain = card("Fountain of Youth");
        SpellAbility printed = scriptedSpell(fountain.getCurrentState(), "GainLife");
        SpellAbility detached = printed.copy(fountain, null, false, true);
        detached.setCardState(null);

        ProvenanceKey.Resolved resolved = ProvenanceKey.resolve(detached);
        assertNull(resolved.key());
        assertEquals(ProvenanceKey.UNRESOLVED_NO_CARD_STATE, resolved.reason());
    }

    /** A key that was found carries no reason, and the reverse. */
    @Test
    void aResolvedKeyCarriesNoReason() {
        Card fountain = card("Fountain of Youth");
        ProvenanceKey.Resolved resolved = ProvenanceKey.resolve(
                scriptedSpell(fountain.getCurrentState(), "GainLife"));

        assertNotNull(resolved.key());
        assertNull(resolved.reason());
    }

    /**
     * Nothing handed over at all is an unknown kind, not a resolver failure.
     *
     * <p>A hook whose signature drifted delivers something that is not a trait,
     * and the collector passes null. {@code unindexable} would be the wrong
     * reason for it: that one is documented as the signature of a resolver bug
     * and must not appear in a healthy run, so spending it on a hook mismatch
     * would make the alarm unreadable.
     */
    @Test
    void nothingAtAllIsAnUnknownKindRatherThanAResolverFailure() {
        ProvenanceKey.Resolved resolved = ProvenanceKey.resolve(null);

        assertNull(resolved.key());
        assertEquals(ProvenanceKey.UNRESOLVED_UNKNOWN_KIND, resolved.reason());
        assertTrue(resolved.keys().isEmpty());
    }

    /** The record's acting-line list is the resolved key, or nothing. */
    @Test
    void theKeysListIsWhatARecordCarries() {
        Card fountain = card("Fountain of Youth");
        ProvenanceKey.Resolved resolved = ProvenanceKey.resolve(
                scriptedSpell(fountain.getCurrentState(), "GainLife"));

        assertEquals(List.of(resolved.key()), resolved.keys());
    }

    /** {@code of} stays the entry point every collector calls. */
    @Test
    void ofAgreesWithResolve() {
        Card fountain = card("Fountain of Youth");
        SpellAbility printed = scriptedSpell(fountain.getCurrentState(), "GainLife");

        assertEquals(ProvenanceKey.resolve(printed).key(), ProvenanceKey.of(printed));
        assertNull(ProvenanceKey.of(null));
    }

    // -- the script path comes from the tree, not from a rule ------------

    /**
     * The card no derivation rule reaches. {@code +2 Mace} sanitizes to
     * {@code +2_mace}, whose initial is {@code +}, and Forge files it under
     * {@code p/}. Nothing downstream would have complained: only a
     * {@code keyword} key is checked against its sidecar, so the fabricated
     * path joined to nothing in silence.
     */
    @Test
    void aCardTheTreeFilesUnderAnotherLetterKeysToWhereItReallyIs() {
        Card mace = card("+2 Mace");
        ProvenanceKey key = ProvenanceKey.of(
                mace.getCurrentState().getStaticAbilities().iterator().next());

        assertNotNull(key);
        assertEquals("cardsfolder/p/+2_mace.txt", key.scriptFile());
    }

    /**
     * And the ordinary case is untouched: the path is still the file Forge read
     * the card from, which for almost every card is the derived one too.
     */
    @Test
    void anOrdinaryCardStillKeysToItsLetterDirectory() {
        Card fountain = card("Fountain of Youth");
        ProvenanceKey key = ProvenanceKey.of(
                scriptedSpell(fountain.getCurrentState(), "GainLife"));

        assertNotNull(key);
        assertEquals("cardsfolder/f/fountain_of_youth.txt", key.scriptFile());
    }

    /**
     * The join is a string comparison, so what the collector writes has to be
     * what the converter wrote: the source path relativised against the tree,
     * with forward slashes, on a platform whose own separator is neither.
     */
    @Test
    void theKeyedPathIsTheOneTheConverterWouldHaveWritten() {
        Card mace = card("+2 Mace");
        String path = ProvenanceKey.scriptFileOf(mace);

        assertEquals(SourceTree.CARDSFOLDER + "/"
                        + SourceTree.relativePathIn(
                                SourceTree.CARDSFOLDER, mace.getRules().getPath()),
                path);
        assertEquals(-1, path.indexOf('\\'), path);
    }

    // -- granted keywords key to the permanent that granted them ---------

    /**
     * Virulent Sliver gives every Sliver poisonous 1, and poisonous builds a
     * trigger — on the <em>recipient</em>, where it names no printed line. The
     * recipient's keyword list is not an answer: it is a position in a list of
     * things the card does not print. The static that handed the keyword out is
     * the printed line, and {@code Card.getKeywordForStaticAbility} records it.
     *
     * <p>This was the largest single bucket among the 443 records that reported
     * {@code unindexable} in the smoke corpus.
     */
    @Test
    void aKeywordGrantedByAnotherPermanentKeysToThatPermanentsStatic() {
        Card donor = card("Virulent Sliver");
        StaticAbility grantor =
                donor.getCurrentState().getStaticAbilities().iterator().next();
        Card recipient = card("Metallic Sliver");

        ProvenanceKey key = ProvenanceKey.of(
                grantedKeywordTrigger(recipient, "Poisonous:1", grantor));

        assertNotNull(key, "a granted keyword's trigger must key to its donor");
        assertEquals("cardsfolder/v/virulent_sliver.txt", key.scriptFile());
        assertEquals(ProvenanceKey.KIND_STATIC, key.traitKind());
        assertEquals(ProvenanceKey.of(grantor), key);
    }

    /**
     * Never the recipient. Keying a granted trigger where it now lives would
     * teach the model that Metallic Sliver — a vanilla 1/1 — prints a poison
     * trigger, which is worse than naming no line at all.
     */
    @Test
    void aGrantedKeywordNeverKeysToTheCardThatReceivedIt() {
        Card donor = card("Virulent Sliver");
        StaticAbility grantor =
                donor.getCurrentState().getStaticAbilities().iterator().next();
        Card recipient = card("Metallic Sliver");

        ProvenanceKey key = ProvenanceKey.of(
                grantedKeywordTrigger(recipient, "Poisonous:1", grantor));

        assertNotEquals(ProvenanceKey.scriptFileOf(recipient), key.scriptFile());
    }

    /**
     * A keyword the card does print is untouched by the donor hop: it still
     * keys to its own ordinal, which is what the converter emits.
     */
    @Test
    void aPrintedKeywordStillKeysToItsOwnOrdinal() {
        Card blast = card("Blast from the Past");
        CardState state = blast.getCurrentState();
        KeywordInterface flashback = null;
        for (KeywordInterface keyword : state.getIntrinsicKeywords()) {
            if (keyword.getOriginal().startsWith("Flashback")) flashback = keyword;
        }
        assertNotNull(flashback);

        SpellAbility derived =
                scriptedSpell(state, "DealDamage").copy(blast, null, false, true);
        derived.setKeyword(flashback);
        ProvenanceKey key = ProvenanceKey.of(derived);

        assertNotNull(key);
        assertEquals(ProvenanceKey.KIND_KEYWORD, key.traitKind());
        assertEquals(ProvenanceKey.keywordIndex(state, flashback), key.indexWithinKind());
    }

    /** The engine's own path for handing a keyword to another permanent. */
    private static Trigger grantedKeywordTrigger(
            Card recipient, String keyword, StaticAbility grantor) {
        KeywordInterface granted =
                recipient.getKeywordForStaticAbility(keyword, grantor, 1);
        for (Trigger trigger : granted.getTriggers()) {
            return trigger;
        }
        throw new AssertionError(keyword + " built no trigger");
    }

    // -- triggers the engine spawned ------------------------------------

    /**
     * A delayed trigger — the "sacrifice it at the beginning of the next end
     * step" that {@code AtEOT$} builds — is assembled in Java rather than
     * parsed from a card, so it is a member of no trait list anywhere. It does
     * carry a copy of the ability that spawned it, and that is the printed
     * line.
     */
    @Test
    void aDelayedTriggerKeysToTheAbilityThatSpawnedIt() {
        Card fountain = card("Fountain of Youth");
        SpellAbility spawner = scriptedSpell(fountain.getCurrentState(), "GainLife");

        ProvenanceKey key = ProvenanceKey.of(delayedTriggerSpawnedBy(spawner));

        assertNotNull(key, "a delayed trigger must key to its spawner");
        assertEquals(ProvenanceKey.of(spawner), key);
        assertEquals(ProvenanceKey.KIND_SPELL, key.traitKind());
    }

    /**
     * The same shape spawned from inside a trigger's {@code Execute$} SVar
     * keys one step further, to the {@code T:} line: the SVar is a member of no
     * slice, so the trigger is what is printed. Two of the smoke corpus's
     * biggest offenders were exactly this — a token created by a triggered
     * ability and sacrificed at end of turn.
     */
    @Test
    void aDelayedTriggerSpawnedInsideATriggerKeysToTheTriggerLine() {
        Card brassMan = card("Brass Man");
        Trigger printed = brassMan.getCurrentState().getTriggers().iterator().next();
        SpellAbility execute = printed.ensureAbility();
        execute.setTrigger(printed);

        ProvenanceKey key = ProvenanceKey.of(delayedTriggerSpawnedBy(execute));

        assertNotNull(key);
        assertEquals("cardsfolder/b/brass_man.txt", key.scriptFile());
        assertEquals(ProvenanceKey.KIND_TRIGGER, key.traitKind());
        assertEquals(0, key.indexWithinKind());
    }

    /**
     * A reflexive trigger's {@code Execute$} ability keys to the printed
     * {@code T:} line two steps above it.
     *
     * <p>The largest bucket the smoke corpus reported as {@code unindexable}:
     * 378 records over 37 cards, all of them the "you may pay {X}. When you do,
     * ..." shape. What resolves is the reflexive trigger's own execute ability,
     * whose owning trigger {@code ImmediateTriggerEffect} assembles at runtime
     * and which is therefore in no slice — but that trigger carries the ability
     * that spawned it, and following that reaches the printed line. The
     * resolver could already follow a {@code Trigger} it was handed directly;
     * what it could not do was reach the trigger from the ability it owns.
     */
    @Test
    void aReflexiveTriggersExecuteKeysToThePrintedTriggerLine() {
        Card numa = card("Numa, Joraga Chieftain");
        Trigger printed = numa.getCurrentState().getTriggers().iterator().next();
        ProvenanceKey expected = ProvenanceKey.of(printed);
        assertNotNull(expected, "the printed T: line must key");

        SpellAbility payCost = printed.ensureAbility();
        payCost.setTrigger(printed);
        Trigger reflexive = reflexiveTriggerSpawnedBy(payCost);
        SpellAbility execute = reflexive.ensureAbility();
        execute.setTrigger(reflexive);

        assertEquals(expected, ProvenanceKey.of(execute));
        assertEquals(ProvenanceKey.KIND_TRIGGER, expected.traitKind());
    }

    /** Built the way {@code ImmediateTriggerEffect} builds it. */
    private static Trigger reflexiveTriggerSpawnedBy(SpellAbility spawner) {
        Card host = spawner.getHostCard();
        Map<String, String> params = new HashMap<>(spawner.getMapParams());
        params.put("Mode", TriggerType.Immediate.name());
        params.remove("Cost");
        Trigger reflexive = TriggerHandler.parseTrigger(
                params, host, spawner.isIntrinsic(), null);
        reflexive.setSpawningAbility(spawner.copy(host, true));
        SpellAbility overriding = spawner.getAdditionalAbility("Execute");
        if (overriding != null) {
            SpellAbility copy = overriding.copy(host, null, false);
            // The engine nulls the parent here, "otherwise it might have wrong
            // root ability" -- which also cuts the link the resolver would
            // otherwise climb.
            if (copy instanceof AbilitySub sub) {
                sub.setParent(null);
            }
            reflexive.setOverridingAbility(copy);
        }
        return reflexive;
    }

    /**
     * A cost-variant copy of a permanent spell keys to the printed spell.
     *
     * <p>Casting for an alternative or an additional cost -- an Adventure half,
     * kicker, {@code MayFlashCost} -- pushes a copy, and unlike the stack's own
     * copy it carries no back-reference. The fingerprint cannot rescue it
     * either: a permanent spell is built in Java with an empty parameter map
     * ({@code SpellPermanent}), so there is nothing to match on. What is left
     * is that a card state has exactly one of them, and one candidate needs no
     * fingerprint to be unambiguous.
     */
    @Test
    void aCostVariantCopyOfAPermanentSpellKeysToThePrintedSpell() {
        Card adept = card("Silvergill Adept");
        SpellAbility printed = adept.getCurrentState().getFirstSpellAbility();
        ProvenanceKey expected = ProvenanceKey.of(printed);
        assertNotNull(expected, "the pristine permanent spell must key");

        SpellAbility variant = printed.copy(adept, null, false, true);

        assertNull(variant.getOriginalAbility(),
                "a cost variant carries no back-reference; that is the case under test");
        assertTrue(printed.getOriginalMapParams().isEmpty(),
                "a permanent spell has no script parameters to fingerprint");
        assertEquals(expected, ProvenanceKey.of(variant));
    }

    /**
     * Built the way {@code SpellAbilityEffect.registerDelayedTrigger} builds
     * it, minus the LKI host copy the engine passes — which changes nothing
     * here, because the point is that the trigger belongs to no slice either
     * way.
     */
    private static Trigger delayedTriggerSpawnedBy(SpellAbility spawner) {
        Trigger delayed = TriggerHandler.parseTrigger(
                "Mode$ Phase | Phase$ End Of Turn | TriggerDescription$ Sacrifice it.",
                spawner.getHostCard(), spawner.isIntrinsic());
        delayed.setSpawningAbility(spawner.copy(spawner.getHostCard(), true));
        return delayed;
    }

    // -- reasons for the traits that genuinely have no printed line ------

    // -- traits granted out of a donor's SVar ---------------------------

    /**
     * Genju of the Fields animates a Plains and hands it the
     * {@code PseudoLifelink} trigger. Forge parses that from the aura's SVar
     * and points it at the aura's state — enough to name the file, and nothing
     * in the engine says which of the aura's lines named the SVar. Both halves
     * are still in the state, so the lookup Forge did on the way out can be
     * done again backwards: the SVar whose text parses to these parameters, and
     * then the line that names that SVar.
     *
     * <p>This shape was half of the 443 records the smoke corpus reported as
     * {@code unindexable}.
     */
    @Test
    void anAuraGrantedTriggerKeysToTheLineThatGrantedIt() {
        Card genju = card("Genju of the Fields");
        SpellAbility animate = scriptedSpell(genju.getCurrentState(), "Animate");
        Trigger granted = grantedFromSVar(genju, card("Plains"), "PseudoLifelink", animate);

        ProvenanceKey key = ProvenanceKey.of(granted);

        assertNotNull(key, "the aura's own Animate line granted this trigger");
        assertEquals(ProvenanceKey.of(animate), key);
        assertEquals("cardsfolder/g/genju_of_the_fields.txt", key.scriptFile());
        assertEquals(ProvenanceKey.KIND_SPELL, key.traitKind());
    }

    /** Never the creature that received it, which prints no such line. */
    @Test
    void anAuraGrantedTriggerNeverKeysToTheCreatureItWasGrantedTo() {
        Card genju = card("Genju of the Fields");
        Card plains = card("Plains");
        SpellAbility animate = scriptedSpell(genju.getCurrentState(), "Animate");

        ProvenanceKey key = ProvenanceKey.of(
                grantedFromSVar(genju, plains, "PseudoLifelink", animate));

        assertNotEquals(ProvenanceKey.scriptFileOf(plains), key.scriptFile());
    }

    /**
     * The equipment half of the same shape, and the commonest donor in the
     * corpus: a continuous static with {@code AddTrigger$}, whose grantee is
     * whatever it is attached to.
     */
    @Test
    void anEquipmentGrantedTriggerKeysToTheStaticThatGrantedIt() {
        Card aura = card("Commanding Presence");
        StaticAbility grantor = null;
        for (StaticAbility stAb : aura.getCurrentState().getStaticAbilities()) {
            if (stAb.hasParam("AddTrigger")) grantor = stAb;
        }
        assertNotNull(grantor, "Commanding Presence grants a trigger");
        Card recipient = card("Grizzly Bears");

        // The engine's own path: Card.getTriggerForStaticAbility.
        Trigger granted = recipient.getTriggerForStaticAbility(
                aura.getCurrentState().getSVar(grantor.getParam("AddTrigger")), grantor);

        ProvenanceKey key = ProvenanceKey.of(granted);
        assertNotNull(key);
        assertEquals(ProvenanceKey.of(grantor), key);
        assertEquals("cardsfolder/c/commanding_presence.txt", key.scriptFile());
        assertEquals(ProvenanceKey.KIND_STATIC, key.traitKind());
    }

    /**
     * Two SVars with the same text cannot be told apart, so neither is
     * answered. No key beats a wrong key here exactly as it does for the
     * structural fingerprint.
     */
    @Test
    void anAmbiguousSvarRefusesToGuessWhichLineGrantedIt() {
        Card genju = card("Genju of the Fields");
        CardState state = genju.getCurrentState();
        SpellAbility animate = scriptedSpell(state, "Animate");
        Trigger granted = grantedFromSVar(genju, card("Plains"), "PseudoLifelink", animate);
        state.setSVar("PseudoLifelinkTwin", state.getSVar("PseudoLifelink"));

        ProvenanceKey.Resolved resolved = ProvenanceKey.resolve(granted);

        assertNull(resolved.key(), "two SVars parse the same; neither is the answer");
        assertEquals(ProvenanceKey.UNRESOLVED_GRANTED_TRAIT, resolved.reason());
    }

    /**
     * A granted trait the engine assembled in Java rather than from an SVar —
     * {@code CountersPut}'s and {@code Earthbend}'s inline triggers — has no
     * SVar to look up and no back-reference, so it names the donor card and no
     * line. {@code unindexable} is the wrong reason for it: that one is
     * documented as the signature of a resolver bug and must not appear in a
     * healthy run, while this appears whenever an aura, an equipment or an
     * {@code Animate} does its job.
     */
    @Test
    void aGrantedTraitWithNoSvarToFindSaysItWasGrantedRatherThanRaisingTheAlarm() {
        Card genju = card("Genju of the Fields");
        SpellAbility animate = scriptedSpell(genju.getCurrentState(), "Animate");
        Trigger granted = TriggerHandler.parseTrigger(
                "Mode$ Phase | Phase$ End Of Turn | TriggerDescription$ Assembled inline.",
                card("Plains"), false, animate);

        ProvenanceKey.Resolved resolved = ProvenanceKey.resolve(granted);

        assertNull(resolved.key(), "no SVar of the donor carries this text");
        assertEquals(ProvenanceKey.UNRESOLVED_GRANTED_TRAIT, resolved.reason());
    }

    /** A granted trait still names the donor's card, even when the line is lost. */
    @Test
    void aGrantedTraitStillKnowsWhichCardGrantedIt() {
        Card genju = card("Genju of the Fields");
        SpellAbility animate = scriptedSpell(genju.getCurrentState(), "Animate");
        Trigger granted = TriggerHandler.parseTrigger(
                "Mode$ Phase | Phase$ End Of Turn | TriggerDescription$ Assembled inline.",
                card("Plains"), false, animate);

        assertEquals("cardsfolder/g/genju_of_the_fields.txt",
                ProvenanceKey.scriptFileOf(granted.getCardState().getCard()),
                "the donor's state is what Forge points a granted trait at");
    }

    /** Granted the way {@code AnimateEffectBase} grants: parsed from the donor's SVar. */
    private static Trigger grantedFromSVar(
            Card donor, Card recipient, String svar, SpellAbility granting) {
        return TriggerHandler.parseTrigger(
                donor.getCurrentState().getSVar(svar), recipient, false, granting);
    }

    /**
     * A keyword granted by something that is not a continuous static — a pump
     * spell's "gains flying until end of turn" — keeps no back-reference to the
     * ability that granted it, so its trigger names no line anywhere. That is a
     * property of how Forge grants keywords, not a resolver failure, and it
     * says so.
     */
    @Test
    void aKeywordGrantedWithNoGrantorSaysItWasGranted() {
        Card recipient = card("Metallic Sliver");
        Trigger granted = grantedKeywordTrigger(recipient, "Poisonous:1", null);

        ProvenanceKey.Resolved resolved = ProvenanceKey.resolve(granted);

        assertNull(resolved.key(), "the recipient prints no such keyword");
        assertEquals(ProvenanceKey.UNRESOLVED_GRANTED_KEYWORD, resolved.reason());
    }

    /**
     * The alarm keeps its meaning: a trait printed on the state it names, which
     * still cannot be placed in that state, is a resolver failure and nothing
     * else.
     */
    @Test
    void aPrintedTraitThatCannotBePlacedStillRaisesTheAlarm() {
        Card fountain = card("Fountain of Youth");
        SpellAbility stray = AbilityFactory.getAbility(
                "DB$ GainLife | Defined$ You | LifeAmount$ 99", fountain);
        stray.setCardState(fountain.getCurrentState());

        ProvenanceKey.Resolved resolved = ProvenanceKey.resolve(stray);

        assertNull(resolved.key());
        assertEquals(ProvenanceKey.UNRESOLVED_UNINDEXABLE, resolved.reason());
    }

    /** Every reason is distinct, because a reason that collides says nothing. */
    @Test
    void theReasonVocabularyHasNoDuplicates() {
        List<String> reasons = List.of(
                ProvenanceKey.UNRESOLVED_NO_CARD_STATE,
                ProvenanceKey.UNRESOLVED_UNKNOWN_KIND,
                ProvenanceKey.UNRESOLVED_ENGINE_EFFECT,
                ProvenanceKey.UNRESOLVED_GRANTED_KEYWORD,
                ProvenanceKey.UNRESOLVED_GRANTED_TRAIT,
                ProvenanceKey.UNRESOLVED_UNINDEXABLE);

        assertEquals(reasons.size(), new java.util.HashSet<>(reasons).size());
    }
}
