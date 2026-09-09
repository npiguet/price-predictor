package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.card.CardStateName;
import forge.card.GamePieceType;
import forge.game.ability.AbilityFactory;
import forge.game.card.Card;
import forge.game.card.CardState;
import forge.game.keyword.KeywordInterface;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;
import forge.game.trigger.WrappedAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.ArrayList;
import java.util.List;

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
}
