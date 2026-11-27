package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.RulesParser;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.card.CardState;
import forge.game.keyword.Keyword;
import forge.game.keyword.KeywordInterface;
import forge.game.phase.PhaseType;
import forge.game.spellability.SpellAbility;
import forge.game.staticability.StaticAbility;
import forge.game.trigger.Trigger;
import forge.game.trigger.WrappedAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A trait a card grants itself keys to the line that granted it, not to its
 * place in the live list.
 *
 * <p>Forge lists a state's traits printed first, then the ones a layer change
 * granted, then the keyword-built ones. A trait a card's own static or effect
 * granted (a Class level's, a "has ... as long as" trigger, a manland's
 * animated trigger or activated ability) sits on the card's own state, so its live index past the
 * printed lines named no line of the sidecar, or another one. Every case reads
 * the key against the card's real sidecar: the line the key declares, else,
 * for a key with {@code built}, the line its bare key declares.
 */
@ExtendWith(ForgeExtension.class)
class SelfGrantedTraitKeyTest {

    private final RulesParser parser = new RulesParser();

    /** Ranger Class at level 3: each level's granted trait joins its own level's line. */
    @Test
    void aClassLevelsGrantedTraitsJoinTheirLevelsLine() throws IOException {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card ranger = atLevel(board, "Ranger Class", 3);
        ProvenanceSidecar sidecar = sidecarOf(ranger);
        Trigger attack = trigger(ranger, each -> "AttackersDeclared".equals(each.getParam("Mode")));

        assertEquals(classLine(sidecar, ranger, 2), lineJoining(sidecar, ProvenanceResolver.of(attack)),
                "the level-2 trigger: " + ProvenanceResolver.of(attack));
        assertEquals(classLine(sidecar, ranger, 2), lineJoining(sidecar, ProvenanceResolver.of(
                        new WrappedAbility(attack, attack.ensureAbility(), board.you))),
                "the level-2 trigger resolving");
        assertNull(ProvenanceResolver.of(attack).built(),
                "the level's static and its level-up ability both name the trigger: the bare key");
        for (String granted : List.of("You may look", "You may cast creature spells")) {
            StaticAbility level3 = staticAbility(ranger, granted);
            assertEquals(classLine(sidecar, ranger, 3), lineJoining(sidecar, ProvenanceResolver.of(level3)),
                    "the level-3 static \"" + granted + "\": " + ProvenanceResolver.of(level3));
        }
    }

    /** Barbarian Class at level 3: its haste static is not its printed line's {@code static 0}. */
    @Test
    void barbarianClassesHasteStaticJoinsLevelThree() throws IOException {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card barbarian = atLevel(board, "Barbarian Class", 3);
        ProvenanceSidecar sidecar = sidecarOf(barbarian);
        StaticAbility haste = staticAbility(barbarian, "Creatures you control have haste");
        Trigger roll = trigger(barbarian, each -> "RolledDieOnce".equals(each.getParam("Mode")));

        ProvenanceKey key = ProvenanceResolver.of(haste);
        assertNotEquals(new ProvenanceKey(ScriptLocator.scriptFileOf(barbarian), 0, TraitKind.STATIC, 0),
                key, "the haste static");
        assertEquals(classLine(sidecar, barbarian, 3), lineJoining(sidecar, key), "the haste static: " + key);
        assertEquals(classLine(sidecar, barbarian, 2), lineJoining(sidecar, ProvenanceResolver.of(roll)),
                "the level-2 trigger: " + ProvenanceResolver.of(roll));
    }

    /** Expedition Diviner's "has 'When this creature dies, draw a card'" joins its static's line. */
    @Test
    void aTriggerACardsOwnStaticGrantsJoinsTheStaticsLine() throws IOException {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card diviner = board.battlefield("Expedition Diviner", board.you);
        board.battlefield("Prodigal Sorcerer", board.you);
        board.game.getAction().checkStaticAbilities();
        ProvenanceSidecar sidecar = sidecarOf(diviner);
        Trigger dies = trigger(diviner, each -> "ChangesZone".equals(each.getParam("Mode")));
        int staticLine = lineDeclaring(sidecar, ProvenanceResolver.of(staticAbility(diviner, "As long as")));

        assertTrue(staticLine >= 0, "the granting static has a line");
        assertEquals(staticLine, lineJoining(sidecar, ProvenanceResolver.of(dies)),
                "the granted trigger: " + ProvenanceResolver.of(dies));
        assertEquals(staticLine, lineJoining(sidecar, ProvenanceResolver.of(
                new WrappedAbility(dies, dies.ensureAbility(), board.you))), "the granted trigger resolving");
    }

    /** Raging Ravine animated: the attack trigger its animation gives it joins the animation's line. */
    @Test
    void aTriggerACardsOwnAnimationGrantsJoinsTheAnimationsLine() throws IOException {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card ravine = board.battlefield("Raging Ravine", board.you);
        SpellAbility animate = board.ability(ravine, ApiType.Animate);
        board.resolve(animate, board.you);
        board.game.getAction().checkStaticAbilities();
        ProvenanceSidecar sidecar = sidecarOf(ravine);
        Trigger attacks = trigger(ravine, each -> "Attacks".equals(each.getParam("Mode")));
        int animateLine = lineDeclaring(sidecar, ProvenanceResolver.of(animate));

        assertTrue(animateLine >= 0, "the animation has a line");
        assertEquals(animateLine, lineJoining(sidecar, ProvenanceResolver.of(attacks)),
                "the granted trigger: " + ProvenanceResolver.of(attacks));
    }

    /**
     * Lavaclaw Reaches and Wandering Fumarole animated: the activated ability
     * their animation gives them ({@code Abilities$}) joins the animation's
     * line, as built and as the copy the stack pushes in its place, which
     * resolves.
     */
    @Test
    void anAbilityACardsOwnAnimationGrantsJoinsTheAnimationsLine() throws IOException {
        for (String manland : List.of("Lavaclaw Reaches", "Wandering Fumarole")) {
            CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
            Card land = board.battlefield(manland, board.you);
            SpellAbility animate = board.ability(land, ApiType.Animate);
            board.resolve(animate, board.you);
            board.game.getAction().checkStaticAbilities();
            ProvenanceSidecar sidecar = sidecarOf(land);
            SpellAbility granted = board.ability(land, ApiType.Pump);
            int animateLine = lineDeclaring(sidecar, ProvenanceResolver.of(animate));

            assertTrue(animateLine >= 0, manland + "'s animation has a line");
            assertEquals(animateLine, lineJoining(sidecar, ProvenanceResolver.of(granted)),
                    manland + "'s granted ability: " + ProvenanceResolver.of(granted));
            SpellAbility resolving = onTheStack(board, granted);
            assertEquals(animateLine, lineJoining(sidecar, ProvenanceResolver.of(resolving)),
                    manland + "'s granted ability's stack copy: " + ProvenanceResolver.of(resolving));
        }
    }

    /**
     * Iymrith's own static grants it ward: the trigger ward builds has no
     * keyword line of the card's, and joins the static's line.
     */
    @Test
    void aTraitAKeywordTheCardGrantsItselfBuiltJoinsTheGrantingStaticsLine() throws IOException {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card iymrith = board.battlefield("Iymrith, Desert Doom", board.you);
        board.game.getAction().checkStaticAbilities();
        ProvenanceSidecar sidecar = sidecarOf(iymrith);
        Trigger ward = trigger(iymrith, each -> each.getKeyword() != null
                && each.getKeyword().getKeyword() == Keyword.WARD);
        int staticLine = lineDeclaring(sidecar, ProvenanceResolver.of(staticAbility(iymrith, "CARDNAME has ward")));

        assertTrue(staticLine >= 0, "the granting static has a line");
        assertEquals(staticLine, lineJoining(sidecar, ProvenanceResolver.of(ward)),
                "the ward trigger: " + ProvenanceResolver.of(ward));
    }

    /**
     * Blood Moon makes Lonely Sandbar a Mountain: the Mountain's mana ability
     * keys past every line the sidecar declares, and not to the cycling line.
     */
    @Test
    void theManaAbilityBloodMoonGivesACyclingLandKeysPastEveryDeclaredLine() throws IOException {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card sandbar = board.battlefield("Lonely Sandbar", board.you);
        board.battlefield("Blood Moon", board.you);
        board.game.getAction().checkStaticAbilities();
        ProvenanceSidecar sidecar = sidecarOf(sandbar);
        List<SpellAbility> red = sandbar.getManaAbilities().stream()
                .filter(sa -> sa.getManaPart().getOrigProduced().contains("R")).toList();
        assertEquals(1, red.size(), "the Mountain's mana ability: " + sandbar.getManaAbilities());

        ProvenanceKey key = ProvenanceResolver.of(red.get(0));
        assertFalse(declares(sidecar, key), "the Mountain's mana ability keys inside the sidecar: " + key);
        assertEquals(-1, lineJoining(sidecar, key), "the Mountain's mana ability: " + key);
    }

    // ── the board ───────────────────────────────────────────────────────

    private static Card atLevel(CharacterizationBoard board, String name, int level) {
        Card card = board.battlefield(name, board.you);
        card.setClassLevel(level);
        board.game.getAction().checkStaticAbilities();
        return card;
    }

    /**
     * The copy {@code MagicStack.add} pushes in place of an activated
     * ability, the one that resolves: a fresh instance whose original is the
     * ability activated.
     */
    private static SpellAbility onTheStack(CharacterizationBoard board, SpellAbility ability) {
        ability.setActivatingPlayer(board.you);
        board.game.getStack().add(ability);
        SpellAbility copy = board.game.getStack().peekAbility();
        assertTrue(copy != ability && copy.getOriginalAbility() == ability,
                "the stack holds a copy of the ability: " + copy);
        return copy;
    }

    private static Trigger trigger(Card card, Predicate<Trigger> matches) {
        List<Trigger> found = new ArrayList<>();
        card.getCurrentState().getTriggers().forEach(each -> {
            if (matches.test(each)) found.add(each);
        });
        assertEquals(1, found.size(), card + "'s matching triggers: " + card.getCurrentState().getTriggers());
        return found.get(0);
    }

    /**
     * The card's one live static whose description starts with
     * {@code description}, other than a keyword's: a Class level's own static
     * describes itself by the lines it grants.
     */
    private static StaticAbility staticAbility(Card card, String description) {
        List<StaticAbility> found = new ArrayList<>();
        card.getCurrentState().getStaticAbilities().forEach(each -> {
            String text = each.getKeyword() == null ? each.getParam("Description") : null;
            if (text != null && text.startsWith(description)) found.add(each);
        });
        assertEquals(1, found.size(), card + "'s statics \"" + description + "\": "
                + card.getCurrentState().getStaticAbilities());
        return found.get(0);
    }

    // ── the sidecar ─────────────────────────────────────────────────────

    /** The card's sidecar as {@code python -m price_predictor convert} writes it. */
    private ProvenanceSidecar sidecarOf(Card card) throws IOException {
        String scriptFile = ScriptLocator.scriptFileOf(card);
        List<String> source = Files.readAllLines(Path.of(card.getRules().getPath()));
        return ConvertedScript.of(parser.parseScript(source, scriptFile), scriptFile).sidecar();
    }

    /** The index of the line that declares {@code key}, or -1. */
    private static int lineDeclaring(ProvenanceSidecar sidecar, ProvenanceKey key) {
        for (ProvenanceSidecar.Line line : sidecar.lines()) {
            if (line.provenance().contains(key)) return line.lineIndex();
        }
        return -1;
    }

    /**
     * The line a key joins, by the sidecar contract's rule: the line that
     * declares it, else, for a key with {@code built}, the line that declares
     * its bare key; -1 for none.
     */
    private static int lineJoining(ProvenanceSidecar sidecar, ProvenanceKey key) {
        if (key == null) return -1;
        int exact = lineDeclaring(sidecar, key);
        return exact >= 0 || key.built() == null ? exact : lineDeclaring(sidecar, key.withoutBuilt());
    }

    /** The line of the card's {@code Class:level} keyword, which declares its bare key. */
    private static int classLine(ProvenanceSidecar sidecar, Card card, int level) {
        CardState state = card.getCurrentState();
        for (KeywordInterface keyword : state.getIntrinsicKeywords()) {
            if (keyword.getOriginal().startsWith("Class:" + level + ":")) {
                int line = lineDeclaring(sidecar, new ProvenanceKey(ScriptLocator.scriptFileOf(card), 0,
                        TraitKind.KEYWORD, FaceOrder.keywordIndex(state, keyword)));
                assertTrue(line >= 0, "the Class:" + level + " line of " + sidecar);
                return line;
            }
        }
        throw new AssertionError("no Class:" + level + " on " + card);
    }

    /**
     * Whether a key falls inside a range the sidecar declares: its face and
     * trait kind, at an index no greater than the last the sidecar lists, on
     * a line or dropped.
     */
    private static boolean declares(ProvenanceSidecar sidecar, ProvenanceKey key) {
        if (key == null) return false;
        List<ProvenanceKey> declared = new ArrayList<>(sidecar.droppedKeys());
        sidecar.lines().forEach(line -> declared.addAll(line.provenance()));
        return declared.stream().anyMatch(each -> each.scriptFile().equals(key.scriptFile())
                && each.face() == key.face() && each.traitKind() == key.traitKind()
                && key.indexWithinKind() <= each.indexWithinKind());
    }
}
