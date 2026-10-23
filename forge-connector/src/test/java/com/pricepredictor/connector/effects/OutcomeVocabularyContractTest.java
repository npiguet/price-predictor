package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Every outcome type the patched engine can send is one this collector knows.
 *
 * <p>The engine's names are read reflectively, because this module builds
 * against an unpatched checkout too; there the test is skipped. A name the
 * collector does not know would otherwise surface only as the unknown-outcome
 * warning in a collection run's log.
 */
@ExtendWith(ForgeExtension.class)
class OutcomeVocabularyContractTest {

    /** Stands in for the engine's outcome types when the checkout is unpatched. */
    private static final String UNPATCHED = "<unpatched>";

    @TempDir
    Path tempDir;

    private static Method onOutcome() {
        return HookShapes.method(HookShapes.Outcome.class, "onOutcome");
    }

    private static Stream<String> engineOutcomeTypes() throws Exception {
        try {
            return stringConstants(Class.forName(PatchHooks.EFFECT_RECORD_OUTCOMES)).stream();
        } catch (ClassNotFoundException unpatched) {
            return Stream.of(UNPATCHED);
        }
    }

    private static Set<String> stringConstants(Class<?> owner) throws IllegalAccessException {
        Set<String> values = new TreeSet<>();
        for (Field field : owner.getFields()) {
            int modifiers = field.getModifiers();
            if (Modifier.isStatic(modifiers) && Modifier.isFinal(modifiers)
                    && field.getType() == String.class) {
                values.add((String) field.get(null));
            }
        }
        return values;
    }

    /**
     * An outcome type the engine declares and {@code KNOWN_OUTCOME_TYPES} omits
     * is dropped with one stderr warning in a collection run, so the type
     * never reaches a shard. Driving each declared type through the handler and
     * reading the shard back is what notices.
     */
    @ParameterizedTest
    @MethodSource("engineOutcomeTypes")
    void everyEngineOutcomeTypeReachesTheLiveShard(String type) throws Throwable {
        assumeTrue(!UNPATCHED.equals(type), "the sibling Forge checkout is unpatched");
        SpellAbility liveAbility = TestCards.scriptedAbility("Alchemist's Gambit", "AddTurn");

        List<String> records = ShardRun.records(tempDir, TestCards.game(), (bracket, collectors) -> {
            bracket.beginBracket(liveAbility);
            collectors.clauses().outcomeHandler().invoke(null, onOutcome(),
                    new Object[]{liveAbility, type, Map.of()});
            bracket.endBracket(liveAbility.getId(), false);
            bracket.onNextAction();
        });

        assertTrue(String.join("\n", records).contains("\"type\":\"" + type + "\""),
                "the engine's " + type + " outcome never reached the shard -- is it an"
                        + " EventType, and in KNOWN_OUTCOME_TYPES? " + records);
    }

    /** A Card and a Player in {@code subjects} become entity and player refs in the shard. */
    @Test
    void outcomeSubjectsReachTheShardAsEntityAndPlayerRefs() throws Throwable {
        Card card = TestCards.build("Grizzly Bears");
        Player player = new Player("active", TestCards.game(), 94001);
        SpellAbility liveAbility = TestCards.scriptedAbility("Alchemist's Gambit", "AddTurn");

        List<String> records = ShardRun.records(tempDir, TestCards.game(), (bracket, collectors) -> {
            bracket.beginBracket(liveAbility);
            collectors.clauses().outcomeHandler().invoke(null, onOutcome(),
                    new Object[]{liveAbility, EventType.TEXT_CHANGE.wireName(),
                            Map.of("subjects", List.of(card, player))});
            bracket.endBracket(liveAbility.getId(), false);
            bracket.onNextAction();
        });

        String shard = String.join("\n", records);
        assertTrue(shard.contains(Refs.of(card)), shard);
        assertTrue(shard.contains(Refs.of(player)), shard);
    }
}
