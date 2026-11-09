package com.pricepredictor.connector.effects;

import forge.ai.simulation.GameSimulator;
import forge.ai.simulation.GameStateEvaluator;
import forge.game.Game;
import forge.game.player.Player;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * What a fork whose score disagrees with the live game's differs in, printed
 * beside the discard: where the game stood, whether the evaluator agrees with
 * itself on the live game, and the lines of the evaluator's own breakdown that
 * one game has and the other lacks. Every scoring draws from the score
 * check's own seeded stream ({@link Forks#seeded}), so a difference left is
 * the copy's and not the evaluator's simulated combat.
 *
 * <p>The breakdown is the evaluator's debug output ({@code
 * GameSimulator.debugLines}, a JVM static), captured for the span of one
 * scoring: diagnostic text only, read nowhere.
 */
final class ScoreMismatchReport {

    /** At most this many differing lines a side, so a large divergence stays readable. */
    static final int MAX_LINES = 25;

    private ScoreMismatchReport() {
    }

    static void print(Game live, Player perspective, Game fork, Player forkPerspective) {
        StringBuilder out = new StringBuilder("Effect records: score mismatch detail: ");
        out.append("phase ").append(live.getPhaseHandler().getPhase())
                .append(", turn ").append(live.getPhaseHandler().getTurn())
                .append(", stack ").append(live.getStack().size())
                .append(", caller ").append(caller());
        List<String> liveLines = breakdown(live, perspective);
        List<String> liveAgain = breakdown(live, perspective);
        List<String> forkLines = breakdown(fork, forkPerspective);
        out.append(", live re-score ").append(liveLines.equals(liveAgain) ? "same" : "DIFFERS");
        appendOnlyIn(out, "live only", liveLines, forkLines);
        appendOnlyIn(out, "fork only", forkLines, liveLines);
        System.err.println(out);
    }

    private static List<String> breakdown(Game game, Player perspective) {
        List<String> was = GameSimulator.debugLines;
        List<String> lines = new ArrayList<>();
        GameSimulator.debugLines = lines;
        try {
            GameStateEvaluator evaluator = new GameStateEvaluator();
            evaluator.setDebugging(true);
            lines.add("score " + Forks.seeded(
                    () -> evaluator.getScoreForGameState(game, perspective).value));
        } catch (RuntimeException e) {
            lines.add("threw " + e);
        } finally {
            GameSimulator.debugLines = was;
        }
        return lines;
    }

    /** The lines of {@code mine} that {@code theirs} lacks, counted as a multiset. */
    private static void appendOnlyIn(StringBuilder out, String label, List<String> mine, List<String> theirs) {
        Map<String, Integer> left = new HashMap<>();
        for (String line : theirs) {
            left.merge(line, 1, Integer::sum);
        }
        List<String> only = new ArrayList<>();
        for (String line : mine) {
            if (left.merge(line, -1, Integer::sum) < 0) {
                only.add(line.strip());
            }
        }
        out.append("\n  ").append(label).append(" (").append(only.size()).append("):");
        for (String line : only.subList(0, Math.min(MAX_LINES, only.size()))) {
            out.append("\n    ").append(line);
        }
    }

    /** The first class outside the fork machinery on the stack: which kind of fork asked. */
    static String caller() {
        return StackWalker.getInstance().walk(frames -> frames
                .map(StackWalker.StackFrame::getClassName)
                .filter(name -> !name.contains(".Forks") && !name.contains(".ScoreMismatchReport"))
                .findFirst().map(name -> name.substring(name.lastIndexOf('.') + 1))
                .orElse("?"));
    }
}
