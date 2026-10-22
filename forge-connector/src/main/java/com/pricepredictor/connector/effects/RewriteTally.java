package com.pricepredictor.connector.effects;

import com.google.common.collect.Multiset;
import com.google.common.collect.TreeMultiset;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.StringJoiner;

/**
 * What the rewrite channel wrote, for the whole worker process.
 *
 * <p>It counted drops until the payload could say which of the five
 * {@code ReplacementResult}s happened. That rule took 87% of the
 * candidates, and it had to: a substitution runs another ability rather
 * than editing the parameter map, so "exile it instead" and "the
 * replacement declined" arrived as the same pair of identical halves. With
 * the result recorded there is nothing left to drop, so this counts what
 * the channel produced instead — and the question it now answers, in the
 * first hour of a run rather than at the end of one, is whether the channel
 * filled: how many records, and in what proportions of result.
 *
 * <p>The raw-parameter breakdown survives the change of subject, aimed at
 * the records that carry no {@code outgoing}. A key whose <em>value</em>
 * moved while the rendered halves stayed identical still names a parameter
 * this collector does not read, and that is still the difference between
 * "this replacement rewrote nothing" and "this normaliser cannot express
 * what it rewrote". What is new is that such a record is now written rather
 * than discarded, so the diagnostic says what a written record is missing
 * rather than what a discarded one contained.
 *
 * <p>Per JVM rather than per game: one game's replacements are too few to
 * read a share off, and a worker plays many games. Synchronized because
 * nothing promises Forge keeps one thread for a whole worker, and the cost
 * is paid once per replacement rather than once per event.
 */
final class RewriteTally {

    /** Entries kept per breakdown before it stops taking new ones. */
    private static final int MAX_KEYS = 512;
    /** Entries printed per breakdown. */
    private static final int TOP = 8;
    /** Games between two summaries, after the first. */
    private static final int GAMES_PER_SUMMARY = 10;

    /**
     * The result of a record that carries none.
     *
     * <p>Two causes, and the record cannot tell them apart: the replacement
     * threw, or this worker is running against a Forge jar whose listener
     * predates the argument. A run where this is the whole breakdown is the
     * second one, and it is the shape to look for in the first summary a
     * fresh deployment prints.
     */
    private static final String NO_RESULT = "?";

    private final Multiset<String> byResult = TreeMultiset.create();
    private final Multiset<String> byModeAndResult = TreeMultiset.create();
    private final Multiset<String> rewrittenByMode = TreeMultiset.create();
    private final Multiset<String> movedButUnread = TreeMultiset.create();
    private final Multiset<String> unreadableByMode = TreeMultiset.create();
    private long records;
    private long rewritten;
    private long named;
    private long unreadable;
    private long games;

    /**
     * One replacement, and what its record says.
     *
     * <p>The raw maps are compared as rendered strings rather than by
     * {@code equals}: the engine deep-copies the containers of the incoming
     * map and leaves the leaves as identity references, so a copied
     * {@code Multiset} that was rewritten in place is a different object
     * reading the same text when nothing changed and a different text when
     * something did. Two distinct cards sharing one name read alike, which
     * understates rather than invents — the right way round for a diagnostic
     * whose job is to say "there is more here than the record shows".
     *
     * @param rewritten whether the record carries an {@code outgoing}
     * @param named     whether it names the ability that ran instead
     */
    synchronized void record(
            String mode, String result, boolean rewritten, boolean named,
            Object beforeParams, Object afterParams) {
        records++;
        String modeName = mode == null ? RunParamNormalizer.UNKNOWN_MODE : mode;
        String resultName = result == null ? NO_RESULT : result;
        byResult.add(resultName);
        byModeAndResult.add(modeName + "." + resultName);
        if (named) {
            this.named++;
        }
        if (rewritten) {
            this.rewritten++;
            rewrittenByMode.add(modeName);
            // A record that rewrote something in place already shows what
            // moved, so there is nothing for the gap diagnostic to find.
            return;
        }
        Map<String, Object> before = RunParamNormalizer.paramsByName(beforeParams);
        Map<String, Object> after = RunParamNormalizer.paramsByName(afterParams);
        Set<String> keys = new LinkedHashSet<>(before.keySet());
        keys.addAll(after.keySet());
        boolean moved = false;
        for (String key : keys) {
            if (String.valueOf(before.get(key))
                    .equals(String.valueOf(after.get(key)))) {
                continue;
            }
            moved = true;
            String entry = modeName + "." + key;
            if (movedButUnread.elementSet().size() < MAX_KEYS
                    || movedButUnread.contains(entry)) {
                movedButUnread.add(entry);
            }
        }
        if (moved) {
            unreadable++;
            unreadableByMode.add(modeName);
        }
    }

    /**
     * Close a game, printing the tally at a cadence the first hour answers.
     *
     * <p>The first game always, so a run that is wrong is visibly wrong a
     * minute after it starts rather than eight hours later; every tenth
     * after that, so a worker's log stays readable. A process that saw no
     * replacement at all prints nothing — that is what
     * {@code PatchHooks.report()} already says, and repeating it per game
     * would be noise.
     */
    synchronized void endOfGame() {
        games++;
        if (records == 0) {
            return;
        }
        if (games == 1 || games % GAMES_PER_SUMMARY == 0) {
            System.out.println(summary());
            System.out.flush();
        }
    }

    /** The rendered summary, its own method so a test can read it. */
    synchronized String summary() {
        StringBuilder out = new StringBuilder();
        out.append("Effect rewrites [worker, ").append(games)
                .append(games == 1 ? " game]: " : " games]: ")
                .append(records).append(" records, ")
                .append(rewritten).append(" rewrote a parameter in place (")
                .append(share(rewritten, records))
                .append("%), ").append(named)
                .append(" named the ability that ran instead");
        // The headline of the run. A first summary whose results are all
        // "?" is a listener the patched jar does not have; one that is all
        // not_replaced is a format whose replacements never apply; one with
        // no records at all never reaches here.
        out.append("\n  by result: ").append(top(byResult));
        out.append("\n  by mode and result: ").append(top(byModeAndResult));
        out.append("\n  rewrote in place, by mode: ").append(top(rewrittenByMode));
        // The normaliser gap, and the list of keys that would close it.
        out.append("\n  ").append(unreadable)
                .append(" carried no outgoing while a raw parameter moved: ")
                .append(top(movedButUnread));
        out.append("\n  of those, by mode: ").append(top(unreadableByMode));
        return out.toString();
    }

    /** A percentage to one decimal, or zero rather than a division by it. */
    private static double share(long part, long whole) {
        return whole == 0 ? 0.0 : Math.round(1000.0 * part / whole) / 10.0;
    }

    /** The heaviest entries, commonest first; the tail is a count. */
    private static String top(Multiset<String> counts) {
        if (counts.isEmpty()) {
            return "none";
        }
        List<Multiset.Entry<String>> entries = new ArrayList<>(counts.entrySet());
        entries.sort((a, b) -> {
            int byCount = Integer.compare(b.getCount(), a.getCount());
            return byCount != 0 ? byCount : a.getElement().compareTo(b.getElement());
        });
        StringJoiner joiner = new StringJoiner(", ");
        for (int i = 0; i < Math.min(TOP, entries.size()); i++) {
            joiner.add(entries.get(i).getElement() + "=" + entries.get(i).getCount());
        }
        if (entries.size() > TOP) {
            joiner.add("and " + (entries.size() - TOP) + " more");
        }
        return joiner.toString();
    }
}
