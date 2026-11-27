package com.pricepredictor.connector;

import org.junit.jupiter.api.extension.ParameterContext;
import org.junit.jupiter.params.aggregator.ArgumentsAccessor;
import org.junit.jupiter.params.aggregator.ArgumentsAggregator;

/**
 * One case of a parameterized parser test: a card script in Forge's card
 * folder, and the text the test looks for in what the parser makes of it.
 *
 * @param file the script's path relative to the card folder
 * @param text the text the converted card holds
 */
record ScriptText(String file, String text) {

    /** Reads a case from a CSV row of the script, then the text. */
    static final class FromCsv implements ArgumentsAggregator {
        @Override
        public ScriptText aggregateArguments(ArgumentsAccessor row, ParameterContext context) {
            return new ScriptText(row.getString(0), row.getString(1));
        }
    }

    /** Reads a case from a CSV row whose first column only names the case. */
    static final class FromNamedCsv implements ArgumentsAggregator {
        @Override
        public ScriptText aggregateArguments(ArgumentsAccessor row, ParameterContext context) {
            return new ScriptText(row.getString(1), row.getString(2));
        }
    }
}
