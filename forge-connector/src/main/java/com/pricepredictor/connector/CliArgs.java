package com.pricepredictor.connector;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Simple CLI argument parser for {@code --flag value} pairs.
 * Exits with an error message if a flag has no following value.
 */
public final class CliArgs {

    private final Map<String, String> values;

    private CliArgs(Map<String, String> values) {
        this.values = values;
    }

    public static CliArgs parse(String[] args) {
        Map<String, String> values = new LinkedHashMap<>();
        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            if (arg.startsWith("--")) {
                if (i + 1 >= args.length || args[i + 1].startsWith("--")) {
                    System.err.println("Error: " + arg + " requires a value");
                    System.exit(1);
                }
                values.put(arg, args[++i]);
            }
        }
        return new CliArgs(values);
    }

    public String get(String flag, String defaultValue) {
        return values.getOrDefault(flag, defaultValue);
    }
}
