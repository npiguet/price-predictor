package com.pricepredictor.connector;

import forge.util.Lang;
import forge.util.Localizer;

import java.lang.reflect.Field;
import java.util.Collections;
import java.util.Enumeration;
import java.util.ResourceBundle;

/**
 * Shared Forge initialization — used by both ConvertMain and tests.
 */
public class ForgeEnvironmentInitializer {
    private static volatile boolean initialized = false;

    public static synchronized void initialize(String langDir) {
        if (initialized) return;
        Lang.createInstance("en-US");
        try {
            Localizer.getInstance().initialize("en-US", langDir);
        } catch (Exception e) {
            // Language directory not available (e.g. in test environment) —
            // inject a key-as-value ResourceBundle so Forge internals don't NPE
            installDummyBundle();
        }
        initialized = true;
    }

    private static void installDummyBundle() {
        try {
            Localizer localizer = Localizer.getInstance();
            ResourceBundle dummy = new ResourceBundle() {
                @Override
                protected Object handleGetObject(String key) { return key; }
                @Override
                public Enumeration<String> getKeys() { return Collections.emptyEnumeration(); }
            };
            setField(localizer, "resourceBundle", dummy);
            setField(localizer, "englishBundle", dummy);
        } catch (Exception ignored) {
        }
    }

    private static void setField(Object target, String fieldName, Object value) throws Exception {
        Field f = target.getClass().getDeclaredField(fieldName);
        f.setAccessible(true);
        if (f.get(target) == null) {
            f.set(target, value);
        }
    }
}
