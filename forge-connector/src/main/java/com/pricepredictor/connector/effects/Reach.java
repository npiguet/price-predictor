package com.pricepredictor.connector.effects;

import java.lang.reflect.Method;

/**
 * Where a hook's method is looked up: among its owner's public methods, or
 * among its own declared ones, made accessible, for the one private method
 * the patch adds ({@link PatchHooks#NOTE_CONFIRM}).
 */
enum Reach {
    PUBLIC, PRIVATE;

    Method[] candidates(Class<?> owner) {
        return this == PUBLIC ? owner.getMethods() : owner.getDeclaredMethods();
    }

    Method reachable(Method method) {
        if (this == PRIVATE) {
            method.setAccessible(true);
        }
        return method;
    }
}
