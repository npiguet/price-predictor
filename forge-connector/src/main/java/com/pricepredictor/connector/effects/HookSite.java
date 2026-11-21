package com.pricepredictor.connector.effects;

/**
 * Where a hook's method is declared: the Forge class, by its binary name, and
 * the method's name.
 *
 * <p>The name is the whole contract. The collectors look the method up by
 * string, so a rename degrades a channel silently rather than failing a build.
 *
 * @param owner  the Forge class it is declared on, as {@link Class#forName} takes it
 * @param method its name
 */
record HookSite(String owner, String method) {

    /** The owner's simple name and the method's, as the report and the failure counts name a hook. */
    @Override
    public String toString() {
        return owner.substring(owner.lastIndexOf('.') + 1) + "." + method;
    }
}
