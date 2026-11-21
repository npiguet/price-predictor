package com.pricepredictor.connector.effects;

/** What a listener does with one callback, once its arguments are there. */
@FunctionalInterface
interface ListenerBody {
    void on(Callback callback, Args args);
}
