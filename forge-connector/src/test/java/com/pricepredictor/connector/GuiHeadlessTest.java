package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** A worker has no event-dispatch thread, so what Forge defers to it runs inline. */
class GuiHeadlessTest {

    @Test
    void invokeInEdtNowRunsTheRunnable() {
        AtomicInteger runs = new AtomicInteger();
        new GuiHeadless("").invokeInEdtNow(runs::incrementAndGet);
        assertEquals(1, runs.get());
    }

    @Test
    void invokeInEdtAndWaitRunsTheRunnable() {
        AtomicInteger runs = new AtomicInteger();
        new GuiHeadless("").invokeInEdtAndWait(runs::incrementAndGet);
        assertEquals(1, runs.get());
    }

    @Test
    void invokeInEdtLaterRunsTheRunnable() {
        AtomicInteger runs = new AtomicInteger();
        new GuiHeadless("").invokeInEdtLater(runs::incrementAndGet);
        assertEquals(1, runs.get());
    }
}
