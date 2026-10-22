package com.pricepredictor.connector;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class ScriptHeaderTest {

    @Test
    void readsTheFrontFaceOnly() {
        ScriptHeader header = ScriptHeader.of(List.of(
                "Name:Front",
                "Types:Creature Human",
                "Oracle:Front text.",
                "ALTERNATE",
                "Name:Back",
                "Oracle:Back text.",
                "Draft:Back draft."));
        assertEquals("Front", header.name());
        assertEquals("Front text.", header.oracle());
        assertEquals(List.of(), header.draftLines());
    }

    @Test
    void keepsTheFirstNameAndOracleAndEveryDraftLine() {
        ScriptHeader header = ScriptHeader.of(List.of(
                "  Name: Cogwork Librarian ",
                "Draft:Draft CARDNAME face up.",
                "Oracle:First.",
                "Oracle:Second.",
                "Draft:Second draft line."));
        assertEquals("Cogwork Librarian", header.name());
        assertEquals("First.", header.oracle());
        assertEquals(List.of("Draft CARDNAME face up.", "Second draft line."),
                header.draftLines());
    }

    @Test
    void aScriptWithoutHeaderLinesHasNone() {
        ScriptHeader header = ScriptHeader.of(List.of("CopyFaceFrom:Bind", "AlternateMode:Split"));
        assertNull(header.name());
        assertNull(header.oracle());
        assertEquals(List.of(), header.draftLines());
    }
}
