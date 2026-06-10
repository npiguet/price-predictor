# May 28, 2026 — Code review and DDD stabilisation

**TL;DR:** I ran two sequential code-review passes on the `sealed`
pipeline — first a general stabilisation pass, then a DDD-focused
pass — and had Claude implement all approved findings and commit them.

The day started with me asking for a general review of the Python
codebase, noting that several features had been added since the last
cleanup and things probably needed a stabilisation round. Claude read
through the source tree and came back with five findings, all
concentrated in `sealed` where the newer picker and scorer features
had been bolted on over time. The four substantive ones were: the
40-card deck assembly steps duplicated verbatim across `build_decks`,
`pick_decks`, and `evaluate_scorer`; the generated-decks line format
written inline in two places that had already drifted from each other
(no `format_generated_deck` counterpart to the parser); the CLI
resume-resolution logic nearly duplicated between `run_train_scorer`
and `run_train_picker`; and some bare integer literals in
`feature_engineering` where named width constants already existed. I
approved all five and had Claude implement and commit them. 676 unit
tests passed.

With the structural duplication gone, I asked for a second pass
focused specifically on DDD — missing entities, value objects, domain
concepts that might be useful to introduce. Claude came back with five
findings again. The ones I approved were: a `SealedPool` type (pools
were bare `(set_code, cards)` tuples while their sibling
`GeneratedDeck` was already typed); a shared delimited-grammar helper
module so the `;` / `|` parsing wouldn't drift between readers —
Claude also caught a latent bug here, where `match_data_loader` was
missing the empty-field guard the other three readers had, which would
produce `[""]` instead of `[]` on an empty pipe-list; a `Deck` type
scoped to the IO and use-case layer (the tensor-facing code stays on
`list[str]`); a `Side` enum with a `match_winner()` function to
replace the triplicated `wins_a > wins_b` winner logic; and unifying
the `WUBRG` / `COLOR_ORDER` constant that was defined in two places.
All five were implemented, ruff was clean, and I asked Claude to commit
and run the integration tests. Those passed too: 32 passed, 1 skipped.

The one finding Claude explicitly did not recommend was a `Color` enum
— the concept is correct, but it touches about eight cross-package
sites in performance-sensitive code, so the churn wasn't judged worth
it. I didn't push back on that call.

The empty-field bug in `match_data_loader` was the only latent defect
found; everything else was about giving names to concepts that already
existed implicitly in the code.
