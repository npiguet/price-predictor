---
description: Review existing code for complexity, readability, and missed domain concepts
---

This is a read-only review. Edit nothing until the user has read the findings and said which ones to act on:
the review is input to a decision about what to refactor, and that decision is theirs.

## Task

Review the code specified by: $ARGUMENTS

If no specific file or module is given, explore the project structure first and identify the most complex or
critical areas to review, then focus your review there.

## What this review is for

The reviewer this command stands in for cares most that code is easy for a human to read and hold in mind. The
preferences behind the checklist below:

- Long methods, and methods with many branches and loops, are a smell in themselves, because a reader cannot keep
  them whole in working memory.
- A concept that exists in the problem domain should exist in the code as a named type. Look for the ones that
  were missed.
- A formal data structure such as a tree or a graph is preferred over ad-hoc nested loops or recursion that
  encode the same structure implicitly.
- One change that removes several findings at once is worth more than several one-off fixes, so look for the
  pattern across instances before proposing a fix for each.
- Logic that an existing dependency already provides should use the dependency, not a manual reimplementation.

## What to look for

The bold headline of each item below is the rule; the bulleted examples are illustrative, not exhaustive.
Flag any code that matches the headline even if it doesn't fit one of the listed shapes, and don't force-fit
findings into a category just because an example matches superficially.

**Long methods or functions** that are hard to hold in working memory. Suggest extracting named sub-methods
when you see:
- A function over ~50 lines, or one that needs scrolling to read end-to-end
- Multiple distinct phases (load → transform → validate → save) that could each become a named helper
- Mixed levels of abstraction in one body (high-level orchestration interleaved with low-level parsing)

**Excessive branching or nesting** (high cognitive complexity). Examples to flag:
- More than 2–3 levels of nested `if`/`for`/`try`
- Long `if/elif` chains dispatching on a type, set code, or string — usually a polymorphism, dict-dispatch,
  or `match` opportunity
- Boolean parameter explosion (`def foo(card, *, normalize: bool, with_metadata: bool, dry_run: bool)`) —
  two or more functions hiding inside one
- Deep nesting that would flatten with guard clauses / early returns

**Duplicate or near-duplicate code.** Examples to flag:
- The same 3+ lines repeated in two or more places
- Two functions whose bodies differ only by a constant or a one-line operation (extract a parameter)
- Test setup copy-pasted across many tests (pull into a fixture)
- Subtle copies that have already drifted out of sync — a strong signal the abstraction is overdue

**Procedural code that belongs in a domain object** (apply DDD thinking). Examples to flag:
- Free functions that always take the same bundle of primitives — that bundle wants to be an object
- Logic operating on an entity's fields from the outside (`if card.power is not None and card.toughness is not
  None and card.power > card.toughness: ...`) when it belongs as a method on `Card`
- Anemic data classes whose validation, parsing, or derived properties live in the calling code
- This codebase has a clear `domain/` layer with entities and value objects — new domain logic that landed in
  `application/` or `infrastructure/` is a relocation candidate

**Missing domain concepts:** unnamed things that exist but have no type or class. Examples to flag:
- A tuple, dict, or list-of-strings used to represent a thing that has a name in the problem domain
  (mana cost, printing key, deck archetype, set rotation, price bucket)
- Magic numbers/strings without a named constant or enum (rarity codes, build-method tags like `forge-best`,
  bucket boundaries)
- Repeated string parsing/formatting (split on `;`, split on `|`) that screams "value object" — the
  match-outcome line and pool line formats are good candidates if they're parsed inline

**Comments that describe WHAT the code does rather than WHY** — prefer a well-named method over a comment.
Examples to flag:
- A comment that summarizes the next 5 lines (turn it into a function whose name is that summary)
- Block comments labeling sections of a function (`# --- preprocessing ---`) — those are phase boundaries
  begging to be extracted methods
- Stale comments referencing removed code, renamed classes, or obsolete TODOs (recommend deletion)
- Keep comments that explain a non-obvious invariant, a constraint from outside the codebase, or a workaround
  for a specific bug — those earn their keep

**Null and absence handling (Java):** Prefer empty collections over null. Return `Optional` when absence is meaningful.
Avoid null checks where `Optional` map/flatMap/ifPresent would read more clearly.

**None and absence handling (Python):** Prefer empty collections or empty strings over `None` where the caller
won't distinguish. Where absence is meaningful, say so in the type as `T | None`, the form this codebase uses.
Avoid `if x is not None` chains where a guard clause or early return would be cleaner.

**Dead code:** Flag code that is no longer reachable or referenced and recommend deletion (don't comment it out
— git history is the archive). Look for:
- Unused functions, classes, methods, imports, parameters, and private fields
- Branches whose condition is always true or always false (including feature flags pinned to one value)
- `if __name__ == "__main__"` blocks or CLI subcommands that nothing invokes
- Code paths guarded by configuration that no longer exists
- Leftover scaffolding from earlier refactors (TODO stubs, "old_*" or "*_v2" siblings of the real implementation)
- Test helpers and fixtures that no test uses anymore

Search for usages (`ruff`, then a grep over the whole repository) before declaring something dead. Public API
surfaces (anything imported by another package, exposed via the CLI in `infrastructure/cli.py`, or used by
`forge-connector` across the subprocess boundary) need extra care — search the Java side too.

**Low-value tests:** A test earns its keep by catching real regressions. Recommend deleting tests that:
- Assert on getters/setters or trivial constructors with no logic
- Re-implement the production code in the test body and assert it equals itself (tautological tests)
- Mock so heavily that they only verify the mocks are called in the order the test set up
- Exercise the standard library or a third-party framework rather than our code
- Duplicate coverage already provided by another test (often a unit test made redundant by an integration test)
- Are pinned snapshots that no one reads and that get blindly regenerated on every change
- Pass regardless of the behaviour under test (no meaningful assertion, or assertion on a value the test itself just set)

Distinguish "trivial" from "important boundary" — a one-liner test on a parser edge case can be high value.
When in doubt, ask whether the test would catch a plausible bug; if not, suggest removal.

**Simple performance problems:** Look only for low-effort, high-confidence wins — not micro-optimization.
Examples to flag:
- Quadratic scans where a `set`/`dict` lookup would do (e.g. `if x in some_list` inside a loop over many items)
- Repeated work hoistable out of a loop (compiling the same regex, re-opening the same file, recomputing an
  invariant value, repeated attribute lookups in a tight Python loop)
- N+1 patterns: per-iteration disk reads, subprocess calls, HTTP requests, or DB queries that could be batched
- String concatenation in a loop where `"".join(...)` (Python) or `StringBuilder` (Java) is the idiom
- Loading a whole file/dataset into memory when streaming or chunked reads suffice (`ijson` is already a
  dependency for MTGJSON)
- Loading a model, tokenizer, or `AllPrintings.json` inside a request handler or per-card loop instead of once
  at startup
- Pandas/NumPy: row-by-row `.iterrows()` / `.apply()` where a vectorized operation exists
- PyTorch: missing `torch.no_grad()` on inference paths; `.item()` / `.cpu()` / `.tolist()` inside a hot loop
  that forces a GPU→CPU sync per element; per-card forward passes where batching is possible
- Sorting or building large intermediate collections only to take the first/last element

Skip speculative perf concerns ("this could be slow at scale"). Only flag when the fix is small and the cost
pattern is visibly present in the code being reviewed.

## Output format

Start with a short summary of the overall state of the reviewed code, then list the findings, most valuable
first. Each finding has:

- Location (file and line range)
- The problem observed
- A concrete suggestion for improvement, including any refactoring pattern or dependency to consider
- Severity: how much the problem costs a reader or a maintainer
- Confidence: how sure you are that it is a real problem and that the suggestion is right

Report every finding, including the ones you are unsure about or consider minor, and let the severity and
confidence say so. The user filters; a finding left out is one they never get to weigh. If a finding needs
more context to resolve, say what is missing.

End by asking the user which findings they want to act on.