# Wednesday, 1 April 2026 — Java refinement and Stage 2 launch

**TL;DR:** Spent most of the day running repeated `/review` cycles on the
forge-connector Java module, progressively raising its type safety until null
was essentially gone from the domain. Capped the evening by implementing the
entire Stage 2 heuristic-gate trainer in one commit.

The day started by adding the `/review` command itself — a slash command that
gives Claude a "rigorous senior developer" persona and a structured workflow
for reviewing and then planning fixes. The first review session targeted the
forge-connector module generally: mutable title strings, mixed null/Optional
patterns, hand-rolled JSON parsing in `PricePredictorClient`. That produced
commits for Gson, a shared `CliArgs` parser, `KeywordFields` for colon
splitting, and a new `ForgeParams` constants class to name all the Forge
sentinel strings that had been literals scattered across the code.

From there the sessions kept narrowing on one specific problem: the module was
full of null checks that were either redundant (because Forge's own APIs
guarantee non-null) or actively spreading null through otherwise clean Optional
chains. I asked Claude to trace Forge source code to determine what could
actually return null. That analysis found exactly three unnecessary guards
across the entire codebase — `getSVar()` returns `""` not null, `getTitle()`
on keyword implementations is always non-null, `getOriginal()` always set.
Three guards deleted, 212 tests still green.

The deeper observation was that scattered null checks were a symptom of missing
type-level enforcement. The response was to introduce `NonBlankString` — a
record whose constructor trims and rejects blank input, with `of()` returning
`Optional<NonBlankString>` for boundary crossing and `require()` for
known-safe sites. `AbilityDescription.normalize()` was changed to return
`Optional<NonBlankString>`, which propagated the type through all 16 ability
records, `CardFace` fields, `SaDescription`, and `SpellEffect`. The commit
touched 30 files but the result was that every place where a string was
guaranteed non-blank now said so at the type level instead of with a guard.

The `KeywordFields` discussion surfaced a useful constraint: Forge keyword
fields are inherently positional — field 1 means something completely
different in `Haunt:SvarName` versus `Craft:costPart:typeDesc` versus
`Class:N:...`.
There is no shared schema, so a named-field API would have needed that schema
to come from somewhere. The conclusion was that `KeywordFields` stays
positional, and per-keyword logic stays where it is. Two methods
(`keywordName()` and `parseAll()`) were added speculatively and then removed
after it turned out only one of them had any real call sites; `parseAll()` was
later restored when a `from(ki, Integer.MAX_VALUE)` smell was identified.

Late in the day the focus shifted to spec 013 (Stage 2 heuristic-gate). The
tasks were generated (34 tasks across 6 phases), the spec analysis found a
gap where FR-002 "encoder MUST stay frozen during Stage 2" had no test task
covering it, and that was added explicitly. Then the full Stage 2 implementation
landed in a single evening commit: a `mana_scorer` domain module (pip counting,
ideal distribution, source counting, reward shaping), a `TrainStage2UseCase`
with a PPO loop and mana-score reward override, a `SampleStage2UseCase`, CLI
routing for `--stage 2 / --init-from`, and test fixtures corrected to match
actual production card format. 181 tests passing at commit.

The review command itself got two small iterative improvements during the day
based on noticing gaps in the initial prompt.
