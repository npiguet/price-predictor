# March 8, 2026 — Card Converter Edge-Case Marathon

**TL;DR:** I spent essentially the entire day hunting and fixing
edge-case failures in the Java `CardScriptConverter`. The work was
driven by running the converter on the full card corpus and piping
the broken output back to Claude, one category of breakage at a
time. Sixteen commits landed before midnight.

The session opened in the middle of the night (00:05) with an
early push to use Forge's built-in classes instead of ad-hoc string
manipulation. I had pushed back on `formatRawCost` because past
experience had shown that hand-rolling cost formatting is fragile;
the same lesson applied to cost descriptions on spell abilities,
where I insisted we extract the proper domain object from the
`SpellAbility` rather than substring-ing `getDescription()`. Three
successive commits captured that refactor.

By 10:40 I had found that Saga cards were printing their chapter
triggers twice, and that the roman numerals were lowercase. The root
cause turned out to be an UNDEFINED keyword condition that was
re-emitting keyword-derived traits that had already been handled
elsewhere in the converter. I rolled back an over-broad fix when it
felt smelly, then asked Claude to keep only the Saga-specific
correction and drop the rest.

The late-morning block tackled one card archetype after another.
Class enchantments were outputting their level-up abilities in the
wrong shape, so I gave Claude a before/after example and asked it to
amend the spec. Planeswalker loyalty costs were not enclosed in
square brackets the way Oracle text is, and negative-X costs were
being lowercased — both wrong. I confirmed the Oracle format myself
and pushed for an exact match. Battle cards (which carry a
"defense" stat instead of power/toughness) were missing from the
output folder entirely, which meant the converter was silently
dropping them; a new `defense:` key fixed that. "Paw print" charm
cards (like Season of the Burrow) needed their chooseable-option
lines rendered with the paw-print cost symbol; they were first
broken, then briefly doubled, before the third attempt landed clean.

The afternoon block was about cost semantics. I noticed that cards
with "As an additional cost to cast" were not separating that cost
from the spell's effects, so I asked for an `additional cost:` line
above the ability lines, later trimming the prefix text so only the
cost description itself appears. Aether Tide and Analyze the Pollen
exposed gaps in that logic. Then came parameterized keywords: Gift
was printing its keyword line without describing *what* is being
gifted, so I asked Claude to audit all parameterized keywords and
confirm their parameters made it into the output. Companion reminder
text was leaking into the keyword line, which needed stripping.

The evening was the most conceptually interesting part. I realized
the converter was lumping together three distinct cost categories
that spell cards can have: *additional* costs (you must pay this on
top), *alternate* costs (pay this instead — Cleave, Ninjutsu), and
*cost reductions* (Convoke, Affinity). I pushed for them to be
separated into distinct output lines. Claude surfaced a nuance I
cared about: a static ability that reduces the cost of *other*
spells (not this one) should appear under a `static:` line, not a
`cost reduction:` line. That distinction required checking whether
the cost modification applies to the card that carries it, which
became its own commit. The final fix of the day silenced spurious
warnings left over from earlier passes.

Throughout, I pushed back consistently on code quality: no naked
conditionals or loops, use `Keywords` constants instead of string
literals, add null-guard comments naming at least one card that
requires them, and always test the new behavior. Each batch of fixes
was followed by a test expansion in `CardScriptConverterTest`, which
grew by hundreds of lines over the course of the day.

*Note: reconstructed from prompt history + git log; full session
transcripts were auto-deleted after 30 days.*
