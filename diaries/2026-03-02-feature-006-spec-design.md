# March 2, 2026 — Designing the Card-Script Conversion Spec

**TL;DR:** I spent the session designing the specification for feature
006, which converts Forge card scripts into an LLM-friendly text format.
The work was entirely spec-and-clarification: no code was written and no
commits landed. By the end of the night I had driven through most of the
speckit lifecycle — clarify, specify, plan, analyze, tasks — and kicked
off implementation.

The session opened just after midnight with `/speckit.clarify` and a
prompt that framed the core motivation: the oracle text field on a card
is just an aggregated dump of all the card's abilities and replacement
effects. I wanted those abilities broken out individually, each as its
own line, so that the structure of the card would be visible to an LLM
rather than buried in a blob of text. That meant the oracle text field
itself would disappear from the model; the collection of typed ability
lines would replace it.

From there I had a series of design decisions to make, each encoded into
the spec via follow-up prompts. The most consequential were: the output
format should stay as close as possible to the Forge card script style
(key–value pairs, the ALTERNATE separator for multi-face cards) since
that format would eventually be fed to an LLM as input for features 008
and 009; player-activatable abilities should carry a unique number
(e.g., `activated[1]:`, `spell[2]:`) so a future AI agent can refer to
them unambiguously; all keyword prefixes should use the full word
("activated", not "A") so that the same word in rules text and in a
type prefix resolves to the same token for the LLM; and CARDNAME should
remain as a literal uppercase placeholder rather than being replaced by
the card's actual name, because the `name:` property already supplies
it and CARDNAME acts as a consistent self-reference token.

I also reversed a decision mid-session: an initial answer had excluded
batch processing, but I then re-read the question and realized I had
misunderstood — I did want each card in its own file, mirroring the
Forge cardsfolder directory structure exactly.

The late-night block, starting around 10:52 pm, shifted from first-pass
clarification to refining the spec details. I pushed through several
rounds of `/speckit.specify` amendments: ability-counter numbering as an
amendment to FR-004, using ability-style lines for keyword costs (e.g.,
`keyword[2]: Cycling {2}`), converting cost parameters to oracle-text
format (`{1}{B}`, `pay 2 life`, `{T}`, `sacrifice a creature`), and
keeping loyalty costs in oracle-text bracket notation (`[N]`, `[0]`,
`[-N]`). A further amendment locked in that CARDNAME must not be
substituted and that all `K:` keyword lines, whether player-activated or
not, collapse to a single `keyword:` prefix in the output.

After each amendment I asked Claude to commit the step, but none of
those commits appear in the git log for this date — they likely landed
just after midnight on March 3, or the commit commands themselves were
deferred. The session closed with `/speckit.implement`, meaning I had
driven the spec all the way through plan, analysis, and task generation
and handed it off to start implementation, all in a single sitting.

*Note: reconstructed from prompt history + git log; full session
transcripts were auto-deleted after 30 days.*
