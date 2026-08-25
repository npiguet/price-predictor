---
name: feature-workflow
description: Binding rules for this repo's three kinds of design document — experiment records (`experiments/*.md`), hand-written normative specs (`specs/YYYY-MM-DD-*.md`), and spec-kit feature dirs (`specs/NNN-name/`). Load BEFORE every create, edit, reword or reformat of any file under `experiments/` or `specs/`, however small — a one-sentence prose change and a single retitled header both count, because `experiments/*.md` carries a binding prose style that applies to each edit rather than only to new documents. Also load when asked to check, review or fix the wording, tone or structure of one of these files.
---

# Specs, experiments & the feature workflow

Three kinds of design document, each with a distinct audience, purpose, tone, and location. Keep them in their lanes — don't blur experiment rationale into specs, or benchmark numbers into specs.

## `experiments/*.md` — experiment / design records (ADRs)

- **Audience:** the user (and Claude) reasoning about *what happened* and *what to try next*.
- **Purpose:** record the outcome of a run (results, metrics, what worked / failed) and the design rationale for the next iteration — the "why" behind a feature. This is the **only** place for benchmark numbers, run logs, post-mortems, rejected alternatives, and cross-references to prior results.
- **Format:** a fluent, discursive ADR. Prose is fine; explain reasoning, trade-offs, and the mechanism behind a result. Each design doc carries an **Outcome / Result** section to fill in once the experiment runs. Named `YYYY-MM-DD-<topic>-design.md` (design/plan) or `YYYY-MM-DD-<topic>.md` (results).
- Convert relative dates to absolute. Link related docs.

### Prose style

**These rules are binding on every write to `experiments/*.md`, not advice to weigh.** They apply to a one-paragraph amendment as much as to a new document, to section headers as much as to body text, and to prose you are moving or lightly editing as much as to prose you are inventing. A section that already breaks them is not precedent — bring it into line while you are there. Once the edit is written, check it with the second pass below.

Discursive does not mean dense. These docs get re-read months later by someone who no longer holds the run in their head, so write for a reader who is skimming and will stop at the first sentence that costs effort.

- **Lead with the finding.** The first sentence of a paragraph states the conclusion. Corpus names, candidate labels, caveats and figures come after it. A reader who stops there should still have the result.
- **Headers state the finding too, not the topic.** "What separates and what does not" and "On the exploration band" name a subject and make the reader read on to learn the answer. "Only the `T = 3` field separates from the rest" is the same header carrying the result. A table of contents built from the headers should read as a list of conclusions.
- **Open a section with a sentence, not a table.** The table supports a claim that has already been made. A section whose first element is a table has made the reader derive the finding themselves.
- **One claim per sentence, in the order the reader would check it.** Short declarative sentences beat one sentence carrying three comparisons in subordinate clauses.
- **Unpack a chain of reasoning into one sentence per step, in causal order.** An argument with four steps is four sentences, not one sentence with three subordinate clauses. "A sampled frozen agent passes downstream cards a properly-playing agent would have kept, which weakens the training field in the dimension the yardstick tests, and transfer suffers for it" asks the reader to hold three things at once; the same argument as four sentences asks nothing. A reader should be able to stop after any step and still be following.
- **Use concrete nouns, not invented abstractions.** "What the two choices trade is run-control fidelity against colour discipline" names two things the reader has never met and cannot picture. "Field at argmax predicts the yardstick better; field at T drafts better" names what actually differs. If a phrase would need its own definition to be understood, write the thing it stands for.
- **Say the direction, not the tally.** "Three corpora out of four" makes the reader count; "both argmax candidates go off-lane more often than the seats beside them" tells them what happened. Give the score only after the direction.
- **Numbers live in tables; prose gets qualitative comparisons.** In the surrounding text prefer "nearly twice", "close to half the time", "the fewest in the sweep". Quote an exact figure in prose only when that specific figure is the point — a threshold, a sign flip, a value the argument turns on.
- **When a section is too long, cut claims, not figures.** A section that has outgrown its job is making claims it was never asked to make: a finer-grained restatement of a finding already given, evidence for something nobody disputes, a defence of a decision against an objection you raised yourself. Drop those whole. Never shorten by keeping every claim and thinning its numbers into "roughly" and "somewhat" — this is the only place the numbers exist, and a claim worth keeping is worth its figures. Length follows from how many things the section has to establish, which follows from what the section is for.
- **Don't rebut objections nobody raised.** Defending a choice against an imagined doubt adds a claim the section was not making, purely so it can be knocked down, and it produces the least readable paragraphs in these documents. "Mix balance is not what drives it. Both families here are evenly split and their biases differ by a factor of two…" answers a question no reader had asked. The urge is strongest right after you have been corrected on something, which is exactly when it is least useful. If a limit is genuinely load-bearing, state it in one sentence and stop; leave the rest to whoever actually objects.
- **Name a baseline the first time you lean on it.** Say what the reference columns are and how they were measured. Don't rely on possessives (`its references`) to carry a definition the reader has to reconstruct.
- **Call things by the names you gave them.** Once a label exists, use it every time. "The one that drafted its own pool" and "the same agent building its own decks" both mean `forge-full`, and each makes the reader resolve a description back to a name already on the page. Repeating the label reads as repetitive to the writer and as clarity to the reader; the reader is the one to serve.
- **Give each paragraph one job, and make consecutive paragraphs' relationship explicit.** If the second paragraph explains why the first is not an artefact, say so in its first clause.
- **Drop the rhetorical scaffolding.** Em-dash asides, "the exception is…", "the same split", callbacks to a previous paragraph's phrasing — these read as style and cost the reader a lookup. Cut them and restate the thing.
- **Name the mechanism; don't gesture at it with a figure of speech.** "A single game turns on the shuffle as much as on the decks, and a race to four averages most of that out" makes the reader decode two metaphors and reassemble three steps. "Which deck wins a single game depends heavily on what each player happens to draw. A match only ends when one deck has won four games, so most of that luck averages out" says the same thing and asks nothing. A sentence that sounds well-turned is the warning sign: check whether it is carrying an image in place of the mechanism.
- **Don't sprinkle emphasis.** Bold and italics are not for words that feel important. Bolding *forced* and *late* in a sentence whose whole subject is those two properties adds nothing, and scattered markup teaches the reader to ignore it. Reserve emphasis for a contrast the sentence genuinely turns on, a few times per document at most; if a point needs weight, give it its own sentence instead of bolding a word.

Plain is not casual. Sentences get short because each carries one step, not because the register drops. Keep the vocabulary technical and the tone level: cut conversational filler ("here's the thing", "it turns out", "its reasoning went like this"), don't narrate the writing as it happens, and don't slip into first-person cheerleading ("drafting better is what we want"). The target is a reader who never has to re-read a sentence, not one who is being talked to.

### The second pass

**Writing to one of these files is not done when the text is written.** Re-read what you just wrote as a separate pass, against the rules above, before reporting the edit as finished. Drafting and checking are different activities, and the check does not happen on its own: the failure mode is writing naturally, seeing that it reads well, and never testing it against the list.

Run the pass over what you changed rather than the whole document, and go looking for the things that most often survive a first draft:

- **The first sentence of every paragraph.** Does it state the finding, or does it name a subject and make the reader read on to get the result?
- **Every header you touched.** Same test.
- **Every figure in the prose.** Is it already in a table the reader can see? Then cut it and give the direction instead.
- **Every paragraph opening.** Does it start with "this", "that", "those" or a bare pronoun pointing back into the previous paragraph? Name the thing instead.
- **Every sentence longer than two lines.** Split it, or confirm it carries one claim.
- **Every word doing summary work in place of a mechanism** — "loosens", "tightens", "judgement", "the dynamic", "one scale up". Say what actually happens.
- **Every claim against the table beneath it.** Does the table show what the sentence asserts, including any trend, symmetry or monotonicity the wording implies?

The last check is not about style, and it is the one that catches errors rather than infelicities. A sentence that reads well and asserts a pattern the table does not contain is the most expensive thing you can leave behind. Do it anyway.

## `specs/YYYY-MM-DD-<name>.md` — root-level human-readable specs

- **Audience:** the user, to track and understand what is being built.
- **Purpose:** a **normative** specification — *what* to build and how it behaves from the outside (commands, CLI surface, contracts, records), based on the conclusions reached in the experiment doc. Stops short of an implementation manual.
- **Format & tone:** tight, direct, WHAT-not-WHY. Prefer **short bullet points and tables over long paragraphs**. Minimal, deliberate bold — only structural labels (list lead-ins, table columns, mini-headers), never mid-sentence emphasis. Keep rationale, gen-over-gen comparisons, and benchmark numbers **out** — those live in `experiments/` (link to them instead). Use timeless present tense.
- These are hand-written (not spec-kit), and are the source the speckit `spec.md` is derived from.

## `specs/NNN-name/` — spec-kit feature directories

- **Audience:** primarily Claude, to drive implementation (`spec.md` → `plan.md` → `research.md` → `tasks.md`).
- **Purpose:** the machine-workable spec that ultimately produces the code. Optimise these for implementation clarity, not for human browsing — do whatever is most useful for producing correct code (detailed FRs, acceptance scenarios, edge cases, cross-references).
- **Format:** follow the spec-kit templates. Invoke them via the `speckit.*` skills (`speckit.specify`, `speckit.clarify`, `speckit.plan`, `speckit.tasks`, `speckit.implement`) rather than editing the `.specify/` templates by hand. Numbered `NNN-name/`, next number = highest existing + 1.
- Before starting non-trivial work in an area, read the relevant `spec.md` / `plan.md` / `research.md`.

## The per-feature workflow

1. **Experiment doc** (`experiments/`) — record what happened in the last run and, based on those results, discuss and decide the next improvement.
2. **Root spec** (`specs/YYYY-MM-DD-*.md`) — write the normative, human-readable spec for that improvement, drawing its conclusions from step 1.
3. **Speckit** (`specs/NNN-name/`) — derive the spec-kit `spec.md` from the root spec and drive the implementation through `speckit.*`.
4. **Run & analyse** — run the experiment the spec enables, record the outcome back in the step-1 doc's Outcome section, and repeat from step 1.
