---
name: feature-workflow
description: How to write and place this repo's three kinds of design document — experiment records (experiments/), hand-written normative specs (specs/YYYY-MM-DD-*.md), and spec-kit feature dirs (specs/NNN-name/). Load before writing or amending any of them.
---

# Specs, experiments & the feature workflow

Three kinds of design document, each with a distinct audience, purpose, tone, and location. Keep them in their lanes — don't blur experiment rationale into specs, or benchmark numbers into specs.

## `experiments/*.md` — experiment / design records (ADRs)

- **Audience:** the user (and Claude) reasoning about *what happened* and *what to try next*.
- **Purpose:** record the outcome of a run (results, metrics, what worked / failed) and the design rationale for the next iteration — the "why" behind a feature. This is the **only** place for benchmark numbers, run logs, post-mortems, rejected alternatives, and cross-references to prior results.
- **Format:** a fluent, discursive ADR. Prose is fine; explain reasoning, trade-offs, and the mechanism behind a result. Each design doc carries an **Outcome / Result** section to fill in once the experiment runs. Named `YYYY-MM-DD-<topic>-design.md` (design/plan) or `YYYY-MM-DD-<topic>.md` (results).
- Convert relative dates to absolute. Link related docs.

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
