---
name: maintaining-claude-md
description: Use when creating, editing, trimming or splitting any CLAUDE.md or a topic file it points to (`src/*/docs/*.md`), and when a code change renames, removes or changes the behaviour of something one of those files names — a command, flag, file, class, file format or rule. Also use when a CLAUDE.md has grown large or reads like documentation.
---

# Maintaining CLAUDE.md and its topic files

A module's `CLAUDE.md` is loaded whole every time a session touches that module, so every line in it is paid
for on every such session. It therefore holds only what a session cannot get from the code, the specs or the
experiment records: the rules that must hold, and where to find the rest.

## Where each kind of content lives

| Content | Home |
|---|---|
| A rule that must hold, and what breaks when it does not | the module `CLAUDE.md` if it applies to any work in the module, else a topic file |
| A fact no single source file states, because two languages or two packages share it (a file format, a join key, flag signatures) | a topic file, or the `CLAUDE.md` when other modules point at it |
| A command that cannot be guessed (required flags, build order, environment setup) | the `CLAUDE.md` |
| What the code does: flags and defaults, what a run logs, exit codes, how a mechanism works | the code and `--help`; the file names the source file at most |
| What a feature specifies | `specs/`; the file names the spec and section |
| Why a design was chosen, benchmark numbers, rejected alternatives | `experiments/`; the file names the record |
| How a defect was found and fixed | nowhere; git history holds it. Keep only the one clause of reason its guard needs |
| The meaning of a domain term | the module's `GLOSSARY.md` or the root one |

## Shape of a module CLAUDE.md

In this order:

1. Entry point and one paragraph on what the package is, with its import rule.
2. A table of topic files: `Read | Before`. Each row names one file under `<module>/docs/` and the concrete
   situation that calls for it ("before reading or writing shards", not "if needed").
3. One line per subcommand, naming its topic file.
4. Rules that apply to any work in the module.
5. File formats other modules point at, or a pointer to the topic file that holds them.

Name a topic file by plain path in backticks. Never write `@path`: that form loads the file with the
`CLAUDE.md` and defeats the split.

Split a `CLAUDE.md` into topic files when it passes about 10,000 characters, or when a block serves one
activity only (running a collection, changing a trainer).

## Shape of a topic file

It opens by saying what it is for and where the description, the specification and the rationale live. The
rest is rules, grouped under topic headers.

Each rule is one bullet:

- It states what must hold first, then what breaks otherwise, in a clause or a second sentence.
- It names the file, class or function involved.
- It is written in the present tense, as if the current rule had always been the rule. No "now", "no longer"
  or "used to".
- It carries no incident numbers and no account of how the problem was found.

A table is the right form for a cross-language fact with several cases: envelope members, the flag signature
of each record kind, how a key resolves.

## Writing or rewriting a rule

- Carry over only what the existing text or the code establishes. A consequence you inferred yourself is
  checked against the code before it is written, or left out.
- When removing rationale, confirm an experiment record carries it; if none does and a guard depends on it,
  keep one clause of it beside the guard.
- State each rule once. A second file that needs it points to the first.
- Delete a sentence when a reader could get the same from `--help` or one source file.

## Keeping the files in sync with the code

A change updates these files in the same commit when it does any of the following:

| Change | Update |
|---|---|
| Adds a subcommand | its one line in the `CLAUDE.md`, and its topic file |
| Adds a topic file | its row in the `CLAUDE.md` table |
| Renames or removes a file, class, function, flag or command | every place these files name it |
| Changes a behaviour a rule describes | that rule |
| Removes the reason for a guard | the guard |
| Renames a section header | every `file.md § Section` reference to it, in other modules too |
| Changes a file format | the format's one home, and nothing else |

## Checks before finishing

Run these on every file you touched:

- Every path, class, function and script the file names exists. Search for each; do not trust memory.
- Every `file.md § Section` reference names a section that exists.
- The `CLAUDE.md` table and the files under `docs/` list the same topic files.
- Nothing that was the only copy of a rule or a format was deleted.
- `forge-connector/CLAUDE.md` and the other modules' files still point at sections that exist.
