---
name: codescene-refactoring
description: Use when asked about CodeScene, code health scores, hotspots or technical debt in this project; when finding the worst-scoring or most complex files; when refactoring a file to raise its CodeScene score (to 10.0 or otherwise); when a review mentions Bumpy Road, Complex Method, Complex Conditional, Large Method, Deep Nested Complexity, Excess Number of Function Arguments, Low Cohesion or Lines of Code in a Single File; or before running `cs review`, `cs delta` or calling api.codescene.io.
---

# CodeScene refactoring

CodeScene scores each file 1–10 (10 = no smells). The cloud analysis ranks the whole project; the
local `cs` CLI scores any file on disk in a second. Raise a score by refactoring until `cs review`
lists no smell, while a characterisation run proves behaviour unchanged byte for byte.

## Access (all verified)

- Project: codescene.io project **87853**. Token: env var `CS_ACCESS_TOKEN` (never print it; it
  is set at user level, so a shell started by Claude Code sees it).
- CLI `cs` (CodeScene CLI 1.0.x) is on PATH. The MCP server is deliberately not installed: it
  duplicates the CLI and API and costs context tokens every session.

### REST API — the project-wide view

```bash
H=(-H "Authorization: Bearer $CS_ACCESS_TOKEN" -H "Accept: application/json")
curl -s "${H[@]}" https://api.codescene.io/v2/projects/87853                  # code_health, hotspot_code_health
curl -s "${H[@]}" https://api.codescene.io/v2/projects/87853/analyses/latest  # revision analysed, file counts
curl -s "${H[@]}" "https://api.codescene.io/v2/projects/87853/analyses/latest/files?page_size=200&page=1"
```

- `files` is paged: read `max_pages` from page 1 (`page_size=200` → 7 pages). Each file has
  `path` (prefixed `price-predictor/`), `code_health.current_score` (a number, or `"-"` when not
  scored), `lines_of_code`, `hotspot`, `change_frequency`, `code_health_rule_violations`.
- `order_by` does **not** sort the result: download every page and sort yourself.
- `/analyses/latest/hotspots`, `/technical-debt/files`, `/code-health` → 404. Use `files`.
- The cloud analysis is of a past commit (`analysis_repo_revisions[0].revision`). Before naming
  the worst files, run `git diff --stat <that revision> HEAD` and re-score with `cs review` every
  changed file among the candidates: a file refactored since then still ranks at its old score.
- "Worst" means lowest score unless the user says otherwise. When they ask what is worth fixing,
  weigh in `hotspot` and `change_frequency`: a low score in a script nobody edits costs little.

Ranking, lowest first (`S` = a directory in the scratchpad):

```bash
B="https://api.codescene.io/v2/projects/87853/analyses/latest/files?page_size=200"
pages=$(curl -s "${H[@]}" "$B&page=1" | python -c "import json,sys; print(json.load(sys.stdin)['max_pages'])")
for p in $(seq 1 "$pages"); do curl -s "${H[@]}" "$B&page=$p" -o "$S/files_$p.json"; done
python -I -c "import json,glob,sys; r=[(f['code_health']['current_score'],f['path']) for g in glob.glob(sys.argv[1]+'/files_*.json') for f in json.load(open(g))['files'] if isinstance(f['code_health']['current_score'],(int,float))]; [print(*x) for x in sorted(r)[:10]]" "$S"
```

### CLI — scoring files on disk

| Command | Use |
|---|---|
| `cs review path/to/file.py` | score + smells with function names and lines |
| `cs review 'scripts/pkg/*.py' --output-format jsonl` | one JSON line per file: `{"score", "review", "path"}`; `score` null = no scorable code |
| `cs review master:./path/to/file.py` | the score at a branch or commit: the "before" of a refactor |
| `cs review --include-descriptions file.py` | each smell's explanation |
| `cs delta [--file f]` | did uncommitted changes make code health worse |

- One file: `--output-format json`. **`jsonl` crashes on a single file** (a Clojure
  `ArityException` stack trace, exit code still 0); it works only on a glob or several files.
- Human output starts with a **blank line**: grep for `score`, don't take `head -1`.
- All CLI output is cp1252 on Windows (descriptions hold non-ASCII): decode as `cp1252`, not UTF-8.
- File-level smells (Number of Functions, Overall Code Complexity, Lines of Code) list no
  function: they clear only by splitting the file.

## Python thresholds (defaults; no `.codescene/` rules file in this repo)

| Smell | Flagged at | Keep each function at |
|---|---|---|
| Complex Method | cyclomatic complexity ≥ 9 | ≤ 8 |
| Large Method | ≥ 70 lines | ≤ 60 (margin) |
| Excess Number of Function Arguments | ≥ 5 (`self` not counted) | ≤ 4 |
| Deep, Nested Complexity | nesting depth ≥ 4 | ≤ 3 |
| Bumpy Road Ahead | ≥ 2 bumps (a bump = one block of conditionals/loops nested ≥ 2 deep; blocks separated by flat code count separately) | ≤ 1 |
| Complex Conditional | an `if`/`elif`/`while` test with ≥ 2 `and`/`or` operators | ≤ 1 operator |
| Lines of Code in a Single File; Number of Functions in a Single Module | either may fire for a big file (790 LoC flagged as LoC; 3,200 lines with 107 functions flagged as function count instead); limits unknown | modules ≲ 300 LoC and ≲ 25 functions, then re-score |
| Overall Code Complexity | high mean cc over the file | follows from small functions |
| Low Cohesion | functions sharing no data or calls (LCOM4) | group by what they share |

What counts, in Python:
- `x.get("k") or ()` **is an `or`**: `if a and "c" in (p.get("t") or ())` is a Complex
  Conditional. Move the defaulting into a helper (`def removes_creature(e): return "creature" in (...)`).
- Lambdas, comprehension `if`s and `and`/`or` in a `return` all add to the **enclosing**
  function's complexity. A dict of 40 lambdas is one huge Complex Method: make each a named
  module-level function, or a factory (`player_flag("monarch")`) where several differ by a name.
- Complex Conditional only reads branch tests: `return a and b and c` is fine there (still cc).
- Two `for … if …` blocks in one function are 2 bumps: extract one of them.

## Method

1. **Score the target**: `cs review file` — the smell list is your checklist.
2. **Safety net before any edit.** If the file has no tests that pin its behaviour, build a
   characterisation run in the scratchpad:
   - Feed it real inputs, small enough to run in seconds (e.g. rebuild input shards from the
     committed test fixtures). Capture every output file, stdout/stderr and exit code.
   - Run the **unchanged** code twice and diff: fix every nondeterminism first (gzip headers hold a
     timestamp — compare decompressed bytes; logs echo output paths — normalise the run label).
   - Where the code dispatches over a table of predicates or handlers, also snapshot, per input
     record, **every** entry that matches — selection hides a predicate that changed.
   - Exercise every mode/flag path (append modes, generations) and the argument errors.
   - Images (matplotlib PNGs) may not be byte-stable: if the two unchanged runs differ, compare
     decoded pixels instead. Redirect output directories to the scratchpad, never over committed
     outputs.
   - Import-time side effects (`matplotlib.use("Agg")`, rcParams, `sys.path` edits) must still run
     first after a split: `ruff check --fix` may reorder `# noqa: E402` imports, so re-run the
     characterisation after it.
3. **Refactor**, re-running the characterisation after each step. Done = byte-identical output.
   - Long function → small helpers; a stateful loop → a small class holding the loop's state
     (one method per step); a function with 5+ arguments → a dataclass for the ones that travel
     together.
   - Nested `if` in a loop → guard clauses, `next((x for x in xs if p(x)), None)`, or `any(...)`.
   - Too big a file → a package beside it (`scripts/foo.py` + `scripts/foo_parts/`), the entry
     script kept at its path with its CLI and docstring. Run by path, `scripts/` is on
     `sys.path`, so `from foo_parts.x import y` works (`tests/unit/scripts/conftest.py` does the same).
4. **Re-score every resulting file**: all must be 10.0 (`__init__.py` with only a docstring reports
   "No scorable code"). A split that moves smells into a new module has not fixed anything.
5. `ruff check --fix` (import order), the tests that touch the area, then commit.

## Order and access patterns are behaviour

A refactor that "only restructures" still breaks these; keep them exact, the characterisation
run is what catches them:
- **First-match tables**: `{**gen3, **gen2, **shared}` dicts where a record lands in the first
  category it fits — the key order is the behaviour. Rebuild the dict in the same order.
- **`defaultdict` key order**: `chosen[name]` creates `name` on first *read*; later iteration
  (summaries, output order, stable-sort ties) follows when each key was first touched. Keep reads
  in the same order and the same short-circuit positions.
- **Short-circuit order**: a reordered `and` can raise `KeyError` where the original never read
  the key.

## Common mistakes

- Trusting the cloud ranking for a file changed after the analysed revision.
- Guessing API endpoints or sort parameters: use the ones above.
- Declaring done when the target scores 10.0 but a new module doesn't.
- Comparing gzip files byte for byte (header timestamps differ every run).
- In a worktree, three `run_manifest` tests fail (`test_effect_embedding_probes`,
  `test_knowledge_probe_compare`, `test_effect_sweep_arm`): the shared `.venv` resolves the repo
  root to the main checkout. Not caused by a refactor; the user has chosen to leave them.
