# Handoff: resume the gen-3 collection review (2026-11-09)

Read `README.md` in this folder first: why the review exists, the process, the status table and every
user decision so far. This file says exactly where the review stopped and how to continue it. The fixing
round (implementation) does not start in the next session: only the review continues.

## Where it stopped
- Done and verified, findings committed:
  - `connector-findings.md` (forge-connector: four slices);
  - `forge-hooks-findings.md` (Forge `effect-record-hooks`: hooks and copier).
- Decided by the user and written into the gen-3 specs (commit 71c7e03b): the record-moments principle.
  Each record's board is read just before the moment it records (root spec § 8.14, delta contract,
  spec.md FR-042a–f, tasks.md phase 12).
- Interrupted: the review of the Python side of collection. Its reviewer was stopped before reporting,
  so relaunch it with the prompt below.

## Next steps, in order
1. Relaunch the Python collection review (prompt below), then verify its findings with a fresh
   adversarial verifier. Write the verified findings, with evidence and the test for each, to
   `python-collection-findings.md` here. Update the README's status table, commit, and report to the user
   in plain text. Never use the question pop-up tool. Ask what to do next.
2. With the user's go: review `src/effects` beyond collection (corpus build and curation, training,
   evaluation, probes, readers) and `rules/` plus `scripts/`. These only read records. The README allows
   them to run during the re-collection, so ask the user whether to do them now or then.
3. Then the fixing round, in a fresh session, from the README and tasks.md phase 12 (T179–T181).

## How each module review runs (keep to this)
- Split a large module into slices. Each slice goes to a fresh general-purpose subagent on Opus (never
  Fable; at most 3 Opus agents at once) with `review-brief.md` (plus `review-brief-forge.md` for Forge)
  and `review-hints.md`.
- Reviewers are read-only: no builds, tests, JVMs or pytest. They do not read `~/.claude`,
  `experiments/`, `diaries/` or git history. Bug hunting comes first; the `/review` rubric second, kept
  brief.
- Save each reviewer report verbatim-in-substance to the session scratchpad. Then send it to a fresh
  Opus verifier told to refute each finding against the code, the Forge source and the Python readers.
  The verifier gives per finding: verdict, decisive evidence, training impact, frequency, and the
  smallest test for NEEDS-TEST. Verifiers may count real shards (read-only scripts in the scratchpad).
- Write only verified results into this folder's findings files, in the same layout as
  `connector-findings.md`, then commit. Check both repos for stray files afterwards.

## Relaunch prompt for the Python collection review
Use the general brief, with this slice description:

> Your slice is the PYTHON SIDE OF COLLECTION: the supervisors that launch Forge worker JVMs, pass them
> their settings, choose the decks and cards they play, and decide when a collection round ends. Files
> (under `src/`):
> - `sealed/infrastructure/effect_collection_flags.py` (the shared collection flags, startup checks,
>   -D properties, dropped-events report), `sealed/application/match_outcomes.py`, and the
>   effects-related parts of `sealed/infrastructure/cli.py` (the match-outcomes command and its flags);
> - `price_predictor/infrastructure/forge_jvm.py` (classpath, JVM launch, ForgeWorkerPool: restarts,
>   recycling, status, stop conditions);
> - `effects/application/collect_coverage.py`, `collect_variants.py`, `collector_common.py`,
>   `holdout_cards.py`; `effects/domain/collection_caps.py`, `coverage.py`, `script_variants.py`,
>   `text_holdout.py`; `effects/infrastructure/collector_connector.py`, `deck_file.py`,
>   `one_shot_workers.py`; and the CLI wiring of collect-coverage, collect-variants and holdout-cards
>   in `effects/infrastructure/cli.py`.
>
> Their tests are under `tests/unit` and `tests/integration`.
>
> Hunt, beyond the brief's list:
> - Settings that reach the JVM wrong:
>   - a flag whose -D property name or default differs between Python and Java (Java side:
>     `forge-connector/.../effects/CollectionCaps.java`, `WorkerProperties.java`);
>   - a flag parsed but never passed;
>   - a startup check that passes a run it should refuse, or refuses a good one;
>   - a default that silently disables a record kind.
> - Corpus contamination and holdout leaks:
>   - held-out cards or texts reaching decks they must not;
>   - coverage or variant decks leaking into `match-outcomes.txt` or `cards-played.txt`;
>   - a variant printing a held-out text;
>   - the holdout computed with flags that differ from build-corpus's.
> - Run control:
>   - worker restarts and recycling losing or duplicating work;
>   - run ids reused;
>   - rounds that never end or end early;
>   - progress counted from the wrong file;
>   - two runs writing one directory;
>   - resume logic.
> - Deck construction for coverage and variants: decks that fail to load silently, weighting that
>   starves or over-samples, retirement charged to the wrong cards.
>
> Report findings with file:line evidence (Python and Java), severity and confidence as the brief
> defines, and the smallest test that would confirm each.

## Session facts the next session needs
- The machine is idle: no collection runs, no JVMs. The old full-strength run (7c8cb070) was stopped by
  the user and will not be resumed; everything is collected again from scratch after the fixes.
- Forge `../forge` is on `effect-record-hooks` at 04e72e0518e. The connector fat JAR is built from
  2acdc5d7 (later commits are docs only). Nothing is pushed.
- The fork fidelity audit (`ForkFidelity`, `fork fidelity:` lines in the worker logs) is on in every
  collection. The last audit run showed about 5% of forks differing; see the table at the end of
  `connector-findings.md`.
- User preferences that apply here:
  - plain-text questions, never the question pop-up;
  - collection bugs matter more than code style;
  - the user wants everything found before re-collecting;
  - commit freely, never push unless asked;
  - heavy jobs one at a time (reviews are read-only and fine in parallel).
