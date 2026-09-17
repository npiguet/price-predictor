# Audit: every token-script access in the collector (2026-09-16)

Scope: Task 3 of the `2026-09-16-token-script-keys` plan, following the FR-150
fix (`ProvenanceKey.tokenScriptStem` — Tasks 1/2). Enumerates every site that
touches a token, a token script, or the `tokenscripts` tree, and checks each
against the shared resolver (`ProvenanceKey.scriptFileOf` on the Java side,
`SidecarCache` with a `tokenscripts` root on the Python side).

## Commands run

```bash
grep -rnE 'isToken\(|PaperToken|tokenscripts|TOKENSCRIPTS|getAllTokens|TokenInfo|token_script_id|_token' forge-connector/src/main/java --include=*.java | grep -vE ':[0-9]+:\s*(//|\*|/\*)'
grep -rnE 'tokenscripts|token_script|_token\.txt' src/effects --include=*.py | grep -v tests
```

(Re-run against the current tree, the Java command above reproduces exactly the fifteen
non-comment hits the table below collapses into fourteen rows — `SnapshotBuilder.java`'s
`509` and `510` share one row. Line numbers in the table were re-checked against this run
and updated where the javadoc rewrite in this fix wave shifted them.)

## Java (`forge-connector/src/main/java`)

| site | what it does | through the shared resolver? | verdict |
|---|---|---|---|
| `ConvertMain.java:43` `tokensPath = defaultSibling(cardsPath, "tokenscripts")` | default token-source directory next to `--cards-path` | n/a — CLI directory wiring, not a per-card lookup | consistent by construction |
| `ConvertMain.java:46` `tokensOutputPath = defaultSibling(outputPath, "tokenscripts")` | default token-output directory next to `--output-path` | n/a — same | consistent by construction |
| `ConvertMain.java:63` `SourceTree.TOKENSCRIPTS` | tags the batch conversion of the tokens directory with its tree | uses `SourceTree` constant | consistent by construction |
| `ConvertMain.java:64` `report("tokenscripts", tokens)` | prints the conversion summary label | cosmetic | consistent by construction |
| `ConvertMain.java:67` `"  tokenscripts: skipped, no such directory (...)"` | prints when no tokens directory is found | cosmetic | consistent by construction |
| `ProvenanceKey.java:19` `import forge.item.PaperToken;` | import for `tokenScriptStem` | — | shared resolver (Task 1/2) |
| `ProvenanceKey.java:592` `CardFilenames.scriptFileForStem(SourceTree.TOKENSCRIPTS, stem)` | `scriptFileOf` files a token under the token tree once `tokenScriptStem` returns a stem | is the resolver | shared resolver |
| `ProvenanceKey.java:628` `if (paper instanceof PaperToken token)` | first path: a paper token's own image filename | is the resolver | shared resolver |
| `ProvenanceKey.java:630` `else if (host.isToken())` | second path: a `TokenInfo`-rebuilt token (forked board) reads its image key | is the resolver | shared resolver |
| `ProvenanceKey.java:638` `!StaticData.instance().getAllTokens().containsRule(stem)` | gates the stem against the token rules DB before it is trusted (FR-150) | is the resolver | shared resolver |
| `SnapshotBuilder.java:509-510` `"token_script_id": card.isToken() ? card.getName() : null` | writes the corpus's `token_script_id` field from the printed **name**, not the script stem | does not call `scriptFileOf`/`tokenScriptStem` | **needs decision** — see below |
| `SourceTree.java:46` `TOKENSCRIPTS` constant | names the flat token tree | is part of the resolver's vocabulary | consistent by construction |
| `SourceTree.java:50` `isFlat(tree)` includes `TOKENSCRIPTS` | tree-layout predicate `scriptFileOf`/`CardFilenames` rely on | is part of the resolver's vocabulary | consistent by construction |
| `PlayedCardCollector.java:108` `boolean isToken = card.isToken();` | filters tokens out of the played-card-name log (`shouldRecord`) | no script/path lookup at all | not token-script related; note and move on (same shape as the `isBasicLand` filters called out in the brief) |

### `token_script_id` — needs decision (not changed here)

`SnapshotBuilder.entityToJson` writes `token_script_id` as `card.getName()` — the
printed display name — not `tokenScriptStem(card)`. The created-objects
vocabulary (FR-078) is meant to be keyed by token-script id, but a printed name
collapses distinct scripts that render under related names, e.g. `b_2_2_zombie`,
`b_2_2_zombie_decayed`, and `b_x_x_zombie` (all "Zombie Token" print variants)
would share one vocabulary entry today.

**Recommendation:** change `SnapshotBuilder.entityToJson` to write
`ProvenanceKey.tokenScriptStem(card)` (falling back to `null` for a non-token,
same as today) instead of `card.getName()`. This is a corpus-schema change
(the `token_script_id` field's value space) and a created-objects vocabulary
change, both of which need a decision from the user and a migration/versioning
plan for existing collected records — out of scope for this audit. Per the
task ruling, no code change is made for this row; it is recorded here only.

## Python (`src/effects`)

| site | what it does | through the shared resolver? | verdict |
|---|---|---|---|
| `application/build_corpus.py:371` `cards_folders: tuple[str, ...] = ("output/cardsfolder", "output/tokenscripts")` | `build-corpus` default cards-folders | lists `output/tokenscripts` | consistent; confirms FR-096 |
| `application/build_vocab.py:35` `Path("output/tokenscripts/")` in `DEFAULT_CARDS_FOLDERS` | `build-vocab` default | lists `output/tokenscripts/` | consistent; confirms FR-096 |
| `application/collect_coverage.py:48` `Path("output/tokenscripts/")` in `DEFAULT_CARDS_FOLDERS` | `collect-coverage` default | lists `output/tokenscripts/` | consistent; confirms FR-096 |
| `application/encode_abilities.py:38` `Path("output/tokenscripts/")` in `DEFAULT_CARDS_FOLDERS` | `encode-abilities` default | lists `output/tokenscripts/` | consistent; confirms FR-096 |
| `application/evaluate_effect_model.py:680` `Path("output/tokenscripts/")` in `DEFAULT_CARDS_FOLDERS` | `evaluate-effect-model` default | lists `output/tokenscripts/` | consistent; confirms FR-096 |
| `application/train_effect_model.py:752` `Path("output/tokenscripts/")` in `DEFAULT_CARDS_FOLDERS` | `train-effect-model` default | lists `output/tokenscripts/` | consistent; confirms FR-096 |
| `domain/event_schema.py:148` `EventType.TOKEN_CREATED: ("token_script_id", "characteristics", "count")` | declares the `token_script_id` param name for a `TOKEN_CREATED` event | schema declaration, not a path lookup; the value it carries is whatever `SnapshotBuilder`/`record_io` put there | consistent by construction (mirrors the Java field; inherits the same needs-decision value above, not a separate issue) |
| `domain/provenance.py:22-24` (comment) | explains `cardsfolder` letter-keyed vs. `tokenscripts`/`variant-scripts` flat | documentation | consistent by construction |
| `domain/provenance.py:25` `SOURCE_TREES: tuple[str, ...] = ("cardsfolder", "tokenscripts", "variant-scripts")` | the tree vocabulary `SidecarCache` roots are checked against | is part of the resolver's vocabulary | consistent by construction |
| `domain/state_snapshot.py:207` `token_script_id: str | None = None` | the `EntityState` field the record schema carries | passthrough field, not a path lookup | consistent by construction (inherits the same needs-decision value) |
| `infrastructure/cli.py:35` `DEFAULT_CARDS_FOLDERS = ("output/cardsfolder/", "output/tokenscripts/")` | shared CLI default (contracts/cli.md) | lists `output/tokenscripts/` | consistent; confirms FR-096 |
| `infrastructure/record_io.py:189` `"token_script_id": entity.token_script_id` | serializes the field to JSON | passthrough | consistent by construction |
| `infrastructure/record_io.py:244` `token_script_id=data.get("token_script_id")` | deserializes the field from JSON | passthrough | consistent by construction |
| `infrastructure/sidecar_io.py:209-210` (docstring) `{"cardsfolder": ..., "tokenscripts": Path("output/tokenscripts")}` | documents `SidecarCache.__init__`'s `roots` mapping | is the Python-side shared resolver | consistent by construction / shared resolver |

Every `cards_folders` / `DEFAULT_CARDS_FOLDERS` default across the six CLI
entry points above lists `output/tokenscripts/` (FR-096) — no default is
missing it.

## Observations carried over from Tasks 1/2 reviews

| site | what it does | verdict |
|---|---|---|
| Endure hotfix tokens `w_2_2_spirit` / `w_3_3_spirit` | These tokens register in the token rules DB under key `w_x_x_spirit`, not their own stem. After the FR-150 `containsRule` gate (`ProvenanceKey.java:638`), `w_2_2_spirit`/`w_3_3_spirit` fail the check and `tokenScriptStem` returns null, so `scriptFileOf` falls back to a `cardsfolder` path derived from the printed name instead of filing under `tokenscripts`. | needs change in principle, but **pre-existing and harmless today**: `cardsfolder/w/w_x_x_spirit.txt` is the wrong tree, but that script has no abilities, so no ability text is lost. Left as-is; flagging for whoever next touches Endure-hotfix tokens. |
| Food token `c_a_food_sac` | **CLOSED.** Renders two printed keys (`spell#0` and `spell#1`) for what is one scripted ability, reproduced on both the mainline board and the FR-150 forked-board test. `output/tokenscripts/c_a_food_sac.provenance.json` lists `{"face":0,"trait_kind":"spell","index_within_kind":0}` under `dropped_keys` and carries exactly one line, `{"face":0,"trait_kind":"spell","index_within_kind":1}` (the `GainLife` line, script text `AB$ GainLife \| Cost$ 2 T Sac<1/CARDNAME/this token> \| LifeAmount$ 3 \| SpellDescription$ You gain 3 life.`). So `spell#0` is the dropped key and `spell#1` is the one line — both keys join by design, per the sidecar's own join rule (`src/effects/infrastructure/sidecar_io.py:10-17`: a key in `dropped_keys` resolves to no line and is kept because the trait is still live at runtime, while the surviving key resolves to the rendered line). Nothing to fix. | closed; verified by reading the sidecar directly. |

## Follow-up (parked)

Items raised by this audit and the whole-plan review that are worth doing but are
out of scope for a cheap fix here:

- An unknown `t:` stem (a token image key naming a stem the token database does
  not know — including the Endure hotfix pair above) should yield an unresolved
  reason from `scriptFileOf`/`tokenScriptStem` rather than silently falling
  through to a fabricated name-derived `cardsfolder` path. Today it is
  indistinguishable from a real card whose printed name happens to match.
- `scriptFileOf` should prefer `SourceTree.relativePathIn(SourceTree.TOKENSCRIPTS,
  rules.getPath())` for a `PaperToken`, the same way it already prefers
  `CardRules.getPath()` for a printed card, rather than deriving the token's
  filename from its image key alone. This closes the Endure hotfix pair
  generically — `rules.getPath()` names the file Forge actually read
  (`tokenscripts/w_x_x_spirit.txt`) regardless of what the image key or the
  rules-DB key happen to be — instead of requiring a one-off fix per
  differently-keyed token family.
- `containsRule` (the token-rules-DB gate `tokenScriptStem` checks the stem
  against) is case-insensitive, but the path `scriptFileOf` emits from a stem
  is not case-normalized. A stem differing from the rules-DB key only in case
  passes the gate today and could still emit a path that does not match the
  file on disk.
- `isRealToken()` is the predicate the `scriptFileOf` javadoc means when it
  talks about a token copy of a real permanent (`isToken()` is true for that
  case too, but the card's abilities are still the printed card's, not a
  token script's) — worth using by name wherever that distinction is drawn in
  prose or in code, on merged permanents in particular.

## Step 3 — fixes made

None. No hit found is a one-line default missing `output/tokenscripts/`
(FR-096) — all six Python CLI defaults already list it — and no hit found
derives a token path from a name instead of calling `scriptFileOf` /
`tokenScriptStem` (the one name-based site, `SnapshotBuilder.entityToJson`'s
`token_script_id`, is a corpus-schema/vocabulary decision, explicitly excluded
from a cheap fix by the task ruling, and is recorded above as a
recommendation only). Nothing in this audit qualifies for the cheap-fix path,
so no code change accompanies this note.
