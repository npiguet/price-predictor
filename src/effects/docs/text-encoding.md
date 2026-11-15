# Ability text: sidecars, tokenization, vocabulary and the ability cache

Rules to respect when touching provenance sidecars, the tokenizer, the vocabulary, keyword definitions, the ability cache or the variant-script tree, and when running `extract-keyword-definitions`, `build-vocab` or `encode-abilities`.

- Sidecar contract: `specs/023-ability-effect-model/contracts/provenance-sidecar.md`.
- Code: `domain/ability_tokenizer.py`, `domain/keyword_formatting.py`, the sidecar cache, and Java `ProvenanceResolver` and `VariantSidecarMain`.

## Provenance sidecars

`<name>.provenance.json` sits beside each converted `.txt`, written by `python -m price_predictor convert`. It is the join between runtime Forge trait objects and converted ability lines.

- The key is printed provenance, `(script_file, face, trait_kind, index_within_kind, option?)`, never a converted-line ordinal. The converter runs one per-face counter across five mixed line kinds, emits keyword-derived lines first though Forge appends them last, and merges and deduplicates relative to the runtime objects.
- `script_file` includes its tree: `cardsfolder` is letter-keyed, `tokenscripts` and `variant-scripts` are flat, and one filename occurs in more than one.
- `option` is a charm mode's 0-based position in its root's `Choices$`, written only when set.
- A line can carry several keys, where it merged several traits.
- Where a keyword prints one line per trait it generates (a saga's chapter lines, haunt's two lines), a generated trait joins its own line alone. Sidecars written before 2026-11-14 gave every such line every generated trait, so chapter II and III records read chapter I's text: re-run `convert` on such a tree.
- `prose_offset` says where a line's prose starts after its `kind[n]: ` prefix. Do not cut at the first `": "`: a continuation line of a multi-line description has no prefix and can hold a `{T}: ` of its own. A sidecar written before the field existed is read by that cut, which is right for every line it holds.
- Role spans (`cost`, `effect`, `trigger-condition`, `target-spec`) are offsets into the prose after `prose_offset`.
- Row *i* of the ability cache corresponds to `lines[i]`.

How a key resolves, by case:

| Key | Result |
|---|---|
| In `dropped_keys` | A trait with no text of its own, such as the implicit permanent spell on every permanent. No line. |
| Inside a declared `(face, trait_kind)` range, in neither list | Fails loudly: the sidecar does not describe the card the record was collected against. |
| Outside every declared range | A trait Forge attached at runtime. No line; the record is kept. |
| Names a script file with no sidecar | Does not fail; the ability contributes no text. `SidecarCache` raises `UnconvertedScript` and tallies it per script file. |

- `build-corpus` refuses a resolution record acting only through dropped keys, except a permanent spell's, which it keeps keyed to the permanent's card.
- `build-corpus` refuses an in-range unjoinable record as `unjoinable-key` and fails only past 20 of them and 0.1% of the records read. A charm cast through Flashback produces a few legitimately.
- The no-sidecar case is mostly token entities: the collector files a token's provenance under `cardsfolder/<letter>/<sanitized token name>.txt` while tokens convert into the token tree (`tokenscripts/` beside `cardsfolder/` in a generation's `converted/`) under Forge's own script filenames. The trainer logs the tally; a conversion that broke would raise it without otherwise changing the shape of a run.
- The collector counts a trait's index over the list as the card prints it, not the live one. A line a static took away stays in its place, and a trait a type change attached (Blood Moon's Mountain mana ability on a nonbasic land) is counted past every printed line and flagged `granted`. Counted over the live list, such a trait would take a printed line's key.

## The text a line encodes to

- A sidecar line's `script_text` is the trait's whole chain (root segment, a trigger's `Execute$` ability, every sub-ability, a replacement's `ReplaceWith$` ability) joined by ` [SEG] `.
- Chain labels are renamed `SV1…` by first appearance per line, so two cards differing only in an author's SVar name share a text.
- A charm root carries its own segment, and each mode is an `option` line with its own chain and API type.
- The converted `.txt` must stay byte-identical; the connector's `golden-024` tests guard it.

## Tokenization

- There is one grammar, `tokenize`, for both surfaces. The surface is fixed at construction from `surface_of(vocab_path)`.
- A checkpoint records the tokenizer rules it trained under, and every loader picks the grammar through `tokenizer_surface`. A gen-1 checkpoint records none and is read with the prose grammar it trained on; the script rules would turn about one of its script tokens in thirty into `[UNK]`.
- On the script surface camel case splits and `$` is always its own token. These stay whole: the values of `Execute$`, `SubAbility$`, `RepeatSubAbility$`, `ReplaceWith$` and `CounterType$`, segment openers, and `Choices$` items of the five chooser APIs (`Charm`, `GenericChoice`, `AssignGroup`, `VillainousChoice`, `Vote`). Elsewhere `Choices$` is a selector.
- A keyword line expands by its display name, filled through `domain/keyword_formatting.py`, which mirrors Forge's per-`Keyword.type` formatting.
- Validation, evaluation and the cache expand at `INFERENCE_KEYWORD_EXPAND_P = 0`, so only unknown keywords expand there.

## Keyword definitions

- The keyword-definition file `extract-keyword-definitions --output` writes is keyed by display name, because that is what a card's keyword line and the converted text both spell.
- The encoder expands an unknown keyword unconditionally. This file is what makes a set the vocabulary predates readable rather than a bare `[UNK]`.
- Both surfaces expand a keyword to its reminder template. `generated_script` is captured but neither scanned into a vocabulary nor used by any expansion; it is null for the engine-coded minority.

## Vocabulary

- `build-vocab` scans three sources: converted cards, converted token scripts and the keyword-definition file. Unscanned reminder text would expand to `[UNK]`.
- `[PAD]`, `[UNK]`, `cardname`, `[MASK]`, `[CLS]` and `[SEG]` are seeded at low ids so `--target-size` cannot truncate them away. On the script surface so are `sv1…svN`, every template word and every key part, and the build fails when a definition would expand to `[UNK]`.
- Set codes are not seeded: an ability line never names a set.
- The surface is read from the vocabulary's file name and from nothing else (`ability_encoder.surface_of`): a name containing `vocab-script` is the script surface, any other the prose surface. A script vocabulary passed with `--vocab-path` must keep such a name (`vocab-script.txt`), else every command reading it tokenizes on the prose surface. The two surfaces' vocabularies are two files, so a script-surface rebuild never overwrites the vocabulary a prose-surface checkpoint recorded and can still load against.

## Ability cache

- The cache directory (`encode-abilities --output-root`, a `cache` run named after the checkpoint's run) holds under `<tree>/…` one `.npz` per source file, a `float32` array `(n_lines, e_dim)` under the key `e`, row-aligned with that source's sidecar.
- `encode-abilities` computes it once, offline; every downstream consumer reads the cache rather than the encoder.
- `--variant` resolves the checkpoint and the output suffix together, so a baseline is never encoded with another variant's weights and never lands on the shipping cache. A variant writes `<name>.{variant}.npz` beside the shipping file.
- The `taxonomy` variant has no encoder and emits a hash embedding into the same row layout, so every geometry check reads one file shape.
- The vocabulary and keyword-definition hashes are checked against the checkpoint before anything is written.

## Variant scripts

- The variant tree (`--variant-scripts`, one `inputs` run directory every variant run of a generation extends) holds the perturbed Forge source scripts, flat. The filename is the path a variant's provenance key resolves to, so it cannot be derived from the source file's stem. `loadable.lst` is the only file in the tree that is not a variant script.
- Each variant gets its sidecar from `VariantSidecarMain`, which runs the ordinary converter with the text output suppressed. A key's `index_within_kind` is the trait's position in Forge's runtime trait list, so only the parser that builds that list can number it the way a record will.
- The tree has no prose surface: `SidecarCache.prose_for` returns None for it, and the prose surface falls back to the script.
- Variants are staged into Forge's custom-cards directory before `FModel.initialize`. The card database is read once at startup, and a script that appears afterwards is invisible for the life of the JVM.
- Staging registers every name with `VariantRegistry`, which is what makes `ProvenanceKey` say `variant-scripts/`. Once loaded, a variant is an ordinary non-token card and nothing on it says otherwise.
