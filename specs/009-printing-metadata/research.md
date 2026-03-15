# Research: 009 Printing Data Fields

## R-001: AllPrintings.json Metadata Field Locations

**Decision**: Extract metadata from the per-set card objects in `data.<setCode>.cards[]`.

**Rationale**: Each card object within a set contains:
- `isReserved` (boolean, only present when `true`; absent = `false`)
- `rarity` (string: "common", "uncommon", "rare", "mythic", "special", "bonus")
- `legalities` (object: `{ "standard": "Legal"|"Banned"|"Restricted"|"Not Legal", ... }`)
- `printings` (list of set codes the card has been printed in, e.g. `["LEA", "LEB", "2ED", ...]`)
- `setCode` (string: the set this printing belongs to)

The `printings` list is the same on every printing of a given card (it's card-level, not printing-level). The `rarity` and `setCode` are printing-specific. The `isReserved` and `legalities` are card-level (same across all printings).

**Alternatives considered**:
- Using the MTGJSON `meta.cards` top-level object: Not available in the AllPrintings structure, which is set-keyed.
- Extracting metadata in a separate pass: Unnecessary overhead; it can be captured during the existing `build_name_to_uuids` traversal.

## R-002: Cheapest Printing Metadata Resolution

**Decision**: During `build_price_map`, track which UUID produced the cheapest price per card. Then join that UUID back to the card-in-set object to extract printing-specific metadata (rarity, setCode).

**Rationale**: The current `build_price_map` discards UUID identity after selecting the cheapest price. To retrieve printing-specific fields (rarity, setCode), we need to know which UUID was cheapest. Card-level fields (isReserved, legalities, printings) are the same regardless of which printing is cheapest.

**Implementation approach**:
1. Extend `build_name_to_uuids` to also return a `dict[str, dict]` mapping UUID → card metadata (rarity, setCode, isReserved, legalities, printings list).
2. Modify `build_price_map` to return `dict[str, tuple[float, str]]` — (price, cheapest_uuid) instead of just `dict[str, float]`.
3. New function `build_metadata_map` combines the two: for each card name, look up the cheapest UUID's metadata → produce `dict[str, PrintingData]`.

**Alternatives considered**:
- Loading AllPrintings.json twice (once for UUIDs, once for metadata): Wastes memory and time on a 536MB file.
- Storing all metadata in the Card entity during parse: Not possible — Card entities are parsed from `.txt` files which don't contain printing metadata.

## R-003: Text Enrichment Format

**Decision**: Append five lines at the end of the card text, using the existing `key: value` format:
```
reserved: false
rarity: rare
printings: 5
set: uma
legalities: standard, pioneer, modern, legacy, vintage, commander
```

**Rationale**: This matches the existing converted card text format (lowercase keys, colon-separated values). Appending at the end means existing text parsing is unaffected — the parser already skips unrecognized key:value lines. The transformer tokenizer sees additional tokens that encode printing context.

**Field format details**:
- `reserved`: `true` or `false` (string, lowercased boolean)
- `rarity`: lowercase rarity name (`common`, `uncommon`, `rare`, `mythic`, `special`, `bonus`)
- `printings`: integer as string (count of sets in the `printings` list)
- `set`: lowercase set code (e.g., `uma`, `2xm`, `lea`)
- `legalities`: comma-space separated list of format names where the card is "Legal", lowercased, from the 10 recognized formats. Empty string if legal in none.

**Alternatives considered**:
- JSON blob: Breaks the line-based format convention.
- Separate section header: Over-engineering; five flat lines suffice.
- Adding fields inline among existing fields: Risky — could confuse the ability-line regex or type parser.

## R-004: Sklearn Feature Engineering for Metadata

**Decision**: Add the following features to the sklearn dense feature vector:
1. `is_reserved` (binary: 0.0 or 1.0)
2. `rarity` (one-hot: 4 features for common/uncommon/rare/mythic, with rare as fallback for special/bonus)
3. `printings_count` (numeric: integer count)
4. `legalities_count` (numeric: count of legal formats)
5. Per-format legality (multi-hot: 10 features, one per recognized format)

Total: 1 + 4 + 1 + 1 + 10 = **17 new dense features**.

**Rationale**: These are all structured, low-cardinality features that fit naturally into the existing dense feature vector. Set code is intentionally NOT included as a dense feature because there are ~800 unique set codes — too high-cardinality for one-hot encoding and not ordinal. The set code is still in the text for the transformer to learn from via BERT tokenization, and indirectly contributes to the sklearn model via TF-IDF on oracle text (the set code line is not part of oracle text, so it won't appear in TF-IDF — this is acceptable since set code's price signal overlaps heavily with rarity + printings count).

**Alternatives considered**:
- Including set code as a feature: Too many categories (~800) for one-hot; frequency encoding adds complexity without clear benefit.
- Including legalities as a single comma-separated text feature in TF-IDF: Loses structural signal; multi-hot is cleaner.

## R-005: Prediction-Time Auto-Fill Architecture

**Decision**: Load a pre-built metadata lookup at server startup. At prediction time, parse the incoming card text, check for the 5 metadata fields, and fill missing ones.

**Implementation**:
1. `run_serve` in `cli.py` loads AllPrintings + AllPricesToday → builds metadata map → passes to `create_app`.
2. `create_app` stores the metadata map in `app.state.metadata_map`.
3. In the predict endpoint: after parsing the card text, check if metadata lines are present. If any are missing, look up card name in the metadata map. If found, auto-fill from cheapest printing. If not found, apply defaults.
4. Rebuild enriched text and pass to both models.

**Rationale**: Loading at startup amortizes the cost of parsing 536MB AllPrintings.json once. The metadata map is a simple `dict[str, PrintingData]` that's fast to look up (~20k entries, negligible memory).

**Alternatives considered**:
- Loading AllPrintings on every request: Unacceptable latency.
- Pre-enriching all card text files on disk: Would couple the conversion pipeline (Java) to the Python metadata logic. The conversion output should remain a clean representation of the card's game text.

## R-006: Recognized Formats List

**Decision**: Hard-code the 10 recognized constructed formats as a constant:
```python
RECOGNIZED_FORMATS = (
    "standard", "pioneer", "modern", "brawl", "legacy",
    "vintage", "pauper", "commander", "penny", "oathbreaker",
)
```

**Rationale**: The spec states these are stable and explicitly excludes online-only formats. A constant is simpler than a config file or CLI argument. If a new format needs to be added later, it's a one-line code change.

**Alternatives considered**:
- Configuration file: Over-engineering for a stable list of 10 items.
- CLI argument: Same — premature configurability.
