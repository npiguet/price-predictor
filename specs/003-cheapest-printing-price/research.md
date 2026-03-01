# Research: Cheapest Printing Price for Training

**Feature**: 003-cheapest-printing-price
**Date**: 2026-03-01

## R-001: Existing Cheapest-Price Logic

**Context**: The spec requires selecting the cheapest price across all printings of a card. Before designing new code, we need to understand what already exists.

**Finding**: The cheapest-price selection logic already exists in `src/price_predictor/infrastructure/mtgjson_loader.py`:

- `build_price_map()` (line 80–115): Iterates over all UUIDs for each card name, collects prices from both `normal` and `foil` Cardmarket retail finishes, picks the latest date's price for each finish, and returns `min(all_prices)`. This already implements FR-001 through FR-003.
- `get_cheapest_price()` (line 49–77): Standalone function with the same logic for a single card's UUIDs.
- Paper-only and English-only filters are applied upstream in `build_name_to_uuids()` (lines 28–36), which already satisfies FR-002's constraint that only English paper printings are compared.
- FR-004 (ignore missing prices) is already handled — `prices_data.get(uuid, {})` returns empty dicts for missing UUIDs, and the `if price and price > 0` guard skips null/zero prices.
- FR-005 (exclude cards with all-missing prices) is already handled — `if all_prices:` only adds cards with at least one valid price.

**Decision**: Extend the existing functions rather than rewriting. The core selection logic is correct; only the price floor and logging are missing.

**Alternatives considered**:
- Rewrite price loading from scratch → Rejected: existing code is well-structured and tested.
- Move cheapest-price logic to domain layer → Rejected: price selection is a data normalization step at the ingestion boundary; it doesn't involve domain entities.

---

## R-002: Price Floor Implementation (€0.01)

**Context**: FR-006 requires a price floor of €0.01. The current code has `if price and price > 0` which filters out €0.00 prices before they even enter the comparison. The spec says €0.00 should participate in comparison but be clamped to €0.01 after selection.

**Finding**: The current filter `if price and price > 0` is doing two things:
1. `if price` — guards against `None` (Python truthiness: `None` is falsy, `0` is also falsy)
2. `price > 0` — excludes zero and negative prices

Both conditions inadvertently exclude €0.00 printings from comparison. The spec says these should be included (the cheapest price might be €0.00, which then gets clamped to €0.01).

**Decision**: Change the condition to `if price is not None and price >= 0`. This allows zero-price printings to participate in the comparison. After computing `min(all_prices)`, apply `max(min_price, 0.01)` as the floor.

**Rationale**: The floor prevents `log(0)` errors during training (log-transform on line 99 of `train.py`). Using `>= 0` instead of `> 0` follows the spec literally: "if the cheapest price is below €0.01 (including €0.00), it is clamped to €0.01."

**Alternatives considered**:
- Apply floor in `train.py` instead of `mtgjson_loader.py` → Rejected: the floor is a data normalization concern, not a training concern. Applying it at the data loading boundary keeps the training pipeline agnostic to price edge cases.
- Keep `> 0` and add a separate "zero price with floor" path → Rejected: unnecessarily complex. Changing the condition to `>= 0` and adding a floor is simpler.

---

## R-003: Multi-Printing Transparency Logging

**Context**: FR-007 requires a summary log of how many cards had multiple printings. FR-008 requires per-card logging of the selected price and alternative count.

**Finding**: The existing `build_price_map()` function already iterates over all card names and their UUIDs, collecting `all_prices` for each. The data needed for FR-007/FR-008 is available during this iteration:
- `len(all_prices) > 1` → card had multiple price points (from different printings/finishes)
- `min(all_prices)` → the selected price
- `len(all_prices) - 1` → number of alternative prices not selected

The existing logging pattern in the codebase uses `logger.info()` with human-readable messages to stderr.

**Decision**: Add logging directly inside `build_price_map()`:
- FR-008: Inside the per-card loop, when `len(all_prices) > 1`, log the card name, selected price, and alternative count.
- FR-007: After the loop, log a summary line with the count of cards that required price selection.

This keeps the change localized and doesn't require changing the function's return type.

**Rationale**: Logging within the function is the simplest approach. The function already logs start/end messages. Adding per-card and summary logging is consistent with the existing pattern (see `forge_parser.py` which logs per-5000-cards progress).

**Alternatives considered**:
- Return a richer data structure (e.g., `dict[str, PriceSelection]`) from `build_price_map()` and log in `train.py` → Rejected: requires changing the function signature and updating all callers. Over-engineered for a logging requirement.
- Return `tuple[dict[str, float], dict[str, int]]` (price map + printing count map) → Rejected: same coupling concern. Logging in-place is simpler.
- Create a domain event/callback pattern → Rejected: violates Simplicity First principle for a logging requirement.

---

## R-004: Impact on `get_cheapest_price()` Standalone Function

**Context**: `get_cheapest_price()` is a standalone function separate from `build_price_map()`. It has the same `price > 0` logic. Should it also get the floor and logging?

**Finding**: `get_cheapest_price()` is used in tests and could be used by other callers. It should be consistent with `build_price_map()` regarding the price floor.

**Decision**: Apply the same `>= 0` condition change and €0.01 floor to `get_cheapest_price()`. Do NOT add multi-printing logging to this function (it operates on a single card's UUIDs, not the full training set — FR-007/FR-008 are training-time concerns).

**Rationale**: Consistency between the two functions prevents confusion. The floor is a data integrity rule that should apply everywhere, not just during training.

**Alternatives considered**:
- Leave `get_cheapest_price()` unchanged → Rejected: would create inconsistent behavior between the two paths to the same data.

---

## R-005: Impact on `TrainingExample` Entity Validation

**Context**: `TrainingExample.actual_price_eur` has a validation `> 0` (strictly greater than zero). The new floor of €0.01 means the minimum possible price is 0.01. Does the entity need updating?

**Finding**: `TrainingExample` is defined in `entities.py` (line 72–81) with validation `actual_price_eur <= 0 → ValueError`. Since the floor is €0.01, and 0.01 > 0, the existing validation is compatible. Additionally, `TrainingExample` is never actually instantiated in the current training pipeline — `train.py` uses raw `list[float]` for prices.

**Decision**: No change to `TrainingExample`. The existing `> 0` validation is correct and compatible with the €0.01 floor.

**Alternatives considered**:
- Change validation to `>= 0.01` → Rejected: unnecessarily couples the entity to the specific floor value. The entity's contract is "positive price," which is a more stable invariant than "price >= current floor constant."

---

## R-006: Test Fixture Updates

**Context**: The existing `allprices_sample.json` test fixture needs entries to test the new price floor behavior.

**Finding**: The current fixture has prices ranging from €0.05 (Island) to €115.00 (Jace foil). There are no zero-price or sub-cent entries. Edge cases from the spec that need test coverage:
1. A card with a €0.00 price (should be clamped to €0.01)
2. A card with multiple printings at different prices (already covered — Grizzly Bears, Lightning Bolt, etc.)
3. A card where the cheapest printing has a very low price like €0.005 (sub-cent — if present in data)

**Decision**: Add new UUID entries to `allprices_sample.json` for:
- A printing with price €0.00 (tests the floor clamping)
- A printing with a very small positive price below €0.01 (tests sub-cent floor clamping)

Also add corresponding entries to `allprintings_sample.json` if new card names are used.

**Rationale**: Testing edge cases with real fixture data is more reliable than mocking. The existing fixture pattern uses static JSON files, so extending them is straightforward.
