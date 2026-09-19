"""Build the per-unique-text table every probe in this directory reads.

Walks every provenance sidecar under ``output/cardsfolder`` and
``output/tokenscripts``, joins each line to its row of the shipping ability
cache (``<name>.npz``) and of the taxonomy baseline's cache
(``<name>.taxonomy.npz``), and keeps one row per unique encoding text on the
script surface. For each text it records the hand-parsed script features
(API type, line kind, trigger/static mode, parameter-key presence, cost,
target, amounts, who is named as affected, duration, length), the facts of the
first card carrying it (type line, mana value, colours), how many cards carry
it, whether the curated corpus holds it out, and the corpus's rarity count.

It also writes ``cache/keymap.pkl``: every provenance key (including the
perturbed-variant tree's) mapped to its encoding text, which
``effect_profiles.py`` uses to name a record's acting text without building a
``SidecarCache`` per worker.

It checks, and reports, that every card carrying a text got the same vector
for it: the cache is computed per card, so a disagreement would mean the
encoder is not a function of the text.

Run: ``python scripts/effect_embedding_probes/build_texts.py`` (CPU, ~2 min).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ABILITIES,
    CACHE,
    CARD_TYPES,
    OUT,
    ROOT,
    TEXT_TABLE,
    TREES,
    affected_features,
    card_facts,
    cost_features,
    count_amounts,
    load_manifest_sets,
    parse_script,
    target_type,
)

from effects.domain.ability_encoder import SURFACE_SCRIPT, encoding_text  # noqa: E402
from effects.infrastructure.sidecar_io import (  # noqa: E402
    converted_text_path,
    prose_for,
    prose_lines,
    read_sidecar,
)

VARIANT_TREE = ROOT / "output" / "effects" / "variant-scripts"


def key_tuple(script_file: str, key) -> tuple[str, int, str, int]:
    return (script_file, key.face, key.trait_kind, key.index_within_kind)


def main() -> None:
    held_out, rarity = load_manifest_sets()
    rows: dict[str, dict] = {}
    keymap: dict[tuple, str] = {}
    mismatch_max = 0.0
    skipped = {"no_cache": 0, "row_mismatch": 0, "no_text": 0}
    for tree in TREES:
        for sidecar_path in sorted((ROOT / "output" / tree).rglob("*.provenance.json")):
            sidecar = read_sidecar(sidecar_path)
            rel = Path(sidecar.script_file)
            base = ABILITIES / rel.parent / rel.stem
            full_path = base.with_suffix(".npz")
            tax_path = base.parent / f"{rel.stem}.taxonomy.npz"
            if not full_path.exists() or not tax_path.exists():
                skipped["no_cache"] += 1
                continue
            full = np.load(full_path)["e"]
            tax = np.load(tax_path)["e"]
            if len(full) != len(sidecar.lines) or len(tax) != len(sidecar.lines):
                skipped["row_mismatch"] += 1
                continue
            converted = converted_text_path(sidecar_path)
            rendered = prose_lines(converted)
            facts = None
            for row_full, row_tax, line in zip(full, tax, sidecar.lines):
                prose = prose_for(line, rendered)
                text = encoding_text(line, prose, SURFACE_SCRIPT)
                if not text:
                    skipped["no_text"] += 1
                    continue
                for key in line.provenance:
                    keymap[key_tuple(sidecar.script_file, key)] = text
                existing = rows.get(text)
                if existing is not None:
                    mismatch_max = max(
                        mismatch_max, float(np.abs(existing["e_full"] - row_full).max()),
                    )
                    existing["carriers"].append(sidecar.card)
                    continue
                if facts is None:
                    facts = card_facts(converted)
                rows[text] = {
                    "text": text,
                    "prose": prose or "",
                    "card": sidecar.card,
                    "tree": tree,
                    "line_kind": line.line_kind,
                    "api": line.script_api_type or "(none)",
                    "param_keys": tuple(line.script_param_keys),
                    "has_script": line.script_text is not None,
                    "facts": facts,
                    "carriers": [sidecar.card],
                    "e_full": row_full.astype(np.float32),
                    "e_tax": row_tax.astype(np.float32),
                }
    # Variant keys: only the keymap, for joining records to their text.
    for sidecar_path in sorted(VARIANT_TREE.glob("*.provenance.json")):
        sidecar = read_sidecar(sidecar_path)
        for line in sidecar.lines:
            text = encoding_text(line, None, SURFACE_SCRIPT)
            if text:
                for key in line.provenance:
                    keymap[key_tuple(sidecar.script_file, key)] = text

    table = pd.DataFrame(list(rows.values()))
    table["n_carriers"] = table.carriers.map(lambda c: len(set(c)))
    # CV group: the alphabetically first carrying card, so texts from one
    # card stay on one side of a fold.
    table["group"] = table.carriers.map(lambda c: sorted(set(c))[0])
    table = table.drop(columns=["carriers"])
    table["held_out"] = table.text.isin(held_out).astype(int)
    table["rarity_games"] = table.text.map(lambda t: rarity.get(t, 0))
    table["pairs"] = table.text.map(parse_script)
    table = add_features(table)
    CACHE.mkdir(parents=True, exist_ok=True)
    table.to_pickle(TEXT_TABLE)
    with open(CACHE / "keymap.pkl", "wb") as handle:
        pickle.dump(keymap, handle)
    lines = [
        f"unique texts: {len(table)}",
        f"keymap keys: {len(keymap)}",
        f"skipped: {skipped}",
        f"max |e| disagreement between carriers of one text: {mismatch_max:.2e}",
        f"held-out texts in table: {int(table.held_out.sum())} "
        f"of {len(held_out)} in the manifest",
        f"texts with a rarity entry: {int((table.rarity_games > 0).sum())}",
    ]
    (OUT / "build_texts.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def add_features(table: pd.DataFrame) -> pd.DataFrame:
    feats = []
    for pairs, api, kind, text in zip(table.pairs, table.api, table.line_kind, table.text):
        row: dict[str, object] = {}
        row.update(cost_features(pairs.get("Cost")))
        row["target"] = target_type(pairs)
        row.update(affected_features(pairs, api))
        row.update(count_amounts(pairs, api))
        mode = pairs.get("Mode") or pairs.get("Event") or ""
        row["mode"] = f"{kind}:{mode}" if mode else "(none)"
        row["trigger_etb"] = float(
            kind == "triggered" and pairs.get("Mode") == "ChangesZone"
            and pairs.get("Destination") == "Battlefield"
        )
        row["trigger_dies"] = float(
            kind == "triggered" and pairs.get("Mode") == "ChangesZone"
            and pairs.get("Origin") == "Battlefield"
            and pairs.get("Destination") == "Graveyard"
        )
        lowered = text.lower()
        row["dur_eot"] = float("until end of turn" in lowered
                               or pairs.get("Duration") == "UntilEndOfTurn")
        row["dur_permanent"] = float(pairs.get("Duration") == "Permanent")
        row["script_head"] = text.split("$", 1)[0] if "$" in text else "(keyword)"
        row["keyword_name"] = (
            text.split(":", 1)[0].strip() if api == "Keyword" else "(none)"
        )
        row["n_params"] = float(len(pairs))
        row["len_chars"] = float(len(text))
        row["log_len"] = float(np.log1p(len(text)))
        row["has_subability"] = float("SubAbility" in pairs or "Execute" in pairs)
        row["has_condition"] = float(any(k.startswith("Condition") for k in pairs))
        row["optional"] = float("Optional" in pairs or "OptionalDecider" in pairs)
        row["is_curse"] = float(pairs.get("IsCurse") == "True")
        feats.append(row)
    feats = pd.DataFrame(feats, index=table.index)
    facts = pd.DataFrame(list(table.facts), index=table.index)
    for card_type in CARD_TYPES:
        feats[f"card_{card_type}"] = facts.type_line.str.split().map(
            lambda words, t=card_type: float(t in (words or []))
        )
    feats["card_token"] = (table.tree == "tokenscripts").astype(float)
    feats["card_cmc"] = facts.cmc.astype(float)
    for colour in "WUBRG":
        feats[f"card_colour_{colour}"] = facts[f"colour_{colour}"].astype(float)
    feats["card_n_colours"] = facts.n_colours.astype(float)
    feats["log_carriers"] = np.log1p(table.n_carriers.astype(float))
    feats["log_rarity"] = np.log1p(table.rarity_games.astype(float))
    return pd.concat([table.drop(columns=["facts"]), feats], axis=1)


if __name__ == "__main__":
    main()
