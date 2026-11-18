"""Build the per-unique-text table every probe in this directory reads.

Walks every provenance sidecar under the checkpoint's sidecar trees, joins
each line to its row of the checkpoint's ability cache (``<name>.npz``) and,
where one exists, of the taxonomy baseline's cache (``<name>.taxonomy.npz``),
and keeps one row per unique encoding text on the checkpoint's surface. A
sidecar with no taxonomy cache keeps its rows with ``e_tax`` empty, and the
scripts that read it omit those columns (FR-072). For each text it records
the hand-parsed script features (API type, line kind, trigger/static mode,
parameter-key presence, cost, target, amounts, who is named as affected,
duration, length), the facts of the first card carrying it (type line, mana
value, colours), how many cards carry it, whether the curated corpus holds it
out, and the corpus's rarity count.

It also writes ``cache/keymap.pkl``: every provenance key a line declares
(including the perturbed-variant tree's), a ``ProvenanceKey``, mapped to its
encoding text through the join (``ability_text.texts_by_key``), which
``effect_profiles.py`` uses to name a record's acting text without building a
``SidecarCache`` per worker. A saga's chapter keys, which differ only in
``built``, are one entry each.

It checks, and reports, that every card carrying a text got the same vector
for it: the cache is computed per card, so a disagreement would mean the
encoder is not a function of the text.

Run: ``python scripts/effect_embedding_probes/build_texts.py --checkpoint PATH
--abilities-root DIR`` (CPU, ~2 min).
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache, partial
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from effect_embedding_probes.common import (  # noqa: E402
    CARD_TYPES,
    VARIANT_TREE,
    ProbePaths,
    add_probe_arguments,
    affected_features,
    card_facts,
    cost_features,
    count_amounts,
    load_manifest_sets,
    resolve_paths,
    target_type,
)
from effects.application.ability_text import texts_by_key  # noqa: E402
from effects.domain.baselines import Baseline  # noqa: E402
from effects.domain.provenance import ProvenanceKey  # noqa: E402
from effects.domain.script_text import params, root  # noqa: E402
from effects.infrastructure.ability_cache_store import AbilityCacheStore  # noqa: E402
from effects.infrastructure.sidecar_io import (  # noqa: E402
    converted_text_path,
    encoded_lines,
    iter_sidecars,
    sidecar_roots,
    tree_of_folder,
)


def keymap_of(paths: ProbePaths) -> dict[ProvenanceKey, str]:
    """Every key a line under the checkpoint's trees or the variant tree
    declares, to its encoding text through the join; a line with no text is
    left out, so a record acting through it names none."""
    roots = sidecar_roots(paths.cards_folders, VARIANT_TREE)
    return {key: text for key, text in texts_by_key(roots, paths.surface).items() if text}


def main() -> None:
    parser = add_probe_arguments(argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
    ))
    build(resolve_paths(parser.parse_args()))


@dataclass(frozen=True)
class _SidecarRows:
    """One sidecar's cache rows and the lazily read facts of its card."""

    tree: str
    sidecar: object
    full: np.ndarray
    tax: np.ndarray | None
    facts: Callable[[], dict]


class _TextRows:
    """The per-unique-text rows the sidecars of a run's trees add up to."""

    def __init__(self, paths: ProbePaths) -> None:
        self.paths = paths
        self.rows: dict[str, dict] = {}
        self.mismatch_max = 0.0
        self.skipped = {"no_cache": 0, "row_mismatch": 0, "no_text": 0, "no_taxonomy": 0}
        self._empty = np.zeros(0, dtype=np.float32)
        self._full_cache = AbilityCacheStore(paths.abilities_root)
        self._taxonomy_cache = AbilityCacheStore(paths.abilities_root, variant=Baseline.TAXONOMY)

    def read_trees(self) -> None:
        """Add every sidecar of the run's trees."""
        for folder in self.paths.cards_folders:
            tree = tree_of_folder(folder)
            for sidecar_path, sidecar in iter_sidecars(folder):
                self._add_sidecar(tree, sidecar_path, sidecar)

    def _cache_rows(self, sidecar) -> tuple[np.ndarray, np.ndarray | None] | None:
        """The sidecar's full and taxonomy cache rows, None when it is skipped."""
        full = self._full_cache.read(sidecar.script_file)
        if full is None:
            self.skipped["no_cache"] += 1
            return None
        tax = self._taxonomy_cache.read(sidecar.script_file)
        if tax is None:
            self.skipped["no_taxonomy"] += 1
        if not _aligned(full, tax, len(sidecar.lines)):
            self.skipped["row_mismatch"] += 1
            return None
        return full, tax

    def _add_sidecar(self, tree: str, sidecar_path: Path, sidecar) -> None:
        cache_rows = self._cache_rows(sidecar)
        if cache_rows is None:
            return
        converted = converted_text_path(sidecar_path)
        rows = _SidecarRows(tree, sidecar, *cache_rows, cache(partial(card_facts, converted)))
        encoded = encoded_lines(sidecar_path, sidecar, self.paths.surface)
        for index, entry in enumerate(encoded):
            if entry[2]:
                self._add_line(rows, index, entry)
            else:
                self.skipped["no_text"] += 1

    def _add_line(self, rows: _SidecarRows, index: int, entry: tuple) -> None:
        text = entry[2]
        row_full = rows.full[index]
        existing = self.rows.get(text)
        if existing is None:
            self.rows[text] = self._new_row(rows, index, entry)
            return
        self.mismatch_max = max(
            self.mismatch_max, float(np.abs(existing["e_full"] - row_full).max()),
        )
        existing["carriers"].append(rows.sidecar.card)

    def _new_row(self, rows: _SidecarRows, index: int, entry: tuple) -> dict:
        line, prose, text = entry
        row_tax = self._empty if rows.tax is None else rows.tax[index]
        return {
            "text": text,
            "prose": prose or "",
            "card": rows.sidecar.card,
            "tree": rows.tree,
            "line_kind": line.line_kind,
            "api": line.script_api_type or "(none)",
            "param_keys": tuple(line.script_param_keys),
            "has_script": line.script_text is not None,
            "facts": rows.facts(),
            "carriers": [rows.sidecar.card],
            "e_full": rows.full[index].astype(np.float32),
            "e_tax": row_tax.astype(np.float32),
        }


def _aligned(full: np.ndarray, tax: np.ndarray | None, n_lines: int) -> bool:
    """Whether each cache holds one row per sidecar line."""
    return len(full) == n_lines and (tax is None or len(tax) == n_lines)


def build(paths: ProbePaths) -> pd.DataFrame:
    """Write the text table and keymap for ``paths``'s checkpoint, and return it."""
    held_out, rarity = load_manifest_sets(paths)
    texts = _TextRows(paths)
    texts.read_trees()
    keymap = keymap_of(paths)
    table = _finished_table(pd.DataFrame(list(texts.rows.values())), held_out, rarity)
    paths.cache.mkdir(parents=True, exist_ok=True)
    table.to_pickle(paths.text_table)
    with open(paths.keymap, "wb") as handle:
        pickle.dump(keymap, handle)
    lines = [
        f"checkpoint: {paths.checkpoint} (surface {paths.surface}, "
        f"e width {paths.e_width})",
        f"sidecar trees: {', '.join(str(f) for f in paths.cards_folders)}",
        f"unique texts: {len(table)}",
        f"keymap keys: {len(keymap)}",
        f"skipped: {texts.skipped}",
        f"max |e| disagreement between carriers of one text: {texts.mismatch_max:.2e}",
        f"held-out texts in table: {int(table.held_out.sum())} "
        f"of {len(held_out)} in the manifest",
        f"texts with a rarity entry: {int((table.rarity_games > 0).sum())}",
    ]
    (paths.out / "build_texts.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return table


def _finished_table(table: pd.DataFrame, held_out: set[str], rarity: dict[str, int],
                    ) -> pd.DataFrame:
    """The rows with their carrier counts, CV group, corpus columns and features."""
    table["n_carriers"] = table.carriers.map(lambda c: len(set(c)))
    # CV group: the alphabetically first carrying card, so texts from one
    # card stay on one side of a fold.
    table["group"] = table.carriers.map(lambda c: sorted(set(c))[0])
    table = table.drop(columns=["carriers"])
    table["held_out"] = table.text.isin(held_out).astype(int)
    table["rarity_games"] = table.text.map(lambda t: rarity.get(t, 0))
    # The root segment's parameters only: a chained text's later [SEG]
    # segments are its sub-abilities, not the trait's own line.
    table["pairs"] = table.text.map(lambda text: params(root(text)))
    return add_features(table)


def _mode_features(pairs: dict[str, str], kind: str) -> dict[str, object]:
    """The trigger or static mode, and the enters-the-battlefield and dies triggers."""
    mode = pairs.get("Mode") or pairs.get("Event") or ""
    zone_trigger = kind == "triggered" and pairs.get("Mode") == "ChangesZone"
    zone_route = (pairs.get("Origin"), pairs.get("Destination"))
    return {
        "mode": f"{kind}:{mode}" if mode else "(none)",
        "trigger_etb": float(zone_trigger and zone_route[1] == "Battlefield"),
        "trigger_dies": float(zone_trigger and zone_route == ("Battlefield", "Graveyard")),
    }


def _text_features(pairs: dict[str, str], api: str, text: str) -> dict[str, float | str]:
    """Duration, head, length and parameter-key presence of a script text."""
    return {
        "dur_eot": float("until end of turn" in text.lower()
                         or pairs.get("Duration") == "UntilEndOfTurn"),
        "dur_permanent": float(pairs.get("Duration") == "Permanent"),
        "script_head": text.split("$", 1)[0] if "$" in text else "(keyword)",
        "keyword_name": text.split(":", 1)[0].strip() if api == "Keyword" else "(none)",
        "n_params": float(len(pairs)),
        "len_chars": float(len(text)),
        "log_len": float(np.log1p(len(text))),
        "has_subability": float("SubAbility" in pairs or "Execute" in pairs),
        "has_condition": float(any(k.startswith("Condition") for k in pairs)),
        "optional": float("Optional" in pairs or "OptionalDecider" in pairs),
        "is_curse": float(pairs.get("IsCurse") == "True"),
    }


def _script_features(pairs: dict[str, str], api: str, kind: str, text: str) -> dict:
    """One text's hand-parsed script features."""
    row: dict[str, object] = {}
    row.update(cost_features(pairs.get("Cost")))
    row["target"] = target_type(pairs)
    row.update(affected_features(pairs, api))
    row.update(count_amounts(pairs, api))
    row.update(_mode_features(pairs, kind))
    row.update(_text_features(pairs, api, text))
    return row


def add_features(table: pd.DataFrame) -> pd.DataFrame:
    feats = [
        _script_features(pairs, api, kind, text)
        for pairs, api, kind, text in zip(table.pairs, table.api, table.line_kind, table.text)
    ]
    feats = pd.DataFrame(feats, index=table.index)
    return pd.concat([table.drop(columns=["facts"]), _add_card_features(table, feats)], axis=1)


def _add_card_features(table: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """``feats`` with the card-fact, token and corpus columns the table supplies."""
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
    return feats


if __name__ == "__main__":
    from effect_embedding_probes.common import main_with_manifest

    main_with_manifest(main, __file__)
