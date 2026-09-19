"""Nearest neighbours of well-known abilities, and what k-means clusters hold.

Two readings of the shipping embedding that need no feature table:

1. **Neighbours.** For each ability line of a handful of well-known cards
   (Lightning Bolt, Llanowar Elves, Giant Growth, Wrath of God, Divination,
   Pacifism, Counterspell, Glorious Anthem, Serra Angel, Murder), the eight
   nearest other unique texts by cosine similarity of the **centred** vectors.
   Centred because the mean vector is large (about 40% of the average norm),
   so raw cosine mostly measures closeness to the mean.
2. **Clusters.** k-means with k = 40 on the full space, the residual space
   (API-type means removed) and the taxonomy baseline. For each full-space
   cluster, its size, its three most common API types with their shares, and
   the prose of the three texts nearest its centre. For each space, the
   normalized mutual information (NMI, 0 = independent, 1 = identical
   partitions) between the clusters and the API type, line kind and target
   type.

Run: ``python scripts/effect_embedding_probes/neighbours.py`` (CPU, ~1 min).
Writes ``neighbours.md`` and ``clusters.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    OUT,
    ROOT,
    SEED,
    group_means_residual,
    load_texts,
    matrix,
    to_markdown,
    write_markdown,
)

from effects.domain.ability_encoder import SURFACE_SCRIPT, encoding_text  # noqa: E402
from effects.infrastructure.sidecar_io import (  # noqa: E402
    converted_text_path,
    prose_for,
    prose_lines,
    read_sidecar,
)

CARDS = ("lightning bolt", "llanowar elves", "giant growth", "wrath of god",
         "divination", "pacifism", "counterspell", "glorious anthem",
         "serra angel", "murder")
K = 40


def card_texts(card: str) -> list[str]:
    """The script-surface texts of one card, read from its own sidecar."""
    stem = card.replace(" ", "_").replace("'", "")
    path = ROOT / "output" / "cardsfolder" / stem[0] / f"{stem}.provenance.json"
    sidecar = read_sidecar(path)
    rendered = prose_lines(converted_text_path(path))
    return [encoding_text(line, prose_for(line, rendered), SURFACE_SCRIPT)
            for line in sidecar.lines]


def unit(E: np.ndarray) -> np.ndarray:
    centred = E - E.mean(axis=0)
    return centred / np.linalg.norm(centred, axis=1, keepdims=True)


def main() -> None:
    table = load_texts()
    full = matrix(table, "e_full")
    U = unit(full)
    lines = ["# Nearest neighbours of well-known abilities (centred cosine)", ""]
    index = {text: i for i, text in enumerate(table.text)}
    for card in CARDS:
        rows = [index[t] for t in card_texts(card) if t in index]
        if not rows:
            lines.append(f"*{card}: no encoded line*\n")
            continue
        for i in rows:
            lines.append(f"## {card} — `{table.text.iloc[i][:120]}`")
            lines.append(f"API {table.api.iloc[i]}, prose: {table.prose.iloc[i][:100]}\n")
            sims = U @ U[i]
            sims[i] = -np.inf
            lines.append("| cosine | card | API | prose |")
            lines.append("|---:|---|---|---|")
            for j in np.argsort(sims)[::-1][:8]:
                prose = (table.prose.iloc[j] or table.text.iloc[j])[:100]
                lines.append(f"| {sims[j]:.3f} | {table.card.iloc[j]} | "
                             f"{table.api.iloc[j]} | {prose.replace('|', '/')} |")
            lines.append("")
    (OUT / "neighbours.md").write_text("\n".join(lines), encoding="utf-8")

    spaces = {
        "full": full,
        "residual": group_means_residual(full, table["api"]),
        "taxonomy": matrix(table, "e_tax"),
    }
    nmi_rows = []
    labels = {}
    for name, E in spaces.items():
        km = KMeans(n_clusters=K, n_init=4, random_state=SEED).fit(E)
        labels[name] = km.labels_
        nmi_rows.append({
            "space": name,
            "NMI with API type": normalized_mutual_info_score(table.api, km.labels_),
            "NMI with line kind": normalized_mutual_info_score(table.line_kind, km.labels_),
            "NMI with target type": normalized_mutual_info_score(table.target, km.labels_),
        })
        if name == "full":
            centres = km.cluster_centers_
    nmi = pd.DataFrame(nmi_rows)
    out = ["# k-means clusters (k = 40)", "", "## Agreement with script categories",
           "", "Normalized mutual information between the cluster labels and "
           "each category.", ""]
    out.append(to_markdown(nmi))
    out += ["", "## Full-space clusters", "",
            "| cluster | texts | top API types (share) | texts nearest the centre |",
            "|---:|---:|---|---|"]
    label = labels["full"]
    sizes = pd.Series(label).value_counts()
    for c in sizes.index:
        members = np.flatnonzero(label == c)
        apis = table.api.iloc[members].value_counts(normalize=True).head(3)
        dist = np.linalg.norm(full[members] - centres[c], axis=1)
        near = members[np.argsort(dist)[:3]]
        examples = " // ".join(
            (table.prose.iloc[i] or table.text.iloc[i])[:70].replace("|", "/")
            for i in near)
        api_text = ", ".join(f"{a} {s:.0%}" for a, s in apis.items())
        out.append(f"| {c} | {len(members)} | {api_text} | {examples} |")
    (OUT / "clusters.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    write_markdown(nmi, OUT / "clusters_nmi.md", "Cluster / category agreement")
    print(nmi.to_string(index=False))


if __name__ == "__main__":
    main()
