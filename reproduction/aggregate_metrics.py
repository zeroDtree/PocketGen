"""Aggregate PocketGen CrossDocked reproduction metrics into one table."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from reproduction.common import (
    DEFAULT_OUTPUT_DIR,
    load_json,
    load_sample_rows,
    mean_std,
    resolve_repo_path,
    write_json,
)

METRIC_FILES = {
    "affinity": "affinity.json",
    "designability": "designability.json",
    "ligand": "ligand_posebusters.json",
    "interaction": "interaction.json",
    "geometry": "geometry.json",
    "diversity": "diversity.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate reproduction metrics.")
    parser.add_argument("--sample_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--out_json", type=str, default=None)
    return parser.parse_args()


def maybe_load(path: str) -> Optional[Dict[str, Any]]:
    if os.path.exists(path):
        return load_json(path)
    return None


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, dict) and "mean" in value:
        mean = value.get("mean")
        std = value.get("std")
        if mean is None:
            return "-"
        if std is None:
            return f"{mean:.{digits}f}"
        return f"{mean:.{digits}f} +/- {std:.{digits}f}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    args = parse_args()
    sample_dir = resolve_repo_path(args.sample_dir)
    rows = load_sample_rows(sample_dir)
    loaded = {name: maybe_load(os.path.join(sample_dir, filename)) for name, filename in METRIC_FILES.items()}

    aar = mean_std([row.get("aar") for row in rows])
    rmsd = mean_std([row.get("rmsd") for row in rows])
    affinity = (loaded["affinity"] or {}).get("summary", {})
    design = (loaded["designability"] or {}).get("summary", {})
    ligand = (loaded["ligand"] or {}).get("summary", {})
    interaction = (loaded["interaction"] or {}).get("summary", {})
    geometry = loaded["geometry"] or {}
    diversity = (loaded["diversity"] or {}).get("summary", {})

    table = {
        "AAR": aar,
        "RMSD": rmsd,
        "Vina": affinity.get("vina"),
        "Vina_top1": affinity.get("top1_vina"),
        "Vina_top3": affinity.get("top3_vina"),
        "Vina_top5": affinity.get("top5_vina"),
        "Vina_top10": affinity.get("top10_vina"),
        "success_rate_pocket": affinity.get("success_rate_pocket"),
        "success_rate_protein": affinity.get("success_rate_protein"),
        "designability": design.get("designability"),
        "scTM": design.get("scTM"),
        "scRMSD_whole": design.get("scRMSD_whole"),
        "scRMSD_pocket": design.get("scRMSD_pocket"),
        "pLDDT_pocket": design.get("pLDDT_pocket"),
        "delta_scTM": design.get("delta_scTM"),
        "posebusters_pass_rate": ligand.get("pass_rate"),
        "interaction_recovery": interaction.get("interaction_recovery"),
        "diversity": diversity.get("diversity"),
        "geometry_kl": geometry.get("kl_divergence"),
        "n_samples": len(rows),
        "paper_targets": {
            "AAR": "63.40 +/- 1.64%",
            "designability": "0.77 +/- 0.02",
            "Vina": "-7.135 +/- 0.08",
            "success_rate": "~97%",
        },
    }

    lines = [
        "PocketGen CrossDocked reproduction",
        f"samples: {len(rows)}",
        f"AAR: {fmt(aar, 4)}  (paper 0.6340 +/- 0.0164)",
        f"Designability: {fmt(design.get('designability'), 3)}  (paper 0.77 +/- 0.02)",
        f"Vina: {fmt(affinity.get('vina'), 3)}  (paper -7.135 +/- 0.08)",
        f"Vina top-1/3/5/10: {fmt(affinity.get('top1_vina'))} / {fmt(affinity.get('top3_vina'))} / {fmt(affinity.get('top5_vina'))} / {fmt(affinity.get('top10_vina'))}",
        f"Success rate pocket/protein: {fmt(affinity.get('success_rate_pocket'), 3)} / {fmt(affinity.get('success_rate_protein'), 3)}",
        f"scTM / scRMSD_whole / scRMSD_pocket / pLDDT: {fmt(design.get('scTM'))} / {fmt(design.get('scRMSD_whole'))} / {fmt(design.get('scRMSD_pocket'))} / {fmt(design.get('pLDDT_pocket'))}",
        f"Delta scTM: {fmt(design.get('delta_scTM'))}",
        f"PoseBusters pass rate: {fmt(ligand.get('pass_rate'), 3)}",
        f"PLIP interaction recovery: {fmt(interaction.get('interaction_recovery'), 3)}",
        f"Diversity: {fmt(diversity.get('diversity'), 3)}",
        f"Geometry KL: {json.dumps(geometry.get('kl_divergence'), indent=2) if geometry.get('kl_divergence') else '-'}",
    ]
    text = "\n".join(lines)
    print(text)

    out_json = args.out_json or os.path.join(sample_dir, "summary.json")
    write_json(out_json, table)
    out_txt = os.path.splitext(out_json)[0] + ".txt"
    with open(out_txt, "w") as handle:
        handle.write(text + "\n")
    print(f"Wrote {out_json} and {out_txt}")


if __name__ == "__main__":
    main()
