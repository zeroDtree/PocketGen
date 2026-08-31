"""Aggregate PocketGen CrossDocked reproduction metrics into one table."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Set

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from reproduction.common import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    load_json,
    load_sample_rows,
    mean_std,
    resolve_repo_path,
    write_json,
)

CANONICAL_METRICS = (
    "aar",
    "rmsd",
    "affinity",
    "designability",
    "ligand",
    "interaction",
    "geometry",
    "diversity",
)
METRIC_ALIASES = {
    "vina": "affinity",
    "posebusters": "ligand",
}
METRIC_FILES = {
    "affinity": "affinity.json",
    "designability": "designability.json",
    "ligand": "ligand_posebusters.json",
    "interaction": "interaction.json",
    "geometry": "geometry.json",
    "diversity": "diversity.json",
}
PAPER_TARGETS = {
    "AAR": "63.40 +/- 1.64%",
    "designability": "0.77 +/- 0.02",
    "Vina": "-7.135 +/- 0.08",
    "success_rate": "~97%",
}


def parse_metrics(raw: Optional[Sequence[str]]) -> List[str]:
    if not raw:
        return list(CANONICAL_METRICS)
    seen: Set[str] = set()
    unknown: List[str] = []
    for token in raw:
        for part in token.split(","):
            name = part.strip().lower()
            if not name:
                continue
            canonical = METRIC_ALIASES.get(name, name)
            if canonical not in CANONICAL_METRICS:
                unknown.append(part.strip())
                continue
            seen.add(canonical)
    if unknown:
        valid = ", ".join(list(CANONICAL_METRICS) + list(METRIC_ALIASES))
        raise argparse.ArgumentTypeError(
            f"Unknown metric(s): {', '.join(unknown)}. Choose from: {valid}"
        )
    if not seen:
        return list(CANONICAL_METRICS)
    return [name for name in CANONICAL_METRICS if name in seen]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate reproduction metrics.")
    parser.add_argument("--sample_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--out_json", type=str, default=None)
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        metavar="NAME",
        help=(
            "Metric groups to report (comma- or space-separated). "
            "Default: all. Names: aar, rmsd, affinity (alias vina), "
            "designability, ligand (alias posebusters), interaction, "
            "geometry, diversity. Example: --metrics aar,vina"
        ),
    )
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


def summary_of(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not payload:
        return {}
    summary = payload.get("summary")
    if isinstance(summary, dict):
        return summary
    return {}


def main() -> None:
    args = parse_args()
    try:
        selected = set(parse_metrics(args.metrics))
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(str(exc)) from exc
    sample_dir = resolve_repo_path(args.sample_dir)

    rows: List[Dict[str, Any]] = []
    aar: Optional[Dict[str, Optional[float]]] = None
    rmsd: Optional[Dict[str, Optional[float]]] = None
    if "aar" in selected or "rmsd" in selected:
        rows = load_sample_rows(sample_dir, progress=True)
        if "aar" in selected:
            aar = mean_std([row.get("aar") for row in rows])
        if "rmsd" in selected:
            rmsd = mean_std([row.get("rmsd") for row in rows])

    loaded: Dict[str, Optional[Dict[str, Any]]] = {}
    for name in CANONICAL_METRICS:
        filename = METRIC_FILES.get(name)
        if filename is None or name not in selected:
            continue
        loaded[name] = maybe_load(os.path.join(sample_dir, filename))

    affinity = summary_of(loaded.get("affinity"))
    design = summary_of(loaded.get("designability"))
    ligand = summary_of(loaded.get("ligand"))
    interaction = summary_of(loaded.get("interaction"))
    geometry = loaded.get("geometry") or {}
    diversity = summary_of(loaded.get("diversity"))

    table: Dict[str, Any] = {}
    if "aar" in selected:
        table["AAR"] = aar
    if "rmsd" in selected:
        table["RMSD"] = rmsd
    if "affinity" in selected:
        table["Vina"] = affinity.get("vina")
        table["Vina_top1"] = affinity.get("top1_vina")
        table["Vina_top3"] = affinity.get("top3_vina")
        table["Vina_top5"] = affinity.get("top5_vina")
        table["Vina_top10"] = affinity.get("top10_vina")
        table["success_rate_pocket"] = affinity.get("success_rate_pocket")
        table["success_rate_protein"] = affinity.get("success_rate_protein")
        table["n_samples_scored"] = affinity.get("n_samples_scored")
        table["n_complexes"] = affinity.get("n_complexes")
    if "designability" in selected:
        table["designability"] = design.get("designability")
        table["scTM"] = design.get("scTM")
        table["scRMSD_whole"] = design.get("scRMSD_whole")
        table["scRMSD_pocket"] = design.get("scRMSD_pocket")
        table["pLDDT_pocket"] = design.get("pLDDT_pocket")
        table["delta_scTM"] = design.get("delta_scTM")
    if "ligand" in selected:
        table["posebusters_pass_rate"] = ligand.get("pass_rate")
    if "interaction" in selected:
        table["interaction_recovery"] = interaction.get("interaction_recovery")
    if "diversity" in selected:
        table["diversity"] = diversity.get("diversity")
    if "geometry" in selected:
        table["geometry_kl"] = geometry.get("kl_divergence")
    if rows:
        table["n_samples"] = len(rows)

    paper_targets: Dict[str, str] = {}
    if "aar" in selected:
        paper_targets["AAR"] = PAPER_TARGETS["AAR"]
    if "designability" in selected:
        paper_targets["designability"] = PAPER_TARGETS["designability"]
    if "affinity" in selected:
        paper_targets["Vina"] = PAPER_TARGETS["Vina"]
        paper_targets["success_rate"] = PAPER_TARGETS["success_rate"]
    if paper_targets:
        table["paper_targets"] = paper_targets

    lines = ["PocketGen CrossDocked reproduction"]
    if rows:
        lines.append(f"samples: {len(rows)}")
    if "aar" in selected:
        lines.append(f"AAR: {fmt(aar, 4)}  (paper 0.6340 +/- 0.0164)")
    if "rmsd" in selected:
        lines.append(f"RMSD: {fmt(rmsd, 4)}")
    if "designability" in selected:
        lines.append(
            f"Designability: {fmt(design.get('designability'), 3)}  (paper 0.77 +/- 0.02)"
        )
    if "affinity" in selected:
        lines.append(f"Vina: {fmt(affinity.get('vina'), 3)}  (paper -7.135 +/- 0.08)")
        lines.append(
            "Vina top-1/3/5/10: "
            f"{fmt(affinity.get('top1_vina'))} / {fmt(affinity.get('top3_vina'))} / "
            f"{fmt(affinity.get('top5_vina'))} / {fmt(affinity.get('top10_vina'))}"
        )
        lines.append(
            "Success rate pocket/protein: "
            f"{fmt(affinity.get('success_rate_pocket'), 3)} / "
            f"{fmt(affinity.get('success_rate_protein'), 3)}"
        )
        lines.append(
            "Vina scored samples/complexes: "
            f"{affinity.get('n_samples_scored', '-')} / {affinity.get('n_complexes', '-')}"
        )
    if "designability" in selected:
        lines.append(
            "scTM / scRMSD_whole / scRMSD_pocket / pLDDT: "
            f"{fmt(design.get('scTM'))} / {fmt(design.get('scRMSD_whole'))} / "
            f"{fmt(design.get('scRMSD_pocket'))} / {fmt(design.get('pLDDT_pocket'))}"
        )
        lines.append(f"Delta scTM: {fmt(design.get('delta_scTM'))}")
    if "ligand" in selected:
        lines.append(f"PoseBusters pass rate: {fmt(ligand.get('pass_rate'), 3)}")
    if "interaction" in selected:
        lines.append(
            f"PLIP interaction recovery: {fmt(interaction.get('interaction_recovery'), 3)}"
        )
    if "diversity" in selected:
        lines.append(f"Diversity: {fmt(diversity.get('diversity'), 3)}")
    if "geometry" in selected:
        kl = geometry.get("kl_divergence")
        lines.append(
            f"Geometry KL: {json.dumps(kl, indent=2) if kl else '-'}"
        )
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
