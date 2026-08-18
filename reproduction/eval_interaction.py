"""PLIP protein-ligand interaction analysis for generated pockets."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import Any, Dict, List, Optional

from tqdm import tqdm

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from reproduction.common import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TMP_DIR,
    ensure_dir,
    ensure_repo_on_path,
    load_jsonl,
    mean_std,
    resolve_repo_path,
    samples_path,
    write_json,
)

ensure_repo_on_path()

from evaluation.protein_ligand_interaction import (  # noqa: E402
    merge_lig_pkt,
    patter_analysis,
    plip_analysis,
    plip_parser,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PLIP interaction analysis on generated pockets.")
    parser.add_argument("--sample_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tmp_dir", type=str, default=os.path.join(DEFAULT_TMP_DIR, "plip"))
    parser.add_argument("--out_json", type=str, default=None)
    return parser.parse_args()


def run_plip(protein_pdb: str, ligand_sdf: str, work_dir: str) -> Optional[Dict[str, int]]:
    ensure_dir(work_dir)
    merged = os.path.join(work_dir, "merged.pdb")
    try:
        merge_lig_pkt(protein_pdb, ligand_sdf, merged)
        xml_path = plip_analysis(merged, work_dir)
        if not os.path.exists(xml_path):
            return None
        return plip_parser(xml_path)
    except Exception as exc:
        print(f"PLIP failed for {protein_pdb}: {exc}")
        return None
    finally:
        if os.path.isdir(work_dir):
            for name in os.listdir(work_dir):
                path = os.path.join(work_dir, name)
                if name.startswith("plip") or name.endswith(".xml") or name == "merged.pdb":
                    if os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
                    elif os.path.exists(path):
                        os.remove(path)


def main() -> None:
    args = parse_args()
    if shutil.which("plip") is None:
        raise SystemExit("plip is not on PATH. Install it in the pocketgen env with: conda install -c conda-forge plip")
    sample_dir = resolve_repo_path(args.sample_dir)
    tmp_root = ensure_dir(resolve_repo_path(args.tmp_dir))
    rows = load_jsonl(samples_path(sample_dir))
    if not rows:
        raise SystemExit(f"No samples found in {samples_path(sample_dir)}")

    ref_cache: Dict[str, Optional[Dict[str, int]]] = {}
    per_sample: List[Dict[str, Any]] = []
    recovery_scores: List[float] = []
    gen_totals: List[float] = []
    ref_totals: List[float] = []

    for row in tqdm(rows, desc="PLIP"):
        cid = row["complex_id"]
        if cid not in ref_cache:
            ref_dir = ensure_dir(os.path.join(tmp_root, f"ref_{cid}"))
            ref_cache[cid] = run_plip(row["ref_pocket_pdb"], row["ref_ligand_sdf"], ref_dir)
        gen_dir = ensure_dir(os.path.join(tmp_root, f"gen_{cid}_{row['sample_id']}"))
        protein = row["paths"]["pocket_relaxed"]
        ligand = row["paths"]["ligand_sdf"]
        gen_report = run_plip(protein, ligand, gen_dir) if os.path.exists(protein) and os.path.exists(ligand) else None
        compare = None
        num_ori = None
        num_gen = None
        if ref_cache[cid] is not None and gen_report is not None:
            compare, num_ori, num_gen = patter_analysis(ref_cache[cid], gen_report)
            ratios = [v for v in compare.values() if isinstance(v, (int, float))]
            if ratios:
                recovery_scores.append(float(sum(ratios) / len(ratios)))
            if num_gen is not None:
                gen_totals.append(float(num_gen))
            if num_ori is not None:
                ref_totals.append(float(num_ori))
        record = {
            "complex_id": cid,
            "sample_id": row["sample_id"],
            "generated": gen_report,
            "reference": ref_cache[cid],
            "compare": compare,
            "num_ori": num_ori,
            "num_gen": num_gen,
        }
        per_sample.append(record)

    summary = {
        "interaction_recovery": mean_std(recovery_scores),
        "generated_interaction_count": mean_std(gen_totals),
        "reference_interaction_count": mean_std(ref_totals),
        "n": len(per_sample),
    }
    payload = {"summary": summary, "per_sample": per_sample}
    out_json = args.out_json or os.path.join(sample_dir, "interaction.json")
    write_json(out_json, payload)
    import json

    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
