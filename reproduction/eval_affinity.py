"""AutoDock Vina affinity evaluation for PocketGen samples."""

from __future__ import annotations

import argparse
import json
import os
import sys
from functools import partial
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from reproduction.common import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TMP_DIR,
    append_jsonl,
    ensure_dir,
    ensure_repo_on_path,
    group_by_complex,
    load_jsonl,
    mean_std,
    resolve_repo_path,
    samples_path,
    topk_mean,
    write_json,
)

ensure_repo_on_path()

if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]

from openbabel import pybel  # noqa: E402
from rdkit import Chem  # noqa: E402
from vina import Vina  # noqa: E402

from utils.evaluation.docking_vina import PrepLig, PrepProt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Vina scores for PocketGen samples.")
    parser.add_argument("--sample_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tmp_dir", type=str, default=DEFAULT_TMP_DIR)
    parser.add_argument("--out_json", type=str, default=None)
    parser.add_argument("--exhaustiveness", type=int, default=64)
    parser.add_argument("--n_poses", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def convert_pdbqt_to_sdf(pdbqt_file: str, sdf_file: str) -> None:
    mol = next(pybel.readfile("pdbqt", pdbqt_file))
    mol.removeh()
    mol.write("sdf", sdf_file, overwrite=True)


def calculate_vina_score(
    protein_pdb: str,
    ligand_sdf: str,
    tmp_prefix: str,
    exhaustiveness: int = 64,
    n_poses: int = 30,
    write_pose: bool = False,
) -> Optional[float]:
    size_factor = 1.2
    buffer = 8.0
    mol = Chem.MolFromMolFile(ligand_sdf, sanitize=False)
    if mol is None or mol.GetNumConformers() == 0:
        return None
    pos = mol.GetConformer(0).GetPositions()
    center = np.mean(pos, 0)
    ligand_pdbqt = tmp_prefix + "_lig.pdbqt"
    protein_pqr = tmp_prefix + "_pro.pqr"
    protein_pdbqt = tmp_prefix + "_pro.pdbqt"
    try:
        lig = PrepLig(ligand_sdf, "sdf")
        lig.addH()
        lig.get_pdbqt(ligand_pdbqt)
        prot = PrepProt(protein_pdb)
        prot.addH(protein_pqr)
        prot.get_pdbqt(protein_pdbqt)
        v = Vina(sf_name="vina", seed=0, verbosity=0)
        v.set_receptor(protein_pdbqt)
        v.set_ligand_from_file(ligand_pdbqt)
        box = (pos.max(0) - pos.min(0)) * size_factor + buffer
        v.compute_vina_maps(center=center, box_size=[float(box[0]), float(box[1]), float(box[2])])
        v.score()
        v.optimize()
        v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
        score = float(v.energies(n_poses=1)[0][0])
        if write_pose:
            pose_pdbqt = protein_pdb[:-4] + "_docked.pdbqt"
            v.write_poses(pose_pdbqt, n_poses=1, overwrite=True)
            convert_pdbqt_to_sdf(pose_pdbqt, protein_pdb[:-4] + "_docked.sdf")
        return score
    except Exception as exc:
        print(f"Vina failed for {protein_pdb} / {ligand_sdf}: {exc}")
        return None


def _eval_one(task: Dict[str, Any], tmp_dir: str, exhaustiveness: int, n_poses: int) -> Dict[str, Any]:
    work = ensure_dir(os.path.join(tmp_dir, task["tmp_name"]))
    tmp_prefix = os.path.join(work, "vina")
    old_cwd = os.getcwd()
    os.chdir(work)
    try:
        score = calculate_vina_score(
            task["protein_pdb"],
            task["ligand_sdf"],
            tmp_prefix=tmp_prefix,
            exhaustiveness=exhaustiveness,
            n_poses=n_poses,
            write_pose=task.get("write_pose", False),
        )
    finally:
        os.chdir(old_cwd)
    return {"key": task["key"], "vina": score}


def scores_from_affinity(payload: Dict[str, Any]) -> Dict[str, Optional[float]]:
    scores: Dict[str, Optional[float]] = {}
    for item in payload.get("per_sample") or []:
        cid = item["complex_id"]
        sid = item["sample_id"]
        scores[f"gen::{cid}::{sid}"] = item.get("vina")
        if item.get("ref_vina") is not None:
            scores[f"ref::{cid}"] = item.get("ref_vina")
    for item in payload.get("per_complex") or []:
        if not isinstance(item, dict):
            continue
        cid = item.get("complex_id")
        if cid is not None and item.get("ref_vina") is not None:
            scores[f"ref::{cid}"] = item["ref_vina"]
    return scores


def scores_from_jsonl(path: str) -> Dict[str, Optional[float]]:
    scores: Dict[str, Optional[float]] = {}
    for item in load_jsonl(path):
        key = item.get("key")
        if key:
            scores[key] = item.get("vina")
    return scores


def main() -> None:
    args = parse_args()
    sample_dir = resolve_repo_path(args.sample_dir)
    tmp_dir = ensure_dir(resolve_repo_path(args.tmp_dir))
    rows = load_jsonl(samples_path(sample_dir))
    if not rows:
        raise SystemExit(f"No samples found in {samples_path(sample_dir)}")

    existing_path = os.path.join(sample_dir, "affinity.json")
    jsonl_path = os.path.join(sample_dir, "vina_scores.jsonl")
    scores: Dict[str, Optional[float]] = {}
    if os.path.exists(existing_path):
        with open(existing_path) as handle:
            scores.update(scores_from_affinity(json.load(handle)))
    scores.update(scores_from_jsonl(jsonl_path))

    tasks: List[Dict[str, Any]] = []
    ref_done = set()
    for row in rows:
        cid = row["complex_id"]
        sid = int(row["sample_id"])
        if cid not in ref_done:
            ref_done.add(cid)
            tasks.append(
                {
                    "key": f"ref::{cid}",
                    "tmp_name": f"ref_{cid}",
                    "protein_pdb": row["ref_pocket_pdb"],
                    "ligand_sdf": row["ref_ligand_sdf"],
                    "write_pose": False,
                }
            )
        tasks.append(
            {
                "key": f"gen::{cid}::{sid}",
                "tmp_name": f"gen_{cid}_{sid}",
                "protein_pdb": row["paths"]["pocket_relaxed"],
                "ligand_sdf": row["paths"]["ligand_sdf"],
                "write_pose": True,
            }
        )

    if args.skip_existing:
        n_before = len(tasks)
        tasks = [t for t in tasks if t["key"] not in scores]
        print(f"Vina: skipping {n_before - len(tasks)} scored tasks, {len(tasks)} remaining")

    worker = partial(_eval_one, tmp_dir=tmp_dir, exhaustiveness=args.exhaustiveness, n_poses=args.n_poses)
    if args.num_workers <= 1:
        for task in tqdm(tasks, desc="Vina"):
            result = worker(task)
            scores[result["key"]] = result["vina"]
            append_jsonl(jsonl_path, result)
    else:
        import multiprocessing as mp

        with mp.Pool(args.num_workers) as pool:
            for result in tqdm(pool.imap_unordered(worker, tasks), total=len(tasks), desc="Vina"):
                scores[result["key"]] = result["vina"]
                append_jsonl(jsonl_path, result)

    grouped = group_by_complex(rows)
    per_sample = []
    per_complex = []
    protein_success = []
    pocket_success_flags = []
    all_vina = []
    top1, top3, top5, top10 = [], [], [], []

    for cid, items in grouped.items():
        ref_score = scores.get(f"ref::{cid}")
        gen_scores = []
        for item in items:
            sid = int(item["sample_id"])
            vina = scores.get(f"gen::{cid}::{sid}")
            better = None
            if vina is not None and ref_score is not None:
                better = vina < ref_score
                pocket_success_flags.append(float(better))
            if vina is not None:
                gen_scores.append(vina)
                all_vina.append(vina)
            per_sample.append(
                {
                    "complex_id": cid,
                    "sample_id": sid,
                    "vina": vina,
                    "ref_vina": ref_score,
                    "better_than_ref": better,
                    "aar": item.get("aar"),
                    "rmsd": item.get("rmsd"),
                }
            )
        if gen_scores:
            best = min(gen_scores)
            protein_success.append(float(ref_score is not None and best < ref_score))
            top1.append(topk_mean(gen_scores, 1))
            top3.append(topk_mean(gen_scores, 3))
            top5.append(topk_mean(gen_scores, 5))
            top10.append(topk_mean(gen_scores, 10))
        per_complex.append(
            {
                "complex_id": cid,
                "ref_vina": ref_score,
                "n_scored": len(gen_scores),
                "mean_vina": float(np.mean(gen_scores)) if gen_scores else None,
                "best_vina": float(min(gen_scores)) if gen_scores else None,
            }
        )

    summary = {
        "vina": mean_std(all_vina),
        "ref_vina": mean_std([c["ref_vina"] for c in per_complex]),
        "top1_vina": mean_std(top1),
        "top3_vina": mean_std(top3),
        "top5_vina": mean_std(top5),
        "top10_vina": mean_std(top10),
        "success_rate_pocket": mean_std(pocket_success_flags)["mean"],
        "success_rate_protein": mean_std(protein_success)["mean"],
        "n_samples_scored": len(all_vina),
        "n_complexes": len(per_complex),
    }
    payload = {"summary": summary, "per_complex": per_complex, "per_sample": per_sample}
    out_json = args.out_json or os.path.join(sample_dir, "affinity.json")
    write_json(out_json, payload)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
